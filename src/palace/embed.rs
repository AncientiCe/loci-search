//! MiniLM encoder client. In-process ort 2.0.0-rc.15 does not reproduce the
//! archived cosines (a sample rounded to 0.167 against a dumped 0.163), so
//! inference runs in `encoder/minilm-rc11`, pinned to the ort rc.11 that
//! produced the dump. One text is sent per ONNX call. Workers are processes.

use anyhow::{Context, Result, anyhow, ensure};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const DIMENSIONS: usize = 384;
const RECORD: usize = DIMENSIONS * 4;

pub fn pinned_minilm() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(
        r"C:\Users\ancie\.cache\huggingface\hub\models--Xenova--all-MiniLM-L6-v2\snapshots\751bff37182d3f1213fa05d7196b954e230abad9",
    );
    (
        root.join("onnx").join("model_quantized.onnx"),
        root.join("vocab.txt"),
    )
}

pub fn sidecar_executable() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("encoder")
        .join("minilm-rc11")
        .join("target")
        .join("release")
        .join("minilm-rc11.exe")
}

pub struct MiniLm {
    model: PathBuf,
    vocab: PathBuf,
    workers: usize,
}

impl MiniLm {
    pub fn open(model: &Path, vocab: &Path, workers: usize) -> Result<Self> {
        ensure!(
            sidecar_executable().is_file(),
            "build the rc.11 encoder first: cargo build --release --manifest-path encoder/minilm-rc11/Cargo.toml"
        );
        ensure!(
            model.is_file() && vocab.is_file(),
            "MiniLM model files are missing"
        );
        Ok(Self {
            model: model.to_path_buf(),
            vocab: vocab.to_path_buf(),
            workers: workers.max(1),
        })
    }

    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let workers = self.workers.min(texts.len());
        let slots = Mutex::new(vec![None; texts.len()]);
        let next = AtomicUsize::new(0);
        let error = Mutex::new(None::<String>);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let model = self.model.clone();
                let vocab = self.vocab.clone();
                let next = &next;
                let error = &error;
                let slots = &slots;
                scope.spawn(move || {
                    let mut worker = match Worker::spawn(&model, &vocab) {
                        Ok(worker) => worker,
                        Err(err) => {
                            *error.lock().expect("error lock") = Some(err.to_string());
                            return;
                        }
                    };
                    loop {
                        if error.lock().expect("error lock").is_some() {
                            return;
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= texts.len() {
                            break;
                        }
                        match worker.embed(texts[index]) {
                            Ok(vector) => slots.lock().expect("slot lock")[index] = Some(vector),
                            Err(err) => {
                                *error.lock().expect("error lock") = Some(err.to_string());
                                return;
                            }
                        }
                    }
                    let _ = worker.shutdown();
                });
            }
        });
        if let Some(err) = error.lock().expect("error lock").take() {
            return Err(anyhow!(err));
        }
        slots
            .into_inner()
            .expect("slot lock")
            .into_iter()
            .enumerate()
            .map(|(index, vector)| vector.ok_or_else(|| anyhow!("missing embedding {index}")))
            .collect()
    }
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Worker {
    fn spawn(model: &Path, vocab: &Path) -> Result<Self> {
        let mut child = Command::new(sidecar_executable())
            .arg(model)
            .arg(vocab)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting the ort rc.11 MiniLM encoder")?;
        let stdin = child.stdin.take().context("encoder stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("encoder stdout")?);
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let bytes = text.as_bytes();
        self.stdin.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.stdin.write_all(bytes)?;
        self.stdin.flush()?;
        let mut packed = [0u8; RECORD];
        self.stdout.read_exact(&mut packed)?;
        Ok(packed
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn shutdown(mut self) -> Result<()> {
        self.stdin.write_all(&0u32.to_le_bytes())?;
        self.stdin.flush()?;
        let _ = self.child.wait();
        Ok(())
    }
}

/// Encode a passage JSONL file into packed little-endian f32 vectors.
/// An existing output file is resumed on a whole-record boundary.
pub fn embed_passage_jsonl(passages: &Path, output: &Path, workers: usize) -> Result<u64> {
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let existing = std::fs::metadata(output).map_or(0, |meta| meta.len());
    ensure!(
        existing.is_multiple_of(RECORD as u64),
        "packed vector file ends in a partial record"
    );
    let skip = existing / RECORD as u64;
    let (model, vocab) = pinned_minilm();
    let workers = if workers == 0 {
        std::thread::available_parallelism()
            .map(|count| count.get().saturating_sub(2))
            .unwrap_or(1)
            .max(1)
    } else {
        workers
    };
    let encoder = MiniLm::open(&model, &vocab, workers)?;
    let mut packed = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output)?;
    let mut batch = Vec::new();
    let mut seen = 0u64;
    let mut written = skip;
    let started = std::time::Instant::now();
    for line in std::io::BufRead::lines(std::io::BufReader::new(std::fs::File::open(passages)?)) {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        seen += 1;
        if seen <= skip {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)?;
        let text = value
            .get("text")
            .and_then(|text| text.as_str())
            .context("passage is missing text")?
            .to_string();
        batch.push(text);
        if batch.len() == workers.max(1) * 8 {
            written = flush_batch(&encoder, &mut batch, &mut packed, written, &started)?;
        }
    }
    written = flush_batch(&encoder, &mut batch, &mut packed, written, &started)?;
    packed.flush()?;
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    eprintln!(
        "encoded {written} passages in {elapsed:.1}s ({:.0}/s) -> {}",
        (written - skip) as f64 / elapsed,
        output.display()
    );
    Ok(written)
}

fn flush_batch(
    encoder: &MiniLm,
    batch: &mut Vec<String>,
    packed: &mut std::fs::File,
    mut written: u64,
    started: &std::time::Instant,
) -> Result<u64> {
    if batch.is_empty() {
        return Ok(written);
    }
    let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
    let vectors = encoder.embed(&refs)?;
    ensure!(
        vectors.len() == batch.len(),
        "encoder returned a short batch"
    );
    for vector in vectors {
        ensure!(vector.len() == DIMENSIONS, "MiniLM vector width changed");
        for value in vector {
            packed.write_all(&value.to_le_bytes())?;
        }
    }
    written += batch.len() as u64;
    batch.clear();
    if written.is_multiple_of(1000) {
        let elapsed = started.elapsed().as_secs_f64().max(0.001);
        eprintln!(
            "encoded {written} passages ({:.0}/s)",
            written as f64 / elapsed
        );
    }
    Ok(written)
}
