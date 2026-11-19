use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use loci_search::{
    Embedding, Passage, Query,
    bench::{self, Run},
    fixture,
    index::{Engine, ingest},
    ranking::{SearchConfig, Strategy},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "SQLite-backed semantic region search and retrieval benchmarks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate deterministic passages, embeddings, and relevance judgments.
    Fixture {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        chunks: usize,
    },
    /// Atomically replace the index with a complete JSONL snapshot.
    Ingest {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        passages: PathBuf,
        #[arg(long)]
        embedding: PathBuf,
        #[arg(long, default_value_t = 1024)]
        leaf_size: usize,
    },
    /// Search with a query JSON file containing a precomputed embedding.
    Search {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        query: PathBuf,
        #[arg(long, value_enum, default_value = "regional")]
        strategy: Strategy,
        #[arg(long)]
        explain: bool,
        #[command(flatten)]
        settings: Settings,
    },
    /// Compare global and regional retrieval on judged queries or a generated fixture.
    Bench {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, conflicts_with_all = ["db", "queries"], required_unless_present = "db")]
        fixture: bool,
        #[arg(long, requires = "queries")]
        db: Option<PathBuf>,
        #[arg(long, requires = "db")]
        queries: Option<PathBuf>,
        #[arg(long, default_value_t = 100_000)]
        chunks: usize,
        #[arg(long, default_value_t = 1024)]
        leaf_size: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 12)]
        exact_queries: usize,
        #[command(flatten)]
        settings: Settings,
    },
}

#[derive(Args)]
struct Settings {
    #[arg(long, default_value_t = 100)]
    limit: usize,
    #[arg(long, default_value_t = 200)]
    depth: usize,
    #[arg(long, default_value_t = 256)]
    graph_width: usize,
    #[arg(long, default_value_t = 12)]
    max_regions: usize,
    #[arg(long, default_value_t = 6_000)]
    max_chunks: usize,
    #[arg(long, default_value_t = 60_000)]
    max_postings: usize,
    #[arg(long, default_value_t = 40)]
    rescue: usize,
    /// Regional routing time budget in milliseconds. Zero uses work limits alone.
    #[arg(long, default_value_t = 0)]
    routing_ms: u64,
}

impl From<Settings> for SearchConfig {
    fn from(settings: Settings) -> Self {
        Self {
            limit: settings.limit,
            depth: settings.depth,
            graph_width: settings.graph_width,
            max_regions: settings.max_regions,
            max_chunks: settings.max_chunks,
            max_postings: settings.max_postings,
            rescue: settings.rescue,
            routing_ms: settings.routing_ms,
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Fixture { out, chunks } => {
            let (embedding, passages, queries) = fixture::generate(chunks)?;
            output_dir(&out)?;
            write_json(&out.join("embedding.json"), &embedding)?;
            write_jsonl(&out.join("passages.jsonl"), &passages)?;
            write_jsonl(&out.join("queries.jsonl"), &queries)?;
            write_json(&out.join("query.json"), &queries[0])?;
        }
        Command::Ingest {
            db,
            passages,
            embedding,
            leaf_size,
        } => {
            let passages = jsonl::<Passage>(&passages)?;
            let embedding = json::<Embedding>(&embedding)?;
            let manifest = ingest(&db, passages, embedding, leaf_size)?;
            print_json(&manifest)?;
        }
        Command::Search {
            db,
            query,
            strategy,
            explain,
            settings,
        } => {
            let query = json::<Query>(&query)?;
            let result = Engine::open(&db)?.search(&query, strategy, &settings.into())?;
            if explain {
                print_json(&result)?;
            } else {
                let sources = result
                    .hits
                    .iter()
                    .map(|hit| &hit.source)
                    .collect::<Vec<_>>();
                print_json(&sources)?;
            }
        }
        Command::Bench {
            out,
            fixture,
            db,
            queries,
            chunks,
            leaf_size,
            repeats,
            exact_queries,
            settings,
        } => {
            let config = settings.into();
            ensure!(repeats > 0, "repeats must be positive");
            output_dir(&out)?;
            let (db, queries, ingest_ms) = if fixture {
                let (embedding, passages, queries) = fixture::generate(chunks)?;
                let db = out.join("fixture.sqlite");
                let started = Instant::now();
                ingest(&db, passages, embedding, leaf_size)?;
                (db, queries, Some(started.elapsed().as_secs_f64() * 1000.0))
            } else {
                (
                    db.context("provide --db")?,
                    jsonl::<Query>(&queries.context("provide --queries")?)?,
                    None,
                )
            };
            let started = Instant::now();
            let mut engine = Engine::open(&db)?;
            let run = Run {
                fixture,
                repeats,
                exact_queries,
                index_bytes: index_bytes(&db)?,
                ingest_ms,
                open_ms: started.elapsed().as_secs_f64() * 1000.0,
            };
            eprintln!(
                "Comparing {} strategies over {} chunks and {} queries",
                3,
                engine.manifest.chunks,
                queries.len()
            );
            let trace_path = out.join("traces.partial");
            let traces = BufWriter::new(File::create(&trace_path)?);
            let report = bench::evaluate(&mut engine, &queries, &config, &run, traces)?;
            fs::rename(trace_path, out.join("traces.jsonl"))?;
            write_json(&out.join("report.json"), &report)?;
            println!(
                "Benchmark report written to {}",
                out.join("report.json").display()
            );
        }
    }
    Ok(())
}

fn index_bytes(path: &Path) -> Result<u64> {
    let mut wal = path.as_os_str().to_owned();
    wal.push("-wal");
    let wal_bytes = match fs::metadata(wal) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    Ok(fs::metadata(path)?.len() + wal_bytes)
}

fn output_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path).with_context(|| format!("create new output directory {}", path.display()))
}

fn json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

fn jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut records = Vec::new();
    for (line, contents) in BufReader::new(file).lines().enumerate() {
        let contents = contents?;
        if !contents.trim().is_empty() {
            records.push(
                serde_json::from_str(&contents)
                    .with_context(|| format!("{}:{}", path.display(), line + 1))?,
            );
        }
    }
    Ok(records)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_jsonl(path: &Path, values: &[impl Serialize]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn print_json(value: &impl Serialize) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value)?;
    stdout.write_all(b"\n")?;
    Ok(())
}
