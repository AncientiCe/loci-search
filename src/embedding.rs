//! Verified model packs for local text encoding. Downloads are limited to explicit installs.

use crate::{Embedding, Normalization};
use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ModelPack {
    #[value(name = "bge-small-en-v1.5")]
    BgeSmallEn,
    #[value(name = "multilingual-e5-small")]
    MultilingualE5Small,
}

// Files and SHA-256s from the exact publisher revisions. Keep the relative layout
// so the same installer can consume an offline Hugging Face checkout.
struct Asset {
    name: &'static str,
    bytes: u64,
    sha256: &'static str,
}

impl Asset {
    const fn new(name: &'static str, bytes: u64, sha256: &'static str) -> Self {
        Self {
            name,
            bytes,
            sha256,
        }
    }

    fn copy_verified(&self, input: impl Read, mut output: impl Write) -> Result<()> {
        let mut input = input.take(self.bytes + 1);
        let mut hash = Sha256::new();
        let mut size = 0;
        let mut buffer = [0; 32 * 1024];
        loop {
            let count = input.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            size += count as u64;
            output.write_all(&buffer[..count])?;
        }
        ensure!(
            size == self.bytes && crate::hex(&hash.finalize()) == self.sha256,
            "{} does not match its pinned checksum and length",
            self.name
        );
        Ok(())
    }
}

const BGE: &[Asset] = &[
    Asset::new(
        "config.json",
        743,
        "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    ),
    Asset::new(
        "special_tokens_map.json",
        125,
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    Asset::new(
        "tokenizer.json",
        711396,
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    Asset::new(
        "tokenizer_config.json",
        366,
        "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    ),
    Asset::new(
        "onnx/model.onnx",
        133093490,
        "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    ),
];
const E5: &[Asset] = &[
    Asset::new(
        "config.json",
        655,
        "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    ),
    Asset::new(
        "special_tokens_map.json",
        167,
        "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    ),
    Asset::new(
        "tokenizer.json",
        17082730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
    Asset::new(
        "tokenizer_config.json",
        443,
        "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    ),
    Asset::new(
        "onnx/model.onnx",
        470268510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
];

impl ModelPack {
    pub fn id(self) -> &'static str {
        match self {
            Self::BgeSmallEn => "bge-small-en-v1.5",
            Self::MultilingualE5Small => "multilingual-e5-small",
        }
    }

    pub fn repository(self) -> &'static str {
        match self {
            Self::BgeSmallEn => "BAAI/bge-small-en-v1.5",
            Self::MultilingualE5Small => "intfloat/multilingual-e5-small",
        }
    }

    pub fn revision(self) -> &'static str {
        match self {
            Self::BgeSmallEn => "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            Self::MultilingualE5Small => "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        }
    }

    fn assets(self) -> &'static [Asset] {
        match self {
            Self::BgeSmallEn => BGE,
            Self::MultilingualE5Small => E5,
        }
    }

    pub fn bytes(self) -> u64 {
        self.assets().iter().map(|asset| asset.bytes).sum()
    }

    fn prefixes(self) -> (&'static str, &'static str) {
        match self {
            Self::BgeSmallEn => (
                "Represent this sentence for searching relevant passages: ",
                "",
            ),
            Self::MultilingualE5Small => ("query: ", "passage: "),
        }
    }

    pub fn manifest(self) -> Embedding {
        let (query, passage) = self.prefixes();
        let pooling = match self {
            Self::BgeSmallEn => "cls",
            Self::MultilingualE5Small => "mean",
        };
        let mut hash = Sha256::new();
        for field in [
            // Names the codec, not a dependency: the encoder reproduces fastembed 7.8.0
            // exactly, and changing this string would orphan every existing index.
            "fastembed-7.8.0",
            "onnx-fp32",
            "512",
            "384",
            "l2",
            pooling,
            query,
            passage,
        ] {
            hash.update(field.as_bytes());
            hash.update([0]);
        }
        for asset in self.assets() {
            hash.update(asset.sha256.as_bytes());
        }
        Embedding {
            model_revision: format!(
                "{}@{}#{}",
                self.repository(),
                self.revision(),
                crate::hex(&hash.finalize())
            ),
            dimensions: 384,
            normalization: Normalization::L2,
        }
    }

    pub fn from_manifest(manifest: &Embedding) -> Option<Self> {
        [Self::BgeSmallEn, Self::MultilingualE5Small]
            .into_iter()
            .find(|pack| &pack.manifest() == manifest)
    }

    pub fn directory(self, cache: &Path) -> PathBuf {
        cache.join(self.id()).join(self.revision())
    }

    /// Install from the publisher or from a local directory with the publisher's layout.
    pub fn install(self, cache: &Path, source: Option<&Path>) -> Result<PathBuf> {
        let target = self.directory(cache);
        if target.exists() && self.verify(cache).is_ok() {
            return Ok(target);
        }
        let parent = cache.join(self.id());
        fs::create_dir_all(&parent)?;
        let staging = tempfile::tempdir_in(parent)?;
        fs::create_dir(staging.path().join("onnx"))?;
        for asset in self.assets() {
            let mut file = File::create(staging.path().join(asset.name))?;
            if let Some(source) = source {
                let path = source.join(asset.name);
                let input =
                    File::open(&path).with_context(|| format!("reading {}", path.display()))?;
                asset.copy_verified(input, &mut file)?;
            } else {
                let url = format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    self.repository(),
                    self.revision(),
                    asset.name
                );
                let response = ureq::get(&url)
                    .call()
                    .with_context(|| format!("downloading {}", asset.name))?;
                asset.copy_verified(response.into_body().into_reader(), &mut file)?;
            }
            file.sync_all()?;
        }
        match fs::rename(staging.path(), &target) {
            Ok(()) => Ok(target),
            Err(_) if target.is_dir() => {
                // A concurrent installer may have published the same revision. If a
                // previous install is damaged, atomically restore each incorrect file.
                for asset in self.assets() {
                    let path = target.join(asset.name);
                    if verify_file(asset, &path).is_err() {
                        if let Some(parent) = path.parent() {
                            fs::create_dir_all(parent)?;
                        }
                        let mut replacement = tempfile::NamedTempFile::new_in(&target)?;
                        io::copy(
                            &mut File::open(staging.path().join(asset.name))?,
                            &mut replacement,
                        )?;
                        replacement.as_file().sync_all()?;
                        replacement.persist(&path)?;
                    }
                }
                self.verify(cache)?;
                Ok(target)
            }
            Err(error) => Err(error).with_context(|| format!("publishing {}", target.display())),
        }
    }

    pub fn verify(self, cache: &Path) -> Result<()> {
        let directory = self.directory(cache);
        for asset in self.assets() {
            verify_file(asset, &directory.join(asset.name))?;
        }
        Ok(())
    }

    #[cfg(feature = "local-model")]
    pub fn open(self, cache: &Path) -> Result<LocalEncoder> {
        self.open_threads(cache, 4)
    }

    /// `thread_cap` limits ONNX intra-op threads. The service keeps this at 4.
    #[cfg(feature = "local-model")]
    pub fn open_threads(self, cache: &Path, thread_cap: usize) -> Result<LocalEncoder> {
        let directory = self.directory(cache);
        // Verify the actual bytes supplied to inference even if another process has
        // changed the cache since installation or the last verification.
        let read = |name: &str| -> Result<Vec<u8>> {
            let asset = self
                .assets()
                .iter()
                .find(|asset| asset.name == name)
                .context("unknown pack asset")?;
            let file = File::open(directory.join(name)).with_context(|| {
                format!(
                    "reading {}. Install {} in {} first",
                    name,
                    self.id(),
                    cache.display()
                )
            })?;
            let mut bytes = Vec::with_capacity(asset.bytes as usize);
            asset.copy_verified(file, &mut bytes)?;
            Ok(bytes)
        };
        let json = |name: &str| -> Result<serde_json::Value> {
            serde_json::from_slice(&read(name)?).with_context(|| format!("parsing {name}"))
        };
        let tokenizer = TextTokenizer::load(
            &read("tokenizer.json")?,
            &json("config.json")?,
            &json("tokenizer_config.json")?,
        )?;
        let threads =
            std::thread::available_parallelism().map_or(1, |n| n.get().min(thread_cap.max(1)));
        let session = ort::session::Session::builder(&onnx_environment()?)
            .map_err(|error| anyhow::anyhow!("ONNX Runtime session: {error}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("ONNX Runtime optimization level: {error}"))?
            .with_intra_threads(threads)
            .map_err(|error| anyhow::anyhow!("ONNX Runtime threads: {error}"))?
            .commit_from_memory(&read("onnx/model.onnx")?)
            .map_err(|error| anyhow::anyhow!("loading the ONNX model: {error}"))?;
        let token_types = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Ok(LocalEncoder {
            pack: self,
            tokenizer,
            session,
            token_types,
        })
    }

    #[cfg(not(feature = "local-model"))]
    pub fn open(self, _cache: &Path) -> Result<LocalEncoder> {
        anyhow::bail!("this binary was built without local-model support")
    }

    #[cfg(not(feature = "local-model"))]
    pub fn open_threads(self, _cache: &Path, _thread_cap: usize) -> Result<LocalEncoder> {
        anyhow::bail!("this binary was built without local-model support")
    }
}

fn verify_file(asset: &Asset, path: &Path) -> Result<()> {
    asset.copy_verified(
        File::open(path).with_context(|| format!("reading {}", path.display()))?,
        io::sink(),
    )
}

/// A pack's `tokenizer.json` with the post-processing, truncation, and padding the
/// manifest records: special tokens around each text, at most 512 tokens (or the
/// tokenizer's own limit) with room kept for those special tokens, truncated on the right.
#[cfg(feature = "local-model")]
struct TextTokenizer {
    model: kitoken::Kitoken,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    max_length: usize,
    pad_id: u32,
}

#[cfg(feature = "local-model")]
impl TextTokenizer {
    fn load(
        tokenizer: &[u8],
        config: &serde_json::Value,
        tokenizer_config: &serde_json::Value,
    ) -> Result<Self> {
        let definition: serde_json::Value =
            serde_json::from_slice(tokenizer).context("parsing tokenizer.json")?;
        let model = kitoken::Kitoken::from_tokenizers_slice(tokenizer)
            .map_err(|error| anyhow::anyhow!("loading tokenizer.json: {error}"))?;
        let (prefix, suffix) = special_tokens(&definition["post_processor"])?;
        let model_max_length = tokenizer_config["model_max_length"]
            .as_f64()
            .context("tokenizer_config.json has no numeric model_max_length")?;
        let max_length = MAX_TOKENS.min(model_max_length as usize);
        ensure!(
            max_length > prefix.len() + suffix.len(),
            "the token limit leaves no room for text"
        );
        let pad_token = tokenizer_config["pad_token"]
            .as_str()
            .or_else(|| tokenizer_config["pad_token"]["content"].as_str())
            .context("tokenizer_config.json has no pad_token")?;
        let pad_id = config["pad_token_id"]
            .as_u64()
            .map(u32::try_from)
            .transpose()?
            .or_else(|| token_id(&definition, pad_token))
            .unwrap_or(0);
        Ok(Self {
            model,
            prefix,
            suffix,
            max_length,
            pad_id,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Special tokens written in the text map to their own ids, as in `tokenizers`.
        let mut content = self
            .model
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("tokenizing: {error}"))?;
        content.truncate(self.max_length - self.prefix.len() - self.suffix.len());
        let mut ids = Vec::with_capacity(self.prefix.len() + content.len() + self.suffix.len());
        ids.extend_from_slice(&self.prefix);
        ids.extend(content);
        ids.extend_from_slice(&self.suffix);
        Ok(ids)
    }
}

/// The special token ids a single-sequence post-processor puts before and after the text.
#[cfg(feature = "local-model")]
fn special_tokens(processor: &serde_json::Value) -> Result<(Vec<u32>, Vec<u32>)> {
    let id = |value: &serde_json::Value| -> Result<u32> {
        Ok(u32::try_from(
            value.as_u64().context("special token id is not a number")?,
        )?)
    };
    match processor["type"].as_str() {
        Some("TemplateProcessing") => {
            let mut prefix = Vec::new();
            let mut suffix = Vec::new();
            let mut seen_sequence = false;
            for piece in processor["single"]
                .as_array()
                .context("template has no single sequence")?
            {
                if piece.get("Sequence").is_some() {
                    ensure!(!seen_sequence, "template repeats the sequence");
                    seen_sequence = true;
                    continue;
                }
                let name = piece["SpecialToken"]["id"]
                    .as_str()
                    .context("template piece is neither a sequence nor a special token")?;
                ensure!(
                    piece["SpecialToken"]["type_id"].as_u64().unwrap_or(0) == 0,
                    "single-sequence special tokens must have type id 0"
                );
                let ids = processor["special_tokens"][name]["ids"]
                    .as_array()
                    .with_context(|| format!("template special token {name} has no ids"))?;
                let target = if seen_sequence {
                    &mut suffix
                } else {
                    &mut prefix
                };
                for value in ids {
                    target.push(id(value)?);
                }
            }
            ensure!(seen_sequence, "template has no sequence");
            Ok((prefix, suffix))
        }
        Some("BertProcessing" | "RobertaProcessing") => Ok((
            vec![id(&processor["cls"][1])?],
            vec![id(&processor["sep"][1])?],
        )),
        Some(other) => anyhow::bail!("unsupported tokenizer post-processor {other}"),
        None => Ok((Vec::new(), Vec::new())),
    }
}

/// A token's id from the added tokens or the model vocabulary of a `tokenizer.json`.
#[cfg(feature = "local-model")]
fn token_id(definition: &serde_json::Value, token: &str) -> Option<u32> {
    let added = definition["added_tokens"].as_array().and_then(|tokens| {
        tokens
            .iter()
            .find(|added| added["content"].as_str() == Some(token))
            .and_then(|added| added["id"].as_u64())
    });
    let vocabulary = &definition["model"]["vocab"];
    let id = added.or_else(|| vocabulary[token].as_u64()).or_else(|| {
        vocabulary.as_array().and_then(|pieces| {
            pieces
                .iter()
                .position(|piece| piece[0].as_str() == Some(token))
                .map(|index| index as u64)
        })
    })?;
    u32::try_from(id).ok()
}

/// CLS takes each row's first token. Mean averages the tokens the attention mask keeps.
#[cfg(feature = "local-model")]
fn pool(
    pack: ModelPack,
    values: &[f32],
    mask: &[i64],
    rows: usize,
    width: usize,
    dimensions: usize,
) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            let token = |index: usize| {
                let start = (row * width + index) * dimensions;
                &values[start..start + dimensions]
            };
            match pack {
                ModelPack::BgeSmallEn => token(0).to_vec(),
                ModelPack::MultilingualE5Small => {
                    let mut sum = vec![0.0f32; dimensions];
                    let mut kept = 0.0f32;
                    for index in 0..width {
                        let weight = mask[row * width + index] as f32;
                        kept += weight;
                        for (total, value) in sum.iter_mut().zip(token(index)) {
                            *total += weight * value;
                        }
                    }
                    let kept = if kept == 0.0 { 1.0 } else { kept };
                    sum.into_iter().map(|total| total / kept).collect()
                }
            }
        })
        .collect()
}

/// Encodes raw text entirely offline after a model pack has been installed.
pub struct LocalEncoder {
    #[cfg(feature = "local-model")]
    pack: ModelPack,
    #[cfg(feature = "local-model")]
    tokenizer: TextTokenizer,
    #[cfg(feature = "local-model")]
    session: ort::session::Session,
    #[cfg(feature = "local-model")]
    token_types: bool,
    #[cfg(not(feature = "local-model"))]
    _unavailable: std::convert::Infallible,
}

/// Token limit for every pack, recorded in the embedding manifest.
#[cfg(feature = "local-model")]
const MAX_TOKENS: usize = 512;

/// One ONNX Runtime environment per process, shared by every encoder.
#[cfg(feature = "local-model")]
fn onnx_environment() -> Result<ort::environment::Environment> {
    static ENVIRONMENT: std::sync::Mutex<Option<ort::environment::Environment>> =
        std::sync::Mutex::new(None);
    let mut shared = ENVIRONMENT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(environment) = shared.as_ref() {
        return Ok(environment.clone());
    }
    let environment = ort::init()
        .build()
        .map_err(|error| anyhow::anyhow!("starting ONNX Runtime: {error}"))?;
    *shared = Some(environment.clone());
    Ok(environment)
}

impl LocalEncoder {
    #[cfg(feature = "local-model")]
    fn encode(&mut self, texts: &[String], query: bool, batch: usize) -> Result<Vec<Vec<f32>>> {
        let (query_prefix, passage_prefix) = self.pack.prefixes();
        let prefix = if query { query_prefix } else { passage_prefix };
        let mut vectors = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch.max(1)) {
            let inputs = chunk
                .iter()
                .map(|text| format!("{prefix}{text}"))
                .collect::<Vec<_>>();
            vectors.extend(self.run(inputs)?);
        }
        ensure!(
            vectors.len() == texts.len(),
            "encoder returned an unexpected number of vectors"
        );
        Ok(vectors)
    }

    /// Tokenize one padded batch, run the model, and pool each row.
    #[cfg(feature = "local-model")]
    fn run(&mut self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        use ort::value::Tensor;
        let encodings = inputs
            .iter()
            .map(|text| self.tokenizer.encode(text))
            .collect::<Result<Vec<_>>>()?;
        let rows = encodings.len();
        // Pad every row to the longest one in the batch.
        let width = encodings.iter().map(Vec::len).max().unwrap_or(0);
        ensure!(rows > 0 && width > 0, "the tokenizer produced no tokens");
        let mut ids = Vec::with_capacity(rows * width);
        let mut mask = Vec::with_capacity(rows * width);
        for encoding in &encodings {
            let padding = width - encoding.len();
            ids.extend(encoding.iter().map(|&id| i64::from(id)));
            ids.extend(std::iter::repeat_n(
                i64::from(self.tokenizer.pad_id),
                padding,
            ));
            mask.extend(std::iter::repeat_n(1, encoding.len()));
            mask.extend(std::iter::repeat_n(0, padding));
        }
        // A single sequence has type id 0 throughout, padding included.
        let types = vec![0i64; rows * width];
        let shape = [rows as i64, width as i64];
        let mut feed = ort::inputs![
            "input_ids" => Tensor::from_array((shape, ids))?,
            "attention_mask" => Tensor::from_array((shape, mask.clone()))?,
        ];
        if self.token_types {
            feed.push((
                "token_type_ids".into(),
                Tensor::from_array((shape, types))?.into(),
            ));
        }
        let outputs = self
            .session
            .run(feed)
            .map_err(|error| anyhow::anyhow!("running the ONNX model: {error}"))?;
        // A sole output, else the first of the names sentence encoders use.
        let name = if outputs.len() == 1 {
            outputs.keys().next().map(str::to_owned)
        } else {
            ["text_embeds", "last_hidden_state", "sentence_embedding"]
                .into_iter()
                .find(|name| outputs.contains_key(name))
                .map(str::to_owned)
        }
        .context("the model has no embedding output")?;
        let output = &outputs[name.as_str()];
        let (dims, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow::anyhow!("reading the model output: {error}"))?;
        let dims = dims.iter().map(|&dim| dim as usize).collect::<Vec<_>>();
        let pooled = match dims.as_slice() {
            [batch, dimensions] if *batch == rows => values
                .chunks(*dimensions)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>(),
            [batch, tokens, dimensions] if *batch == rows && *tokens == width => {
                pool(self.pack, values, &mask, rows, width, *dimensions)
            }
            _ => anyhow::bail!("unexpected model output shape {dims:?}"),
        };
        pooled
            .into_iter()
            .map(|vector| crate::normalize(&vector, 384))
            .collect()
    }

    #[cfg(not(feature = "local-model"))]
    fn encode(&mut self, _texts: &[String], _query: bool, _batch: usize) -> Result<Vec<Vec<f32>>> {
        match self._unavailable {}
    }

    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.encode(texts, false, 16)
    }

    /// Encode passage text in larger batches for a corpus rebuild.
    pub fn embed_passages(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.encode(texts, false, 64)
    }

    pub fn embed_queries(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.encode(texts, true, 64)
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.encode(&[text.to_owned()], true, 1)?
            .pop()
            .context("encoder returned no query vector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_corrupt_short_and_long_assets() {
        let asset = Asset::new(
            "fixture",
            3,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        for input in [b"abc".as_slice(), b"ab", b"abd", b"abcd"] {
            let mut out = Vec::new();
            assert_eq!(
                asset.copy_verified(input, &mut out).is_ok(),
                input == b"abc"
            );
        }
    }

    #[test]
    fn matching_dimensions_do_not_mix_model_spaces() {
        let bge = ModelPack::BgeSmallEn.manifest();
        let mut altered = bge.clone();
        altered.model_revision.push_str("-different-codec");
        assert_eq!(ModelPack::from_manifest(&bge), Some(ModelPack::BgeSmallEn));
        assert_eq!(ModelPack::from_manifest(&altered), None);
        assert_ne!(bge, ModelPack::MultilingualE5Small.manifest());
    }

    #[test]
    fn interrupted_offline_install_has_no_visible_model() {
        let cache = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let pack = ModelPack::BgeSmallEn;
        fs::write(source.path().join("config.json"), "damaged").unwrap();
        assert!(pack.install(cache.path(), Some(source.path())).is_err());
        assert!(!pack.directory(cache.path()).exists());
        assert!(pack.verify(cache.path()).is_err());
    }

    #[cfg(feature = "local-model")]
    #[test]
    #[ignore = "downloads and verifies both local model packs (about 620 MB) on first run"]
    fn actual_models_rank_evidence_above_a_distractor() {
        let root = std::env::var_os("LOCI_TEST_MODEL_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/model-test-cache"));
        let cases = [
            (ModelPack::BgeSmallEn, "What is the capital of France?"),
            (
                ModelPack::MultilingualE5Small,
                "Was ist die Hauptstadt von Frankreich?",
            ),
        ];
        for (pack, query) in cases {
            pack.install(&root, None).unwrap();
            let mut encoder = pack.open(&root).unwrap();
            let query = encoder.embed_query(query).unwrap();
            let docs = encoder
                .embed_documents(&[
                    "Paris is the capital city of France.".into(),
                    "Fruit flies usually lay eggs on ripe bananas.".into(),
                ])
                .unwrap();
            assert!(crate::cosine(&query, &docs[0]) > crate::cosine(&query, &docs[1]));
            assert!((crate::cosine(&query, &query) - 1.0).abs() < 1e-4);
        }
    }
}
