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
            size == self.bytes && format!("{:x}", hash.finalize()) == self.sha256,
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
                "{}@{}#{:x}",
                self.repository(),
                self.revision(),
                hash.finalize()
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
        use fastembed::{
            InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
            UserDefinedEmbeddingModel,
        };
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
        let tokenizer = TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        let pooling = match self {
            Self::BgeSmallEn => Pooling::Cls,
            Self::MultilingualE5Small => Pooling::Mean,
        };
        let model = UserDefinedEmbeddingModel::new(read("onnx/model.onnx")?, tokenizer)
            .with_pooling(pooling);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
        let options = InitOptionsUserDefined::default()
            .with_max_length(512)
            .with_intra_threads(threads);
        let runtime = TextEmbedding::try_new_from_user_defined(model, options)?;
        Ok(LocalEncoder {
            pack: self,
            runtime,
        })
    }

    #[cfg(not(feature = "local-model"))]
    pub fn open(self, _cache: &Path) -> Result<LocalEncoder> {
        anyhow::bail!("this binary was built without local-model support")
    }
}

fn verify_file(asset: &Asset, path: &Path) -> Result<()> {
    asset.copy_verified(
        File::open(path).with_context(|| format!("reading {}", path.display()))?,
        io::sink(),
    )
}

/// Encodes raw text entirely offline after a model pack has been installed.
pub struct LocalEncoder {
    #[cfg(feature = "local-model")]
    pack: ModelPack,
    #[cfg(feature = "local-model")]
    runtime: fastembed::TextEmbedding,
    #[cfg(not(feature = "local-model"))]
    _unavailable: std::convert::Infallible,
}

impl LocalEncoder {
    #[cfg(feature = "local-model")]
    fn encode(&mut self, texts: &[String], query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let (query_prefix, passage_prefix) = self.pack.prefixes();
        let prefix = if query { query_prefix } else { passage_prefix };
        let inputs = texts
            .iter()
            .map(|text| format!("{prefix}{text}"))
            .collect::<Vec<_>>();
        let vectors = self.runtime.embed(inputs, Some(16))?;
        ensure!(
            vectors.len() == texts.len(),
            "encoder returned an unexpected number of vectors"
        );
        vectors
            .into_iter()
            .map(|vector| crate::normalize(&vector, 384))
            .collect()
    }

    #[cfg(not(feature = "local-model"))]
    fn encode(&mut self, _texts: &[String], _query: bool) -> Result<Vec<Vec<f32>>> {
        match self._unavailable {}
    }

    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.encode(texts, false)
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.encode(&[text.to_owned()], true)?
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
