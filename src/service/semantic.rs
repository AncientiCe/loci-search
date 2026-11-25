use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use loci_search::{
    Embedding, Passage, Query,
    embedding::{LocalEncoder, ModelPack},
    generation,
    index::Engine,
    ranking::{SearchConfig, Strategy},
    store::{Document, EncodedPassage, Filter, Mutation, PreparedDocument, Snapshot},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
};
use tempfile::TempDir;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Budget {
    regions: Option<usize>,
    passages: Option<usize>,
    postings: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    text: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    budget: Option<Budget>,
}
fn default_limit() -> usize {
    10
}

pub(super) fn validate_settings(settings: &Value) -> Result<()> {
    let fields = settings
        .as_object()
        .context("index settings must be an object")?;
    ensure!(
        fields.keys().all(|key| matches!(
            key.as_str(),
            "model" | "mappings" | "stream" | "default_visibility" | "retrieval_profile"
        )),
        "unsupported index setting"
    );
    let palace = fields.get("retrieval_profile").and_then(Value::as_str) == Some("palace");
    if let Some(profile) = fields.get("retrieval_profile") {
        ensure!(
            profile.as_str() == Some("palace"),
            "retrieval_profile must be palace"
        );
    }
    if let Some(model) = fields.get("model") {
        let name = model.as_str().context("model must be a string")?;
        if name == generation::MODEL_ID {
            ensure!(palace, "minilm-l6-v2-q8 requires retrieval_profile palace");
        } else if palace || name == "lexical" {
            ensure!(!palace, "palace text indexes use minilm-l6-v2-q8");
        } else {
            ModelPack::from_str(name, false).map_err(anyhow::Error::msg)?;
        }
    } else if palace {
        anyhow::bail!("palace text indexes set model to minilm-l6-v2-q8");
    }
    Ok(())
}

type Slot = Arc<Mutex<Option<Arc<Projection>>>>;
struct Projection {
    _workspace: TempDir,
    db: PathBuf,
    readers: Mutex<Vec<Engine>>,
}

pub(super) struct RegionalService {
    models: PathBuf,
    encoders: Mutex<BTreeMap<String, Arc<Mutex<LocalEncoder>>>>,
    recent: Mutex<VecDeque<(Vec<u8>, Slot)>>,
}

fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| anyhow::anyhow!("search worker lock unavailable"))
}

impl RegionalService {
    pub(super) fn encode_changes(
        &self,
        pack: ModelPack,
        ops: &[Mutation],
    ) -> Result<Vec<PreparedDocument>> {
        let mut current = BTreeMap::new();
        for op in ops {
            match op {
                Mutation::Upsert { document } => {
                    current.insert(&document.id, document);
                }
                Mutation::Delete { id } => {
                    current.remove(id);
                }
            }
        }
        self.prepare(pack, current.into_values())
    }

    pub(super) fn prepare<'a>(
        &self,
        pack: ModelPack,
        docs: impl IntoIterator<Item = &'a Document>,
    ) -> Result<Vec<PreparedDocument>> {
        let encoder = self.encoder(pack)?;
        let mut enc = lock(&encoder)?;
        docs.into_iter()
            .filter(|doc| doc.embed)
            .map(|doc| {
                let mut passages = Vec::new();
                for batch in spans(&doc.text).chunks(32) {
                    let texts = batch
                        .iter()
                        .map(|&(a, b)| doc.text[a..b].to_owned())
                        .collect::<Vec<_>>();
                    let vectors = enc.embed_documents(&texts)?;
                    ensure!(
                        vectors.len() == batch.len(),
                        "incomplete passage embedding batch"
                    );
                    passages.extend(batch.iter().zip(vectors).map(|(&(a, b), vector)| {
                        EncodedPassage {
                            start: a as u64,
                            end: b as u64,
                            vector,
                        }
                    }));
                }
                Ok(PreparedDocument {
                    id: doc.id.clone(),
                    revision: doc.revision.clone(),
                    passages,
                })
            })
            .collect()
    }
    pub(super) fn new(models: &Path) -> Result<Self> {
        Ok(Self {
            models: models.to_owned(),
            encoders: Mutex::default(),
            recent: Mutex::default(),
        })
    }

    fn encoder(&self, pack: ModelPack) -> Result<Arc<Mutex<LocalEncoder>>> {
        let mut models = lock(&self.encoders)?;
        if let Some(encoder) = models.get(pack.id()) {
            return Ok(encoder.clone());
        }
        let encoder = Arc::new(Mutex::new(pack.open(&self.models)?));
        models.insert(pack.id().to_string(), encoder.clone());
        Ok(encoder)
    }

    fn slot(&self, key: Vec<u8>) -> Result<Slot> {
        let mut recent = lock(&self.recent)?;
        if let Some(i) = recent.iter().position(|(k, _)| *k == key) {
            let entry = recent
                .remove(i)
                .context("projection cache entry disappeared")?;
            let slot = entry.1.clone();
            recent.push_front(entry);
            return Ok(slot);
        }
        let slot = Arc::new(Mutex::new(None));
        recent.push_front((key, slot.clone()));
        recent.truncate(4);
        Ok(slot)
    }

    pub(super) fn search(
        &self,
        name: &str,
        snapshot: &Snapshot,
        settings: &Value,
        body: Value,
        principals: &[String],
        strategy: Strategy,
    ) -> Result<Value> {
        let req: Request = serde_json::from_value(body)?;
        ensure!(
            strategy == Strategy::Regional || req.budget.is_none(),
            "budget applies to regional search; send \"mode\": \"regional\" with it"
        );
        let budget = req.budget.unwrap_or_default();
        ensure!(
            !req.text.trim().is_empty() && req.text.len() <= 16_384,
            "text must contain 1–16384 bytes"
        );
        ensure!(
            (1..=200).contains(&req.limit),
            "semantic search limit must be 1–200"
        );
        let count = budget.passages.unwrap_or(6000);
        let regions = budget.regions.unwrap_or(12);
        let postings = budget.postings.unwrap_or(60_000);
        ensure!(
            (2..=12000).contains(&count)
                && (1..=64).contains(&regions)
                && (1..=120000).contains(&postings),
            "regional work budget outside supported bounds"
        );
        let model = settings
            .get("model")
            .and_then(Value::as_str)
            .context("semantic search needs an index with a local model")?;
        let pack = ModelPack::from_str(model, false).map_err(anyhow::Error::msg)?;
        // The topology itself contains landmark IDs. Build only from eligible documents so
        // a route, candidate ID, or explanation cannot disclose a different reader's data.
        let profile = settings
            .get("retrieval_profile")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let key = serde_json::to_vec(&(
            name,
            snapshot.generation,
            model,
            profile,
            principals,
            &req.filter,
        ))?;
        let slot = self.slot(Sha256::digest(key).to_vec())?;
        let encoder = self.encoder(pack)?;
        let manifest = pack.manifest();
        let projection = {
            let mut entry = lock(&slot)?;
            if entry.is_none() {
                let documents = snapshot.eligible(req.filter.as_ref(), principals)?;
                if documents.iter().all(|d| d.text.trim().is_empty()) {
                    return Ok(
                        json!({"generation":snapshot.generation,"strategy":"regional","hits":[],"expanded":[],"work":{},"stop_reason":"empty_eligible_set"}),
                    );
                }
                let (persisted, missing) = snapshot.encoded_passages(&documents, &manifest)?;
                let mut enc = lock(&encoder)?;
                *entry = Some(Arc::new(build(
                    missing.into_iter().map(|i| &documents[i]),
                    persisted,
                    &manifest,
                    &mut enc,
                    profile == "palace",
                )?));
            }
            entry.as_ref().context("missing search projection")?.clone()
        };
        let query = Query {
            id: "http".into(),
            text: req.text.clone(),
            embedding: manifest,
            vector: lock(&encoder)?.embed_query(&req.text)?,
            group: "general".into(),
            judgments: BTreeMap::new(),
            required: vec![],
        };
        let config = if strategy == Strategy::Regional {
            let rescue = (count / 4).clamp(1, 40);
            SearchConfig {
                limit: req.limit,
                depth: req.limit.max(40),
                max_regions: regions,
                max_chunks: count - rescue,
                max_postings: postings,
                rescue,
                rescue_max_scored: Some(rescue),
                ..Default::default()
            }
        } else {
            SearchConfig::hybrid(req.limit)
        };
        let idle = lock(&projection.readers)?.pop();
        let mut engine = match idle {
            Some(e) => e,
            None => Engine::open(&projection.db)?,
        };
        let mut response = serde_json::to_value(engine.search(&query, strategy, &config)?)?;
        let mut pool = lock(&projection.readers)?;
        if pool.len() < 2 {
            pool.push(engine);
        }
        response["generation"] = json!(snapshot.generation);
        Ok(response)
    }
}

fn build<'a>(
    documents: impl Iterator<Item = &'a Document>,
    mut persisted: Vec<Passage>,
    embedding: &Embedding,
    encoder: &mut LocalEncoder,
    palace: bool,
) -> Result<Projection> {
    let workspace = tempfile::tempdir()?;
    let db = workspace.path().join("regional.sqlite");
    let mut passages = Vec::new();
    for d in documents {
        for (start, end) in spans(&d.text) {
            passages.push(Passage {
                id: format!("{}:{start}", d.id),
                document: d.id.clone(),
                revision: d.revision.clone(),
                start: start as u64,
                end: end as u64,
                text: d.text[start..end].to_owned(),
                metadata: d.fields.clone(),
                vector: vec![],
            });
        }
    }
    for batch in passages.chunks_mut(32) {
        let texts = batch.iter().map(|p| p.text.clone()).collect::<Vec<_>>();
        let vectors = encoder.embed_documents(&texts)?;
        ensure!(
            vectors.len() == batch.len(),
            "incomplete passage embedding batch"
        );
        for (passage, vector) in batch.iter_mut().zip(vectors) {
            passage.vector = vector;
        }
    }
    persisted.extend(passages);
    let profile = palace.then_some(loci_search::index::RetrievalProfile::Palace);
    loci_search::index::ingest_compact_with_profile(
        &db,
        persisted,
        embedding.clone(),
        64,
        profile,
    )?;
    Ok(Projection {
        _workspace: workspace,
        db,
        readers: Mutex::default(),
    })
}

fn spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let rest = &text[start..];
        let hard = rest.char_indices().nth(320).map_or(rest.len(), |(i, _)| i);
        let slice = &rest[..hard];
        let end = if hard == rest.len() {
            hard
        } else {
            slice
                .char_indices()
                .rev()
                .find(|(i, c)| *i > hard / 2 && c.is_whitespace())
                .map_or(hard, |(i, c)| i + c.len_utf8())
        };
        if !rest[..end].trim().is_empty() {
            spans.push((start, start + end));
        }
        start += end;
    }
    spans
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spans_preserve_utf8_source_offsets() {
        let text = "βeta résumé ".repeat(130);
        let pieces = spans(&text);
        assert!(pieces.len() > 1);
        assert_eq!(
            pieces.iter().map(|&(a, b)| &text[a..b]).collect::<String>(),
            text
        );
    }
}
