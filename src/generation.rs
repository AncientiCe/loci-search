//! Build a palace text index from PostgreSQL and query one published generation.
//!
//! Writes commit to PostgreSQL before a generation directory is renamed into
//! place. Search opens that directory's SQLite snapshot and optional `.hnsw`
//! sidecar. It does not open PostgreSQL.

use crate::durable::{
    Durable, GenerationManifest, Generations, PassageWrite, SourceCommit, Span, SpanKind,
    StoredPassage, hnsw_file_name,
};
use crate::index::{self, Engine, RetrievalProfile};
use crate::palace::{self, DIMENSIONS, MiniLm};
use crate::ranking::{SearchConfig, Strategy};
use crate::{Embedding, Normalization, Passage, Query};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "minilm-l6-v2-q8";
const SQLITE_NAME: &str = "retrieval.sqlite";
const LEAF_SIZE: usize = 64;
/// One sidecar process. The HotpotQA embed already uses the other workers.
const EMBED_WORKERS: usize = 1;

/// New text indexes use Palace and `minilm-l6-v2-q8`.
///
/// The 470-question re-embed reported 464/470 any-support@5, 423/470
/// all-support@5, and the same counts with dates omitted. Indexes that
/// already omit `retrieval_profile` keep the coverage ranker.
pub fn default_palace_for_new_text_indexes() -> bool {
    true
}

/// Fill model and profile only when a new index specifies neither.
pub fn apply_new_text_defaults(settings: &mut Value) {
    if !default_palace_for_new_text_indexes() {
        return;
    }
    let Some(fields) = settings.as_object_mut() else {
        return;
    };
    if !fields.contains_key("model") && !fields.contains_key("retrieval_profile") {
        fields.insert(MODEL_ID_KEY.into(), Value::String(MODEL_ID.into()));
        fields.insert("retrieval_profile".into(), Value::String("palace".into()));
    }
}

const MODEL_ID_KEY: &str = "model";

pub fn minilm_embedding() -> Embedding {
    Embedding {
        model_revision:
            "Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9#minilm-l6-v2-q8"
                .into(),
        dimensions: DIMENSIONS,
        normalization: Normalization::L2,
    }
}

pub fn is_palace(settings: &Value) -> bool {
    settings
        .get("retrieval_profile")
        .and_then(Value::as_str)
        == Some("palace")
}

pub fn index_root(data: &Path, name: &str) -> PathBuf {
    data.join("indexes").join(name)
}

pub fn schema_for_index(name: &str) -> Result<String> {
    let mut schema = String::from("ix_");
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() {
            schema.push(byte.to_ascii_lowercase() as char);
        } else if byte == b'-' || byte == b'_' {
            schema.push('_');
        } else {
            bail!("index name {name} cannot be used as a PostgreSQL schema");
        }
    }
    ensure!(
        schema.len() <= 63,
        "index name {name} is too long for a PostgreSQL schema"
    );
    Ok(schema)
}

pub fn postgres_url() -> Result<String> {
    std::env::var("LOCI_POSTGRES_URL").context("LOCI_POSTGRES_URL is required for a palace index")
}

pub fn open_index(data: &Path, name: &str) -> Result<Durable> {
    Durable::open_schema(
        &postgres_url()?,
        &schema_for_index(name)?,
        index_root(data, name),
    )
}

/// One document change applied to the durable log before the generation publish.
#[derive(Clone, Debug)]
pub enum TextChange {
    Upsert(TextUpdate),
    Delete { id: String },
}

#[derive(Clone, Debug)]
pub struct TextUpdate {
    pub id: String,
    pub revision: String,
    pub text: String,
    pub metadata: Value,
}

/// Commit the changes, then publish one generation from the live passages.
pub fn commit_and_publish(durable: &mut Durable, changes: &[TextChange]) -> Result<PublishedIds> {
    ensure!(!changes.is_empty(), "at least one document change is required");
    let mut pending = Vec::new();
    for change in changes {
        if let TextChange::Upsert(update) = change {
            pending.push(pieces(&update.text)?);
        }
    }
    let texts = pending
        .iter()
        .flat_map(|pieces| pieces.iter().map(|piece| piece.text.as_str()))
        .collect::<Vec<_>>();
    let vectors = if texts.is_empty() {
        Vec::new()
    } else {
        embed_texts(&texts)?
    };
    let mut vector = vectors.into_iter();
    let mut prepared = pending.into_iter();
    let fingerprint = minilm_embedding().model_revision;
    for change in changes {
        match change {
            TextChange::Upsert(update) => {
                let pieces = prepared
                    .next()
                    .context("missing prepared passages for an upsert")?;
                let mut passages = Vec::with_capacity(pieces.len());
                for (index, piece) in pieces.into_iter().enumerate() {
                    let values = vector
                        .next()
                        .context("encoder returned fewer vectors than passages")?;
                    let mut passage = PassageWrite::new(
                        format!("{}:{index}", update.id),
                        piece.text,
                        Span {
                            kind: SpanKind::Byte,
                            start: piece.start,
                            end: piece.end,
                        },
                        pack_vector(&values),
                    );
                    passage.metadata = update.metadata.clone();
                    passages.push(passage);
                }
                let mut commit = SourceCommit::new(
                    update.id.clone(),
                    update.revision.clone(),
                    fingerprint.clone(),
                    passages,
                );
                commit.metadata = update.metadata.clone();
                let _ = durable.commit(&commit)?;
            }
            TextChange::Delete { id } => delete_source(durable, id)?,
        }
    }
    publish_live(durable)
}

/// Ignore partial directories, then publish the passages still committed in PostgreSQL.
pub fn publish_live(durable: &mut Durable) -> Result<PublishedIds> {
    let _resume = durable.resume()?;
    let stored = durable.live_passages()?;
    ensure!(
        !stored.is_empty(),
        "no live passages are committed for this index"
    );
    let seq = durable.changelog_head()?;
    let passages = stored_to_passages(&stored)?;
    let staging = durable.create_staging()?;
    let sqlite = staging.join(SQLITE_NAME);
    let manifest = index::ingest_compact_with_profile(
        &sqlite,
        passages,
        minilm_embedding(),
        LEAF_SIZE,
        Some(RetrievalProfile::Palace),
    )?;
    checkpoint(&sqlite)?;
    {
        let mut engine = Engine::open(&sqlite)?;
        engine.build_dense_graph()?;
    }
    GenerationManifest::new(seq, SQLITE_NAME, serde_json::to_value(&manifest)?)
        .with_hnsw(hnsw_file_name(SQLITE_NAME))
        .write(&staging)?;
    let published = durable.publish(&staging)?;
    Ok(PublishedIds {
        id: published.id,
        changelog_seq: published.manifest.changelog_seq,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PublishedIds {
    pub id: String,
    pub changelog_seq: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CitedPage {
    pub generation: String,
    pub changelog_seq: i64,
    pub hits: Vec<CitedHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CitedHit {
    pub id: String,
    pub source: String,
    pub score: f64,
    pub excerpt: String,
    pub citation: String,
}

/// Score `text` against one published generation.
///
/// `before_score` runs after the snapshot is open and before the query is
/// embedded, so a replacement can publish while this reader still sees the
/// snapshot it opened.
pub fn search_text(
    root: &Path,
    generation: Option<&str>,
    text: &str,
    limit: usize,
    before_score: impl FnOnce(),
) -> Result<CitedPage> {
    let text = text.trim();
    ensure!(
        !text.is_empty() && text.len() <= 16_384,
        "text must contain 1–16384 bytes"
    );
    ensure!((1..=200).contains(&limit), "limit must be 1–200");
    let generations = Generations::open(root)?;
    let reader = match generation {
        Some(id) => generations.open_reader_id(id)?,
        None => generations
            .open_reader()?
            .context("no published generation")?,
    };
    let mut engine = Engine::open(reader.sqlite_path())?;
    before_score();
    let vector = embed_texts(&[text])?.pop().context("missing query vector")?;
    let query = Query {
        id: "http".into(),
        text: text.to_string(),
        embedding: engine.manifest.embedding.clone(),
        vector,
        group: "general".into(),
        judgments: BTreeMap::new(),
        required: Vec::new(),
    };
    let mut config = SearchConfig::default();
    config.limit = limit;
    config.depth = limit.max(40);
    config.rescue = 40.min(config.depth);
    let result = engine.search(&query, Strategy::Exact, &config)?;
    let mut hits = Vec::with_capacity(result.hits.len());
    for hit in result.hits {
        let citation = reader
            .citation(&hit.source.id)?
            .with_context(|| format!("no citation for {}", hit.source.id))?;
        hits.push(CitedHit {
            id: hit.source.id,
            source: hit.source.document,
            score: hit.score,
            excerpt: hit.source.text,
            citation,
        });
    }
    Ok(CitedPage {
        generation: reader.id().to_string(),
        changelog_seq: reader.changelog_seq(),
        hits,
    })
}

struct Piece {
    text: String,
    start: u64,
    end: u64,
}

fn pieces(text: &str) -> Result<Vec<Piece>> {
    let trimmed = text.trim();
    ensure!(!trimmed.is_empty(), "document text is empty");
    let chunks = palace::chunk_text(trimmed);
    let chunks = if chunks.is_empty() {
        vec![(trimmed.to_string(), 0)]
    } else {
        chunks
    };
    let mut from = 0usize;
    let mut pieces = Vec::with_capacity(chunks.len());
    for (index, (chunk, _)) in chunks.into_iter().enumerate() {
        let relative = trimmed[from..]
            .find(&chunk)
            .with_context(|| format!("chunk {index} is not in the document"))?;
        let start = from + relative;
        let end = start + chunk.len();
        ensure!(end <= trimmed.len(), "chunk exceeds the document");
        from = start.saturating_add(1);
        pieces.push(Piece {
            text: chunk,
            start: start as u64,
            end: end as u64,
        });
    }
    Ok(pieces)
}

fn delete_source(durable: &mut Durable, id: &str) -> Result<()> {
    let live = durable.live_passages()?;
    let Some(row) = live.iter().find(|row| row.source_id == id) else {
        return Ok(());
    };
    let _ = durable.delete(id, &row.revision, &row.checksum)?;
    Ok(())
}

fn stored_to_passages(rows: &[StoredPassage]) -> Result<Vec<Passage>> {
    rows.iter()
        .map(|row| {
            Ok(Passage {
                id: row.passage_id.clone(),
                document: row.source_id.clone(),
                revision: row.revision.clone(),
                start: row.span.start,
                end: row.span.end,
                text: row.text.clone(),
                metadata: metadata_map(&row.metadata)?,
                vector: unpack_vector(&row.embedding)?,
            })
        })
        .collect()
}

fn metadata_map(value: &Value) -> Result<BTreeMap<String, Value>> {
    let object = value
        .as_object()
        .context("passage metadata must be a JSON object")?;
    Ok(object
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

fn embed_texts(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let (model, vocab) = palace::pinned_minilm();
    let encoder = MiniLm::open(&model, &vocab, EMBED_WORKERS)?;
    let vectors = encoder.embed(texts)?;
    ensure!(
        vectors.len() == texts.len(),
        "encoder returned a short batch"
    );
    for vector in &vectors {
        ensure!(
            vector.len() == DIMENSIONS,
            "MiniLM vector dimension mismatch"
        );
    }
    Ok(vectors)
}

fn pack_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn unpack_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() == DIMENSIONS * 4,
        "stored embedding has the wrong length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

fn checkpoint(sqlite: &Path) -> Result<()> {
    let connection = rusqlite::Connection::open(sqlite)
        .with_context(|| format!("checkpoint {}", sqlite.display()))?;
    connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_text_indexes_default_to_palace_after_the_gate() {
        assert!(default_palace_for_new_text_indexes());
        let mut settings = serde_json::json!({});
        apply_new_text_defaults(&mut settings);
        assert_eq!(settings["model"].as_str(), Some(MODEL_ID));
        assert_eq!(settings["retrieval_profile"].as_str(), Some("palace"));
        let mut lexical = serde_json::json!({"model": "lexical"});
        apply_new_text_defaults(&mut lexical);
        assert_eq!(lexical["model"].as_str(), Some("lexical"));
        assert!(lexical.get("retrieval_profile").is_none());
    }

    #[test]
    fn schema_names_follow_the_index_name() {
        assert_eq!(schema_for_index("Notes").unwrap(), "ix_notes");
        assert_eq!(schema_for_index("palace-notes").unwrap(), "ix_palace_notes");
    }
}
