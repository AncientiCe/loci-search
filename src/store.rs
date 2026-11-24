//! Durable named indexes, atomic document batches and permission-scoped snapshots.

mod cursor;
mod encoded;
mod query;
pub use encoded::{EncodedPassage, PreparedDocument};
mod mapping;
mod stream;
pub use query::{
    Bucket, Direction, Filter, Highlight, Metric, MetricKind, SearchPage, SearchRequest, Sort,
};
pub use stream::StreamPolicy;

use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn embed_by_default() -> bool {
    true
}

fn embed_is_default(embed: &bool) -> bool {
    *embed
}

/// A source document. An empty readers list makes the document public.
/// `embed: false` keeps the text lexical and leaves it out of vector writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub id: String,
    pub revision: String,
    pub text: String,
    #[serde(default)]
    pub fields: BTreeMap<String, Value>,
    #[serde(default)]
    pub readers: Vec<String>,
    #[serde(default = "embed_by_default", skip_serializing_if = "embed_is_default")]
    pub embed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mutation {
    Upsert { document: Document },
    Delete { id: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteResult {
    pub generation: u64,
    pub changed: usize,
}

#[derive(Debug)]
pub struct Catalog {
    path: PathBuf,
}

fn connect(path: &Path) -> Result<Connection> {
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(5))?;
    db.pragma_update(None, "foreign_keys", true)?;
    Ok(db)
}

fn valid_name(name: &str) -> Result<()> {
    ensure!(!name.trim().is_empty(), "empty index name");
    ensure!(
        name.len() <= 200 && !name.chars().any(char::is_control),
        "invalid index name"
    );
    Ok(())
}

fn generation(db: &Connection, index: &str) -> Result<u64> {
    db.query_row(
        "SELECT generation FROM loci_indexes WHERE name = ?1",
        [index],
        |row| row.get(0),
    )
    .optional()?
    .with_context(|| format!("unknown index {index}"))
}

impl Catalog {
    /// Open or initialize a catalog at a persistent filesystem path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(
            !path.as_os_str().is_empty() && path != Path::new(":memory:"),
            "catalog requires a file path"
        );
        let db = connect(path)?;
        let journal: String = db.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
        ensure!(
            journal.eq_ignore_ascii_case("wal"),
            "catalog requires WAL support"
        );
        db.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS loci_indexes (
                 name TEXT PRIMARY KEY NOT NULL,
                 generation INTEGER NOT NULL DEFAULT 0,
                 settings TEXT NOT NULL DEFAULT '{}'
             );
             CREATE TABLE IF NOT EXISTS loci_documents (
                 idx TEXT NOT NULL REFERENCES loci_indexes(name),
                 id TEXT NOT NULL,
                 revision TEXT NOT NULL,
                 body TEXT NOT NULL,
                 readers TEXT NOT NULL,
                 rowid INTEGER PRIMARY KEY,
                 UNIQUE(idx, id)
             );
             CREATE TABLE IF NOT EXISTS loci_write_receipts (
                 idx TEXT NOT NULL REFERENCES loci_indexes(name),
                 key TEXT NOT NULL,
                 request TEXT NOT NULL,
                 result TEXT NOT NULL,
                 PRIMARY KEY(idx, key)
             ) WITHOUT ROWID;
             CREATE VIRTUAL TABLE IF NOT EXISTS loci_fts USING fts5(
                 idx UNINDEXED, doc_id UNINDEXED, identifiers, title, text, attributes, tokenize='unicode61'
             );
             COMMIT;",
        )?;
        cursor::initialize(&db)?;
        encoded::initialize(&db)?;
        stream::initialize(&db)?;
        Ok(Self {
            path: path.to_owned(),
        })
    }

    /// Returns true only when this call created the index.
    pub fn create_index(&self, name: &str) -> Result<bool> {
        self.create_index_with_settings(name, &serde_json::json!({}))
    }

    pub fn create_index_with_settings(&self, name: &str, settings: &Value) -> Result<bool> {
        valid_name(name)?;
        ensure!(settings.is_object(), "index settings must be an object");
        mapping::validate_settings(settings)?;
        let mut db = connect(&self.path)?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let created = tx.execute(
            "INSERT INTO loci_indexes(name,settings) VALUES(?1,?2) ON CONFLICT(name) DO NOTHING",
            params![name, serde_json::to_string(settings)?],
        )? == 1;
        if !created {
            let current: String = tx.query_row(
                "SELECT settings FROM loci_indexes WHERE name=?1",
                [name],
                |row| row.get(0),
            )?;
            ensure!(
                &serde_json::from_str::<Value>(&current)? == settings,
                "conflicting settings for existing index {name}"
            );
        }
        tx.commit()?;
        Ok(created)
    }

    pub fn list_indexes(&self) -> Result<Vec<String>> {
        let db = connect(&self.path)?;
        let mut statement = db.prepare("SELECT name FROM loci_indexes ORDER BY name")?;
        let names = statement
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(names)
    }

    pub fn bulk(&self, index: &str, mutations: &[Mutation]) -> Result<WriteResult> {
        self.write(index, mutations, None, None, None)
    }

    /// A repeated key replays its committed result, including after intervening writes.
    pub fn bulk_with_key(
        &self,
        index: &str,
        key: &str,
        mutations: &[Mutation],
    ) -> Result<WriteResult> {
        ensure!(
            !key.is_empty() && key.len() <= 256,
            "invalid idempotency key"
        );
        self.write(index, mutations, Some(key), None, None)
    }

    /// Reject the batch when the index model changed after the caller chose a lexical write.
    pub fn bulk_lexical(
        &self,
        index: &str,
        mutations: &[Mutation],
        key: Option<&str>,
    ) -> Result<WriteResult> {
        self.write(index, mutations, key, Some("lexical"), None)
    }

    fn write(
        &self,
        index: &str,
        mutations: &[Mutation],
        key: Option<&str>,
        expected_model: Option<&str>,
        prepared: Option<(&crate::Embedding, &[PreparedDocument])>,
    ) -> Result<WriteResult> {
        ensure!(
            key.is_none_or(|k| !k.is_empty() && k.len() <= 256),
            "invalid idempotency key"
        );
        let mut db = connect(&self.path)?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let initial = generation(&tx, index)?;
        let raw_settings: String = tx.query_row(
            "SELECT settings FROM loci_indexes WHERE name=?1",
            [index],
            |r| r.get(0),
        )?;
        let settings: Value = serde_json::from_str(&raw_settings)?;
        if let Some(model) = expected_model {
            ensure!(
                settings
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("lexical")
                    == model,
                "stale generation: index model changed"
            );
        }
        let request = key
            .map(|_| {
                serde_json::to_vec(mutations)
                    .map(|payload| format!("{:x}", Sha256::digest(payload)))
            })
            .transpose()?;
        if let Some(key) = key {
            let receipt: Option<(String, String)> = tx
                .query_row(
                    "SELECT request, result FROM loci_write_receipts WHERE idx=?1 AND key=?2",
                    params![index, key],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((original, result)) = receipt {
                ensure!(
                    Some(&original) == request.as_ref(),
                    "idempotency key already used for a different batch"
                );
                return Ok(serde_json::from_str(&result)?);
            }
        }
        let mut changed = 0;
        for mutation in mutations {
            changed += match mutation {
                Mutation::Upsert { document } => {
                    mapping::validate_document(&settings, document)?;
                    upsert(&tx, index, document)?
                }
                Mutation::Delete { id } => {
                    ensure!(!id.trim().is_empty(), "empty document ID");
                    tx.execute("DELETE FROM loci_fts WHERE rowid=(SELECT rowid FROM loci_documents WHERE idx=?1 AND id=?2)", params![index,id])?;
                    tx.execute(
                        "DELETE FROM loci_documents WHERE idx=?1 AND id=?2",
                        params![index, id],
                    )?
                }
            };
        }
        stream::sync(&tx, index, mutations)?;
        if let Some((codec, docs)) = prepared {
            encoded::save(&tx, index, codec, docs)?;
        }
        let next = if changed > 0 {
            tx.execute(
                "UPDATE loci_indexes SET generation=generation+1 WHERE name=?1",
                [index],
            )?;
            generation(&tx, index)?
        } else {
            initial
        };
        let result = WriteResult {
            generation: next,
            changed,
        };
        if let (Some(key), Some(request)) = (key, request) {
            tx.execute(
                "INSERT INTO loci_write_receipts(idx,key,request,result) VALUES(?1,?2,?3,?4)",
                params![index, key, request, serde_json::to_string(&result)?],
            )?;
        }
        tx.commit()?;
        Ok(result)
    }

    /// A snapshot pins all subsequent reads to the same committed generation.
    pub fn snapshot(&self, index: &str) -> Result<Snapshot> {
        let db = connect(&self.path)?;
        db.execute_batch("BEGIN DEFERRED")?;
        let generation = generation(&db, index)?;
        Ok(Snapshot {
            db,
            index: index.to_owned(),
            generation,
        })
    }

    pub fn snapshot_at(&self, index: &str, expected: u64) -> Result<Snapshot> {
        let snapshot = self.snapshot(index)?;
        ensure!(
            snapshot.generation == expected,
            "stale generation: expected {expected}, current {}",
            snapshot.generation
        );
        Ok(snapshot)
    }
}

fn upsert(db: &Connection, index: &str, document: &Document) -> Result<usize> {
    ensure!(
        !document.id.trim().is_empty() && !document.revision.trim().is_empty(),
        "document requires an ID and revision"
    );
    ensure!(
        document.readers.iter().all(|r| !r.trim().is_empty()),
        "empty reader principal"
    );
    let mut document = document.clone();
    document.readers.sort();
    document.readers.dedup();
    let body = serde_json::to_string(&document)?;
    let existing: Option<(String, String)> = db
        .query_row(
            "SELECT revision,body FROM loci_documents WHERE idx=?1 AND id=?2",
            params![index, document.id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    if let Some((revision, previous)) = existing {
        if previous == body {
            return Ok(0);
        }
        if revision == document.revision {
            bail!(
                "document {} has conflicting content for revision {}",
                document.id,
                revision
            );
        }
    }
    let changed = db.execute(
        "INSERT INTO loci_documents(idx,id,revision,body,readers) VALUES(?1,?2,?3,?4,?5)
         ON CONFLICT(idx,id) DO UPDATE SET revision=excluded.revision,body=excluded.body,readers=excluded.readers",
        params![index, document.id, document.revision, body, serde_json::to_string(&document.readers)?],
    )?;
    let rowid: i64 = db.query_row(
        "SELECT rowid FROM loci_documents WHERE idx=?1 AND id=?2",
        params![index, document.id],
        |r| r.get(0),
    )?;
    db.execute("DELETE FROM loci_fts WHERE rowid=?1", [rowid])?;
    let mut identifiers = document.id.clone();
    if let Some(sku) = document.fields.get("sku").and_then(Value::as_str) {
        identifiers.push(' ');
        identifiers.push_str(sku);
    }
    let mut attributes = String::new();
    for (name, value) in &document.fields {
        if matches!(name.as_str(), "sku" | "title") {
            continue;
        }
        let values = value
            .as_array()
            .map_or(std::slice::from_ref(value), Vec::as_slice);
        for value in values.iter().filter_map(Value::as_str) {
            attributes.push_str(value);
            attributes.push(' ');
        }
    }
    db.execute("INSERT INTO loci_fts(rowid,idx,doc_id,identifiers,title,text,attributes) VALUES(?1,?2,?3,?4,?5,?6,?7)",
        params![rowid, index, document.id, identifiers, document.fields.get("title").and_then(Value::as_str).unwrap_or(""), document.text, attributes])?;
    Ok(changed)
}

pub struct Snapshot {
    db: Connection,
    index: String,
    pub generation: u64,
}

// The second parameter is the authenticated principal set. Authorization runs before
// selecting document bodies, so counts, aggregations and hydration share one boundary.
const ACCESS: &str = " AND ((json_array_length(d.readers)=0 AND
    COALESCE((SELECT json_extract(settings,'$.default_visibility') FROM loci_indexes WHERE name=d.idx),'public')='public') OR EXISTS (
    SELECT 1 FROM json_each(d.readers) AS grantee
    JOIN json_each(?2) AS principal ON principal.value=grantee.value))";

impl Snapshot {
    pub fn settings(&self) -> Result<Value> {
        let raw: String = self.db.query_row(
            "SELECT settings FROM loci_indexes WHERE name=?1",
            [&self.index],
            |r| r.get(0),
        )?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn document(&self, id: &str, principals: &[String]) -> Result<Option<Document>> {
        let sql = format!("SELECT d.body FROM loci_documents d WHERE d.idx=?1 AND d.id=?3{ACCESS}");
        let body: Option<String> = self
            .db
            .query_row(
                &sql,
                params![self.index, serde_json::to_string(principals)?, id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(body.map(|json| serde_json::from_str(&json)).transpose()?)
    }

    pub fn documents(&self, principals: &[String]) -> Result<Vec<Document>> {
        let mut documents = Vec::new();
        self.visit_documents(principals, |document| {
            documents.push(document);
            Ok(())
        })?;
        Ok(documents)
    }

    pub fn visit_documents(
        &self,
        principals: &[String],
        mut visit: impl FnMut(Document) -> Result<()>,
    ) -> Result<()> {
        let sql =
            format!("SELECT d.body FROM loci_documents d WHERE d.idx=?1{ACCESS} ORDER BY d.id");
        let mut statement = self.db.prepare(&sql)?;
        let mut rows = statement.query(params![self.index, serde_json::to_string(principals)?])?;
        while let Some(row) = rows.next()? {
            let body: String = row.get(0)?;
            visit(serde_json::from_str(&body)?)?;
        }
        Ok(())
    }
}
