use super::{Catalog, Document, Mutation, Snapshot, WriteResult, connect, generation};
use crate::{Embedding, Passage};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, TransactionBehavior, params};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// Model outputs prepared outside the catalog writer transaction.
pub struct EncodedPassage {
    pub start: u64,
    pub end: u64,
    pub vector: Vec<f32>,
}

pub struct PreparedDocument {
    pub id: String,
    pub revision: String,
    pub passages: Vec<EncodedPassage>,
}

pub(super) fn initialize(db: &Connection) -> Result<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS loci_vectors (
        idx TEXT NOT NULL, doc TEXT NOT NULL, codec TEXT NOT NULL, start INTEGER NOT NULL,
        end INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(idx,doc,codec,start),
        FOREIGN KEY(idx,doc) REFERENCES loci_documents(idx,id) ON DELETE CASCADE
    ) WITHOUT ROWID;
    CREATE TRIGGER IF NOT EXISTS loci_forget_updated_vectors AFTER UPDATE OF body ON loci_documents
    BEGIN DELETE FROM loci_vectors WHERE idx=NEW.idx AND doc=NEW.id; END;",
    )?;
    Ok(())
}

pub(super) fn save(
    db: &Connection,
    index: &str,
    codec: &Embedding,
    docs: &[PreparedDocument],
) -> Result<()> {
    let mut put=db.prepare("INSERT OR REPLACE INTO loci_vectors(idx,doc,codec,start,end,vector) VALUES(?1,?2,?3,?4,?5,?6)")?;
    let mut read=db.prepare("SELECT revision,length(CAST(json_extract(body,'$.text') AS BLOB)) FROM loci_documents WHERE idx=? AND id=?")?;
    for doc in docs {
        let (revision, length): (String, u64) =
            read.query_row(params![index, doc.id], |r| Ok((r.get(0)?, r.get(1)?)))?;
        ensure!(
            revision == doc.revision,
            "stale document revision during encoding"
        );
        let mut previous = 0;
        for p in &doc.passages {
            ensure!(
                p.start >= previous && p.end > p.start && p.end <= length,
                "invalid encoded source interval"
            );
            ensure!(
                p.vector.len() == codec.dimensions && p.vector.iter().all(|v| v.is_finite()),
                "invalid encoded vector"
            );
            let blob = p
                .vector
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            put.execute(params![
                index,
                doc.id,
                codec.model_revision,
                p.start,
                p.end,
                blob
            ])?;
            previous = p.end;
        }
    }
    Ok(())
}

impl Catalog {
    /// Commit prepared vectors with the source edits they represent.
    pub fn bulk_encoded(
        &self,
        index: &str,
        ops: &[Mutation],
        key: Option<&str>,
        model: &str,
        codec: &Embedding,
        docs: &[PreparedDocument],
    ) -> Result<WriteResult> {
        self.write(index, ops, key, Some(model), Some((codec, docs)))
    }

    /// Switch model and vectors in one transaction if the source generation still matches.
    pub fn publish_model(
        &self,
        index: &str,
        expected: u64,
        model: &str,
        codec: &Embedding,
        docs: &[PreparedDocument],
    ) -> Result<u64> {
        let mut db = connect(&self.path)?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        ensure!(
            generation(&tx, index)? == expected,
            "stale generation while upgrading model"
        );
        let mut required = HashSet::new();
        {
            let mut stored = tx.prepare("SELECT body FROM loci_documents WHERE idx=?")?;
            let mut rows = stored.query([index])?;
            while let Some(row) = rows.next()? {
                let doc: Document = serde_json::from_str(&row.get::<_, String>(0)?)?;
                if doc.embed {
                    required.insert(doc.id);
                }
            }
        }
        let prepared_ids = docs
            .iter()
            .map(|doc| doc.id.clone())
            .collect::<HashSet<_>>();
        ensure!(
            prepared_ids.len() == docs.len() && prepared_ids == required,
            "model upgrade must encode every embedded document once"
        );
        tx.execute("DELETE FROM loci_vectors WHERE idx=?", [index])?;
        save(&tx, index, codec, docs)?;
        tx.execute("UPDATE loci_indexes SET settings=json_set(settings,'$.model',?2), generation=generation+1 WHERE name=?1",params![index,model])?;
        let next = generation(&tx, index)?;
        tx.commit()?;
        Ok(next)
    }
}

impl Snapshot {
    /// Internal data for an index-wide model rebuild. Call only from an administrative operation.
    pub fn documents_for_model_upgrade(&self) -> Result<Vec<Document>> {
        let mut s = self
            .db
            .prepare("SELECT body FROM loci_documents WHERE idx=? ORDER BY id")?;
        let mut rows = s.query([&self.index])?;
        let mut docs = Vec::new();
        while let Some(r) = rows.next()? {
            docs.push(serde_json::from_str(&r.get::<_, String>(0)?)?);
        }
        Ok(docs)
    }

    /// Hydrate persisted vectors only for documents already selected by the permission filter.
    pub fn encoded_passages(
        &self,
        docs: &[Document],
        codec: &Embedding,
    ) -> Result<(Vec<Passage>, Vec<usize>)> {
        let mut s=self.db.prepare("SELECT start,end,vector FROM loci_vectors WHERE idx=?1 AND doc=?2 AND codec=?3 ORDER BY start")?;
        let mut passages = Vec::new();
        let mut missing = Vec::new();
        for (pos, d) in docs.iter().enumerate() {
            let mut rows = s.query(params![self.index, d.id, codec.model_revision])?;
            let before = passages.len();
            while let Some(r) = rows.next()? {
                let start: usize = r.get(0)?;
                let end: usize = r.get(1)?;
                let bytes: Vec<u8> = r.get(2)?;
                ensure!(
                    bytes.len() == codec.dimensions * 4,
                    "invalid stored vector size"
                );
                let vector = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                passages.push(Passage {
                    id: format!("{}:{start}", d.id),
                    document: d.id.clone(),
                    revision: d.revision.clone(),
                    start: start as u64,
                    end: end as u64,
                    text: d
                        .text
                        .get(start..end)
                        .context("invalid stored source range")?
                        .into(),
                    metadata: BTreeMap::<String, Value>::clone(&d.fields),
                    vector,
                });
            }
            if passages.len() == before && d.embed && !d.text.trim().is_empty() {
                missing.push(pos);
            }
        }
        Ok((passages, missing))
    }
}
