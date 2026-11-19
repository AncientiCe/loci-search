use crate::{
    Embedding, Passage, Query, cosine,
    graph::{Graph, Neighbor},
    normalize,
    ranking::{self, SearchConfig, SearchResult, Strategy},
    terms,
    topology::{Region, Topology},
};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Duration,
};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS manifest (json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
  rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, document TEXT NOT NULL,
  revision TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
  text TEXT NOT NULL, metadata TEXT NOT NULL, vector BLOB NOT NULL, length INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(terms, tokenize='ascii', content='');
CREATE TABLE IF NOT EXISTS regions (id INTEGER PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memberships (
  region INTEGER NOT NULL REFERENCES regions(id),
  artifact INTEGER NOT NULL REFERENCES artifacts(rowid),
  similarity REAL NOT NULL, reason TEXT NOT NULL,
  PRIMARY KEY (region, artifact)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS postings (
  region INTEGER NOT NULL, term TEXT NOT NULL, artifact INTEGER NOT NULL, frequency INTEGER NOT NULL,
  PRIMARY KEY (region, term, artifact),
  FOREIGN KEY (region, artifact) REFERENCES memberships(region, artifact)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS term_stats (term TEXT PRIMARY KEY, df INTEGER NOT NULL) WITHOUT ROWID;
";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    pub schema: u32,
    pub embedding: Embedding,
    pub chunks: usize,
    pub regions: usize,
    pub leaf_size: usize,
    pub average_length: f64,
}

#[derive(Debug, Serialize)]
pub struct Source {
    pub id: String,
    pub document: String,
    pub revision: String,
    pub start: u64,
    pub end: u64,
    pub text: String,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

pub(crate) fn frequencies(text: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for term in terms(text) {
        *counts.entry(term).or_default() += 1;
    }
    counts
}

pub fn ingest(
    path: &Path,
    mut passages: Vec<Passage>,
    embedding: Embedding,
    leaf_size: usize,
) -> Result<Manifest> {
    ensure!(!passages.is_empty(), "at least one passage is required");
    ensure!(
        !embedding.model_revision.trim().is_empty(),
        "model revision is required"
    );
    ensure!(leaf_size > 0, "leaf size must be positive");
    passages.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    ensure!(
        passages.windows(2).all(|pair| pair[0].id != pair[1].id),
        "duplicate chunk ID"
    );
    let mut vectors = Vec::with_capacity(passages.len());
    let mut term_stats = BTreeMap::<String, usize>::new();
    let mut term_lists = Vec::with_capacity(passages.len());
    let mut total_length = 0;
    for passage in &passages {
        ensure!(
            !passage.id.is_empty() && !passage.document.is_empty() && !passage.revision.is_empty(),
            "passage ID, document and revision are required"
        );
        ensure!(
            passage.start <= passage.end && passage.end <= i64::MAX as u64,
            "invalid source span for {}",
            passage.id
        );
        vectors.push(
            normalize(&passage.vector, embedding.dimensions)
                .with_context(|| format!("passage {}", passage.id))?,
        );
        let frequencies = frequencies(&passage.text);
        for term in frequencies.keys() {
            *term_stats.entry(term.clone()).or_default() += 1;
        }
        total_length += frequencies.values().sum::<usize>();
        term_lists.push(frequencies);
    }
    let topology = Topology::build(&vectors, leaf_size);
    let manifest = Manifest {
        schema: 1,
        embedding,
        chunks: passages.len(),
        regions: topology.regions.len(),
        leaf_size,
        average_length: total_length as f64 / passages.len() as f64,
    };
    let mut conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute_batch(SCHEMA)?;
    tx.execute_batch(
        "
        INSERT INTO lexical(lexical) VALUES ('delete-all');
        DELETE FROM postings; DELETE FROM memberships; DELETE FROM regions;
        DELETE FROM artifacts; DELETE FROM term_stats; DELETE FROM manifest;
    ",
    )?;
    tx.execute(
        "INSERT INTO manifest VALUES (?)",
        [serde_json::to_string(&manifest)?],
    )?;
    {
        let mut artifact =
            tx.prepare("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")?;
        let mut lexical = tx.prepare("INSERT INTO lexical(rowid, terms) VALUES (?, ?)")?;
        let mut region = tx.prepare("INSERT INTO regions VALUES (?, ?)")?;
        let mut membership = tx.prepare("INSERT INTO memberships VALUES (?, ?, ?, ?)")?;
        let mut posting = tx.prepare("INSERT INTO postings VALUES (?, ?, ?, ?)")?;
        let mut stat = tx.prepare("INSERT INTO term_stats VALUES (?, ?)")?;
        for (id, passage) in passages.iter().enumerate() {
            let vector = vectors[id]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            let length = term_lists[id].values().sum::<usize>();
            artifact.execute(params![
                id,
                passage.id,
                passage.document,
                passage.revision,
                passage.start,
                passage.end,
                passage.text,
                serde_json::to_string(&passage.metadata)?,
                vector,
                length
            ])?;
            let tokenized = terms(&passage.text).collect::<Vec<_>>().join(" ");
            lexical.execute(params![id, tokenized])?;
        }
        let mut best = vec![f32::NEG_INFINITY; passages.len()];
        for (region, ids) in topology.members.iter().enumerate() {
            for &id in ids {
                best[id] = best[id].max(cosine(&vectors[id], &topology.regions[region].centroid));
            }
        }
        for (id, node) in topology.regions.iter().enumerate() {
            region.execute(params![id, serde_json::to_string(node)?])?;
            for &chunk in &topology.members[id] {
                let similarity = cosine(&vectors[chunk], &node.centroid);
                let reason = if similarity == best[chunk] {
                    "best_centroid"
                } else {
                    "within_margin"
                };
                membership.execute(params![id, chunk, similarity, reason])?;
                for (term, &frequency) in &term_lists[chunk] {
                    posting.execute(params![id, term, chunk, frequency])?;
                }
            }
        }
        for (term, df) in term_stats {
            stat.execute(params![term, df])?;
        }
    }
    tx.commit()?;
    Ok(manifest)
}

pub struct Engine {
    pub manifest: Manifest,
    pub(crate) conn: Connection,
    pub(crate) ids: Vec<String>,
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) lengths: Vec<usize>,
    pub(crate) topology: Topology,
    graph: Option<Graph>,
}

impl Engine {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        // Keep lexical lookups on the same snapshot as the in-memory vectors and topology.
        conn.execute_batch("BEGIN")?;
        let json: String = conn.query_row("SELECT json FROM manifest", [], |row| row.get(0))?;
        let manifest: Manifest = serde_json::from_str(&json)?;
        ensure!(
            manifest.schema == 1,
            "unsupported index schema {}",
            manifest.schema
        );
        let mut ids = Vec::with_capacity(manifest.chunks);
        let mut vectors = Vec::with_capacity(manifest.chunks);
        let mut lengths = Vec::with_capacity(manifest.chunks);
        let mut regions = Vec::<Region>::with_capacity(manifest.regions);
        let mut members = vec![Vec::new(); manifest.regions];
        {
            let mut statement =
                conn.prepare("SELECT rowid, id, vector, length FROM artifacts ORDER BY rowid")?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                ensure!(
                    row.get::<_, usize>(0)? == ids.len(),
                    "unexpected artifact row ID"
                );
                ids.push(row.get(1)?);
                let bytes: Vec<u8> = row.get(2)?;
                let (components, remainder) = bytes.as_chunks::<4>();
                ensure!(
                    components.len() == manifest.embedding.dimensions && remainder.is_empty(),
                    "invalid stored vector length"
                );
                vectors.push(components.iter().copied().map(f32::from_le_bytes).collect());
                lengths.push(row.get(3)?);
            }
            let mut statement = conn.prepare("SELECT id, json FROM regions ORDER BY id")?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                ensure!(
                    row.get::<_, usize>(0)? == regions.len(),
                    "unexpected region ID"
                );
                regions.push(serde_json::from_str(&row.get::<_, String>(1)?)?);
            }
            let mut statement =
                conn.prepare("SELECT region, artifact FROM memberships ORDER BY region, artifact")?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let region: usize = row.get(0)?;
                let artifact: usize = row.get(1)?;
                ensure!(
                    region < regions.len() && artifact < ids.len(),
                    "invalid stored membership"
                );
                members[region].push(artifact);
            }
        }
        ensure!(
            ids.len() == manifest.chunks && regions.len() == manifest.regions,
            "incomplete index"
        );
        Ok(Self {
            manifest,
            conn,
            ids,
            vectors,
            lengths,
            topology: Topology { regions, members },
            graph: None,
        })
    }

    pub fn prepare_global(&mut self) {
        if self.graph.is_none() {
            self.graph = Some(Graph::build(&self.vectors));
        }
    }

    pub fn search(
        &mut self,
        query: &Query,
        strategy: Strategy,
        config: &SearchConfig,
    ) -> Result<SearchResult> {
        config.validate()?;
        let vector = self.query_vector(query)?;
        if strategy == Strategy::Global {
            self.prepare_global();
        }
        ranking::search(self, query, &vector, strategy, config)
    }

    pub(crate) fn query_vector(&self, query: &Query) -> Result<Vec<f32>> {
        ensure!(
            query.embedding == self.manifest.embedding,
            "query {} embedding manifest differs from the index",
            query.id
        );
        normalize(&query.vector, self.manifest.embedding.dimensions)
            .with_context(|| format!("query {}", query.id))
    }

    pub(crate) fn nearest(
        &self,
        vector: &[f32],
        depth: usize,
        width: usize,
    ) -> (Vec<Neighbor>, usize) {
        self.graph
            .as_ref()
            .expect("prepared graph")
            .search(&self.vectors, vector, depth, width)
    }

    pub(crate) fn exact(&self, vector: &[f32], depth: usize) -> Vec<Neighbor> {
        let mut ranked = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, candidate)| Neighbor {
                id,
                similarity: cosine(vector, candidate),
            })
            .collect::<Vec<_>>();
        if depth < ranked.len() {
            ranked.select_nth_unstable_by(depth, |a, b| b.cmp(a));
            ranked.truncate(depth);
        }
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        ranked
    }

    pub(crate) fn lexical(&self, terms: &BTreeSet<String>, limit: usize) -> Result<Vec<usize>> {
        if terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let matches = terms
            .iter()
            .map(|term| format!("\"{term}\""))
            .collect::<Vec<_>>()
            .join(" OR ");
        let mut stmt = self.conn.prepare_cached(
            "SELECT rowid FROM lexical WHERE lexical MATCH ? ORDER BY rank, rowid LIMIT ?",
        )?;
        let rows = stmt.query_map(params![matches, limit], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub(crate) fn source(&self, id: usize) -> Result<Source> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, document, revision, start, end, text, metadata FROM artifacts WHERE rowid=?")?;
        let (id, document, revision, start, end, text, metadata): (_, _, _, _, _, _, String) = stmt
            .query_row([id], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })?;
        Ok(Source {
            id,
            document,
            revision,
            start,
            end,
            text,
            metadata: serde_json::from_str(&metadata)?,
        })
    }
}
