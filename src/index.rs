use crate::{
    Embedding, Passage, Query, cosine,
    graph::{Graph, Neighbor},
    normalize,
    ranking::{self, SearchConfig, SearchResult, Strategy},
    regional::RegionalIndex,
    terms,
    topology::{Region, Topology},
};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Deref,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

/// Read-only SQLite connections pinned to one index snapshot.
///
/// A connection serves one search at a time, so concurrent searches over a shared
/// `Engine` each take their own. Extra connections open lazily and must see the
/// same manifest as the first, so every reader stays on one generation.
pub(crate) struct Readers {
    path: PathBuf,
    manifest: String,
    idle: Mutex<Vec<Connection>>,
}

pub(crate) struct Reader<'a> {
    pool: &'a Readers,
    conn: Option<Connection>,
}

impl Readers {
    fn new(conn: Connection, path: &Path, manifest: String) -> Self {
        Self {
            path: path.to_path_buf(),
            manifest,
            idle: Mutex::new(vec![conn]),
        }
    }

    pub(crate) fn get(&self) -> Result<Reader<'_>> {
        let idle = self
            .idle
            .lock()
            .map_err(|_| anyhow::anyhow!("index reader pool poisoned"))?
            .pop();
        let conn = match idle {
            Some(conn) => conn,
            None => {
                let conn = open_snapshot(&self.path)?;
                let json: String =
                    conn.query_row("SELECT json FROM manifest", [], |row| row.get(0))?;
                ensure!(
                    json == self.manifest,
                    "index {} changed after it was opened",
                    self.path.display()
                );
                conn
            }
        };
        Ok(Reader {
            pool: self,
            conn: Some(conn),
        })
    }
}

impl Deref for Reader<'_> {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("reader connection")
    }
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        if let (Some(conn), Ok(mut idle)) = (self.conn.take(), self.pool.idle.lock()) {
            idle.push(conn);
        }
    }
}

fn open_snapshot(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    // Keep lexical lookups on the same snapshot as the in-memory vectors and topology.
    conn.execute_batch("BEGIN")?;
    Ok(conn)
}

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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalProfile {
    Palace,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Manifest {
    pub schema: u32,
    pub embedding: Embedding,
    pub chunks: usize,
    pub regions: usize,
    pub leaf_size: usize,
    pub average_length: f64,
    /// Absent keeps the coverage ranker, including indexes built before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_profile: Option<RetrievalProfile>,
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
    passages: Vec<Passage>,
    embedding: Embedding,
    leaf_size: usize,
) -> Result<Manifest> {
    ingest_with_layout(path, passages, embedding, leaf_size, false, None)
}

/// Store canonical text and memberships without duplicating reconstructed regional postings.
pub fn ingest_compact(
    path: &Path,
    passages: Vec<Passage>,
    embedding: Embedding,
    leaf_size: usize,
) -> Result<Manifest> {
    ingest_with_layout(path, passages, embedding, leaf_size, true, None)
}

pub fn ingest_with_profile(
    path: &Path,
    passages: Vec<Passage>,
    embedding: Embedding,
    leaf_size: usize,
    profile: Option<RetrievalProfile>,
) -> Result<Manifest> {
    ingest_with_layout(path, passages, embedding, leaf_size, false, profile)
}

pub fn ingest_compact_with_profile(
    path: &Path,
    passages: Vec<Passage>,
    embedding: Embedding,
    leaf_size: usize,
    profile: Option<RetrievalProfile>,
) -> Result<Manifest> {
    ingest_with_layout(path, passages, embedding, leaf_size, true, profile)
}

fn ingest_with_layout(
    path: &Path,
    mut passages: Vec<Passage>,
    embedding: Embedding,
    leaf_size: usize,
    compact: bool,
    profile: Option<RetrievalProfile>,
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
    let mut term_lists = (!compact).then(|| Vec::with_capacity(passages.len()));
    let mut lengths = Vec::with_capacity(passages.len());
    let mut total_length = 0;
    for passage in &mut passages {
        ensure!(
            !passage.id.is_empty() && !passage.document.is_empty() && !passage.revision.is_empty(),
            "passage ID, document and revision are required"
        );
        ensure!(
            passage.start <= passage.end && passage.end <= i64::MAX as u64,
            "invalid source span for {}",
            passage.id
        );
        let vector = std::mem::take(&mut passage.vector);
        vectors.push(
            normalize(&vector, embedding.dimensions)
                .with_context(|| format!("passage {}", passage.id))?,
        );
        let frequencies = frequencies(&passage.text);
        for term in frequencies.keys() {
            *term_stats.entry(term.clone()).or_default() += 1;
        }
        let length = frequencies.values().sum::<usize>();
        total_length += length;
        lengths.push(length);
        if let Some(lists) = &mut term_lists {
            lists.push(frequencies);
        }
    }
    if compact {
        eprintln!("Building the topology for {} passages", passages.len());
    }
    let topology = Topology::build(&vectors, leaf_size);
    if compact {
        eprintln!("Writing canonical passages and memberships to SQLite");
    }
    let manifest = Manifest {
        schema: if compact { 2 } else { 1 },
        embedding,
        chunks: passages.len(),
        regions: topology.regions.len(),
        leaf_size,
        average_length: total_length as f64 / passages.len() as f64,
        retrieval_profile: profile,
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
                lengths[id]
            ])?;
            let tokenized = terms(&passage.text).collect::<Vec<_>>().join(" ");
            lexical.execute(params![id, tokenized])?;
            if compact && (id + 1).is_multiple_of(100_000) {
                eprintln!("Wrote {} of {} passages", id + 1, passages.len());
            }
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
                if let Some(lists) = &term_lists {
                    for (term, &frequency) in &lists[chunk] {
                        posting.execute(params![id, term, chunk, frequency])?;
                    }
                }
            }
        }
        for (term, df) in term_stats {
            stat.execute(params![term, df])?;
        }
    }
    if profile == Some(RetrievalProfile::Palace) {
        write_palace_stats(&tx, &passages)?;
    }
    tx.commit()?;
    Ok(manifest)
}

fn write_palace_stats(tx: &rusqlite::Transaction<'_>, passages: &[Passage]) -> Result<()> {
    tx.execute_batch(
        "
        CREATE TABLE palace_doc (artifact INTEGER PRIMARY KEY, length INTEGER NOT NULL);
        CREATE TABLE palace_df (term TEXT PRIMARY KEY, df INTEGER NOT NULL) WITHOUT ROWID;
        ",
    )?;
    let texts: Vec<&str> = passages
        .iter()
        .map(|passage| passage.text.as_str())
        .collect();
    let (corpus, lengths) = crate::palace::build_corpus(&texts);
    let mut doc = tx.prepare("INSERT INTO palace_doc VALUES (?, ?)")?;
    for (artifact, length) in lengths.iter().enumerate() {
        doc.execute(params![artifact, *length as i64])?;
    }
    let mut frequency = tx.prepare("INSERT INTO palace_df VALUES (?, ?)")?;
    for (term, df) in corpus.document_frequency {
        frequency.execute(params![term, df])?;
    }
    Ok(())
}

pub(crate) struct PalaceStats {
    pub doc_len: Vec<u32>,
    pub document_frequency: HashMap<String, u32>,
    pub documents: f64,
    pub average_length: f64,
}

fn load_palace_stats(conn: &Connection, chunks: usize) -> Result<PalaceStats> {
    let mut doc_len = vec![0u32; chunks];
    let mut total = 0u64;
    let mut seen = 0usize;
    let mut statement = conn.prepare("SELECT artifact, length FROM palace_doc")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let artifact: usize = row.get(0)?;
        let length: u32 = row.get(1)?;
        ensure!(artifact < chunks, "palace document id is outside the index");
        doc_len[artifact] = length;
        total += u64::from(length);
        seen += 1;
    }
    ensure!(
        seen == chunks,
        "palace document lengths do not cover the index"
    );
    let mut document_frequency = HashMap::new();
    let mut statement = conn.prepare("SELECT term, df FROM palace_df")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        document_frequency.insert(row.get(0)?, row.get(1)?);
    }
    let documents = chunks as f64;
    Ok(PalaceStats {
        doc_len,
        document_frequency,
        documents,
        average_length: if chunks == 0 {
            0.0
        } else {
            total as f64 / documents
        },
    })
}

pub struct Engine {
    pub manifest: Manifest,
    pub(crate) conn: Readers,
    pub(crate) ids: Vec<String>,
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) lengths: Vec<usize>,
    pub(crate) topology: Topology,
    graph: Option<Graph>,
    dense_graph: Option<Graph>,
    regional: Option<RegionalIndex>,
    pub(crate) palace: Option<PalaceStats>,
    path: PathBuf,
}

impl Engine {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = open_snapshot(path)?;
        let json: String = conn.query_row("SELECT json FROM manifest", [], |row| row.get(0))?;
        let manifest: Manifest = serde_json::from_str(&json)?;
        ensure!(
            matches!(manifest.schema, 1 | 2),
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
        let palace = if manifest.retrieval_profile == Some(RetrievalProfile::Palace) {
            Some(load_palace_stats(&conn, ids.len())?)
        } else {
            None
        };
        Ok(Self {
            manifest,
            conn: Readers::new(conn, path, json),
            ids,
            vectors,
            lengths,
            topology: Topology { regions, members },
            graph: None,
            dense_graph: None,
            regional: None,
            palace,
            path: path.to_path_buf(),
        })
    }

    pub fn prepare_global(&mut self) {
        if self.graph.is_none() {
            self.graph = Some(Graph::build(&self.vectors));
        }
    }

    pub fn database_path(&self) -> &Path {
        &self.path
    }

    /// Path of the persisted batched HNSW graph for this index.
    pub fn graph_path(&self) -> PathBuf {
        let mut path = self.path.clone().into_os_string();
        path.push(".hnsw");
        path.into()
    }

    /// Load the persisted batched HNSW graph, building and saving it if needed.
    pub(crate) fn prepare_dense_graph(&mut self) -> Result<()> {
        if self.dense_graph.is_none() {
            let identity = self.graph_identity()?;
            let path = self.graph_path();
            let graph = match Graph::load(&path, &identity)? {
                Some(graph) => graph,
                None => {
                    let graph = Graph::build_batched(&self.vectors);
                    graph.save(&path, &identity)?;
                    graph
                }
            };
            ensure!(
                graph.len() == self.ids.len(),
                "graph size differs from the index"
            );
            self.dense_graph = Some(graph);
        }
        Ok(())
    }

    /// Rebuild and save the batched HNSW graph, replacing any saved copy.
    pub fn build_dense_graph(&mut self) -> Result<PathBuf> {
        let identity = self.graph_identity()?;
        let path = self.graph_path();
        let graph = Graph::build_batched(&self.vectors);
        graph.save(&path, &identity)?;
        self.dense_graph = Some(graph);
        Ok(path)
    }

    /// Saved graphs are tied to these vectors, not only to the passage count.
    fn graph_identity(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(&self.manifest)?);
        for vector in &self.vectors {
            hasher.update((vector.len() as u32).to_le_bytes());
            for value in vector {
                hasher.update(value.to_le_bytes());
            }
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub(crate) fn nearest_dense(
        &self,
        vector: &[f32],
        depth: usize,
        width: usize,
    ) -> (Vec<Neighbor>, usize) {
        self.dense_graph
            .as_ref()
            .expect("prepared dense graph")
            .search(&self.vectors, vector, depth, width)
    }

    pub(crate) fn prepare_regional(&mut self) -> Result<()> {
        if self.regional.is_none() {
            self.regional = Some(RegionalIndex::load(self)?);
        }
        Ok(())
    }

    pub(crate) fn regional_index(&self) -> &RegionalIndex {
        self.regional.as_ref().expect("prepared regional index")
    }

    pub fn search(
        &mut self,
        query: &Query,
        strategy: Strategy,
        config: &SearchConfig,
    ) -> Result<SearchResult> {
        self.prepare(strategy, config)?;
        self.search_prepared(query, strategy, config)
    }

    /// Load the in-memory structures `strategy` needs under `config`.
    pub fn prepare(&mut self, strategy: Strategy, config: &SearchConfig) -> Result<()> {
        config.validate()?;
        if strategy == Strategy::Global {
            self.prepare_global();
        }
        if strategy == Strategy::Regional {
            self.prepare_regional()?;
            if config.graph_candidates.is_some() {
                self.prepare_dense_graph()?;
            }
        }
        Ok(())
    }

    /// Search through a shared engine after [`Engine::prepare`] ran for this strategy and config.
    pub fn search_prepared(
        &self,
        query: &Query,
        strategy: Strategy,
        config: &SearchConfig,
    ) -> Result<SearchResult> {
        config.validate()?;
        let ready = match strategy {
            Strategy::Global => self.graph.is_some(),
            Strategy::Regional => {
                self.regional.is_some()
                    && (config.graph_candidates.is_none() || self.dense_graph.is_some())
            }
            _ => true,
        };
        ensure!(
            ready,
            "prepare the engine for {strategy:?} before a shared search"
        );
        let vector = self.query_vector(query)?;
        ranking::search(self, query, &vector, strategy, config)
    }

    pub(crate) fn profile_regional(
        &mut self,
        query: &Query,
        config: &SearchConfig,
        phases: &mut crate::profile::Phases,
    ) -> Result<SearchResult> {
        config.validate()?;
        let vector = self.query_vector(query)?;
        self.prepare_regional()?;
        if config.graph_candidates.is_some() {
            self.prepare_dense_graph()?;
        }
        ranking::profile_regional(self, query, &vector, config, phases)
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
        let conn = self.conn.get()?;
        let mut stmt = conn.prepare_cached(
            "SELECT rowid FROM lexical WHERE lexical MATCH ? ORDER BY rank, rowid LIMIT ?",
        )?;
        let rows = stmt.query_map(params![matches, limit], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub(crate) fn source(&self, id: usize) -> Result<Source> {
        let conn = self.conn.get()?;
        let mut stmt = conn.prepare_cached(
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
