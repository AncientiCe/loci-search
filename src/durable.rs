//! Durable passages in PostgreSQL and atomic publication of a retrieval generation.
//!
//! PostgreSQL stores sources, revisions, passage text, byte or turn spans, metadata,
//! the model fingerprint, embedding bytes, and an append-only changelog. Serving a
//! query does not open PostgreSQL. Open a [`Generations`] directory and read citation
//! text from the SQLite snapshot already published there.
//!
//! A generation directory contains `manifest.json`, the existing retrieval SQLite
//! file (`manifest` and `artifacts` tables, plus the engine's other tables), and an
//! optional `.hnsw` sidecar. The sidecar name is the SQLite file name with `.hnsw`
//! appended, the same path the indexer already uses. [`Generations::publish`] renames
//! a completed staging directory into `generations/`. A partial directory is ignored.
//! A [`GenerationReader`] keeps the directory it opened.
//!
//! Indexer order:
//! 1. [`Durable::commit`] or [`Durable::delete`]. The changelog transaction commits
//!    before either call returns.
//! 2. Build the retrieval SQLite file, and the optional `.hnsw` file, inside
//!    [`Durable::create_staging`].
//! 3. [`GenerationManifest::write`].
//! 4. [`Durable::publish`], which checks that the manifest's changelog sequence is
//!    already committed and then renames the directory into place.
//!
//! A crash between the changelog commit and the rename leaves the changelog committed.
//! [`Durable::resume`] reports those commits and ignores partial directories. Rebuild
//! from [`Durable::live_passages`], or publish a staging directory listed in
//! [`ResumeState::ready_staging`] when that directory is already complete.

use anyhow::{Context, Result, bail, ensure};
use postgres::types::ToSql;
use postgres::{Client, Config, NoTls, Transaction};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_SCHEMA: &str = "loci";

/// One passage written into a source revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassageWrite {
    pub id: String,
    pub text: String,
    pub span: Span,
    pub metadata: Value,
    pub embedding: Vec<u8>,
}

impl PassageWrite {
    pub fn new(
        id: impl Into<String>,
        text: impl Into<String>,
        span: Span,
        embedding: Vec<u8>,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            span,
            metadata: Value::Object(serde_json::Map::new()),
            embedding,
        }
    }
}

/// Byte offset or conversation-turn span covered by a passage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub kind: SpanKind,
    pub start: u64,
    pub end: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Byte,
    Turn,
}

impl SpanKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Byte => "byte",
            Self::Turn => "turn",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "byte" => Ok(Self::Byte),
            "turn" => Ok(Self::Turn),
            other => bail!("invalid span kind {other}"),
        }
    }
}

/// A source revision and the passages that belong to it.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceCommit {
    pub source_id: String,
    pub revision: String,
    pub model_fingerprint: String,
    pub metadata: Value,
    pub passages: Vec<PassageWrite>,
}

impl SourceCommit {
    pub fn new(
        source_id: impl Into<String>,
        revision: impl Into<String>,
        model_fingerprint: impl Into<String>,
        passages: Vec<PassageWrite>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            revision: revision.into(),
            model_fingerprint: model_fingerprint.into(),
            metadata: Value::Object(serde_json::Map::new()),
            passages,
        }
    }

    /// Checksum of this revision. A retry with the same revision and checksum is idempotent.
    pub fn checksum(&self) -> Result<String> {
        validate_commit(self)?;
        digest(self)
    }
}

/// Result of an idempotent changelog append.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use]
pub enum CommitOutcome {
    /// This call appended one changelog row.
    Committed { seq: i64, checksum: String },
    /// The same source revision and checksum were already committed.
    AlreadyCommitted { seq: i64, checksum: String },
}

impl CommitOutcome {
    pub fn seq(&self) -> i64 {
        match self {
            Self::Committed { seq, .. } | Self::AlreadyCommitted { seq, .. } => *seq,
        }
    }

    pub fn checksum(&self) -> &str {
        match self {
            Self::Committed { checksum, .. } | Self::AlreadyCommitted { checksum, .. } => checksum,
        }
    }

    pub fn is_new(&self) -> bool {
        matches!(self, Self::Committed { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeOp {
    Upsert,
    Delete,
}

impl ChangeOp {
    fn as_str(self) -> &'static str {
        match self {
            Self::Upsert => "upsert",
            Self::Delete => "delete",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "upsert" => Ok(Self::Upsert),
            "delete" => Ok(Self::Delete),
            other => bail!("invalid changelog op {other}"),
        }
    }
}

/// One appended changelog row.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Change {
    pub seq: i64,
    pub op: ChangeOp,
    pub source_id: String,
    pub revision: String,
    pub checksum: String,
}

/// A passage from a committed revision.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredPassage {
    pub source_id: String,
    pub revision: String,
    pub checksum: String,
    pub passage_id: String,
    pub text: String,
    pub span: Span,
    pub metadata: Value,
    pub source_metadata: Value,
    pub model_fingerprint: String,
    pub embedding: Vec<u8>,
    pub changelog_seq: i64,
}

/// `manifest.json` written beside the retrieval SQLite file.
///
/// `index` is the retrieval manifest (`schema`, `embedding`, `chunks`, `regions`,
/// `leaf_size`, `average_length`). Extra index fields are preserved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationManifest {
    pub changelog_seq: i64,
    pub sqlite: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hnsw: Option<String>,
    pub index: Value,
}

impl GenerationManifest {
    pub fn new(changelog_seq: i64, sqlite: impl Into<String>, index: Value) -> Self {
        Self {
            changelog_seq,
            sqlite: sqlite.into(),
            hnsw: None,
            index,
        }
    }

    pub fn with_hnsw(mut self, file_name: impl Into<String>) -> Self {
        self.hnsw = Some(file_name.into());
        self
    }

    pub fn write(&self, directory: &Path) -> Result<()> {
        ensure!(directory.is_dir(), "generation directory does not exist");
        let path = directory.join("manifest.json");
        let mut file =
            File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        serde_json::to_writer_pretty(&mut file, self)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }
}

/// File name of the HNSW sidecar for a retrieval SQLite file.
///
/// `retrieval.sqlite` becomes `retrieval.sqlite.hnsw`, matching the indexer's
/// graph path.
pub fn hnsw_file_name(sqlite_file_name: &str) -> String {
    format!("{sqlite_file_name}.hnsw")
}

/// A generation directory that has been renamed into `generations/`.
#[derive(Clone, Debug, PartialEq)]
pub struct PublishedGeneration {
    pub id: String,
    pub directory: PathBuf,
    pub manifest: GenerationManifest,
}

/// Filesystem view of staging and published generations. This type does not open PostgreSQL.
#[derive(Debug)]
pub struct Readiness {
    /// Newest complete directory under `generations/`.
    pub current: Option<PublishedGeneration>,
    /// Staging or generation directories that are missing a manifest, the SQLite file, or the declared sidecar.
    pub ignored_partials: Vec<PathBuf>,
    /// Complete directories still under `staging/`. They are not served until [`Generations::publish`].
    pub ready_staging: Vec<PathBuf>,
}

/// Root of staging directories, published generations, and the `current` pointer file.
#[derive(Debug)]
pub struct Generations {
    root: PathBuf,
}

impl Generations {
    /// Create `staging/` and `generations/` under `root`. Does not connect to PostgreSQL.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        ensure!(!root.as_os_str().is_empty(), "generation root is required");
        fs::create_dir_all(root.join("staging"))?;
        fs::create_dir_all(root.join("generations"))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory readers will not open. The indexer writes the SQLite snapshot here.
    pub fn create_staging(&self) -> Result<PathBuf> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let nanos = unix_nanos();
        let path = self
            .root
            .join("staging")
            .join(format!("s{nanos:020}-{n:08}"));
        fs::create_dir(&path)?;
        Ok(path)
    }

    /// Newest complete generation. Partial directories are ignored.
    pub fn published(&self) -> Result<Option<PublishedGeneration>> {
        let (complete, _) = inspect(&self.root.join("generations"))?;
        Ok(newest(complete))
    }

    pub fn readiness(&self) -> Result<Readiness> {
        let (generations, mut partials) = inspect(&self.root.join("generations"))?;
        let (ready, staging_partials) = inspect(&self.root.join("staging"))?;
        partials.extend(staging_partials);
        partials.sort();
        Ok(Readiness {
            current: newest(generations),
            ignored_partials: partials,
            ready_staging: ready
                .into_iter()
                .map(|generation| generation.directory)
                .collect(),
        })
    }

    /// Point `current` at the newest complete generation, or remove it when none exists.
    ///
    /// Readers resolve the newest complete directory directly. This file records that
    /// choice for operators after a crash between the rename and the pointer write.
    pub fn repair_pointer(&self) -> Result<Option<PublishedGeneration>> {
        let published = self.published()?;
        match &published {
            Some(generation) => {
                let expected = format!("generations/{}", generation.id);
                if self.pointer_target()?.as_deref() != Some(expected.as_str()) {
                    self.write_current(generation)?;
                }
            }
            None => {
                let path = self.root.join("current");
                if path.exists() {
                    fs::remove_file(&path)?;
                }
            }
        }
        Ok(published)
    }

    /// Validate a completed staging directory and rename it into `generations/`.
    ///
    /// The staging directory must be a direct child of `staging/`. On success it no
    /// longer exists at the old path. An incomplete directory is left in place and
    /// returns an error whose message starts with `incomplete generation`.
    pub fn publish(&self, staging: &Path) -> Result<PublishedGeneration> {
        let staging = self.staging_child(staging)?;
        let manifest = load_complete(&staging)?;
        sync_generation_files(&staging, &manifest)?;
        let id = new_generation_id(manifest.changelog_seq);
        let dest = self.root.join("generations").join(&id);
        ensure!(
            !dest.exists(),
            "generation id collision at {}",
            dest.display()
        );
        fs::rename(&staging, &dest).with_context(|| format!("publishing {}", dest.display()))?;
        let published = PublishedGeneration {
            id,
            directory: dest,
            manifest,
        };
        self.write_current(&published)?;
        Ok(published)
    }

    /// Open the newest complete generation. Does not connect to PostgreSQL.
    ///
    /// Returns `None` when every candidate directory is partial or missing. The
    /// returned reader is not retargeted by a later publish.
    pub fn open_reader(&self) -> Result<Option<GenerationReader>> {
        self.published()?.map(GenerationReader::open).transpose()
    }

    /// Open one published generation by directory id. Does not connect to PostgreSQL.
    ///
    /// Publishing a newer generation does not retarget the returned reader.
    pub fn open_reader_id(&self, id: &str) -> Result<GenerationReader> {
        valid_generation_id(id)?;
        let directory = self.root.join("generations").join(id);
        ensure!(directory.is_dir(), "no published generation {id}");
        let manifest = load_complete(&directory)?;
        GenerationReader::open(PublishedGeneration {
            id: id.to_string(),
            directory,
            manifest,
        })
    }

    fn staging_child(&self, staging: &Path) -> Result<PathBuf> {
        let root = self
            .root
            .join("staging")
            .canonicalize()
            .context("staging root")?;
        let staging = staging
            .canonicalize()
            .with_context(|| format!("staging {}", staging.display()))?;
        let parent = staging
            .parent()
            .context("staging directory has no parent")?;
        ensure!(
            parent == root,
            "only a directory inside staging/ can be published"
        );
        ensure!(staging.is_dir(), "staging path is not a directory");
        Ok(staging)
    }

    fn pointer_target(&self) -> Result<Option<String>> {
        let path = self.root.join("current");
        if !path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)?;
        let relative = text.trim();
        if relative.is_empty() {
            return Ok(None);
        }
        Ok(Some(relative.to_string()))
    }

    fn write_current(&self, generation: &PublishedGeneration) -> Result<()> {
        let dest = self.root.join("current");
        let tmp = self.root.join(format!(
            "current.{}.{}.tmp",
            std::process::id(),
            unix_nanos()
        ));
        {
            let mut file = File::create(&tmp)?;
            writeln!(file, "generations/{}", generation.id)?;
            file.sync_all()?;
        }
        replace_file(&tmp, &dest)?;
        Ok(())
    }
}

/// Read-only view of one published generation.
///
/// The SQLite connection stays on the directory given at construction. Publishing
/// another generation does not move this reader.
pub struct GenerationReader {
    id: String,
    directory: PathBuf,
    sqlite: PathBuf,
    manifest: GenerationManifest,
    conn: Connection,
}

impl std::fmt::Debug for GenerationReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerationReader")
            .field("id", &self.id)
            .field("directory", &self.directory)
            .field("sqlite", &self.sqlite)
            .field("changelog_seq", &self.manifest.changelog_seq)
            .finish()
    }
}

impl GenerationReader {
    fn open(published: PublishedGeneration) -> Result<Self> {
        let sqlite = require_file(
            &published.directory,
            &published.manifest.sqlite,
            "retrieval sqlite",
        )?;
        let conn = Connection::open_with_flags(&sqlite, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("opening {}", sqlite.display()))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA query_only=ON; BEGIN;")?;
        Ok(Self {
            id: published.id,
            directory: published.directory,
            sqlite,
            manifest: published.manifest,
            conn,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn sqlite_path(&self) -> &Path {
        &self.sqlite
    }

    pub fn manifest(&self) -> &GenerationManifest {
        &self.manifest
    }

    pub fn changelog_seq(&self) -> i64 {
        self.manifest.changelog_seq
    }

    /// Citation text from this snapshot's `artifacts.text`. Missing ids return `None`.
    pub fn citation(&self, passage_id: &str) -> Result<Option<String>> {
        let mut statement = self
            .conn
            .prepare("SELECT text FROM artifacts WHERE id = ?1")
            .context("reading citation text")?;
        let mut rows = statement.query([passage_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }
}

/// What is committed but not yet covered by the published generation.
#[derive(Debug)]
pub struct ResumeState {
    pub changelog_head: i64,
    pub published_seq: Option<i64>,
    pub pending: Vec<Change>,
    pub ignored_partials: Vec<PathBuf>,
    pub ready_staging: Vec<PathBuf>,
}

/// PostgreSQL client plus the generation directory for one index.
pub struct Durable {
    client: Client,
    generations: Generations,
}

impl Durable {
    /// Open the `loci` schema, creating it and the tables when needed.
    pub fn open(database_url: &str, generations: impl AsRef<Path>) -> Result<Self> {
        Self::open_schema(database_url, DEFAULT_SCHEMA, generations)
    }

    pub fn open_schema(
        database_url: &str,
        schema: &str,
        generations: impl AsRef<Path>,
    ) -> Result<Self> {
        valid_schema(schema)?;
        let generations = Generations::open(generations)?;
        let mut client = connect(database_url)?;
        client
            .batch_execute(&format!(
                "CREATE SCHEMA IF NOT EXISTS {schema}; SET search_path TO {schema};"
            ))
            .context("creating the durable schema")?;
        client
            .batch_execute(SCHEMA_SQL)
            .context("creating durable tables")?;
        let version: i32 = client
            .query_one("SELECT version FROM schema_version", &empty())?
            .try_get(0)?;
        ensure!(version == 1, "unsupported durable schema version {version}");
        Ok(Self {
            client,
            generations,
        })
    }

    pub fn generations(&self) -> &Generations {
        &self.generations
    }

    pub fn create_staging(&self) -> Result<PathBuf> {
        self.generations.create_staging()
    }

    pub fn open_reader(&self) -> Result<Option<GenerationReader>> {
        self.generations.open_reader()
    }

    /// Append an upsert, or return the existing row when the revision and checksum match.
    ///
    /// The same revision with a different checksum is rejected. Replaying an upsert
    /// after a later delete of that revision does not restore the source; commit a new
    /// revision to replace or restore it. Returns only after the transaction commits.
    pub fn commit(&mut self, commit: &SourceCommit) -> Result<CommitOutcome> {
        validate_commit(commit)?;
        let checksum = digest(commit)?;
        let mut tx = self.client.transaction()?;
        let outcome = apply_upsert(&mut tx, commit, &checksum)?;
        tx.commit()?;
        Ok(outcome)
    }

    /// Append a delete of the current revision.
    ///
    /// Retrying the same source, revision, and checksum returns the original sequence.
    pub fn delete(
        &mut self,
        source_id: &str,
        revision: &str,
        checksum: &str,
    ) -> Result<CommitOutcome> {
        valid_id(source_id, "source id")?;
        valid_id(revision, "revision")?;
        ensure!(!checksum.is_empty(), "checksum is required");
        let mut tx = self.client.transaction()?;
        let outcome = apply_delete(&mut tx, source_id, revision, checksum)?;
        tx.commit()?;
        Ok(outcome)
    }

    pub fn changelog_head(&mut self) -> Result<i64> {
        let seq: i64 = self
            .client
            .query_one(
                "SELECT COALESCE(MAX(seq), 0::bigint) FROM changelog",
                &empty(),
            )?
            .try_get(0)?;
        Ok(seq)
    }

    pub fn changes_after(&mut self, seq: i64) -> Result<Vec<Change>> {
        let rows = self.client.query(
            "SELECT seq, op, source_id, revision, checksum
             FROM changelog WHERE seq > $1 ORDER BY seq",
            &[&seq],
        )?;
        rows.iter().map(map_change).collect()
    }

    /// Passages of sources that have not been deleted, at each source's current revision.
    pub fn live_passages(&mut self) -> Result<Vec<StoredPassage>> {
        let rows = self.client.query(
            &format!("{PASSAGE_FROM} WHERE NOT s.deleted ORDER BY p.source_id, p.passage_id"),
            &empty(),
        )?;
        rows.iter().map(stored_passage).collect()
    }

    /// Passages stored for one revision, including a revision that is no longer current.
    pub fn passages_for(&mut self, source_id: &str, revision: &str) -> Result<Vec<StoredPassage>> {
        let rows = self.client.query(
            &format!(
                "{PASSAGE_FROM_REVISION} WHERE r.source_id = $1 AND r.revision = $2
                 ORDER BY p.passage_id"
            ),
            &[&source_id, &revision],
        )?;
        let passages = rows
            .iter()
            .map(stored_passage)
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !passages.is_empty(),
            "unknown revision {source_id}/{revision}"
        );
        Ok(passages)
    }

    /// Publish a completed staging directory whose changelog sequence is already committed.
    pub fn publish(&mut self, staging: &Path) -> Result<PublishedGeneration> {
        let manifest = load_complete(staging)?;
        let head = self.changelog_head()?;
        ensure!(
            manifest.changelog_seq > 0 && manifest.changelog_seq <= head,
            "generation changelog sequence {} is outside the committed changelog (head {head})",
            manifest.changelog_seq
        );
        let exists: bool = self
            .client
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM changelog WHERE seq = $1)",
                &[&manifest.changelog_seq],
            )?
            .try_get(0)?;
        ensure!(
            exists,
            "generation changelog sequence {} is not in the changelog",
            manifest.changelog_seq
        );
        self.generations.publish(staging)
    }

    /// Repair the generation pointer, then list changelog rows newer than the published sequence.
    ///
    /// Partial directories stay on disk and are not published. Complete staging
    /// directories are reported in `ready_staging` and are not served until `publish`.
    pub fn resume(&mut self) -> Result<ResumeState> {
        let published = self.generations.repair_pointer()?;
        let readiness = self.generations.readiness()?;
        let published_seq = published
            .as_ref()
            .map(|generation| generation.manifest.changelog_seq);
        let pending = self.changes_after(published_seq.unwrap_or(0))?;
        Ok(ResumeState {
            changelog_head: self.changelog_head()?,
            published_seq,
            pending,
            ignored_partials: readiness.ignored_partials,
            ready_staging: readiness.ready_staging,
        })
    }
}

const PASSAGE_FROM_REVISION: &str = "
    SELECT p.source_id, p.revision, r.checksum, p.passage_id, p.passage_text, p.span_kind,
           p.span_start, p.span_end, p.metadata, r.metadata, r.model_fingerprint, p.embedding, c.seq
    FROM revisions r
    JOIN passages p ON p.source_id = r.source_id AND p.revision = r.revision
    JOIN changelog c ON c.source_id = r.source_id AND c.revision = r.revision
        AND c.checksum = r.checksum AND c.op = 'upsert'
";

const PASSAGE_FROM: &str = "
    SELECT p.source_id, p.revision, r.checksum, p.passage_id, p.passage_text, p.span_kind,
           p.span_start, p.span_end, p.metadata, r.metadata, r.model_fingerprint, p.embedding, c.seq
    FROM sources s
    JOIN revisions r ON r.source_id = s.source_id AND r.revision = s.current_revision
    JOIN passages p ON p.source_id = r.source_id AND p.revision = r.revision
    JOIN changelog c ON c.source_id = r.source_id AND c.revision = r.revision
        AND c.checksum = r.checksum AND c.op = 'upsert'
";

struct SourceRow {
    deleted: bool,
    current_revision: Option<String>,
    current_checksum: Option<String>,
}

fn apply_upsert(
    tx: &mut Transaction<'_>,
    commit: &SourceCommit,
    checksum: &str,
) -> Result<CommitOutcome> {
    tx.execute(
        "INSERT INTO sources (source_id, deleted) VALUES ($1, TRUE)
         ON CONFLICT (source_id) DO NOTHING",
        &[&commit.source_id],
    )?;
    let row = tx.query_one(
        "SELECT deleted, current_revision, current_checksum FROM sources
         WHERE source_id = $1 FOR UPDATE",
        &[&commit.source_id],
    )?;
    let _locked = source_row(&row)?;
    if let Some(existing) = tx.query_opt(
        "SELECT checksum FROM revisions WHERE source_id = $1 AND revision = $2",
        &[&commit.source_id, &commit.revision],
    )? {
        let stored: String = existing.try_get(0)?;
        ensure!(
            stored == checksum,
            "revision {}/{} already committed with a different checksum",
            commit.source_id,
            commit.revision
        );
        let seq = changelog_seq(tx, &commit.source_id, &commit.revision, checksum, "upsert")?
            .context("committed revision is missing its changelog row")?;
        return Ok(CommitOutcome::AlreadyCommitted {
            seq,
            checksum: checksum.to_string(),
        });
    }
    tx.execute(
        "INSERT INTO revisions (source_id, revision, checksum, metadata, model_fingerprint)
         VALUES ($1, $2, $3, $4, $5)",
        &[
            &commit.source_id,
            &commit.revision,
            &checksum,
            &commit.metadata,
            &commit.model_fingerprint,
        ],
    )?;
    for passage in &commit.passages {
        let start = i64::try_from(passage.span.start)?;
        let end = i64::try_from(passage.span.end)?;
        let kind = passage.span.kind.as_str();
        tx.execute(
            "INSERT INTO passages (
                source_id, revision, passage_id, passage_text, span_kind,
                span_start, span_end, metadata, embedding
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &commit.source_id,
                &commit.revision,
                &passage.id,
                &passage.text,
                &kind,
                &start,
                &end,
                &passage.metadata,
                &passage.embedding,
            ],
        )?;
    }
    let seq = insert_change(
        tx,
        ChangeOp::Upsert.as_str(),
        &commit.source_id,
        &commit.revision,
        checksum,
    )?;
    tx.execute(
        "UPDATE sources
         SET current_revision = $2, current_checksum = $3, deleted = FALSE,
             metadata = $4, updated_at = now()
         WHERE source_id = $1",
        &[
            &commit.source_id,
            &commit.revision,
            &checksum,
            &commit.metadata,
        ],
    )?;
    Ok(CommitOutcome::Committed {
        seq,
        checksum: checksum.to_string(),
    })
}

fn apply_delete(
    tx: &mut Transaction<'_>,
    source_id: &str,
    revision: &str,
    checksum: &str,
) -> Result<CommitOutcome> {
    let Some(row) = tx.query_opt(
        "SELECT deleted, current_revision, current_checksum FROM sources
         WHERE source_id = $1 FOR UPDATE",
        &[&source_id],
    )?
    else {
        bail!("unknown source {source_id}");
    };
    let current = source_row(&row)?;
    if let Some(seq) = changelog_seq(tx, source_id, revision, checksum, "delete")? {
        return Ok(CommitOutcome::AlreadyCommitted {
            seq,
            checksum: checksum.to_string(),
        });
    }
    let Some(existing) = tx.query_opt(
        "SELECT checksum FROM revisions WHERE source_id = $1 AND revision = $2",
        &[&source_id, &revision],
    )?
    else {
        bail!("unknown revision {source_id}/{revision}");
    };
    let stored: String = existing.try_get(0)?;
    ensure!(
        stored == checksum,
        "revision {source_id}/{revision} already committed with a different checksum"
    );
    ensure!(
        !current.deleted
            && current.current_revision.as_deref() == Some(revision)
            && current.current_checksum.as_deref() == Some(checksum),
        "cannot delete {source_id}/{revision}: it is not the current revision"
    );
    tx.execute(
        "UPDATE sources
         SET deleted = TRUE, current_revision = NULL, current_checksum = NULL, updated_at = now()
         WHERE source_id = $1",
        &[&source_id],
    )?;
    let seq = insert_change(tx, ChangeOp::Delete.as_str(), source_id, revision, checksum)?;
    Ok(CommitOutcome::Committed {
        seq,
        checksum: checksum.to_string(),
    })
}

fn insert_change(
    tx: &mut Transaction<'_>,
    op: &str,
    source_id: &str,
    revision: &str,
    checksum: &str,
) -> Result<i64> {
    let seq = tx
        .query_one(
            "INSERT INTO changelog (op, source_id, revision, checksum)
             VALUES ($1, $2, $3, $4) RETURNING seq",
            &[&op, &source_id, &revision, &checksum],
        )?
        .try_get(0)?;
    Ok(seq)
}

fn changelog_seq(
    tx: &mut Transaction<'_>,
    source_id: &str,
    revision: &str,
    checksum: &str,
    op: &str,
) -> Result<Option<i64>> {
    let row = tx.query_opt(
        "SELECT seq FROM changelog
         WHERE source_id = $1 AND revision = $2 AND checksum = $3 AND op = $4",
        &[&source_id, &revision, &checksum, &op],
    )?;
    row.map(|row| row.try_get(0))
        .transpose()
        .map_err(Into::into)
}

fn source_row(row: &postgres::Row) -> Result<SourceRow> {
    Ok(SourceRow {
        deleted: row.try_get(0)?,
        current_revision: row.try_get(1)?,
        current_checksum: row.try_get(2)?,
    })
}

fn map_change(row: &postgres::Row) -> Result<Change> {
    Ok(Change {
        seq: row.try_get(0)?,
        op: ChangeOp::parse(&row.try_get::<_, String>(1)?)?,
        source_id: row.try_get(2)?,
        revision: row.try_get(3)?,
        checksum: row.try_get(4)?,
    })
}

fn stored_passage(row: &postgres::Row) -> Result<StoredPassage> {
    let kind = SpanKind::parse(&row.try_get::<_, String>(5)?)?;
    let start = u64::try_from(row.try_get::<_, i64>(6)?)?;
    let end = u64::try_from(row.try_get::<_, i64>(7)?)?;
    Ok(StoredPassage {
        source_id: row.try_get(0)?,
        revision: row.try_get(1)?,
        checksum: row.try_get(2)?,
        passage_id: row.try_get(3)?,
        text: row.try_get(4)?,
        span: Span { kind, start, end },
        metadata: row.try_get(8)?,
        source_metadata: row.try_get(9)?,
        model_fingerprint: row.try_get(10)?,
        embedding: row.try_get(11)?,
        changelog_seq: row.try_get(12)?,
    })
}

fn validate_commit(commit: &SourceCommit) -> Result<()> {
    valid_id(&commit.source_id, "source id")?;
    valid_id(&commit.revision, "revision")?;
    valid_label(&commit.model_fingerprint, "model fingerprint", 2048)?;
    valid_object(&commit.metadata, "source")?;
    ensure!(
        !commit.passages.is_empty(),
        "at least one passage is required"
    );
    let order = passage_order(commit);
    if let Some(pair) = order
        .windows(2)
        .find(|pair| commit.passages[pair[0]].id == commit.passages[pair[1]].id)
    {
        bail!("duplicate passage id {}", commit.passages[pair[0]].id);
    }
    for passage in &commit.passages {
        valid_id(&passage.id, "passage id")?;
        valid_object(&passage.metadata, "passage")?;
        ensure!(
            passage.span.start <= passage.span.end,
            "span start exceeds span end for {}",
            passage.id
        );
        ensure!(
            i64::try_from(passage.span.end).is_ok(),
            "span end exceeds i64 for {}",
            passage.id
        );
    }
    Ok(())
}

/// SHA-256 of a version byte and length-prefixed fields. Passages are hashed in id order.
fn digest(commit: &SourceCommit) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update([1u8]);
    write_len(&mut hasher, commit.source_id.as_bytes());
    write_len(&mut hasher, commit.revision.as_bytes());
    write_len(&mut hasher, commit.model_fingerprint.as_bytes());
    write_len(&mut hasher, &serde_json::to_vec(&commit.metadata)?);
    let order = passage_order(commit);
    hasher.update((order.len() as u64).to_le_bytes());
    for index in order {
        let passage = &commit.passages[index];
        write_len(&mut hasher, passage.id.as_bytes());
        write_len(&mut hasher, passage.text.as_bytes());
        hasher.update([match passage.span.kind {
            SpanKind::Byte => 1u8,
            SpanKind::Turn => 2u8,
        }]);
        hasher.update(passage.span.start.to_le_bytes());
        hasher.update(passage.span.end.to_le_bytes());
        write_len(&mut hasher, &serde_json::to_vec(&passage.metadata)?);
        write_len(&mut hasher, &passage.embedding);
    }
    Ok(crate::hex(&hasher.finalize()))
}

fn write_len(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn passage_order(commit: &SourceCommit) -> Vec<usize> {
    let mut order: Vec<usize> = (0..commit.passages.len()).collect();
    order.sort_by(|&left, &right| commit.passages[left].id.cmp(&commit.passages[right].id));
    order
}

fn valid_id(value: &str, what: &str) -> Result<()> {
    valid_label(value, what, 1024)
}

fn valid_label(value: &str, what: &str, max: usize) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= max,
        "{what} is empty or too long"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "{what} contains a control character"
    );
    Ok(())
}

fn valid_object(value: &Value, what: &str) -> Result<()> {
    ensure!(value.is_object(), "{what} metadata must be a JSON object");
    Ok(())
}

fn valid_generation_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id.len() <= 80
            && id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "invalid generation id"
    );
    Ok(())
}

fn valid_schema(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        bail!("postgres schema name is empty");
    };
    ensure!(
        name.len() <= 63
            && (first == '_' || first.is_ascii_lowercase())
            && chars.all(|c| c == '_' || c.is_ascii_lowercase() || c.is_ascii_digit()),
        "invalid postgres schema name"
    );
    Ok(())
}

fn connect(database_url: &str) -> Result<Client> {
    let mut config = database_url
        .parse::<Config>()
        .context("parsing PostgreSQL URL")?;
    config.connect_timeout(Duration::from_secs(5));
    if config.get_application_name().is_none() {
        config.application_name("loci-search");
    }
    config.connect(NoTls).context("connecting to PostgreSQL")
}

fn empty() -> [&'static (dyn ToSql + Sync); 0] {
    []
}

fn load_complete(dir: &Path) -> Result<GenerationManifest> {
    let path = dir.join("manifest.json");
    ensure!(
        path.is_file(),
        "incomplete generation: missing manifest.json"
    );
    let file = File::open(&path)?;
    let manifest: GenerationManifest = serde_json::from_reader(file)
        .context("incomplete generation: manifest.json is not a generation manifest")?;
    ensure!(
        manifest.changelog_seq >= 0,
        "incomplete generation: changelog sequence is negative"
    );
    validate_index(&manifest.index)?;
    let sqlite = require_file(dir, &manifest.sqlite, "retrieval sqlite")?;
    require_retrieval_sqlite(&sqlite)?;
    if let Some(hnsw) = &manifest.hnsw {
        require_file(dir, hnsw, "hnsw sidecar")?;
    }
    Ok(manifest)
}

fn validate_index(index: &Value) -> Result<()> {
    ensure!(
        index.is_object(),
        "incomplete generation: index manifest must be an object"
    );
    for field in [
        "schema",
        "embedding",
        "chunks",
        "regions",
        "leaf_size",
        "average_length",
    ] {
        ensure!(
            index.get(field).is_some(),
            "incomplete generation: index manifest missing {field}"
        );
    }
    let embedding = &index["embedding"];
    ensure!(
        embedding.is_object(),
        "incomplete generation: embedding manifest must be an object"
    );
    for field in ["model_revision", "dimensions", "normalization"] {
        ensure!(
            embedding.get(field).is_some(),
            "incomplete generation: embedding manifest missing {field}"
        );
    }
    Ok(())
}

fn require_file(dir: &Path, name: &str, what: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    let single = relative.components().count() == 1
        && matches!(relative.components().next(), Some(Component::Normal(_)));
    ensure!(
        single,
        "incomplete generation: {what} path must be a file name"
    );
    let path = dir.join(relative);
    ensure!(path.is_file(), "incomplete generation: missing {name}");
    ensure!(
        path.metadata()?.len() > 0,
        "incomplete generation: {name} is empty"
    );
    Ok(path)
}

fn require_retrieval_sqlite(path: &Path) -> Result<()> {
    let conn =
        Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).with_context(|| {
            format!(
                "incomplete generation: {} is not a retrieval sqlite",
                path.display()
            )
        })?;
    conn.execute_batch("PRAGMA query_only=ON;")?;
    conn.prepare("SELECT json FROM manifest LIMIT 0")
        .context("incomplete generation: retrieval sqlite has no manifest table")?;
    conn.prepare("SELECT text FROM artifacts LIMIT 0")
        .context("incomplete generation: retrieval sqlite has no artifacts.text column")?;
    Ok(())
}

fn sync_generation_files(dir: &Path, manifest: &GenerationManifest) -> Result<()> {
    sync_file(&dir.join("manifest.json"))?;
    sync_file(&dir.join(&manifest.sqlite))?;
    if let Some(hnsw) = &manifest.hnsw {
        sync_file(&dir.join(hnsw))?;
    }
    Ok(())
}

fn sync_file(path: &Path) -> Result<()> {
    // Windows rejects FlushFileBuffers on a read-only handle.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .with_context(|| format!("syncing {}", path.display()))?;
    file.sync_all()?;
    Ok(())
}

fn inspect(parent: &Path) -> Result<(Vec<PublishedGeneration>, Vec<PathBuf>)> {
    let mut complete = Vec::new();
    let mut partial = Vec::new();
    for dir in child_dirs(parent)? {
        match load_complete(&dir) {
            Ok(manifest) => {
                let Some(id) = dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
                else {
                    partial.push(dir);
                    continue;
                };
                complete.push(PublishedGeneration {
                    id,
                    directory: dir,
                    manifest,
                });
            }
            Err(_) => partial.push(dir),
        }
    }
    Ok((complete, partial))
}

fn newest(mut complete: Vec<PublishedGeneration>) -> Option<PublishedGeneration> {
    complete.sort_by(|left, right| {
        left.manifest
            .changelog_seq
            .cmp(&right.manifest.changelog_seq)
            .then_with(|| left.id.cmp(&right.id))
    });
    complete.pop()
}

fn child_dirs(parent: &Path) -> Result<Vec<PathBuf>> {
    if !parent.exists() {
        return Ok(Vec::new());
    }
    let mut dirs = Vec::new();
    for entry in fs::read_dir(parent).with_context(|| format!("reading {}", parent.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn new_generation_id(seq: i64) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("g{seq:020}-{:020}-{n:08}", unix_nanos())
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn replace_file(from: &Path, to: &Path) -> Result<()> {
    if let Err(error) = fs::rename(from, to) {
        if to.is_file() {
            fs::remove_file(to)?;
            fs::rename(from, to)
                .with_context(|| format!("replacing {} ({error})", to.display()))?;
            return Ok(());
        }
        return Err(error)
            .with_context(|| format!("renaming {} to {}", from.display(), to.display()));
    }
    Ok(())
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    current_revision TEXT,
    current_checksum TEXT,
    deleted BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT sources_current_ok CHECK (
        (deleted AND current_revision IS NULL AND current_checksum IS NULL)
        OR (NOT deleted AND current_revision IS NOT NULL AND current_checksum IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS revisions (
    source_id TEXT NOT NULL REFERENCES sources (source_id),
    revision TEXT NOT NULL,
    checksum TEXT NOT NULL,
    metadata JSONB NOT NULL,
    model_fingerprint TEXT NOT NULL,
    PRIMARY KEY (source_id, revision)
);
CREATE TABLE IF NOT EXISTS passages (
    source_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    passage_id TEXT NOT NULL,
    passage_text TEXT NOT NULL,
    span_kind TEXT NOT NULL CHECK (span_kind IN ('byte', 'turn')),
    span_start BIGINT NOT NULL CHECK (span_start >= 0),
    span_end BIGINT NOT NULL CHECK (span_end >= span_start),
    metadata JSONB NOT NULL,
    embedding BYTEA NOT NULL,
    PRIMARY KEY (source_id, revision, passage_id),
    FOREIGN KEY (source_id, revision) REFERENCES revisions (source_id, revision)
);
CREATE TABLE IF NOT EXISTS changelog (
    seq BIGSERIAL PRIMARY KEY,
    op TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
    source_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    checksum TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS changelog_idempotency
    ON changelog (source_id, revision, checksum, op);
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY CHECK (version = 1)
);
INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
CREATE OR REPLACE FUNCTION loci_reject_changelog_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'changelog is append-only';
END;
$$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'loci_changelog_append_only'
          AND tgrelid = 'changelog'::regclass
    ) THEN
        CREATE TRIGGER loci_changelog_append_only
        BEFORE UPDATE OR DELETE ON changelog
        FOR EACH ROW EXECUTE FUNCTION loci_reject_changelog_mutation();
    END IF;
END $$;
"#;
