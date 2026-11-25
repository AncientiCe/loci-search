//! Storage tests for the PostgreSQL log and sqlite+hnsw generation publisher.
//!
//! Filesystem tests always run. PostgreSQL tests run when `LOCI_POSTGRES=1` and
//! `LOCI_POSTGRES_URL` point at the local container. `cargo test` passes without them.

use loci_search::durable::{
    ChangeOp, CommitOutcome, Durable, GenerationManifest, Generations, PassageWrite, SourceCommit,
    Span, SpanKind,
};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn partial_directories_are_ignored_until_a_complete_rename() {
    let root = tempfile::tempdir().unwrap();
    let generations = Generations::open(root.path()).unwrap();

    let broken = generations.create_staging().unwrap();
    fs::write(broken.join("manifest.json"), b"{").unwrap();
    let err = generations.publish(&broken).unwrap_err();
    assert!(err.to_string().contains("incomplete generation"), "{err:#}");
    assert!(broken.is_dir(), "a partial directory stays where it was");

    let missing_sidecar = stage(&generations, 1, &[("p1", "sidecar")], false);
    GenerationManifest::new(1, "retrieval.sqlite", index_manifest())
        .with_hnsw("retrieval.sqlite.hnsw")
        .write(&missing_sidecar)
        .unwrap();
    let err = generations.publish(&missing_sidecar).unwrap_err();
    assert!(err.to_string().contains("retrieval.sqlite.hnsw"), "{err:#}");

    let bare = generations.create_staging().unwrap();
    {
        let conn = rusqlite::Connection::open(bare.join("retrieval.sqlite")).unwrap();
        conn.execute_batch("CREATE TABLE manifest (json TEXT NOT NULL);")
            .unwrap();
    }
    GenerationManifest::new(1, "retrieval.sqlite", index_manifest())
        .write(&bare)
        .unwrap();
    let err = generations.publish(&bare).unwrap_err();
    assert!(err.to_string().contains("artifacts.text"), "{err:#}");

    let crashed = root.path().join("generations").join("partial-crash");
    fs::create_dir_all(&crashed).unwrap();
    fs::write(crashed.join("retrieval.sqlite"), b"").unwrap();
    fs::write(root.path().join("current"), "generations/partial-crash\n").unwrap();

    let ready = stage(&generations, 4, &[("p1", "kept")], false);
    let before = generations.readiness().unwrap();
    assert!(before.current.is_none());
    assert!(before.ignored_partials.iter().any(|path| path == &broken));
    assert!(before.ignored_partials.iter().any(|path| path == &crashed));
    assert!(before.ready_staging.iter().any(|path| path == &ready));
    assert!(generations.open_reader().unwrap().is_none());
    assert!(generations.repair_pointer().unwrap().is_none());
    assert!(!root.path().join("current").exists());

    let published = generations.publish(&ready).unwrap();
    assert!(!ready.exists());
    assert!(
        published
            .directory
            .starts_with(root.path().join("generations"))
    );
    let reader = generations.open_reader().unwrap().unwrap();
    assert_eq!(reader.citation("p1").unwrap().as_deref(), Some("kept"));
    assert_eq!(reader.citation("missing").unwrap(), None);

    fs::write(root.path().join("current"), "generations/partial-crash\n").unwrap();
    let again = generations.open_reader().unwrap().unwrap();
    assert_eq!(again.id(), published.id);
    assert_eq!(again.citation("p1").unwrap().as_deref(), Some("kept"));
    generations.repair_pointer().unwrap();
    let pointer = fs::read_to_string(root.path().join("current")).unwrap();
    assert!(pointer.contains(published.id.as_str()), "{pointer}");
}

#[test]
fn open_reader_keeps_the_previous_citation_after_replacement() {
    let root = tempfile::tempdir().unwrap();
    let generations = Generations::open(root.path()).unwrap();
    let first_dir = stage(&generations, 1, &[("p1", "citation-v1")], false);
    let first = generations.publish(&first_dir).unwrap();
    let reader = generations.open_reader().unwrap().unwrap();
    assert_eq!(
        reader.citation("p1").unwrap().as_deref(),
        Some("citation-v1")
    );
    assert!(reader.manifest().hnsw.is_none());

    let second_dir = stage(&generations, 2, &[("p1", "citation-v2")], true);
    let second = generations.publish(&second_dir).unwrap();
    assert_ne!(first.directory, second.directory);
    assert!(
        first.directory.is_dir(),
        "the open generation stays on disk"
    );
    assert_eq!(reader.directory(), first.directory);
    assert_eq!(
        reader.citation("p1").unwrap().as_deref(),
        Some("citation-v1")
    );

    let latest = generations.open_reader().unwrap().unwrap();
    assert_eq!(latest.id(), second.id);
    assert_eq!(
        latest.citation("p1").unwrap().as_deref(),
        Some("citation-v2")
    );
    assert_eq!(
        latest.manifest().hnsw.as_deref(),
        Some("retrieval.sqlite.hnsw")
    );
    assert!(latest.directory().join("retrieval.sqlite.hnsw").is_file());
}

#[test]
fn postgres_retry_is_idempotent_for_the_same_revision_and_checksum() {
    let Some((url, schema, dir)) = postgres_setup() else {
        return;
    };
    let _schema = SchemaDrop {
        url: url.clone(),
        schema: schema.clone(),
    };
    let mut durable = Durable::open_schema(&url, &schema, dir.path()).unwrap();
    let mut commit = SourceCommit::new(
        "doc-1",
        "rev-1",
        "minilm-l6-v2-q8@test",
        vec![
            passage("p-turn", "beta", SpanKind::Turn, vec![9, 9]),
            passage("p-byte", "alpha", SpanKind::Byte, vec![1, 2, 3, 4]),
        ],
    );
    commit.metadata = json!({"kind": "note"});
    commit.passages[0].metadata = json!({"speaker": "a"});

    let first = durable.commit(&commit).unwrap();
    assert!(first.is_new());
    let seq = first.seq();
    let checksum = first.checksum().to_string();
    assert_eq!(durable.changelog_head().unwrap(), seq);

    let mut reordered = commit.clone();
    reordered.passages.reverse();
    let retry = durable.commit(&reordered).unwrap();
    assert_eq!(
        retry,
        CommitOutcome::AlreadyCommitted {
            seq,
            checksum: checksum.clone(),
        }
    );
    assert_eq!(durable.changelog_head().unwrap(), seq);

    let mut changed = commit.clone();
    changed.passages[1].text = "mutated".into();
    let err = durable.commit(&changed).unwrap_err();
    assert!(err.to_string().contains("different checksum"), "{err:#}");
    assert_eq!(durable.changelog_head().unwrap(), seq);

    let live = durable.live_passages().unwrap();
    assert_eq!(live.len(), 2);
    assert_eq!(live[0].passage_id, "p-byte");
    assert_eq!(live[0].text, "alpha");
    assert_eq!(live[0].span.kind, SpanKind::Byte);
    assert_eq!(live[0].embedding, vec![1, 2, 3, 4]);
    assert_eq!(live[0].model_fingerprint, "minilm-l6-v2-q8@test");
    assert_eq!(live[0].source_metadata, json!({"kind": "note"}));
    assert_eq!(live[1].passage_id, "p-turn");
    assert_eq!(live[1].span.kind, SpanKind::Turn);
    assert_eq!(
        live[1].span,
        Span {
            kind: SpanKind::Turn,
            start: 0,
            end: 4
        }
    );
    assert_eq!(live[1].metadata, json!({"speaker": "a"}));

    let err = durable
        .delete("doc-1", "rev-1", "not-the-checksum")
        .unwrap_err();
    assert!(err.to_string().contains("different checksum"), "{err:#}");
    let deleted = durable.delete("doc-1", "rev-1", &checksum).unwrap();
    assert!(deleted.is_new());
    assert!(deleted.seq() > seq);
    let deleted_again = durable.delete("doc-1", "rev-1", &checksum).unwrap();
    assert_eq!(
        deleted_again,
        CommitOutcome::AlreadyCommitted {
            seq: deleted.seq(),
            checksum: checksum.clone(),
        }
    );
    assert!(durable.live_passages().unwrap().is_empty());
    let replay = durable.commit(&commit).unwrap();
    assert_eq!(replay.seq(), seq);
    assert!(!replay.is_new());
    assert!(
        durable.live_passages().unwrap().is_empty(),
        "replaying the deleted revision does not restore it"
    );
    assert_eq!(
        durable.passages_for("doc-1", "rev-1").unwrap()[0].text,
        "alpha"
    );

    let restored = SourceCommit::new(
        "doc-1",
        "rev-2",
        "minilm-l6-v2-q8@test",
        vec![passage("p-byte", "restored", SpanKind::Byte, vec![5])],
    );
    assert!(durable.commit(&restored).unwrap().is_new());
    assert_eq!(durable.live_passages().unwrap()[0].text, "restored");
    let changes = durable.changes_after(0).unwrap();
    assert_eq!(
        changes.iter().map(|change| change.op).collect::<Vec<_>>(),
        vec![ChangeOp::Upsert, ChangeOp::Delete, ChangeOp::Upsert]
    );
}

#[test]
fn postgres_interrupted_publish_keeps_the_changelog_and_ignores_partials() {
    let Some((url, schema, dir)) = postgres_setup() else {
        return;
    };
    let _schema = SchemaDrop {
        url: url.clone(),
        schema: schema.clone(),
    };
    let mut durable = Durable::open_schema(&url, &schema, dir.path()).unwrap();
    let commit = SourceCommit::new(
        "doc-1",
        "rev-1",
        "minilm-l6-v2-q8@test",
        vec![passage("p1", "alpha", SpanKind::Byte, vec![1])],
    );
    let seq = durable.commit(&commit).unwrap().seq();

    let partial = durable.create_staging().unwrap();
    fs::write(partial.join("manifest.json"), b"{\"changelog_seq\":1}").unwrap();
    let err = durable.publish(&partial).unwrap_err();
    assert!(err.to_string().contains("incomplete generation"), "{err:#}");
    assert!(partial.is_dir());

    let uncommitted = stage(durable.generations(), 999, &[("p1", "too-new")], false);
    let err = durable.publish(&uncommitted).unwrap_err();
    assert!(
        err.to_string().contains("outside the committed changelog"),
        "{err:#}"
    );
    assert!(uncommitted.is_dir(), "rejected publish does not rename");

    let crashed = dir.path().join("generations").join("partial-crash");
    fs::create_dir_all(&crashed).unwrap();
    fs::write(crashed.join("note.txt"), b"partial").unwrap();

    let resume = durable.resume().unwrap();
    assert_eq!(resume.changelog_head, seq);
    assert_eq!(resume.published_seq, None);
    assert_eq!(resume.pending.len(), 1);
    assert_eq!(resume.pending[0].seq, seq);
    assert_eq!(resume.pending[0].op, ChangeOp::Upsert);
    assert!(resume.ignored_partials.iter().any(|path| path == &partial));
    assert!(resume.ignored_partials.iter().any(|path| path == &crashed));
    assert!(resume.ready_staging.iter().any(|path| path == &uncommitted));
    assert!(durable.open_reader().unwrap().is_none());

    let mut other = Durable::open_schema(&url, &schema, dir.path()).unwrap();
    assert_eq!(other.changelog_head().unwrap(), seq);
    assert_eq!(other.live_passages().unwrap()[0].text, "alpha");
    assert!(other.open_reader().unwrap().is_none());
}

#[test]
fn postgres_replacement_publishes_a_new_generation_without_moving_an_open_reader() {
    let Some((url, schema, dir)) = postgres_setup() else {
        return;
    };
    let _schema = SchemaDrop {
        url: url.clone(),
        schema: schema.clone(),
    };
    let mut durable = Durable::open_schema(&url, &schema, dir.path()).unwrap();
    let first_commit = SourceCommit::new(
        "doc-1",
        "rev-1",
        "minilm-l6-v2-q8@test",
        vec![passage("p1", "citation-v1", SpanKind::Byte, vec![1, 0])],
    );
    let first = durable.commit(&first_commit).unwrap();
    let staged = stage(
        durable.generations(),
        first.seq(),
        &[("p1", "citation-v1")],
        false,
    );
    let published = durable.publish(&staged).unwrap();
    let reader = durable.open_reader().unwrap().unwrap();
    assert_eq!(
        reader.citation("p1").unwrap().as_deref(),
        Some("citation-v1")
    );
    assert_eq!(reader.changelog_seq(), first.seq());

    let retry = durable.commit(&first_commit).unwrap();
    assert_eq!(retry.seq(), first.seq());
    assert!(!retry.is_new());

    let second_commit = SourceCommit::new(
        "doc-1",
        "rev-2",
        "minilm-l6-v2-q8@test",
        vec![passage("p1", "citation-v2", SpanKind::Byte, vec![2, 0])],
    );
    let second = durable.commit(&second_commit).unwrap();
    assert!(second.is_new());
    assert!(second.seq() > first.seq());
    let staged = stage(
        durable.generations(),
        second.seq(),
        &[("p1", "citation-v2")],
        true,
    );
    let next = durable.publish(&staged).unwrap();
    assert_eq!(reader.directory(), published.directory);
    assert_eq!(
        reader.citation("p1").unwrap().as_deref(),
        Some("citation-v1")
    );
    assert!(published.directory.is_dir());

    let resume = durable.resume().unwrap();
    assert_eq!(resume.published_seq, Some(second.seq()));
    assert!(resume.pending.is_empty());
    assert_eq!(durable.live_passages().unwrap().len(), 1);
    assert_eq!(durable.live_passages().unwrap()[0].text, "citation-v2");
    assert_eq!(durable.live_passages().unwrap()[0].embedding, vec![2, 0]);

    drop(durable);
    assert_eq!(
        reader.citation("p1").unwrap().as_deref(),
        Some("citation-v1"),
        "an open reader serves the previous citation without PostgreSQL"
    );
    let generations = Generations::open(dir.path()).unwrap();
    let latest = generations.open_reader().unwrap().unwrap();
    assert_eq!(latest.directory(), next.directory);
    assert_eq!(
        latest.citation("p1").unwrap().as_deref(),
        Some("citation-v2")
    );
    assert_ne!(reader.directory(), latest.directory());
}

fn stage(generations: &Generations, seq: i64, rows: &[(&str, &str)], hnsw: bool) -> PathBuf {
    let staging = generations.create_staging().unwrap();
    write_retrieval(&staging, rows);
    let mut manifest = GenerationManifest::new(seq, "retrieval.sqlite", index_manifest());
    if hnsw {
        let sidecar = "retrieval.sqlite.hnsw";
        fs::write(staging.join(sidecar), b"hnsw-sidecar").unwrap();
        manifest = manifest.with_hnsw(sidecar);
    }
    manifest.write(&staging).unwrap();
    staging
}

fn write_retrieval(dir: &Path, rows: &[(&str, &str)]) {
    let conn = rusqlite::Connection::open(dir.join("retrieval.sqlite")).unwrap();
    conn.execute_batch(
        "CREATE TABLE manifest (json TEXT NOT NULL);
         CREATE TABLE artifacts (
            rowid INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE,
            document TEXT NOT NULL,
            revision TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT NOT NULL,
            vector BLOB NOT NULL,
            length INTEGER NOT NULL
         );",
    )
    .unwrap();
    for (id, text) in rows {
        conn.execute(
            "INSERT INTO artifacts (id, document, revision, start, end, text, metadata, vector, length)
             VALUES (?1, 'doc', 'rev', 0, ?2, ?3, '{}', X'', ?2)",
            rusqlite::params![id, text.len(), text],
        )
        .unwrap();
    }
}

fn index_manifest() -> serde_json::Value {
    json!({
        "schema": 2,
        "embedding": {
            "model_revision": "test@local",
            "dimensions": 1,
            "normalization": "l2"
        },
        "chunks": 1,
        "regions": 1,
        "leaf_size": 8,
        "average_length": 1.0
    })
}

fn passage(id: &str, text: &str, kind: SpanKind, embedding: Vec<u8>) -> PassageWrite {
    PassageWrite::new(
        id,
        text,
        Span {
            kind,
            start: 0,
            end: text.len() as u64,
        },
        embedding,
    )
}

fn postgres_setup() -> Option<(String, String, tempfile::TempDir)> {
    if std::env::var("LOCI_POSTGRES").ok().as_deref() != Some("1") {
        eprintln!(
            "skipped: set LOCI_POSTGRES=1 and LOCI_POSTGRES_URL to run the PostgreSQL durable test"
        );
        return None;
    }
    let url =
        std::env::var("LOCI_POSTGRES_URL").expect("LOCI_POSTGRES=1 requires LOCI_POSTGRES_URL");
    let dir = tempfile::tempdir().expect("temp generation directory");
    Some((url, unique_schema(), dir))
}

fn unique_schema() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("loci_it_{}_{n}", std::process::id())
}

struct SchemaDrop {
    url: String,
    schema: String,
}

impl Drop for SchemaDrop {
    fn drop(&mut self) {
        let Ok(mut client) = postgres::Client::connect(&self.url, postgres::NoTls) else {
            return;
        };
        let _ = client.batch_execute(&format!("DROP SCHEMA IF EXISTS {} CASCADE", self.schema));
    }
}
