use super::{Catalog, Mutation, WriteResult, connect, generation};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Event-time partitions and age-based retention, both measured in seconds.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamPolicy {
    pub timestamp: String,
    pub rollover_seconds: i64,
    pub retention_seconds: i64,
    #[serde(default = "automatic")]
    pub automatic: bool,
}
fn automatic() -> bool {
    true
}

impl StreamPolicy {
    pub fn from_settings(settings: &Value) -> Result<Option<Self>> {
        let Some(raw) = settings.get("stream") else {
            return Ok(None);
        };
        let p: Self = serde_json::from_value(raw.clone())?;
        ensure!(
            !p.timestamp.is_empty() && p.rollover_seconds > 0 && p.retention_seconds > 0,
            "invalid stream policy"
        );
        Ok(Some(p))
    }
}

pub(super) fn timestamp(value: &Value) -> Result<i64> {
    if let Some(n) = value.as_i64() {
        return Ok(n);
    }
    let s = value
        .as_str()
        .context("timestamp must be epoch seconds or RFC3339")?;
    Ok(s.parse::<jiff::Timestamp>()?.as_second())
}

pub(super) fn initialize(db: &Connection) -> Result<()> {
    db.execute_batch("CREATE TABLE IF NOT EXISTS loci_events(
       idx TEXT NOT NULL, id TEXT NOT NULL, occurred INTEGER NOT NULL, partition_start INTEGER NOT NULL,
       PRIMARY KEY(idx,id), FOREIGN KEY(idx,id) REFERENCES loci_documents(idx,id) ON DELETE CASCADE
    ) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS loci_events_by_time ON loci_events(idx,occurred);
    CREATE INDEX IF NOT EXISTS loci_events_by_partition ON loci_events(idx,partition_start);")?;
    Ok(())
}

fn policy(db: &Connection, index: &str) -> Result<Option<StreamPolicy>> {
    let raw: String = db.query_row(
        "SELECT settings FROM loci_indexes WHERE name=?",
        [index],
        |row| row.get(0),
    )?;
    StreamPolicy::from_settings(&serde_json::from_str(&raw)?)
}

pub(super) fn sync(db: &Connection, index: &str, mutations: &[Mutation]) -> Result<()> {
    let Some(p) = policy(db, index)? else {
        return Ok(());
    };
    // The document table already contains the final state of this batch. Look up
    // that state once per ID, including when an upsert is followed by a delete.
    let mut ids = std::collections::BTreeSet::new();
    for op in mutations {
        ids.insert(match op {
            Mutation::Upsert { document } => &document.id,
            Mutation::Delete { id } => id,
        });
    }
    let mut find = db.prepare("SELECT body FROM loci_documents WHERE idx=? AND id=?")?;
    let mut put=db.prepare("INSERT INTO loci_events VALUES(?1,?2,?3,?4) ON CONFLICT(idx,id) DO UPDATE SET occurred=excluded.occurred, partition_start=excluded.partition_start")?;
    for id in ids {
        use rusqlite::OptionalExtension;
        if let Some(raw) = find
            .query_row(params![index, id], |r| r.get::<_, String>(0))
            .optional()?
        {
            let doc: super::Document = serde_json::from_str(&raw)?;
            let time = timestamp(
                doc.fields
                    .get(&p.timestamp)
                    .context("missing event timestamp")?,
            )?;
            let start = time
                .div_euclid(p.rollover_seconds)
                .checked_mul(p.rollover_seconds)
                .context("timestamp partition outside supported range")?;
            put.execute(params![index, id, time, start])?;
        }
    }
    Ok(())
}

impl Catalog {
    /// Sweep automatic event retention, reclaim expired cursors, and advance FTS compaction.
    pub fn maintain(&self, now: i64) -> Result<()> {
        for name in self.list_indexes()? {
            let snapshot = self.snapshot(&name)?;
            let settings = snapshot.settings()?;
            drop(snapshot);
            if StreamPolicy::from_settings(&settings)?.is_some_and(|policy| policy.automatic) {
                self.expire_stream(&name, now)?;
            }
        }
        let db = connect(&self.path)?;
        db.execute("DELETE FROM loci_cursor_results WHERE expires<=?1", [now])?;
        db.execute(
            "INSERT INTO loci_fts(loci_fts, rank) VALUES('merge', 32)",
            [],
        )?;
        db.execute_batch("PRAGMA optimize;")?;
        let _: (i64, i64, i64) = db.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        Ok(())
    }

    /// Return UTC partition starts and current record counts, including late arrivals.
    pub fn stream_partitions(&self, index: &str) -> Result<Vec<(i64, u64)>> {
        let db = connect(&self.path)?;
        ensure!(policy(&db, index)?.is_some(), "index has no stream policy");
        let mut s=db.prepare("SELECT partition_start,count(*) FROM loci_events WHERE idx=? GROUP BY partition_start ORDER BY partition_start")?;
        Ok(s.query_map([index], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Atomically expire eligible events and their lexical entries.
    pub fn expire_stream(&self, index: &str, now: i64) -> Result<WriteResult> {
        let mut db = connect(&self.path)?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let policy = policy(&tx, index)?.context("index has no stream policy")?;
        let cutoff = now
            .checked_sub(policy.retention_seconds)
            .context("retention cutoff outside supported range")?;
        tx.execute("DELETE FROM loci_fts WHERE rowid IN (SELECT d.rowid FROM loci_events e JOIN loci_documents d ON d.idx=e.idx AND d.id=e.id WHERE e.idx=? AND e.occurred<?)",params![index,cutoff])?;
        let changed=tx.execute("DELETE FROM loci_documents WHERE idx=?1 AND id IN (SELECT id FROM loci_events WHERE idx=?1 AND occurred<?2)",params![index,cutoff])?;
        if changed > 0 {
            tx.execute(
                "UPDATE loci_indexes SET generation=generation+1 WHERE name=?",
                [index],
            )?;
        }
        let generation = generation(&tx, index)?;
        tx.commit()?;
        Ok(WriteResult {
            generation,
            changed,
        })
    }
}
