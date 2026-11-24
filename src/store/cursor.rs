use super::{ACCESS, Catalog, SearchPage, SearchRequest, connect};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashSet;

const LIFETIME_SECONDS: u64 = 300;

pub(super) fn initialize(db: &Connection) -> Result<()> {
    db.execute_batch("CREATE TABLE IF NOT EXISTS loci_cursor_results (
        result_id TEXT PRIMARY KEY, idx TEXT NOT NULL REFERENCES loci_indexes(name), principals TEXT NOT NULL,
        expires INTEGER NOT NULL, page TEXT NOT NULL, page_size INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS loci_cursors (
        token TEXT PRIMARY KEY, result_id TEXT NOT NULL REFERENCES loci_cursor_results(result_id) ON DELETE CASCADE,
        next_offset INTEGER NOT NULL, UNIQUE(result_id,next_offset)
    );
    CREATE INDEX IF NOT EXISTS loci_cursors_expiry ON loci_cursor_results(expires);")?;
    db.execute(
        "DELETE FROM loci_cursor_results WHERE expires<=unixepoch()",
        [],
    )?;
    Ok(())
}

impl Catalog {
    /// Stable paging uses an opaque token valid for five minutes. A change in access
    /// invalidates the page and its original totals before any cached data is returned.
    pub fn search(
        &self,
        index: &str,
        request: &SearchRequest,
        principals: &[String],
    ) -> Result<SearchPage> {
        let snapshot = self.snapshot(index)?;
        let mut principals = principals.to_vec();
        principals.sort();
        principals.dedup();
        let scope = serde_json::to_string(&principals)?;
        let db = connect(&self.path)?;
        let (mut page, offset, limit, result_id) = if let Some(token) = &request.cursor {
            let stored: Option<(String,usize,usize,String)> = db.query_row(
                "SELECT page,next_offset,page_size,r.result_id FROM loci_cursors c JOIN loci_cursor_results r ON r.result_id=c.result_id
                 WHERE token=?1 AND idx=?2 AND principals=?3 AND expires>unixepoch()",
                params![token,index,scope], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?)),
            ).optional()?;
            let (raw, offset, limit, id) = stored.context("invalid or expired cursor")?;
            let page: SearchPage = serde_json::from_str(&raw)?;
            if page.generation != snapshot.generation {
                let sql = format!("SELECT d.id FROM loci_documents d WHERE d.idx=?1{ACCESS}");
                let mut stmt = snapshot.db.prepare(&sql)?;
                let visible: HashSet<String> = stmt
                    .query_map(params![index, scope], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?;
                ensure!(
                    page.hits.iter().all(|d| visible.contains(&d.id)),
                    "stale cursor: document visibility changed"
                );
            }
            (page, offset, limit, Some(id))
        } else {
            let full_request = SearchRequest {
                offset: 0,
                limit: usize::MAX,
                ..request.clone()
            };
            let page = snapshot.search(&full_request, &principals)?;
            (page, request.offset, request.limit, None)
        };
        let end = offset.saturating_add(limit).min(page.total);
        if limit > 0 && end < page.total {
            let tx = db.unchecked_transaction()?;
            tx.execute(
                "DELETE FROM loci_cursor_results WHERE expires<=unixepoch()",
                [],
            )?;
            let id = if let Some(id) = result_id {
                id
            } else {
                tx.query_row("INSERT INTO loci_cursor_results(result_id,idx,principals,expires,page,page_size)
                    VALUES(lower(hex(randomblob(24))),?1,?2,unixepoch()+?3,?4,?5) RETURNING result_id",
                    params![index,scope,LIFETIME_SECONDS,serde_json::to_string(&page)?,limit],|r| r.get(0))?
            };
            let token: String = tx.query_row(
                "INSERT INTO loci_cursors(token,result_id,next_offset) VALUES(lower(hex(randomblob(24))),?1,?2)
                 ON CONFLICT(result_id,next_offset) DO UPDATE SET next_offset=excluded.next_offset RETURNING token",
                params![id,end], |r| r.get(0),
            )?;
            tx.commit()?;
            page.next_cursor = Some(token);
        }
        page.hits = page.hits.into_iter().skip(offset).take(limit).collect();
        let visible: HashSet<_> = page.hits.iter().map(|d| d.id.as_str()).collect();
        page.highlights
            .retain(|id, _| visible.contains(id.as_str()));
        Ok(page)
    }
}
