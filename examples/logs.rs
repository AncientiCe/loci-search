//! Logs: a late event lands in an earlier bucket, then retention drops the old ones.
//!
//! `cargo run --locked --example logs`

use anyhow::{Result, ensure};
use loci_search::store::{Catalog, Mutation, SearchRequest};

fn main() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let catalog = Catalog::open(directory.path().join("catalog.sqlite"))?;
    catalog.create_index_with_settings(
        "logs",
        &serde_json::json!({
            "mappings": {"time": "date"},
            "stream": {"timestamp": "time", "rollover_seconds": 200, "retention_seconds": 300, "automatic": false}
        }),
    )?;
    catalog.bulk(
        "logs",
        &[
            event("old", "1", 50, "heartbeat")?,
            event("edge", "1", 500, "trace-42")?,
        ],
    )?;
    catalog.bulk("logs", &[event("late", "1", 350, "trace-42")?])?;
    let before = hours(&catalog)?;
    ensure!(
        before == serde_json::json!([[0, 1], [200, 1], [400, 1]]),
        "buckets before retention were {before}"
    );
    catalog.expire_stream("logs", 600)?;
    let trace = search(
        &catalog,
        serde_json::json!({"filter": {"op": "eq", "field": "trace", "value": "trace-42"}, "sort": [{"field": "time"}], "limit": 10}),
    )?;
    let ids = trace
        .hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<Vec<_>>();
    ensure!(
        ids == ["late", "edge"] && trace.total == 2,
        "retained trace was {ids:?}"
    );
    let after = hours(&catalog)?;
    ensure!(
        after == serde_json::json!([[0, 0], [200, 1], [400, 1]]),
        "buckets after retention were {after}"
    );
    println!("logs: late event joined the middle hour; retention left {ids:?}");
    Ok(())
}

fn event(id: &str, revision: &str, time: i64, trace: &str) -> Result<Mutation> {
    Ok(serde_json::from_value(serde_json::json!({
        "op": "upsert",
        "document": {"id": id, "revision": revision, "text": id, "fields": {"time": time, "trace": trace}}
    }))?)
}

fn hours(catalog: &Catalog) -> Result<serde_json::Value> {
    let page = search(
        catalog,
        serde_json::json!({
            "limit": 1,
            "aggregations": {"hours": {"kind": "date_histogram", "field": "time", "start": 0, "end": 600, "width": 200}}
        }),
    )?;
    Ok(page.aggregations["hours"].clone())
}

fn search(catalog: &Catalog, body: serde_json::Value) -> Result<loci_search::store::SearchPage> {
    let request: SearchRequest = serde_json::from_value(body)?;
    catalog.search("logs", &request, &[])
}
