//! Business records: a permissioned page, and an empty page for the wrong reader.
//!
//! `cargo run --locked --example records`

use anyhow::{Result, ensure};
use loci_search::store::{Catalog, Mutation, SearchRequest};

fn main() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let catalog = Catalog::open(directory.path().join("catalog.sqlite"))?;
    catalog.create_index_with_settings(
        "records",
        &serde_json::json!({"default_visibility": "private", "mappings": {"amount": "integer"}}),
    )?;
    catalog.bulk(
        "records",
        &[
            record("b1", "1", 100, &["sales"])?,
            record("b2", "1", 110, &["sales", "ops"])?,
            record("b3", "1", 112, &["ops"])?,
        ],
    )?;
    let sales = search(
        &catalog,
        serde_json::json!({
            "filter": {"op": "range", "field": "amount", "gte": 100, "lt": 200},
            "sort": [{"field": "amount", "direction": "asc"}],
            "limit": 2
        }),
        &["sales".into()],
    )?;
    let ids = sales
        .hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<Vec<_>>();
    ensure!(
        ids == ["b1", "b2"] && sales.total == 2,
        "sales page was {ids:?} total {}",
        sales.total
    );
    let guest = search(
        &catalog,
        serde_json::json!({"sort": [{"field": "amount"}], "limit": 10}),
        &["guest".into()],
    )?;
    ensure!(guest.total == 0, "guest saw {} records", guest.total);
    println!("records: sales sees {ids:?}; guest sees none");
    Ok(())
}

fn record(id: &str, revision: &str, amount: i64, readers: &[&str]) -> Result<Mutation> {
    Ok(serde_json::from_value(serde_json::json!({
        "op": "upsert",
        "document": {"id": id, "revision": revision, "text": id, "fields": {"amount": amount}, "readers": readers}
    }))?)
}

fn search(
    catalog: &Catalog,
    body: serde_json::Value,
    principals: &[String],
) -> Result<loci_search::store::SearchPage> {
    let request: SearchRequest = serde_json::from_value(body)?;
    catalog.search("records", &request, principals)
}
