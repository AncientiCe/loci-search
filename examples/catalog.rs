//! Product catalog: an exact SKU, then a stock filter after an update.
//!
//! `cargo run --locked --example catalog`

use anyhow::{Result, ensure};
use loci_search::store::{Catalog, Mutation, SearchRequest};

fn main() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let catalog = Catalog::open(directory.path().join("catalog.sqlite"))?;
    catalog.create_index_with_settings(
        "shop",
        &serde_json::json!({"mappings": {"stock": "integer", "price_cents": "integer", "sku": "keyword"}}),
    )?;
    catalog.bulk(
        "shop",
        &[
            item(
                "p1",
                "1",
                "Rechargeable camping lantern",
                "LMP-201",
                3,
                3000,
            )?,
            item("p2", "1", "Pocket hiking torch", "LMP-210", 1, 2000)?,
            item("p3", "1", "Storm lantern", "LMP-202", 0, 1500)?,
        ],
    )?;
    let sku = search(&catalog, serde_json::json!({"text": "LMP-201", "limit": 1}))?;
    ensure!(sku.hits[0].id == "p1", "SKU missed {}", sku.hits[0].id);

    catalog.bulk(
        "shop",
        &[item(
            "p1",
            "2",
            "Rechargeable camping lantern",
            "LMP-201",
            0,
            3000,
        )?],
    )?;
    let available = search(
        &catalog,
        serde_json::json!({
            "filter": {"op": "range", "field": "stock", "gte": 1},
            "sort": [{"field": "price_cents", "direction": "asc"}],
            "limit": 10
        }),
    )?;
    let ids = available
        .hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<Vec<_>>();
    ensure!(ids == ["p2"], "in-stock order changed to {ids:?}");
    println!("catalog: LMP-201 -> p1; after the stock update only {ids:?} remains");
    Ok(())
}

fn item(
    id: &str,
    revision: &str,
    text: &str,
    sku: &str,
    stock: i64,
    price: i64,
) -> Result<Mutation> {
    Ok(serde_json::from_value(serde_json::json!({
        "op": "upsert",
        "document": {
            "id": id,
            "revision": revision,
            "text": text,
            "fields": {"sku": sku, "stock": stock, "price_cents": price}
        }
    }))?)
}

fn search(catalog: &Catalog, body: serde_json::Value) -> Result<loci_search::store::SearchPage> {
    let request: SearchRequest = serde_json::from_value(body)?;
    catalog.search("shop", &request, &[])
}
