//! Knowledge search: an exact identifier and a question that cites two documents.
//!
//! `cargo run --locked --example knowledge`

use anyhow::{Result, ensure};
use loci_search::store::{Catalog, Mutation, SearchRequest};

fn main() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let catalog = Catalog::open(directory.path().join("catalog.sqlite"))?;
    catalog.create_index("knowledge")?;
    let notes = [
        (
            "manual",
            "After replacing a machine, restore the encrypted vault from its backup.",
        ),
        (
            "release",
            "Version 2.4 requires the original recovery key before a vault can be restored.",
        ),
        (
            "misc",
            "The restore dialog reports VLC-019 when the recovery key is missing.",
        ),
        (
            "invoice",
            "Download an invoice from the payment settings panel.",
        ),
    ];
    let batch = notes
        .into_iter()
        .map(|(id, text)| upsert(id, text))
        .collect::<Result<Vec<_>>>()?;
    catalog.bulk("knowledge", &batch)?;

    let exact = search(
        &catalog,
        "knowledge",
        serde_json::json!({"text": "VLC-019", "limit": 1}),
    )?;
    ensure!(
        exact.hits[0].id == "misc",
        "exact identifier missed {}",
        exact.hits[0].id
    );

    let question = search(
        &catalog,
        "knowledge",
        serde_json::json!({"text": "What steps and credential do I need after replacing my machine?", "limit": 3}),
    )?;
    let ids = question
        .hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<Vec<_>>();
    ensure!(
        ids.contains(&"manual") && ids.contains(&"release"),
        "two-document citation missed {ids:?}"
    );
    println!("knowledge: VLC-019 -> misc; question cites {ids:?}");
    Ok(())
}

fn upsert(id: &str, text: &str) -> Result<Mutation> {
    Ok(serde_json::from_value(serde_json::json!({
        "op": "upsert",
        "document": {"id": id, "revision": "1", "text": text}
    }))?)
}

fn search(
    catalog: &Catalog,
    index: &str,
    body: serde_json::Value,
) -> Result<loci_search::store::SearchPage> {
    let request: SearchRequest = serde_json::from_value(body)?;
    catalog.search(index, &request, &[])
}
