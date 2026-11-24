use anyhow::Result;
use loci_search::store::{Catalog, Document, Mutation, SearchRequest};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let catalog = Catalog::open(directory.path().join("catalog.sqlite"))?;
    catalog.create_index("manuals")?;
    catalog.bulk(
        "manuals",
        &[Mutation::Upsert {
            document: Document {
                id: "recovery-guide".into(),
                revision: "1".into(),
                text: "Restore the encrypted vault from its backup.".into(),
                fields: BTreeMap::new(),
                readers: vec![],
                embed: true,
            },
        }],
    )?;
    let snapshot = catalog.snapshot("manuals")?;
    let page = snapshot.search(
        &SearchRequest {
            text: "encrypted vault".into(),
            ..Default::default()
        },
        &[],
    )?;
    println!("{}", serde_json::to_string_pretty(&page)?);
    Ok(())
}
