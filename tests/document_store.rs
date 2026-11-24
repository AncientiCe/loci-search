use anyhow::Result;
use loci_search::store::{
    Catalog, Document, EncodedPassage, Mutation, PreparedDocument, SearchRequest,
};
use loci_search::{Embedding, Normalization};
use serde_json::{Value, json};

fn doc(id: &str, revision: &str, text: &str, fields: Value, readers: &[&str]) -> Mutation {
    Mutation::Upsert {
        document: Document {
            id: id.to_owned(),
            revision: revision.to_owned(),
            text: text.to_owned(),
            fields: serde_json::from_value(fields).unwrap(),
            readers: readers.iter().map(|s| s.to_string()).collect(),
            embed: true,
        },
    }
}

fn request(value: Value) -> SearchRequest {
    serde_json::from_value(value).unwrap()
}

#[test]
fn committed_retries_and_validation_preserve_revisions() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("catalog.sqlite");
    let catalog = Catalog::open(&path)?;
    let settings = json!({"mappings":{"price":"integer","tags":{"type":"keyword","array":true},"when":"date"}});
    assert!(catalog.create_index_with_settings("products", &settings)?);
    assert!(!catalog.create_index_with_settings("products", &settings)?);
    assert!(
        catalog
            .create_index_with_settings("products", &json!({}))
            .is_err()
    );
    let old = doc(
        "p",
        "1",
        "first",
        json!({"price":4,"when":1,"tags":["x"]}),
        &[],
    );
    let written = catalog.bulk_with_key("products", "request", std::slice::from_ref(&old))?;
    assert_eq!(written.generation, 1);
    let reader = catalog.snapshot("products")?;
    let conflict = doc("p", "1", "different", json!({"price":5}), &[]);
    assert!(
        catalog
            .bulk(
                "products",
                &[doc("new", "1", "new", json!({}), &[]), conflict]
            )
            .is_err()
    );
    assert!(
        catalog
            .bulk(
                "products",
                &[doc("bad", "1", "bad", json!({"price":"3"}), &[])]
            )
            .is_err()
    );
    let replaced = doc("p", "2", "second", json!({"price":8}), &[]);
    assert_eq!(catalog.bulk("products", &[replaced])?.generation, 2);
    assert_eq!(reader.document("p", &[])?.unwrap().text, "first");
    assert!(catalog.snapshot_at("products", 1).is_err());
    drop(catalog);
    let catalog = Catalog::open(&path)?;
    assert_eq!(
        catalog.bulk_with_key("products", "request", &[old])?,
        written
    );
    assert!(catalog.bulk_with_key("products", "request", &[]).is_err());
    let now = catalog.snapshot("products")?;
    assert_eq!(now.document("p", &[])?.unwrap().text, "second");
    assert!(now.document("new", &[])?.is_none());
    assert_eq!(now.search(&request(json!({"text":"first"})), &[])?.total, 0);
    Ok(())
}

#[test]
fn scoped_search_returns_full_facets_histograms_and_exact_numeric_order() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index_with_settings(
        "events",
        &json!({"default_visibility":"private","mappings":{"at":"date"}}),
    )?;
    let sales = vec!["sales".to_owned()];
    catalog.bulk("events",&[
        doc("a","1","lamp",json!({"group":["green","green"],"at":"1970-01-01T01:04:59+01:00","value":9007199254740993u64}),&["sales"]),
        doc("b","1","lamp",json!({"group":"green","at":300,"value":9007199254740992u64}),&["sales"]),
        doc("c","1","lamp",json!({"group":"hidden","at":598,"value":1}),&["other"]),
        doc("d","1","lamp",json!({"group":"hidden","at":601,"value":2}),&[]),
    ])?;
    let req = request(
        json!({"text":"lamp","filter":{"op":"range","field":"at","gte":299,"lt":"1970-01-01T00:10:00Z"},
        "sort":[{"field":"value"}],"limit":1,"facets":["group"],"aggregations":{
          "time":{"kind":"date_histogram","field":"at","start":0,"end":900,"width":300},
          "max":{"kind":"max","field":"value"},"sum":{"kind":"sum","field":"value"}}}),
    );
    let page = catalog.search("events", &req, &sales)?;
    assert_eq!(page.total, 2);
    assert_eq!(page.hits[0].id, "b");
    assert_eq!(page.facets["group"][0].count, 2);
    assert_eq!(
        page.aggregations["time"],
        json!([[0, 1], [300, 1], [600, 0]])
    );
    assert_eq!(page.aggregations["max"], json!(9007199254740993u64));
    assert_eq!(page.aggregations["sum"], json!(18014398509481985u64));
    let snapshot = catalog.snapshot("events")?;
    assert!(snapshot.documents(&[])?.is_empty());
    assert!(snapshot.document("c", &sales)?.is_none());
    Ok(())
}

#[test]
fn pinned_cursor_survives_field_changes_and_rechecks_visibility() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog.sqlite"))?;
    catalog.create_index("records")?;
    let team = vec!["team".to_owned()];
    let other = vec!["other".to_owned()];
    let initial = [("a", 1), ("b", 2), ("c", 3)];
    catalog.bulk(
        "records",
        &initial
            .iter()
            .map(|(id, n)| doc(id, "1", "", json!({"x":n}), &["team"]))
            .collect::<Vec<_>>(),
    )?;
    let req = request(json!({"limit":1,"sort":[{"field":"x"}],"facets":["x"]}));
    let first = catalog.search("records", &req, &team)?;
    let resume = request(json!({"cursor":first.next_cursor}));
    assert!(catalog.search("records", &resume, &other).is_err());
    catalog.bulk(
        "records",
        &[
            doc("b", "2", "edited", json!({"x":10}), &["team"]),
            doc("z", "1", "new", json!({"x":0}), &["team"]),
        ],
    )?;
    let next = catalog.search("records", &resume, &team)?;
    assert_eq!(next.total, 3);
    assert_eq!(next.hits[0].revision, "1");
    assert_eq!(next.hits[0].id, "b");
    assert_eq!(
        catalog.search("records", &resume, &team)?.next_cursor,
        next.next_cursor
    );
    catalog.bulk(
        "records",
        &[doc("c", "2", "hidden", json!({"x":3}), &["other"])],
    )?;
    assert!(catalog.search("records", &resume, &team).is_err());
    Ok(())
}

#[test]
fn retrieval_combines_identifiers_synonyms_fuzzy_and_highlights() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog.sqlite"))?;
    catalog.create_index_with_settings("shop", &json!({"synonyms":{"torch":["lantern"]}}))?;
    catalog.bulk(
        "shop",
        &[
            doc(
                "b",
                "1",
                "Rechargeable camping lantern",
                json!({"sku":"LMP-201","title":"Camping lantern"}),
                &[],
            ),
            doc("a", "1", "LMP-201 replacement", json!({}), &[]),
            doc("private", "1", "secret aquarium", json!({}), &["other"]),
        ],
    )?;
    let s = catalog.snapshot("shop")?;
    let exact = s.search(&request(json!({"text":"LMP-201"})), &[])?;
    assert_eq!(exact.hits[0].id, "b");
    let typo = s.search(
        &request(json!({"text":"rechageable campng lantarn","fuzzy":true,"highlight":true})),
        &[],
    )?;
    assert_eq!(typo.hits[0].id, "b");
    assert_eq!(typo.highlights["b"].len(), 3);
    assert_eq!(
        s.search(&request(json!({"text":"torch"})), &[])?.hits[0].id,
        "b"
    );
    assert_eq!(s.suggest("recha", 3, &[])?, vec!["rechargeable"]);
    assert!(s.suggest("aqu", 3, &[])?.is_empty());
    Ok(())
}

#[test]
fn relevance_statistics_and_boosts_obey_the_read_scope() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index("scope")?;
    catalog.bulk(
        "scope",
        &[
            doc("a", "1", "alpha", json!({}), &[]),
            doc("b", "1", "", json!({"tag":"beta"}), &[]),
        ],
    )?;
    let req = request(json!({"text":"alpha beta"}));
    let before = catalog.snapshot("scope")?.search(&req, &[])?;
    assert_eq!(before.hits[0].id, "a");
    let hidden = (0..25)
        .map(|i| doc(&format!("h{i}"), "1", "alpha", json!({}), &["private"]))
        .collect::<Vec<_>>();
    catalog.bulk("scope", &hidden)?;
    let s = catalog.snapshot("scope")?;
    assert_eq!(s.search(&req, &[])?.hits, before.hits);
    assert_eq!(
        s.search(
            &request(json!({"text":"alpha beta","boosts":{"tag":3}})),
            &[]
        )?
        .hits[0]
            .id,
        "b"
    );
    Ok(())
}

#[test]
fn maintenance_expires_automatic_streams_and_reclaims_cursors() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("catalog.sqlite");
    let catalog = Catalog::open(&path)?;
    let policy = json!({"stream":{"timestamp":"at","rollover_seconds":10,"retention_seconds":50}});
    catalog.create_index_with_settings("logs", &policy)?;
    catalog.create_index_with_settings(
        "kept",
        &json!({"stream":{"timestamp":"at","rollover_seconds":10,"retention_seconds":50,"automatic":false}}),
    )?;
    catalog.bulk("logs", &[doc("old", "1", "gone", json!({"at":100}), &[])])?;
    catalog.bulk("kept", &[doc("old", "1", "stay", json!({"at":100}), &[])])?;
    catalog.create_index("pages")?;
    catalog.bulk(
        "pages",
        &[
            doc("a", "1", "lamp", json!({"x":1}), &[]),
            doc("b", "1", "", json!({"x":2}), &[]),
        ],
    )?;
    let page = catalog.search(
        "pages",
        &request(json!({"limit":1,"sort":[{"field":"x"}]})),
        &[],
    )?;
    let cursor = page.next_cursor.expect("page continues");
    let db = rusqlite::Connection::open(&path)?;
    db.execute("UPDATE loci_cursor_results SET expires=0", [])?;
    drop(db);
    catalog.maintain(160)?;
    assert!(catalog.snapshot("logs")?.document("old", &[])?.is_none());
    assert_eq!(
        catalog
            .snapshot("kept")?
            .document("old", &[])?
            .unwrap()
            .text,
        "stay"
    );
    assert!(
        catalog
            .search("pages", &request(json!({"cursor":cursor})), &[])
            .is_err()
    );
    assert_eq!(
        catalog
            .search("pages", &request(json!({"text":"lamp"})), &[])?
            .total,
        1
    );
    Ok(())
}

#[test]
fn model_upgrade_switches_vectors_with_the_source_generation() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog.sqlite"))?;
    catalog.create_index_with_settings("docs", &json!({"model":"lexical"}))?;
    catalog.bulk(
        "docs",
        &[
            doc("a", "1", "text", json!({}), &[]),
            doc("b", "1", "more", json!({}), &[]),
        ],
    )?;
    let generation = catalog.snapshot("docs")?.generation;
    let codec = Embedding {
        model_revision: "codec-1".into(),
        dimensions: 2,
        normalization: Normalization::L2,
    };
    let prepared = vec![
        PreparedDocument {
            id: "a".into(),
            revision: "1".into(),
            passages: vec![EncodedPassage {
                start: 0,
                end: 4,
                vector: vec![1.0, 0.0],
            }],
        },
        PreparedDocument {
            id: "b".into(),
            revision: "1".into(),
            passages: vec![EncodedPassage {
                start: 0,
                end: 4,
                vector: vec![0.0, 1.0],
            }],
        },
    ];
    assert!(generation > 0);
    assert!(
        catalog
            .publish_model(
                "docs",
                generation - 1,
                "bge-small-en-v1.5",
                &codec,
                &prepared
            )
            .is_err()
    );
    let next = catalog.publish_model("docs", generation, "bge-small-en-v1.5", &codec, &prepared)?;
    assert!(next > generation);
    let snapshot = catalog.snapshot("docs")?;
    let settings = snapshot.settings()?;
    assert_eq!(settings["model"], json!("bge-small-en-v1.5"));
    let docs = snapshot.documents(&[])?;
    let (passages, missing) = snapshot.encoded_passages(&docs, &codec)?;
    assert!(missing.is_empty());
    assert_eq!(passages.len(), 2);
    assert!(
        catalog
            .bulk_lexical("docs", &[doc("c", "1", "new", json!({}), &[])], None)
            .is_err()
    );
    let encoded = catalog.bulk_encoded(
        "docs",
        &[doc("c", "1", "note", json!({}), &[])],
        None,
        "bge-small-en-v1.5",
        &codec,
        &[PreparedDocument {
            id: "c".into(),
            revision: "1".into(),
            passages: vec![EncodedPassage {
                start: 0,
                end: 4,
                vector: vec![0.5, 0.5],
            }],
        }],
    )?;
    assert_eq!(encoded.changed, 1);
    Ok(())
}

#[test]
fn a_long_question_borrows_distinctive_terms_from_the_best_hit() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index("knowledge")?;
    catalog.bulk(
        "knowledge",
        &[
            doc(
                "manual",
                "r2",
                "After replacing a machine, restore the encrypted vault from its backup.",
                json!({}),
                &[],
            ),
            doc(
                "release",
                "v2.4",
                "Version 2.4 requires the original recovery key before a vault can be restored.",
                json!({}),
                &[],
            ),
            doc(
                "misc",
                "r7",
                "The restore dialog reports VLC-019 when the recovery key is missing.",
                json!({}),
                &[],
            ),
            doc(
                "invoice",
                "r1",
                "Download an invoice from the payment settings panel.",
                json!({}),
                &[],
            ),
        ],
    )?;
    let page = catalog.search(
        "knowledge",
        &request(json!({
            "text": "What steps and credential do I need after replacing my machine?",
            "limit": 3
        })),
        &[],
    )?;
    let ids: Vec<_> = page
        .hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"manual"), "{ids:?}");
    assert!(ids.contains(&"release"), "{ids:?}");
    let exact = catalog.search(
        "knowledge",
        &request(json!({"text": "VLC-019", "limit": 1})),
        &[],
    )?;
    assert_eq!(exact.hits[0].id, "misc");
    Ok(())
}

#[test]
fn an_uncommitted_write_is_absent_after_reopen() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("catalog.sqlite");
    let catalog = Catalog::open(&path)?;
    catalog.create_index("docs")?;
    catalog.bulk("docs", &[doc("a", "1", "kept", json!({}), &[])])?;
    {
        let mut db = rusqlite::Connection::open(&path)?;
        let tx = db.transaction()?;
        tx.execute(
            "INSERT INTO loci_documents(idx,id,revision,body,readers) VALUES('docs','ghost','1','{\"id\":\"ghost\",\"revision\":\"1\",\"text\":\"nope\",\"fields\":{},\"readers\":[]}','[]')",
            [],
        )?;
        drop(tx);
    }
    let catalog = Catalog::open(&path)?;
    assert!(catalog.snapshot("docs")?.document("ghost", &[])?.is_none());
    assert_eq!(
        catalog.snapshot("docs")?.document("a", &[])?.unwrap().text,
        "kept"
    );
    Ok(())
}

#[test]
fn an_interrupted_compaction_keeps_the_published_generation() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("catalog.sqlite");
    let catalog = Catalog::open(&path)?;
    catalog.create_index("docs")?;
    let batch = (0..12)
        .map(|n| {
            doc(
                &format!("d{n}"),
                "1",
                &format!("ledger entry {n} amber"),
                json!({}),
                &[],
            )
        })
        .collect::<Vec<_>>();
    let written = catalog.bulk("docs", &batch)?;
    let before = catalog.search("docs", &request(json!({"text": "amber", "limit": 12})), &[])?;
    assert_eq!(before.total, 12);
    assert_eq!(before.generation, written.generation);
    {
        let mut db = rusqlite::Connection::open(&path)?;
        let tx = db.transaction()?;
        tx.execute(
            "INSERT INTO loci_fts(loci_fts, rank) VALUES('merge', 32)",
            [],
        )?;
        drop(tx);
    }
    let catalog = Catalog::open(&path)?;
    let after = catalog.search("docs", &request(json!({"text": "amber", "limit": 12})), &[])?;
    assert_eq!(after.generation, written.generation);
    assert_eq!(after.total, 12);
    assert_eq!(after.hits.len(), 12);
    catalog.maintain(0)?;
    let maintained =
        catalog.search("docs", &request(json!({"text": "amber", "limit": 12})), &[])?;
    assert_eq!(maintained.generation, written.generation);
    assert_eq!(maintained.total, 12);
    Ok(())
}

#[test]
fn readers_see_one_generation_while_a_write_commits() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = std::sync::Arc::new(Catalog::open(dir.path().join("catalog"))?);
    catalog.create_index("docs")?;
    catalog.bulk("docs", &[doc("a", "1", "one", json!({}), &[])])?;
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let catalog = catalog.clone();
            scope.spawn(move || {
                for _ in 0..30 {
                    let page = catalog
                        .search("docs", &request(json!({"text": "one"})), &[])
                        .unwrap();
                    assert!(page.hits.iter().all(|hit| hit.id == "a"));
                    assert!(page.generation >= 1);
                }
            });
        }
        catalog
            .bulk("docs", &[doc("a", "2", "one more", json!({}), &[])])
            .unwrap();
    });
    let current = catalog.snapshot("docs")?;
    let document = current.document("a", &[])?.unwrap();
    assert_eq!(document.revision, "2");
    assert_eq!(document.text, "one more");
    assert!(catalog.snapshot_at("docs", 1).is_err());
    Ok(())
}

#[test]
fn private_text_stays_out_of_hits_snippets_suggestions_and_totals() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index("records")?;
    let team = vec!["team".to_owned()];
    catalog.bulk(
        "records",
        &[
            doc("open", "1", "shared lamp", json!({"price": 2}), &["team"]),
            doc(
                "secret",
                "1",
                "hidden xylophone",
                json!({"price": 9}),
                &["other"],
            ),
        ],
    )?;
    let page = catalog.search(
        "records",
        &request(json!({
            "text": "lamp xylophone",
            "highlight": true,
            "facets": ["price"],
            "aggregations": {"sum": {"kind": "sum", "field": "price"}}
        })),
        &team,
    )?;
    assert!(page.exact);
    assert_eq!(page.total, 1);
    assert_eq!(page.hits[0].id, "open");
    assert!(page.highlights["open"].is_empty() || page.highlights.contains_key("open"));
    assert!(!page.highlights.contains_key("secret"));
    let snippet = format!("{:?}", page.highlights);
    assert!(!snippet.contains("xylophone"));
    assert_eq!(page.aggregations["sum"], json!(2));
    assert!(
        catalog
            .snapshot("records")?
            .suggest("xylo", 5, &team)?
            .is_empty()
    );
    assert_eq!(
        catalog.snapshot("records")?.suggest("lamp", 5, &team)?,
        vec!["lamp".to_string()]
    );
    Ok(())
}

#[test]
fn a_match_limit_is_reported_instead_of_an_exact_total() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index("docs")?;
    catalog.bulk(
        "docs",
        &[
            doc("a", "1", "lamp", json!({}), &[]),
            doc("b", "1", "lamp", json!({}), &[]),
        ],
    )?;
    let mut limited = request(json!({"text": "lamp", "limit": 10}));
    limited.match_limit = 1;
    let page = catalog.search("docs", &limited, &[])?;
    assert!(!page.exact);
    assert_eq!(page.limit_reason.as_deref(), Some("match_limit"));
    assert_eq!(page.total, 1);
    Ok(())
}

#[test]
fn a_failed_model_upgrade_leaves_the_previous_generation() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index_with_settings("docs", &json!({"model": "lexical"}))?;
    catalog.bulk("docs", &[doc("a", "1", "text", json!({}), &[])])?;
    let before = catalog.snapshot("docs")?;
    let generation = before.generation;
    let codec = Embedding {
        model_revision: "codec-1".into(),
        dimensions: 2,
        normalization: Normalization::L2,
    };
    let broken = vec![PreparedDocument {
        id: "a".into(),
        revision: "1".into(),
        passages: vec![EncodedPassage {
            start: 0,
            end: 4,
            vector: vec![1.0],
        }],
    }];
    assert!(
        catalog
            .publish_model("docs", generation, "bge-small-en-v1.5", &codec, &broken)
            .is_err()
    );
    let after = catalog.snapshot("docs")?;
    assert_eq!(after.generation, generation);
    assert_eq!(after.settings()?["model"], json!("lexical"));
    Ok(())
}

#[test]
fn an_interrupted_model_upgrade_keeps_the_published_model() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("catalog.sqlite");
    let catalog = Catalog::open(&path)?;
    catalog.create_index_with_settings("docs", &json!({"model": "lexical"}))?;
    catalog.bulk("docs", &[doc("a", "1", "text", json!({}), &[])])?;
    let generation = catalog.snapshot("docs")?.generation;
    let codec = Embedding {
        model_revision: "codec-1".into(),
        dimensions: 2,
        normalization: Normalization::L2,
    };
    let prepared = vec![PreparedDocument {
        id: "a".into(),
        revision: "1".into(),
        passages: vec![EncodedPassage {
            start: 0,
            end: 4,
            vector: vec![1.0, 0.0],
        }],
    }];
    let published =
        catalog.publish_model("docs", generation, "bge-small-en-v1.5", &codec, &prepared)?;
    let reader = catalog.snapshot("docs")?;
    {
        let mut db = rusqlite::Connection::open(&path)?;
        let tx = db.transaction()?;
        tx.execute("DELETE FROM loci_vectors WHERE idx='docs'", [])?;
        tx.execute(
            "UPDATE loci_indexes SET settings=json_set(settings,'$.model','multilingual-e5-small'), generation=generation+1 WHERE name='docs'",
            [],
        )?;
        let during = reader.search(&request(json!({"text": "text"})), &[])?;
        assert_eq!(during.generation, published);
        assert_eq!(during.total, 1);
        drop(tx);
    }
    assert_eq!(reader.settings()?["model"], json!("bge-small-en-v1.5"));
    let catalog = Catalog::open(&path)?;
    let after = catalog.snapshot("docs")?;
    assert_eq!(after.generation, published);
    assert_eq!(after.settings()?["model"], json!("bge-small-en-v1.5"));
    let stored = after.documents(&[])?;
    let (passages, missing) = after.encoded_passages(&stored, &codec)?;
    assert!(missing.is_empty());
    assert_eq!(passages.len(), 1);
    assert_eq!(passages[0].vector, vec![1.0, 0.0]);
    let page = catalog.search("docs", &request(json!({"text": "text"})), &[])?;
    assert_eq!(page.generation, published);
    assert_eq!(page.hits[0].id, "a");
    Ok(())
}

#[test]
fn a_skipped_document_stays_lexical_through_a_model_upgrade() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = Catalog::open(dir.path().join("catalog"))?;
    catalog.create_index_with_settings("logs", &json!({"model": "lexical"}))?;
    let mut skipped = doc("trace", "1", "trace kept lexically", json!({}), &[]);
    if let Mutation::Upsert { document } = &mut skipped {
        document.embed = false;
    }
    catalog.bulk("logs", &[doc("kept", "1", "note", json!({}), &[]), skipped])?;
    let generation = catalog.snapshot("logs")?.generation;
    let codec = Embedding {
        model_revision: "codec-1".into(),
        dimensions: 2,
        normalization: Normalization::L2,
    };
    let embedded = vec![PreparedDocument {
        id: "kept".into(),
        revision: "1".into(),
        passages: vec![EncodedPassage {
            start: 0,
            end: 4,
            vector: vec![1.0, 0.0],
        }],
    }];
    let with_skipped = vec![
        PreparedDocument {
            id: "kept".into(),
            revision: "1".into(),
            passages: vec![EncodedPassage {
                start: 0,
                end: 4,
                vector: vec![1.0, 0.0],
            }],
        },
        PreparedDocument {
            id: "trace".into(),
            revision: "1".into(),
            passages: vec![EncodedPassage {
                start: 0,
                end: 4,
                vector: vec![0.0, 1.0],
            }],
        },
    ];
    assert!(
        catalog
            .publish_model(
                "logs",
                generation,
                "bge-small-en-v1.5",
                &codec,
                &with_skipped
            )
            .is_err()
    );
    assert!(
        catalog
            .publish_model("logs", generation, "bge-small-en-v1.5", &codec, &[])
            .is_err()
    );
    catalog.publish_model("logs", generation, "bge-small-en-v1.5", &codec, &embedded)?;
    let snapshot = catalog.snapshot("logs")?;
    let stored = snapshot.documents(&[])?;
    let (passages, missing) = snapshot.encoded_passages(&stored, &codec)?;
    assert!(missing.is_empty());
    assert_eq!(passages.len(), 1);
    assert_eq!(passages[0].document, "kept");
    assert!(!snapshot.document("trace", &[])?.unwrap().embed);
    drop(snapshot);
    let page = catalog.search("logs", &request(json!({"text": "lexically"})), &[])?;
    assert_eq!(page.total, 1);
    assert_eq!(page.hits[0].id, "trace");
    assert!(!page.hits[0].embed);

    let traced = "trace kept lexically";
    catalog.bulk_encoded(
        "logs",
        &[doc("trace", "2", traced, json!({}), &[])],
        None,
        "bge-small-en-v1.5",
        &codec,
        &[PreparedDocument {
            id: "trace".into(),
            revision: "2".into(),
            passages: vec![EncodedPassage {
                start: 0,
                end: traced.len() as u64,
                vector: vec![0.0, 1.0],
            }],
        }],
    )?;
    let snapshot = catalog.snapshot("logs")?;
    let stored = snapshot.documents(&[])?;
    let (passages, missing) = snapshot.encoded_passages(&stored, &codec)?;
    assert!(missing.is_empty());
    assert_eq!(passages.iter().filter(|p| p.document == "trace").count(), 1);
    drop(snapshot);

    let mut off = doc("trace", "3", traced, json!({}), &[]);
    if let Mutation::Upsert { document } = &mut off {
        document.embed = false;
    }
    catalog.bulk_encoded("logs", &[off], None, "bge-small-en-v1.5", &codec, &[])?;
    let snapshot = catalog.snapshot("logs")?;
    let stored = snapshot.documents(&[])?;
    let (passages, missing) = snapshot.encoded_passages(&stored, &codec)?;
    assert!(missing.is_empty());
    assert!(passages.iter().all(|p| p.document != "trace"));
    assert!(!snapshot.document("trace", &[])?.unwrap().embed);
    drop(snapshot);
    let page = catalog.search("logs", &request(json!({"text": "lexically"})), &[])?;
    assert_eq!(page.hits[0].id, "trace");
    Ok(())
}

#[test]
fn a_full_day_replays_under_search_and_keeps_exact_hours() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let catalog = std::sync::Arc::new(Catalog::open(dir.path().join("catalog"))?);
    catalog.create_index_with_settings(
        "logs",
        &json!({"model":"lexical","mappings":{"time":"date"},"stream":{"timestamp":"time","rollover_seconds":3600,"retention_seconds":86400,"automatic":false}}),
    )?;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let failure = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
    std::thread::scope(|scope| {
        let reader = std::sync::Arc::clone(&catalog);
        let stop_reader = std::sync::Arc::clone(&stop);
        let failure_reader = std::sync::Arc::clone(&failure);
        scope.spawn(move || {
            let query = request(json!({"text": "replay", "limit": 1}));
            while !stop_reader.load(std::sync::atomic::Ordering::Relaxed) {
                match reader.search("logs", &query, &[]) {
                    Ok(page) if page.exact && page.total <= 1441 => {}
                    Ok(page) => {
                        *failure_reader.lock().unwrap() = Some(format!(
                            "inexact search total {} exact {}",
                            page.total, page.exact
                        ));
                        break;
                    }
                    Err(error) => {
                        *failure_reader.lock().unwrap() = Some(error.to_string());
                        break;
                    }
                }
            }
        });
        let mut batch = Vec::new();
        for minute in 0..1440 {
            batch.push(doc(
                &format!("m{minute}"),
                "1",
                "day replay",
                json!({"time": minute * 60}),
                &[],
            ));
            if batch.len() == 120 {
                catalog.bulk("logs", &batch).unwrap();
                batch.clear();
            }
        }
        catalog
            .bulk(
                "logs",
                &[doc("late", "1", "day replay", json!({"time": 600}), &[])],
            )
            .unwrap();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    if let Some(error) = failure.lock().unwrap().clone() {
        anyhow::bail!(error);
    }
    let page = catalog.search(
        "logs",
        &request(json!({
            "text": "replay",
            "limit": 1,
            "aggregations": {
                "hours": {"kind": "date_histogram", "field": "time", "start": 0, "end": 86400, "width": 3600}
            }
        })),
        &[],
    )?;
    assert!(page.exact);
    assert_eq!(page.total, 1441);
    let hours = page.aggregations["hours"].as_array().expect("hour buckets");
    assert_eq!(hours.len(), 24);
    assert_eq!(hours[0], json!([0, 61]));
    for (hour, bucket) in hours.iter().enumerate().skip(1) {
        assert_eq!(*bucket, json!([hour * 3600, 60]), "hour {hour}");
    }
    Ok(())
}
