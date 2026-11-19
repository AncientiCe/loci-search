use loci_search::{
    fixture,
    index::{Engine, ingest, ingest_compact},
    ranking::{RegionalRerank, SearchConfig, SearchResult, Strategy},
};

#[test]
fn lexical_rescue_finds_the_misplaced_identifier_within_routing_budgets() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let db = directory.path().join("test.sqlite");
    let (embedding, passages, queries) = fixture::generate(512)?;
    let manifest = ingest(&db, passages, embedding, 16)?;
    assert_eq!(manifest.chunks, 512);
    let mut engine = Engine::open(&db)?;
    let config = SearchConfig {
        limit: 10,
        depth: 30,
        max_regions: 2,
        max_chunks: 3,
        max_postings: 1,
        rescue: 1,
        ..SearchConfig::default()
    };
    let query = &queries[0];
    let result = engine.search(query, Strategy::Regional, &config)?;
    let expected = query.judgments.keys().next().unwrap();
    assert_eq!(&result.hits[0].source.id, expected);
    assert_eq!(result.hits[0].global_lexical_rank, Some(1));
    assert!(result.lexical_rescue_only_candidates.contains(expected));
    assert!(result.work.regions_expanded <= config.max_regions);
    assert!(result.work.regional_chunks_scored <= config.max_chunks);
    assert!(result.work.regional_postings_read <= config.max_postings);
    Ok(())
}

#[test]
fn repeat_ingest_keeps_one_chunk_and_open_readers_keep_their_snapshot() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let db = directory.path().join("test.sqlite");
    let (embedding, passages, queries) = fixture::generate(128)?;
    ingest(&db, passages, embedding.clone(), 16)?;
    let mut old = Engine::open(&db)?;
    let config = SearchConfig::default();
    let query = &queries[0];
    let before = old.search(query, Strategy::Bm25, &config)?;
    let (_, mut replacement, _) = fixture::generate(128)?;
    replacement.remove(0);
    ingest(&db, replacement, embedding, 16)?;
    let prior = old.search(query, Strategy::Bm25, &config)?;
    assert_eq!(prior.hits[0].source.id, before.hits[0].source.id);
    let mut new = Engine::open(&db)?;
    assert_eq!(new.manifest.chunks, 127);
    assert!(new.search(query, Strategy::Bm25, &config)?.hits.is_empty());
    Ok(())
}

#[test]
fn rejects_query_vectors_from_a_different_embedding_model() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let db = directory.path().join("test.sqlite");
    let (embedding, passages, mut queries) = fixture::generate(64)?;
    ingest(&db, passages, embedding, 16)?;
    let mut engine = Engine::open(&db)?;
    let query = &mut queries[0];
    query.embedding.model_revision.push_str("-different");
    let error = engine
        .search(query, Strategy::Regional, &SearchConfig::default())
        .unwrap_err();
    assert!(error.to_string().contains("embedding manifest differs"));
    Ok(())
}

#[test]
fn regional_reranker_preserves_candidates_and_global_control_scores() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let db = directory.path().join("rerank.sqlite");
    let (embedding, passages, queries) = fixture::generate(128)?;
    ingest(&db, passages, embedding, 16)?;
    let mut engine = Engine::open(&db)?;
    let query = &queries[1];
    let config = SearchConfig::default();
    let tuned = SearchConfig {
        regional_rerank: Some(RegionalRerank {
            coverage: 0.08,
            cosine: 0.92,
            fusion: 0.0,
        }),
        ..config.clone()
    };
    let scores = |result: SearchResult| {
        result
            .hits
            .into_iter()
            .map(|hit| (hit.source.id, hit.score.to_bits()))
            .collect::<Vec<_>>()
    };
    let original = engine.search(query, Strategy::Regional, &config)?;
    let regional = engine.search(query, Strategy::Regional, &tuned)?;
    assert_eq!(original.candidates, regional.candidates);
    assert_eq!(
        original.work.vector_comparisons,
        regional.work.vector_comparisons
    );
    assert_eq!(
        original.work.regional_postings_read,
        regional.work.regional_postings_read
    );
    assert_ne!(scores(original), scores(regional));
    assert_eq!(
        scores(engine.search(query, Strategy::Global, &config)?),
        scores(engine.search(query, Strategy::Global, &tuned)?),
    );
    Ok(())
}

#[test]
fn compact_snapshot_preserves_regional_rankings_and_work() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let normal = directory.path().join("normal.sqlite");
    let compact = directory.path().join("compact.sqlite");
    let (embedding, passages, queries) = fixture::generate(128)?;
    let (_, same_passages, _) = fixture::generate(128)?;
    ingest(&normal, passages, embedding.clone(), 16)?;
    assert_eq!(
        ingest_compact(&compact, same_passages, embedding, 16)?.schema,
        2
    );
    let mut normal = Engine::open(&normal)?;
    let mut compact = Engine::open(&compact)?;
    let config = SearchConfig::default();
    for query in &queries {
        let baseline = normal.search(query, Strategy::Regional, &config)?;
        let reduced = compact.search(query, Strategy::Regional, &config)?;
        assert_eq!(
            serde_json::to_value(baseline)?,
            serde_json::to_value(reduced)?
        );
    }
    Ok(())
}
