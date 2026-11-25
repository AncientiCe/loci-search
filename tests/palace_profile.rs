//! Palace profile scoring on exhaustive, global, and regional candidates.
//! An absent profile keeps today's coverage ranker.

use loci_search::{
    Embedding, Normalization, Passage, Query,
    index::{Engine, RetrievalProfile, ingest, ingest_with_profile},
    ranking::{SearchConfig, Strategy},
};
use std::collections::{BTreeMap, HashSet};

fn unit(values: [f32; 4]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    values.into_iter().map(|value| value / norm).collect()
}

fn passages() -> Vec<Passage> {
    vec![
        passage(
            "b-gold",
            "gold-source",
            "zeolite catalyst membrane",
            [1.0, 1.0, 0.0, 0.0],
        ),
        passage(
            "a-near",
            "near-source",
            "banana smoothie",
            [1.0, 0.0, 0.0, 0.0],
        ),
        passage(
            "c-far",
            "far-source",
            "quartz geology",
            [0.0, 0.0, 1.0, 0.0],
        ),
        passage(
            "d-quiet",
            "quiet-source",
            "orchid petal",
            [0.0, 0.0, 0.0, 1.0],
        ),
    ]
}

fn passage(id: &str, document: &str, text: &str, vector: [f32; 4]) -> Passage {
    Passage {
        id: id.into(),
        document: document.into(),
        revision: "1".into(),
        start: 0,
        end: text.len() as u64,
        text: text.into(),
        metadata: BTreeMap::new(),
        vector: unit(vector),
    }
}

fn embedding() -> Embedding {
    Embedding {
        model_revision: "fixture-palace".into(),
        dimensions: 4,
        normalization: Normalization::L2,
    }
}

fn query(text: &str, vector: [f32; 4]) -> Query {
    Query {
        id: "q".into(),
        text: text.into(),
        embedding: embedding(),
        vector: unit(vector),
        group: "general".into(),
        judgments: BTreeMap::new(),
        required: Vec::new(),
    }
}

fn config(limit: usize, depth: usize) -> SearchConfig {
    SearchConfig {
        limit,
        depth,
        graph_width: 8,
        max_regions: 8,
        max_chunks: 8,
        max_postings: 8,
        rescue: 1,
        ..SearchConfig::default()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Miss {
    NeverVisited,
    OutsideBudget,
    RankedTooLow,
}

fn classify(
    gold: &str,
    tight_hits: &[String],
    tight_candidates: &HashSet<String>,
    wide_candidates: &HashSet<String>,
    exhaustive_rank: usize,
    cutoff: usize,
) -> Option<Miss> {
    if exhaustive_rank > cutoff || tight_hits.iter().any(|id| id == gold) {
        return None;
    }
    if tight_candidates.contains(gold) {
        Some(Miss::RankedTooLow)
    } else if wide_candidates.contains(gold) {
        Some(Miss::OutsideBudget)
    } else {
        Some(Miss::NeverVisited)
    }
}

#[test]
fn absent_profile_keeps_todays_ranker_and_palace_scores_candidates() -> anyhow::Result<()> {
    let legacy_dir = tempfile::tempdir()?;
    let palace_dir = tempfile::tempdir()?;
    let legacy_db = legacy_dir.path().join("legacy.sqlite");
    let palace_db = palace_dir.path().join("palace.sqlite");
    let legacy_manifest = ingest(&legacy_db, passages(), embedding(), 2)?;
    let palace_manifest = ingest_with_profile(
        &palace_db,
        passages(),
        embedding(),
        2,
        Some(RetrievalProfile::Palace),
    )?;
    assert_eq!(legacy_manifest.retrieval_profile, None);
    assert_eq!(
        palace_manifest.retrieval_profile,
        Some(RetrievalProfile::Palace)
    );

    let question = query("zeolite catalyst", [1.0, 0.0, 0.0, 0.0]);
    let mut legacy = Engine::open(&legacy_db)?;
    let mut palace = Engine::open(&palace_db)?;
    let legacy_hit = legacy.search(&question, Strategy::Exact, &config(4, 4))?;
    let exhaustive = palace.search(&question, Strategy::Exact, &config(4, 4))?;
    assert_ne!(
        legacy_hit.hits[0].score, exhaustive.hits[0].score,
        "palace fusion must not reuse the coverage ranker"
    );
    assert_eq!(exhaustive.hits[0].source.document, "gold-source");

    let tight = palace.search(&question, Strategy::Global, &config(1, 1))?;
    let wide = palace.search(&question, Strategy::Global, &config(4, 4))?;
    let capped = palace.search(&question, Strategy::Global, &config(1, 4))?;
    let lexical = palace.search(&question, Strategy::Bm25, &config(4, 4))?;
    let rank_of = |id: &str| {
        exhaustive
            .hits
            .iter()
            .position(|hit| hit.source.id == id)
            .map(|rank| rank + 1)
            .unwrap_or(99)
    };
    let ids = |result: &loci_search::ranking::SearchResult| {
        result
            .hits
            .iter()
            .map(|hit| hit.source.id.clone())
            .collect::<Vec<_>>()
    };
    let candidate_set = |result: &loci_search::ranking::SearchResult| {
        result.candidates.iter().cloned().collect::<HashSet<_>>()
    };
    let gold = classify(
        "b-gold",
        &ids(&tight),
        &candidate_set(&tight),
        &candidate_set(&wide),
        rank_of("b-gold"),
        1,
    );
    let near = classify(
        "a-near",
        &ids(&capped),
        &candidate_set(&capped),
        &candidate_set(&wide),
        rank_of("a-near"),
        2,
    );
    let quiet = classify(
        "d-quiet",
        &ids(&lexical),
        &candidate_set(&lexical),
        &candidate_set(&lexical),
        rank_of("d-quiet"),
        4,
    );
    eprintln!(
        "miss ledger outside_budget={gold:?} ranked_too_low={near:?} never_visited={quiet:?} exhaustive={}",
        exhaustive
            .hits
            .iter()
            .map(|hit| hit.source.id.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    assert_eq!(gold, Some(Miss::OutsideBudget));
    assert_eq!(near, Some(Miss::RankedTooLow));
    assert_eq!(quiet, Some(Miss::NeverVisited));
    let _ = palace.search(&question, Strategy::Regional, &config(1, 4))?;
    Ok(())
}

#[test]
fn development_half_is_even_sha256() {
    use sha2::{Digest, Sha256};
    let id = "e47becba";
    let digest = Sha256::digest(id.as_bytes());
    assert_eq!(
        loci_search::palace::development_half(id),
        digest.last().unwrap() % 2 == 0
    );
}
