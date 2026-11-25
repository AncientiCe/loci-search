//! Miss ledger for the frozen LongMemEval development half.
//!
//! Budgets are fixed here, before the other half is read. Weights are the
//! Palace defaults. Set PALACE_LEDGER=1 to run it. A normal `cargo test` skips it.

use loci_search::palace::{self, MiniLm};
use loci_search::{
    Embedding, Normalization, Passage, Query,
    index::{Engine, RetrievalProfile, ingest_compact_with_profile},
    ranking::{SearchConfig, SearchResult, Strategy},
};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

const CHUNKS: &str = r"D:\Dev\Projects\mempalace-rs\data\out\longmemeval\recall_all_research\minilm-l6-v2-q8-exchange\longmemeval_cases.jsonl";
const QUESTIONS: &str = r"D:\Dev\Projects\mempalace-rs\data\longmemeval\longmemeval_s_cleaned.json";

#[derive(Deserialize)]
struct DumpCase {
    question_id: String,
    #[serde(default)]
    answer_session_ids: Vec<String>,
    #[serde(default)]
    chunks: Vec<DumpChunk>,
}

#[derive(Deserialize)]
struct DumpChunk {
    session_id: String,
    text: String,
}

#[derive(Deserialize)]
struct QuestionFile {
    question_id: String,
    question: String,
    question_date: String,
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_dates: Vec<String>,
}

#[derive(Default)]
struct Counts {
    misses: usize,
    never_visited: usize,
    outside_budget: usize,
    ranked_too_low: usize,
}

#[test]
fn development_half_records_miss_reasons_without_tuning_weights() {
    if std::env::var("PALACE_LEDGER").ok().as_deref() != Some("1") {
        eprintln!("set PALACE_LEDGER=1 to score the frozen LongMemEval development half");
        return;
    }
    let questions = load_questions();
    let (model, vocab) = palace::pinned_minilm();
    let encoder = MiniLm::open(&model, &vocab, 2).unwrap();
    let mut global = Counts::default();
    let mut regional = Counts::default();
    let mut questions_scored = 0usize;
    for line in BufReader::new(File::open(CHUNKS).unwrap()).lines() {
        let case: DumpCase = serde_json::from_str(&line.unwrap()).unwrap();
        if !palace::development_half(&case.question_id) || case.chunks.is_empty() {
            continue;
        }
        let Some(question) = questions.get(&case.question_id) else {
            continue;
        };
        let mut texts = vec![question.question.as_str()];
        texts.extend(case.chunks.iter().map(|chunk| chunk.text.as_str()));
        let vectors = encoder.embed(&texts).unwrap();
        let dates: HashMapDates = question
            .haystack_session_ids
            .iter()
            .zip(&question.haystack_dates)
            .map(|(id, date)| (id.clone(), date.clone()))
            .collect();
        let passages = case
            .chunks
            .iter()
            .enumerate()
            .zip(vectors.iter().skip(1))
            .map(|((index, chunk), vector)| Passage {
                id: format!("{}:{index}", chunk.session_id),
                document: chunk.session_id.clone(),
                revision: "1".into(),
                start: 0,
                end: chunk.text.len() as u64,
                text: chunk.text.clone(),
                metadata: metadata(dates.get(&chunk.session_id), &question.question_date),
                vector: vector.clone(),
            })
            .collect::<Vec<_>>();
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("ledger.sqlite");
        ingest_compact_with_profile(
            &db,
            passages,
            embedding(),
            64,
            Some(RetrievalProfile::Palace),
        )
        .unwrap();
        let mut engine = Engine::open(&db).unwrap();
        let query = Query {
            id: case.question_id.clone(),
            text: question.question.clone(),
            embedding: embedding(),
            vector: vectors[0].clone(),
            group: "general".into(),
            judgments: BTreeMap::new(),
            required: Vec::new(),
        };
        let depth = case.chunks.len().max(5);
        let exhaustive = engine
            .search(&query, Strategy::Exact, &budget(depth, depth, 12, 6000))
            .unwrap();
        let order = sources(&exhaustive);
        let global_tight = engine
            .search(&query, Strategy::Global, &budget(5, 40, 12, 6000))
            .unwrap();
        let global_wide = engine
            .search(&query, Strategy::Global, &budget(5, 400, 12, 6000))
            .unwrap();
        let regional_tight = engine
            .search(&query, Strategy::Regional, &budget(5, 40, 12, 6000))
            .unwrap();
        let regional_wide = engine
            .search(&query, Strategy::Regional, &budget(5, 200, 36, 24_000))
            .unwrap();
        for gold in &case.answer_session_ids {
            let rank = order.iter().position(|id| id == gold).map(|rank| rank + 1);
            let Some(rank) = rank else {
                continue;
            };
            if rank > 5 {
                continue;
            }
            tally(&mut global, gold, &global_tight, &global_wide);
            tally(&mut regional, gold, &regional_tight, &regional_wide);
        }
        questions_scored += 1;
        if questions_scored.is_multiple_of(20) {
            eprintln!(
                "ledger {questions_scored} global {:?} regional {:?}",
                (
                    global.never_visited,
                    global.outside_budget,
                    global.ranked_too_low
                ),
                (
                    regional.never_visited,
                    regional.outside_budget,
                    regional.ranked_too_low
                )
            );
        }
    }
    let report = serde_json::json!({
        "half": "development",
        "split": "sha256(question_id) big-endian last byte even",
        "questions": questions_scored,
        "budgets_frozen_before_the_other_half": {
            "tight": {"limit": 5, "global_depth": 40, "regional_regions": 12, "regional_chunks": 6000},
            "wide": {"global_depth": 400, "regional_regions": 36, "regional_chunks": 24000}
        },
        "weights": "palace defaults, not tuned",
        "global": counts_json(&global),
        "regional": counts_json(&regional),
    });
    let out = std::path::Path::new("runs/longmemeval-minilm/dev-half-ledger.json");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap() + "\n").unwrap();
    eprintln!("{report}");
    assert!(questions_scored > 0, "development half was empty");
}

type HashMapDates = std::collections::HashMap<String, String>;

fn metadata(created: Option<&String>, filed: &str) -> BTreeMap<String, serde_json::Value> {
    let mut metadata = BTreeMap::new();
    if let Some(created) = created {
        metadata.insert(
            "created_at".into(),
            serde_json::Value::String(created.clone()),
        );
    }
    metadata.insert("filed_at".into(), serde_json::Value::String(filed.into()));
    metadata
}

fn embedding() -> Embedding {
    Embedding {
        model_revision: loci_search::generation::minilm_embedding().model_revision,
        dimensions: 384,
        normalization: Normalization::L2,
    }
}

fn budget(limit: usize, depth: usize, regions: usize, chunks: usize) -> SearchConfig {
    SearchConfig {
        limit,
        depth,
        rescue: 40.min(depth),
        max_regions: regions,
        max_chunks: chunks,
        max_postings: chunks.saturating_mul(10),
        ..SearchConfig::default()
    }
}

fn sources(result: &SearchResult) -> Vec<String> {
    let mut seen = HashSet::new();
    result
        .hits
        .iter()
        .filter_map(|hit| {
            seen.insert(hit.source.document.clone())
                .then(|| hit.source.document.clone())
        })
        .collect()
}

fn tally(counts: &mut Counts, gold: &str, tight: &SearchResult, wide: &SearchResult) {
    let hits = sources(tight);
    if hits.iter().take(5).any(|id| id == gold) {
        return;
    }
    counts.misses += 1;
    let tight_candidates = documents(tight);
    let wide_candidates = documents(wide);
    if tight_candidates.contains(gold) {
        counts.ranked_too_low += 1;
    } else if wide_candidates.contains(gold) {
        counts.outside_budget += 1;
    } else {
        counts.never_visited += 1;
    }
}

fn documents(result: &SearchResult) -> HashSet<String> {
    result
        .candidates
        .iter()
        .chain(result.hits.iter().map(|hit| &hit.source.id))
        .map(|id| {
            id.rsplit_once(':')
                .map(|(source, _)| source.to_string())
                .unwrap_or_else(|| id.clone())
        })
        .collect()
}

fn counts_json(counts: &Counts) -> serde_json::Value {
    serde_json::json!({
        "misses": counts.misses,
        "never_visited": counts.never_visited,
        "outside_budget": counts.outside_budget,
        "ranked_too_low": counts.ranked_too_low,
    })
}

fn load_questions() -> std::collections::HashMap<String, QuestionFile> {
    let file = File::open(QUESTIONS).unwrap();
    let questions: Vec<QuestionFile> = serde_json::from_reader(file).unwrap();
    questions
        .into_iter()
        .map(|question| (question.question_id.clone(), question))
        .collect()
}
