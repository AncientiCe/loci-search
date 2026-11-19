use crate::{
    Query, graph,
    index::{Engine, Manifest},
    ranking::{SearchConfig, SearchResult, Strategy},
    topology::{MEMBERSHIP_LIMIT, MEMBERSHIP_MARGIN},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    time::Instant,
};

const STRATEGIES: [Strategy; 3] = [Strategy::Bm25, Strategy::Global, Strategy::Regional];

#[derive(Debug, Serialize)]
pub struct Report {
    version: &'static str,
    platform: (&'static str, &'static str),
    fixture: bool,
    index: Manifest,
    index_bytes: u64,
    ingest_ms: Option<f64>,
    open_ms: f64,
    graph_build_ms: f64,
    graph_degree: usize,
    graph_construction_width: usize,
    membership_limit: usize,
    membership_margin: f32,
    queries: usize,
    repeats: usize,
    config: SearchConfig,
    results: Vec<Summary>,
    exact_reference: Vec<Reference>,
}

#[derive(Debug, Serialize)]
struct Summary {
    strategy: Strategy,
    group: String,
    samples: usize,
    candidate_recall: Option<f64>,
    ndcg_at_10: Option<f64>,
    recall_at_100: Option<f64>,
    all_evidence_at_100: Option<f64>,
    lexical_rescue_relevant_candidates_mean: f64,
    p50_ms: f64,
    p95_ms: f64,
    regions_expanded_mean: f64,
    regional_postings_read_mean: f64,
    vector_comparisons_mean: f64,
    routing_comparisons_mean: f64,
    rerank_vector_comparisons_mean: f64,
}

#[derive(Debug, Serialize)]
struct Reference {
    query: String,
    exact_ids: Vec<String>,
    approximate_ids: Vec<String>,
    neighbor_recall_at_10: f64,
    neighbor_recall_at_100: f64,
    exact_ms: f64,
    approximate_ms: f64,
    approximate_comparisons: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Quality {
    candidate_recall: Option<f64>,
    ndcg_at_10: Option<f64>,
    recall_at_100: Option<f64>,
    all_evidence_at_100: Option<f64>,
    lexical_rescue_relevant_candidates: usize,
}

#[derive(Clone, Copy)]
struct Sample {
    quality: Quality,
    elapsed_ms: f64,
    regions: usize,
    postings: usize,
    comparisons: usize,
    routing_comparisons: usize,
    rerank_comparisons: usize,
}

#[derive(Serialize)]
struct Trace<'a> {
    repeat: usize,
    query: &'a str,
    group: &'a str,
    elapsed_ms: f64,
    quality: Quality,
    ranked: Vec<&'a str>,
    result: &'a SearchResult,
}

pub struct Run {
    pub fixture: bool,
    pub repeats: usize,
    pub exact_queries: usize,
    pub index_bytes: u64,
    pub ingest_ms: Option<f64>,
    pub open_ms: f64,
}

pub fn evaluate(
    engine: &mut Engine,
    queries: &[Query],
    config: &SearchConfig,
    run: &Run,
    mut traces: impl Write,
) -> Result<Report> {
    config.validate()?;
    ensure!(
        config.limit >= 100,
        "benchmark requires at least 100 result slots"
    );
    ensure!(
        !queries.is_empty() && run.repeats > 0,
        "queries and repeats are required"
    );
    let mut query_ids = BTreeSet::new();
    for query in queries {
        ensure!(
            !query.id.is_empty() && query_ids.insert(&query.id),
            "missing or duplicate query ID"
        );
        ensure!(
            !query.group.is_empty(),
            "missing group in query {}",
            query.id
        );
        for id in query.judgments.keys().chain(&query.required) {
            ensure!(
                engine.ids.binary_search(id).is_ok(),
                "unknown judged passage {id} in {}",
                query.id
            );
        }
        for id in &query.required {
            ensure!(
                query.judgments.get(id).is_some_and(|&grade| grade > 0),
                "required passage {id} must have a positive grade in {}",
                query.id
            );
        }
        engine.query_vector(query)?;
    }
    let started = Instant::now();
    engine.prepare_global();
    let graph_build_ms = milliseconds(started);
    // One untimed pass per engine before collecting warm, single-query samples.
    for query in queries {
        for strategy in STRATEGIES {
            engine.search(query, strategy, config)?;
        }
    }
    let mut samples = BTreeMap::<(usize, Option<&str>), Vec<Sample>>::new();
    for repeat in 0..run.repeats {
        for (position, query) in queries.iter().enumerate() {
            for offset in 0..STRATEGIES.len() {
                let index = (position + repeat + offset) % STRATEGIES.len();
                let started = Instant::now();
                let result = engine.search(query, STRATEGIES[index], config)?;
                let elapsed_ms = milliseconds(started);
                let ranked = result
                    .hits
                    .iter()
                    .map(|hit| hit.source.id.as_str())
                    .collect::<Vec<_>>();
                let quality = quality(query, &ranked, &result);
                let sample = Sample {
                    quality,
                    elapsed_ms,
                    regions: result.work.regions_expanded,
                    postings: result.work.regional_postings_read,
                    comparisons: result.work.vector_comparisons,
                    routing_comparisons: result.work.routing_comparisons,
                    rerank_comparisons: result.work.rerank_vector_comparisons,
                };
                samples.entry((index, None)).or_default().push(sample);
                samples
                    .entry((index, Some(&query.group)))
                    .or_default()
                    .push(sample);
                serde_json::to_writer(
                    &mut traces,
                    &Trace {
                        repeat,
                        query: &query.id,
                        group: &query.group,
                        elapsed_ms,
                        quality,
                        ranked,
                        result: &result,
                    },
                )?;
                traces.write_all(b"\n")?;
            }
        }
    }
    traces.flush()?;
    let mut exact_reference = Vec::new();
    let references = run.exact_queries.min(queries.len());
    for position in 0..references {
        let query = &queries[position * queries.len() / references];
        let vector = engine.query_vector(query)?;
        let depth = config.depth.min(100);
        let started = Instant::now();
        let exact = engine.exact(&vector, depth);
        let exact_ms = milliseconds(started);
        let started = Instant::now();
        let (approximate, approximate_comparisons) =
            engine.nearest(&vector, depth, config.graph_width);
        let approximate_ms = milliseconds(started);
        let ids = |neighbors: &[graph::Neighbor]| {
            neighbors
                .iter()
                .map(|entry| engine.ids[entry.id].clone())
                .collect()
        };
        let recall = |limit: usize| {
            let approx = approximate
                .iter()
                .take(limit)
                .map(|item| item.id)
                .collect::<BTreeSet<_>>();
            let overlap = exact
                .iter()
                .take(limit)
                .filter(|item| approx.contains(&item.id))
                .count();
            overlap as f64 / exact.len().min(limit) as f64
        };
        exact_reference.push(Reference {
            query: query.id.clone(),
            exact_ids: ids(&exact),
            approximate_ids: ids(&approximate),
            neighbor_recall_at_10: recall(10),
            neighbor_recall_at_100: recall(100),
            exact_ms,
            approximate_ms,
            approximate_comparisons,
        });
    }
    let results = samples
        .into_iter()
        .map(|((index, group), samples)| {
            summarize(
                STRATEGIES[index],
                group.unwrap_or("(all queries)"),
                &samples,
            )
        })
        .collect();
    Ok(Report {
        version: env!("CARGO_PKG_VERSION"),
        platform: (std::env::consts::OS, std::env::consts::ARCH),
        fixture: run.fixture,
        index: engine.manifest.clone(),
        index_bytes: run.index_bytes,
        ingest_ms: run.ingest_ms,
        open_ms: run.open_ms,
        graph_build_ms,
        graph_degree: graph::DEGREE,
        graph_construction_width: graph::BUILD_WIDTH,
        membership_limit: MEMBERSHIP_LIMIT,
        membership_margin: MEMBERSHIP_MARGIN,
        queries: queries.len(),
        repeats: run.repeats,
        config: config.clone(),
        results,
        exact_reference,
    })
}

fn quality(query: &Query, ranked: &[&str], result: &SearchResult) -> Quality {
    let candidates = result
        .candidates
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let top = ranked.iter().take(100).copied().collect::<BTreeSet<_>>();
    let grades = query.judgments.values().filter(|&&grade| grade > 0);
    let count = grades.clone().count();
    let recall = |set: &BTreeSet<&str>| {
        (count > 0).then(|| {
            query
                .judgments
                .iter()
                .filter(|(id, grade)| **grade > 0 && set.contains(id.as_str()))
                .count() as f64
                / count as f64
        })
    };
    let dcg = |grades: &[u8]| {
        grades
            .iter()
            .take(10)
            .enumerate()
            .map(|(index, &grade)| {
                (2.0_f64.powi(i32::from(grade)) - 1.0) / ((index + 2) as f64).log2()
            })
            .sum::<f64>()
    };
    let mut ideal = grades.copied().collect::<Vec<_>>();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let returned = ranked
        .iter()
        .take(10)
        .map(|id| query.judgments.get(*id).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let rescue = result
        .lexical_rescue_only_candidates
        .iter()
        .filter(|id| query.judgments.get(*id).is_some_and(|&grade| grade > 0))
        .count();
    Quality {
        candidate_recall: recall(&candidates),
        ndcg_at_10: (count > 0).then(|| dcg(&returned) / dcg(&ideal)),
        recall_at_100: recall(&top),
        all_evidence_at_100: (!query.required.is_empty())
            .then(|| f64::from(query.required.iter().all(|id| top.contains(id.as_str())))),
        lexical_rescue_relevant_candidates: rescue,
    }
}

fn summarize(strategy: Strategy, group: &str, samples: &[Sample]) -> Summary {
    let mut latencies = samples
        .iter()
        .map(|sample| sample.elapsed_ms)
        .collect::<Vec<_>>();
    latencies.sort_unstable_by(f64::total_cmp);
    let percentile =
        |fraction: f64| latencies[((latencies.len() as f64 * fraction).ceil() as usize) - 1];
    let mean = |field: fn(&Sample) -> usize| {
        samples.iter().map(field).sum::<usize>() as f64 / samples.len() as f64
    };
    let quality_mean = |field: fn(Quality) -> Option<f64>| {
        let (sum, count) = samples
            .iter()
            .filter_map(|sample| field(sample.quality))
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
        (count > 0).then(|| sum / f64::from(count))
    };
    Summary {
        strategy,
        group: group.into(),
        samples: samples.len(),
        candidate_recall: quality_mean(|q| q.candidate_recall),
        ndcg_at_10: quality_mean(|q| q.ndcg_at_10),
        recall_at_100: quality_mean(|q| q.recall_at_100),
        all_evidence_at_100: quality_mean(|q| q.all_evidence_at_100),
        lexical_rescue_relevant_candidates_mean: mean(|s| {
            s.quality.lexical_rescue_relevant_candidates
        }),
        p50_ms: percentile(0.50),
        p95_ms: percentile(0.95),
        regions_expanded_mean: mean(|s| s.regions),
        regional_postings_read_mean: mean(|s| s.postings),
        vector_comparisons_mean: mean(|s| s.comparisons),
        routing_comparisons_mean: mean(|s| s.routing_comparisons),
        rerank_vector_comparisons_mean: mean(|s| s.rerank_comparisons),
    }
}

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}
