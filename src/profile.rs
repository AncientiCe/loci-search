use crate::{
    Query,
    index::Engine,
    ranking::{SearchConfig, Strategy},
    terms,
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{collections::BTreeSet, time::Instant};

#[derive(Default)]
pub(crate) struct Phases {
    pub rescue: f64,
    pub weights: f64,
    pub routing: f64,
    pub vectors: f64,
    pub postings: f64,
    pub selection: f64,
    pub rerank: f64,
    pub graph: f64,
}

pub(crate) struct Timer(Option<(&'static str, Instant)>);

impl Timer {
    pub(crate) fn start(profile: &Option<&mut Phases>, label: &'static str) -> Self {
        Self(profile.is_some().then(|| (label, Instant::now())))
    }

    pub(crate) fn finish(self, profile: &mut Option<&mut Phases>) {
        if let Some((label, start)) = self.0 {
            let profile = profile.as_deref_mut().expect("active profile");
            let field = match label {
                "rescue" => &mut profile.rescue,
                "weights" => &mut profile.weights,
                "routing" => &mut profile.routing,
                "vectors" => &mut profile.vectors,
                "postings" => &mut profile.postings,
                "selection" => &mut profile.selection,
                "rerank" => &mut profile.rerank,
                "graph" => &mut profile.graph,
                _ => unreachable!(),
            };
            *field += start.elapsed().as_secs_f64() * 1000.0;
        }
    }
}

#[derive(Serialize)]
pub struct Report {
    queries: usize,
    strategy: Strategy,
    config: SearchConfig,
    preparation_ms: f64,
    posting_payload_bytes: usize,
    quantized_vector_bytes: usize,
    phases: Vec<Summary>,
    by_query_terms: Vec<LengthSummary>,
    samples: Vec<QuerySample>,
    signatures: Vec<Signature>,
}

#[derive(Serialize)]
struct Summary {
    phase: &'static str,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
}

/// Per-phase cost for queries with a given number of distinct terms.
#[derive(Serialize)]
struct LengthSummary {
    terms: &'static str,
    queries: usize,
    rescue_mean_ms: f64,
    rescue_p95_ms: f64,
    total_mean_ms: f64,
    total_p95_ms: f64,
    rescue_scored_mean: f64,
}

#[derive(Serialize)]
struct QuerySample {
    query: String,
    terms: usize,
    rescue_ms: f64,
    weights_ms: f64,
    graph_ms: f64,
    rerank_ms: f64,
    total_ms: f64,
    rescue_scored: Option<usize>,
    rescue_budget_hit: Option<bool>,
}

#[derive(Serialize)]
struct Signature {
    query: String,
    ranked: Vec<String>,
    rescue_ranks: Vec<Option<usize>>,
    route_regions: Vec<Vec<Vec<usize>>>,
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[((values.len() as f64 * p).ceil() as usize).max(1) - 1]
}

/// Time each phase of `strategy` (regional or hybrid) for every query.
pub fn run(
    engine: &mut Engine,
    queries: &[Query],
    strategy: Strategy,
    config: &SearchConfig,
) -> Result<Report> {
    ensure!(!queries.is_empty(), "queries are required");
    ensure!(
        strategy.uses_rescue(),
        "profile runs regional or hybrid search"
    );
    let started = Instant::now();
    engine.prepare(strategy, config)?;
    let preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
    let posting_payload_bytes = engine.regional_index().payload_bytes();
    let quantized_vector_bytes = engine.regional_index().quantized_bytes();
    for query in queries.iter().take(32) {
        engine.search(query, strategy, config)?;
    }
    let mut series: [Vec<f64>; 10] = Default::default();
    let mut samples = Vec::with_capacity(queries.len());
    let mut signatures = Vec::new();
    for query in queries {
        let start = Instant::now();
        let mut profile = Phases::default();
        let result = engine.profile_search(query, strategy, config, &mut profile)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let measured = profile.rescue
            + profile.weights
            + profile.routing
            + profile.vectors
            + profile.postings
            + profile.selection
            + profile.rerank
            + profile.graph;
        let values = [
            profile.rescue,
            profile.weights,
            profile.routing,
            profile.vectors,
            profile.postings,
            profile.selection,
            profile.rerank,
            profile.graph,
            (elapsed - measured).max(0.0),
            elapsed,
        ];
        for (values, sample) in series.iter_mut().zip(values) {
            values.push(sample);
        }
        samples.push(QuerySample {
            query: query.id.clone(),
            terms: terms(&query.text).collect::<BTreeSet<_>>().len(),
            rescue_ms: profile.rescue,
            weights_ms: profile.weights,
            graph_ms: profile.graph,
            rerank_ms: profile.rerank,
            total_ms: elapsed,
            rescue_scored: result.work.global_rescue_chunks_scored,
            rescue_budget_hit: result.work.global_rescue_budget_hit,
        });
        signatures.push(Signature {
            query: query.id.clone(),
            rescue_ranks: result
                .hits
                .iter()
                .map(|hit| hit.global_lexical_rank)
                .collect(),
            route_regions: result
                .hits
                .iter()
                .map(|hit| {
                    hit.routes
                        .iter()
                        .map(|route| route.path.iter().map(|step| step.region).collect())
                        .collect()
                })
                .collect(),
            ranked: result.hits.into_iter().map(|hit| hit.source.id).collect(),
        });
    }
    let names = [
        "global_lexical_rescue",
        "coverage_weights",
        "routing",
        "vectors_and_memberships",
        "regional_postings",
        "regional_top_k",
        "rerank_and_hits",
        "global_graph",
        "other",
        "total",
    ];
    let phases = names
        .into_iter()
        .zip(series)
        .map(|(phase, mut values)| Summary {
            phase,
            mean_ms: values.iter().sum::<f64>() / values.len() as f64,
            p50_ms: percentile(&mut values, 0.50),
            p95_ms: percentile(&mut values, 0.95),
        })
        .collect();
    let buckets: [(&'static str, fn(usize) -> bool); 4] = [
        ("1-8", |n| n <= 8),
        ("9-12", |n| (9..=12).contains(&n)),
        ("13-16", |n| (13..=16).contains(&n)),
        ("17+", |n| n >= 17),
    ];
    let by_query_terms = buckets
        .into_iter()
        .filter_map(|(label, keep)| {
            let group = samples.iter().filter(|s| keep(s.terms)).collect::<Vec<_>>();
            if group.is_empty() {
                return None;
            }
            let n = group.len() as f64;
            let mut rescue = group.iter().map(|s| s.rescue_ms).collect::<Vec<_>>();
            let mut total = group.iter().map(|s| s.total_ms).collect::<Vec<_>>();
            Some(LengthSummary {
                terms: label,
                queries: group.len(),
                rescue_mean_ms: rescue.iter().sum::<f64>() / n,
                rescue_p95_ms: percentile(&mut rescue, 0.95),
                total_mean_ms: total.iter().sum::<f64>() / n,
                total_p95_ms: percentile(&mut total, 0.95),
                rescue_scored_mean: group
                    .iter()
                    .map(|s| s.rescue_scored.unwrap_or(0) as f64)
                    .sum::<f64>()
                    / n,
            })
        })
        .collect();
    Ok(Report {
        queries: queries.len(),
        strategy,
        config: config.clone(),
        preparation_ms,
        posting_payload_bytes,
        quantized_vector_bytes,
        phases,
        by_query_terms,
        samples,
        signatures,
    })
}
