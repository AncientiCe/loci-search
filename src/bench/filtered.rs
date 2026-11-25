//! Filtered search: each engine against the exhaustive filtered top 100.
//!
//! A passage's filter attribute is a hash bucket of its id (0..1000), so a filter
//! `bucket < t` admits a fraction `t / 1000` of the corpus independently of topic.
//! Elasticsearch candidates are reranked with Loci's frozen reranker, so every
//! engine is judged by how many of the same exhaustive top 100 it recovers.

use super::{FrozenComparison, milliseconds, validate_queries};
use crate::{
    Query, elastic,
    index::Engine,
    ranking::{SearchConfig, SearchResult, Strategy, coverage_weights},
    regional::bound::{self, Objective},
    terms,
};
use anyhow::{Result, ensure};
use serde::Serialize;
use serde_json::json;
use std::{
    collections::{BTreeSet, HashMap},
    time::Instant,
};

pub const BUCKETS: u32 = 1000;
const K: usize = 100;
const WARMUP: usize = 32;

/// FNV-1a over the UTF-8 bytes of a passage id, reduced to a bucket.
pub fn bucket(id: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in id.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash % BUCKETS
}

/// Build an Elasticsearch comparison index that carries each passage's bucket.
pub fn build_elastic_buckets(engine: &Engine, client: &elastic::Client) -> Result<()> {
    client.build_bucketed(engine, bucket)
}

#[derive(Serialize)]
pub struct FilterReport {
    queries: usize,
    corpus: usize,
    elasticsearch_index: Option<String>,
    levels: Vec<Level>,
}

#[derive(Serialize)]
struct Level {
    selectivity: f64,
    buckets_admitted: u32,
    passages_admitted: usize,
    queries_with_admitted_positives: usize,
    engines: Vec<Row>,
}

#[derive(Serialize)]
struct Row {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<SearchConfig>,
    overlap_at_10: f64,
    overlap_at_100: f64,
    /// Over admitted judged positives, on queries that have one.
    recall_at_100: f64,
    mean_passages_scored: f64,
    mean_regions: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

struct Sample {
    top: Vec<usize>,
    scored: usize,
    regions: usize,
    elapsed_ms: f64,
}

#[derive(Default)]
struct Samples(Vec<Sample>);

pub fn filter_dev(
    engine: &mut Engine,
    queries: &[Query],
    source: &FrozenComparison,
    elastic: Option<&elastic::Client>,
    selectivities: &[f64],
    budget_scales: &[f64],
    elastic_candidates: &[usize],
) -> Result<FilterReport> {
    ensure!(
        budget_scales.iter().all(|&scale| scale >= 1.0),
        "budget scales must be at least 1"
    );
    validate_queries(engine, queries)?;
    let frozen = source.regional_config(engine)?.clone();
    let rerank = frozen
        .regional_rerank
        .ok_or_else(|| anyhow::anyhow!("frozen regional plan has no reranker"))?;
    ensure!(
        rerank.fusion == 0.0,
        "the exhaustive scan assumes no fusion weight"
    );
    let elastic_options = source
        .settings
        .iter()
        .find(|plan| plan.strategy == Strategy::Elasticsearch)
        .and_then(|plan| plan.elastic)
        .ok_or_else(|| anyhow::anyhow!("frozen report has no Elasticsearch options"))?;
    let mut elastic_config = frozen.clone();
    elastic_config.elastic_rerank = Some(rerank);
    elastic_config.regional_rerank = None;
    elastic_config.graph_candidates = None;
    elastic_config.rescue_max_scored = None;
    elastic_config.regional_rescore = None;
    engine.prepare(Strategy::Regional, &frozen)?;
    let engine = &*engine;
    let corpus = engine.ids.len();
    let leaves = engine
        .topology
        .members
        .iter()
        .filter(|members| !members.is_empty())
        .count();
    let rows = engine
        .ids
        .iter()
        .enumerate()
        .map(|(row, id)| (id.as_str(), row))
        .collect::<HashMap<_, _>>();
    let buckets = engine.ids.iter().map(|id| bucket(id)).collect::<Vec<_>>();
    let prepared = queries
        .iter()
        .map(|query| {
            let query_terms = terms(&query.text).collect::<BTreeSet<_>>();
            Ok((
                engine.query_vector(query)?,
                coverage_weights(engine, &query_terms)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut levels = Vec::new();
    for &selectivity in selectivities {
        ensure!(
            selectivity > 0.0 && selectivity <= 1.0,
            "selectivity must be in (0, 1]"
        );
        let admitted_buckets = ((selectivity * f64::from(BUCKETS)).round() as u32).max(1);
        let allow = buckets
            .iter()
            .map(|&b| b < admitted_buckets)
            .collect::<Vec<_>>();
        let admitted = allow
            .iter()
            .enumerate()
            .filter(|&(_, &ok)| ok)
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        let filter = json!({"range": {"bucket": {"lt": admitted_buckets}}});
        let scale = (1.0 / selectivity).max(1.0);
        let mut scaled = frozen.clone();
        scaled.graph_candidates = None;
        scaled.max_regions = ((frozen.max_regions as f64 * scale).ceil() as usize).min(leaves);
        scaled.max_postings = (frozen.max_postings as f64 * scale).ceil() as usize;
        let mut plans = vec![
            ("loci_frozen_graph_postfilter".to_string(), frozen.clone()),
            ("loci_scaled_regions".to_string(), scaled.clone()),
        ];
        for &budget in budget_scales {
            let mut config = scaled.clone();
            config.graph_candidates = frozen.graph_candidates;
            config.max_chunks = (frozen.max_chunks as f64 * budget).ceil() as usize;
            config.max_regions =
                ((frozen.max_regions as f64 * scale * budget).ceil() as usize).min(leaves);
            config.max_postings = (frozen.max_postings as f64 * scale * budget).ceil() as usize;
            plans.push((
                format!("loci_scaled_regions_graph_postfilter_x{budget}"),
                config,
            ));
        }
        let candidate_counts = if elastic_candidates.is_empty() {
            vec![elastic_options.num_candidates]
        } else {
            elastic_candidates.to_vec()
        };
        eprintln!(
            "Selectivity {selectivity}: {} passages admitted",
            admitted.len()
        );

        let mut oracle = Vec::with_capacity(queries.len());
        let mut brute = Samples::default();
        for (vector, weights) in &prepared {
            let objective = Objective { weights, rerank };
            let started = Instant::now();
            let top = bound::exhaustive_over(engine, vector, &objective, K, &admitted);
            brute.0.push(Sample {
                top: top.iter().map(|hit| hit.id).collect(),
                scored: admitted.len(),
                regions: 0,
                elapsed_ms: milliseconds(started),
            });
            oracle.push(top.into_iter().map(|hit| hit.id).collect::<Vec<_>>());
        }

        let mut loci = plans.iter().map(|_| Samples::default()).collect::<Vec<_>>();
        let mut es = candidate_counts
            .iter()
            .map(|_| Samples::default())
            .collect::<Vec<_>>();
        let run_loci = |config: &SearchConfig, query: &Query| -> Result<Sample> {
            let started = Instant::now();
            let result = engine.search_filtered(query, Strategy::Regional, config, &allow)?;
            let elapsed_ms = milliseconds(started);
            sample(&rows, &result, elapsed_ms, |work| {
                (
                    work.regional_or_rescue_chunks_scored
                        .unwrap_or(work.regional_chunks_scored)
                        + work.graph_vector_comparisons.unwrap_or(0),
                    work.regions_expanded,
                )
            })
        };
        let run_elastic =
            |client: &elastic::Client, candidates: usize, query: &Query| -> Result<Sample> {
                let options = elastic::Options {
                    num_candidates: candidates,
                    ..elastic_options
                };
                let started = Instant::now();
                let result =
                    client.search_filtered(engine, query, &elastic_config, options, &filter)?;
                let elapsed_ms = milliseconds(started);
                sample(&rows, &result, elapsed_ms, |_| (0, 0))
            };
        for query in queries.iter().take(WARMUP) {
            for (_, config) in &plans {
                run_loci(config, query)?;
            }
            if let Some(client) = elastic {
                for &candidates in &candidate_counts {
                    run_elastic(client, candidates, query)?;
                }
            }
        }
        // Alternate which engine goes first so drift and caching do not favour one.
        for (i, query) in queries.iter().enumerate() {
            let elastic_first = i % 2 == 1;
            if elastic_first && let Some(client) = elastic {
                for (&candidates, samples) in candidate_counts.iter().zip(&mut es) {
                    samples.0.push(run_elastic(client, candidates, query)?);
                }
            }
            for ((_, config), samples) in plans.iter().zip(&mut loci) {
                samples.0.push(run_loci(config, query)?);
            }
            if !elastic_first && let Some(client) = elastic {
                for (&candidates, samples) in candidate_counts.iter().zip(&mut es) {
                    samples.0.push(run_elastic(client, candidates, query)?);
                }
            }
        }

        let positives = queries
            .iter()
            .map(|query| {
                query
                    .judgments
                    .iter()
                    .filter(|(id, grade)| {
                        **grade > 0 && rows.get(id.as_str()).is_some_and(|&row| allow[row])
                    })
                    .map(|(id, _)| rows[id.as_str()])
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        let mut engines = vec![summarize(
            "brute_force_admitted".into(),
            None,
            &brute,
            &oracle,
            &positives,
        )];
        for ((name, config), samples) in plans.into_iter().zip(&loci) {
            engines.push(summarize(name, Some(config), samples, &oracle, &positives));
        }
        if elastic.is_some() {
            for (candidates, samples) in candidate_counts.iter().zip(&es) {
                engines.push(summarize(
                    format!("elasticsearch_same_reranker_candidates_{candidates}"),
                    Some(elastic_config.clone()),
                    samples,
                    &oracle,
                    &positives,
                ));
            }
        }
        for row in &engines {
            eprintln!(
                "  {}: overlap@100 {:.4}, Recall@100 {:.4}, scored {:.0}, regions {:.0}, p50 {:.2} p95 {:.2} ms",
                row.name,
                row.overlap_at_100,
                row.recall_at_100,
                row.mean_passages_scored,
                row.mean_regions,
                row.p50_ms,
                row.p95_ms
            );
        }
        levels.push(Level {
            selectivity,
            buckets_admitted: admitted_buckets,
            passages_admitted: admitted.len(),
            queries_with_admitted_positives: positives.iter().filter(|set| !set.is_empty()).count(),
            engines,
        });
    }
    Ok(FilterReport {
        queries: queries.len(),
        corpus,
        elasticsearch_index: elastic.map(|client| client.index_name().to_string()),
        levels,
    })
}

fn sample(
    rows: &HashMap<&str, usize>,
    result: &SearchResult,
    elapsed_ms: f64,
    work: impl Fn(&crate::ranking::Work) -> (usize, usize),
) -> Result<Sample> {
    let (scored, regions) = work(&result.work);
    Ok(Sample {
        top: result
            .hits
            .iter()
            .map(|hit| {
                rows.get(hit.source.id.as_str())
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("unknown hit {}", hit.source.id))
            })
            .collect::<Result<_>>()?,
        scored,
        regions,
        elapsed_ms,
    })
}

fn summarize(
    name: String,
    config: Option<SearchConfig>,
    samples: &Samples,
    oracle: &[Vec<usize>],
    positives: &[BTreeSet<usize>],
) -> Row {
    let samples = &samples.0;
    let n = samples.len() as f64;
    let overlap = |top: &[usize], truth: &[usize], k: usize| {
        let truth = truth.iter().take(k).collect::<BTreeSet<_>>();
        if truth.is_empty() {
            return 1.0;
        }
        top.iter().take(k).filter(|id| truth.contains(id)).count() as f64 / truth.len() as f64
    };
    let mut row = Row {
        name,
        config,
        overlap_at_10: 0.0,
        overlap_at_100: 0.0,
        recall_at_100: 0.0,
        mean_passages_scored: samples.iter().map(|s| s.scored as f64).sum::<f64>() / n,
        mean_regions: samples.iter().map(|s| s.regions as f64).sum::<f64>() / n,
        p50_ms: 0.0,
        p95_ms: 0.0,
        p99_ms: 0.0,
    };
    let mut judged = 0;
    for ((sample, truth), relevant) in samples.iter().zip(oracle).zip(positives) {
        row.overlap_at_10 += overlap(&sample.top, truth, 10) / n;
        row.overlap_at_100 += overlap(&sample.top, truth, K) / n;
        if !relevant.is_empty() {
            judged += 1;
            row.recall_at_100 += sample
                .top
                .iter()
                .take(K)
                .filter(|id| relevant.contains(id))
                .count() as f64
                / relevant.len() as f64;
        }
    }
    if judged > 0 {
        row.recall_at_100 /= f64::from(judged);
    }
    let mut latencies = samples.iter().map(|s| s.elapsed_ms).collect::<Vec<_>>();
    latencies.sort_unstable_by(f64::total_cmp);
    let at = |p: f64| latencies[((latencies.len() as f64 * p).ceil() as usize).max(1) - 1];
    row.p50_ms = at(0.50);
    row.p95_ms = at(0.95);
    row.p99_ms = at(0.99);
    row
}
