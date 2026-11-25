//! How much of the corpus a query must score to keep the exhaustive top 100.
//!
//! Every configuration is compared with an exhaustive scan under the same final
//! ranking function, so a miss is a candidate-generation loss, not a ranking choice.

use super::{FrozenComparison, ScoredPassages, milliseconds, validate_queries};
use crate::{
    Query,
    index::Engine,
    ranking::{SearchConfig, Strategy, coverage_weights},
    regional::bound::{self, Cones, Objective, Scored},
    terms,
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    time::Instant,
};

const K: usize = 100;
const WARMUP: usize = 32;
/// A query term in at most this many passages makes the query a rare-term query.
const RARE_DF: usize = 100;

#[derive(Serialize)]
pub struct PruningReport {
    queries: usize,
    corpus: usize,
    k: usize,
    classes: BTreeMap<&'static str, usize>,
    cones: ConeSummary,
    exhaustive: Row,
    regional: Vec<Row>,
    branch_and_bound: Vec<Row>,
    exact_bound_queries: Vec<BoundTrace>,
}

#[derive(Serialize)]
struct ConeSummary {
    leaves: usize,
    radius_degrees_p05: f64,
    radius_degrees_p50: f64,
    radius_degrees_p95: f64,
}

#[derive(Serialize)]
struct Row {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<SearchConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shrink: Option<f64>,
    overlap_at_10: f64,
    overlap_at_100: f64,
    queries_with_exact_top_100: usize,
    recall_at_100: f64,
    ndcg_at_10: f64,
    passages_scored: ScoredPassages,
    mean_regions: f64,
    mean_graph_comparisons: f64,
    mean_leaves_inside_cone: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    by_class: Vec<ClassRow>,
}

#[derive(Serialize)]
struct ClassRow {
    class: &'static str,
    queries: usize,
    overlap_at_100: f64,
    recall_at_100: f64,
    mean_passages_scored: f64,
}

#[derive(Serialize)]
struct BoundTrace {
    query: String,
    class: &'static str,
    passages_scored: usize,
    regions: usize,
    leaves_inside_cone: usize,
    kth_score: f64,
    stop_reason: &'static str,
}

struct Prepared {
    vector: Vec<f32>,
    weights: BTreeMap<String, f64>,
    class: &'static str,
    oracle: Vec<usize>,
}

/// One query's outcome under one configuration.
struct Outcome {
    top: Vec<usize>,
    scored: usize,
    regions: usize,
    graph: usize,
    inside_cone: usize,
    elapsed_ms: f64,
}

pub fn pruning_dev(
    engine: &mut Engine,
    queries: &[Query],
    source: &FrozenComparison,
    shrinks: &[f64],
    ablate: bool,
) -> Result<PruningReport> {
    ensure!(
        shrinks.iter().all(|shrink| (0.0..=1.0).contains(shrink)),
        "radius scales must be in [0, 1]"
    );
    validate_queries(engine, queries)?;
    let frozen = source.regional_config(engine)?.clone();
    let rerank = frozen
        .regional_rerank
        .ok_or_else(|| anyhow::anyhow!("frozen regional plan has no reranker"))?;
    ensure!(
        rerank.fusion == 0.0,
        "the score bound assumes the frozen reranker gives fusion no weight"
    );
    ensure!(frozen.limit == K, "the frozen plan must return {K} hits");
    engine.prepare(Strategy::Regional, &frozen)?;
    let engine = &*engine;
    let corpus = engine.ids.len();

    eprintln!("Measuring leaf cones");
    let cones = Cones::build(engine);
    let radii = cones.radii_degrees();
    let at = |p: f64| radii[((radii.len() as f64 * p).ceil() as usize).clamp(1, radii.len()) - 1];
    let cone_summary = ConeSummary {
        leaves: cones.leaves(),
        radius_degrees_p05: at(0.05),
        radius_degrees_p50: at(0.50),
        radius_degrees_p95: at(0.95),
    };

    eprintln!("Scoring the whole corpus for {} queries", queries.len());
    let mut prepared = Vec::with_capacity(queries.len());
    for query in queries {
        let query_terms = terms(&query.text).collect::<BTreeSet<_>>();
        let index = engine.regional_index();
        let class = if query_terms
            .iter()
            .any(|term| term.chars().any(|c| c.is_ascii_digit()))
        {
            "numeric"
        } else if query_terms
            .iter()
            .any(|term| (1..=RARE_DF).contains(&index.document_frequency(term)))
        {
            "rare_term"
        } else {
            "common"
        };
        prepared.push(Prepared {
            vector: engine.query_vector(query)?,
            weights: coverage_weights(engine, &query_terms)?,
            class,
            oracle: Vec::new(),
        });
    }
    let mut oracle_ms = Vec::with_capacity(queries.len());
    for item in &mut prepared {
        let objective = Objective {
            weights: &item.weights,
            rerank,
        };
        let started = Instant::now();
        let top = bound::exhaustive(engine, &item.vector, &objective, K, None);
        oracle_ms.push(milliseconds(started));
        item.oracle = top.into_iter().map(|hit| hit.id).collect();
    }
    let exhaustive_outcomes = prepared
        .iter()
        .zip(oracle_ms)
        .map(|(item, elapsed_ms)| Outcome {
            top: item.oracle.clone(),
            scored: corpus,
            regions: 0,
            graph: 0,
            inside_cone: 0,
            elapsed_ms,
        })
        .collect::<Vec<_>>();
    let exhaustive = summarize(
        "exhaustive".into(),
        None,
        None,
        engine,
        queries,
        &prepared,
        &exhaustive_outcomes,
    );

    let rows = engine
        .ids
        .iter()
        .enumerate()
        .map(|(row, id)| (id.as_str(), row))
        .collect::<HashMap<_, _>>();
    let mut regional = Vec::new();
    let plans = if ablate {
        ablations(&frozen, corpus)
    } else {
        Vec::new()
    };
    for (name, config) in plans {
        config.validate()?;
        for query in queries.iter().take(WARMUP) {
            engine.search_prepared(query, Strategy::Regional, &config)?;
        }
        let mut outcomes = Vec::with_capacity(queries.len());
        for query in queries {
            let started = Instant::now();
            let result = engine.search_prepared(query, Strategy::Regional, &config)?;
            let elapsed_ms = milliseconds(started);
            let work = &result.work;
            outcomes.push(Outcome {
                top: result
                    .hits
                    .iter()
                    .map(|hit| position(&rows, &hit.source.id))
                    .collect::<Result<_>>()?,
                scored: work
                    .regional_or_rescue_chunks_scored
                    .unwrap_or(work.regional_chunks_scored),
                regions: work.regions_expanded,
                graph: work.graph_vector_comparisons.unwrap_or(0),
                inside_cone: 0,
                elapsed_ms,
            });
        }
        let row = summarize(
            name,
            Some(config),
            None,
            engine,
            queries,
            &prepared,
            &outcomes,
        );
        eprintln!(
            "{}: overlap@100 {:.4}, Recall@100 {:.4}, scored mean {:.0}, p95 {:.2} ms",
            row.name, row.overlap_at_100, row.recall_at_100, row.passages_scored.mean, row.p95_ms
        );
        regional.push(row);
    }

    let mut seen = vec![false; corpus];
    let mut branch_and_bound = Vec::new();
    let mut exact_bound_queries = Vec::new();
    for &shrink in shrinks {
        let mut outcomes = Vec::with_capacity(queries.len());
        let run = |item: &Prepared, seen: &mut [bool]| {
            let objective = Objective {
                weights: &item.weights,
                rerank,
            };
            bound::branch_and_bound(engine, &cones, &item.vector, &objective, K, shrink, seen)
        };
        for item in prepared.iter().take(WARMUP) {
            run(item, &mut seen);
        }
        for (query, item) in queries.iter().zip(&prepared) {
            let started = Instant::now();
            let pruned = run(item, &mut seen);
            let elapsed_ms = milliseconds(started);
            if shrink == 1.0 {
                let top = pruned.top.iter().map(|hit| hit.id).collect::<Vec<_>>();
                ensure!(
                    top == item.oracle,
                    "the safe bound changed the exhaustive top {K} for query {}",
                    query.id
                );
                exact_bound_queries.push(BoundTrace {
                    query: query.id.clone(),
                    class: item.class,
                    passages_scored: pruned.scored,
                    regions: pruned.regions,
                    leaves_inside_cone: pruned.inside_cone,
                    kth_score: pruned.top.last().map_or(0.0, |hit: &Scored| hit.score),
                    stop_reason: pruned.stop_reason,
                });
            }
            outcomes.push(Outcome {
                top: pruned.top.iter().map(|hit| hit.id).collect(),
                scored: pruned.scored,
                regions: pruned.regions,
                graph: 0,
                inside_cone: pruned.inside_cone,
                elapsed_ms,
            });
        }
        let row = summarize(
            format!("branch_and_bound_shrink_{shrink:.1}"),
            None,
            Some(shrink),
            engine,
            queries,
            &prepared,
            &outcomes,
        );
        eprintln!(
            "{}: overlap@100 {:.4}, Recall@100 {:.4}, scored mean {:.0} ({:.2}%), regions {:.0}, leaves inside cone {:.0}, p95 {:.2} ms",
            row.name,
            row.overlap_at_100,
            row.recall_at_100,
            row.passages_scored.mean,
            100.0 * row.passages_scored.mean_corpus_fraction,
            row.mean_regions,
            row.mean_leaves_inside_cone,
            row.p95_ms
        );
        branch_and_bound.push(row);
    }

    let mut classes = BTreeMap::new();
    for item in &prepared {
        *classes.entry(item.class).or_default() += 1;
    }
    Ok(PruningReport {
        queries: queries.len(),
        corpus,
        k: K,
        classes,
        cones: cone_summary,
        exhaustive,
        regional,
        branch_and_bound,
        exact_bound_queries,
    })
}

/// The frozen plan, then global rescue and the global graph removed or capped.
fn ablations(frozen: &SearchConfig, corpus: usize) -> Vec<(String, SearchConfig)> {
    let with = |change: &dyn Fn(&mut SearchConfig)| {
        let mut config = frozen.clone();
        change(&mut config);
        config
    };
    let mut plans = vec![("frozen".to_string(), frozen.clone())];
    plans.push((
        "unbounded_rescue".into(),
        with(&|config| config.rescue_max_scored = None),
    ));
    for percent in [10, 5, 2, 1] {
        let cap = corpus * percent / 100;
        plans.push((
            format!("rescue_cap_{percent}pct"),
            with(&|config| config.rescue_max_scored = Some(cap.max(config.rescue))),
        ));
    }
    let no_rescue = |config: &mut SearchConfig| {
        config.rescue = 0;
        config.rescue_max_scored = None;
    };
    plans.push(("no_rescue".into(), with(&no_rescue)));
    plans.push((
        "no_graph".into(),
        with(&|config| config.graph_candidates = None),
    ));
    plans.push((
        "topology_only".into(),
        with(&|config| {
            no_rescue(config);
            config.graph_candidates = None;
        }),
    ));
    // No regional scan: the global graph, bounded rescue, and the reranker only.
    for (graph, width, rescue) in [
        (100, 256, 40),
        (200, 256, 40),
        (200, 256, 100),
        (200, 128, 100),
    ] {
        plans.push((
            format!("graph_{graph}_width_{width}_rescue_{rescue}_no_regions"),
            with(&|config| {
                config.max_regions = 1;
                config.max_chunks = 1;
                config.max_postings = 1;
                config.graph_candidates = Some(graph);
                config.graph_width = width;
                config.rescue = rescue;
            }),
        ));
    }
    plans
}

fn position(rows: &HashMap<&str, usize>, id: &str) -> Result<usize> {
    rows.get(id)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("hit {id} is not in the index"))
}

fn summarize(
    name: String,
    config: Option<SearchConfig>,
    shrink: Option<f64>,
    engine: &Engine,
    queries: &[Query],
    prepared: &[Prepared],
    outcomes: &[Outcome],
) -> Row {
    let n = queries.len() as f64;
    let overlap = |outcome: &Outcome, oracle: &[usize], k: usize| {
        let truth = oracle.iter().take(k).collect::<BTreeSet<_>>();
        outcome
            .top
            .iter()
            .take(k)
            .filter(|id| truth.contains(id))
            .count() as f64
            / k.min(oracle.len()).max(1) as f64
    };
    let mut row = Row {
        name,
        config,
        shrink,
        overlap_at_10: 0.0,
        overlap_at_100: 0.0,
        queries_with_exact_top_100: 0,
        recall_at_100: 0.0,
        ndcg_at_10: 0.0,
        passages_scored: ScoredPassages::summarize(
            outcomes.iter().map(|outcome| outcome.scored).collect(),
            engine.ids.len(),
        ),
        mean_regions: outcomes.iter().map(|o| o.regions as f64).sum::<f64>() / n,
        mean_graph_comparisons: outcomes.iter().map(|o| o.graph as f64).sum::<f64>() / n,
        mean_leaves_inside_cone: outcomes.iter().map(|o| o.inside_cone as f64).sum::<f64>() / n,
        p50_ms: 0.0,
        p95_ms: 0.0,
        p99_ms: 0.0,
        by_class: Vec::new(),
    };
    let mut classes = BTreeMap::<&'static str, (usize, f64, f64, f64)>::new();
    for ((query, item), outcome) in queries.iter().zip(prepared).zip(outcomes) {
        let at_100 = overlap(outcome, &item.oracle, K);
        row.overlap_at_10 += overlap(outcome, &item.oracle, 10) / n;
        row.overlap_at_100 += at_100 / n;
        row.queries_with_exact_top_100 += usize::from(outcome.top == item.oracle);
        let ranked = outcome
            .top
            .iter()
            .map(|&id| engine.ids[id].as_str())
            .collect::<Vec<_>>();
        let (recall, ndcg) = judged(query, &ranked);
        row.recall_at_100 += recall / n;
        row.ndcg_at_10 += ndcg / n;
        let class = classes.entry(item.class).or_default();
        class.0 += 1;
        class.1 += at_100;
        class.2 += recall;
        class.3 += outcome.scored as f64;
    }
    row.by_class = classes
        .into_iter()
        .map(|(class, (count, overlap, recall, scored))| {
            let count_f = count as f64;
            ClassRow {
                class,
                queries: count,
                overlap_at_100: overlap / count_f,
                recall_at_100: recall / count_f,
                mean_passages_scored: scored / count_f,
            }
        })
        .collect();
    let mut latencies = outcomes.iter().map(|o| o.elapsed_ms).collect::<Vec<_>>();
    latencies.sort_unstable_by(f64::total_cmp);
    let at = |p: f64| latencies[((latencies.len() as f64 * p).ceil() as usize).max(1) - 1];
    row.p50_ms = at(0.50);
    row.p95_ms = at(0.95);
    row.p99_ms = at(0.99);
    row
}

/// Judged Recall@100 and nDCG@10, with the formulas `bench::quality` uses.
fn judged(query: &Query, ranked: &[&str]) -> (f64, f64) {
    let positives = query.judgments.values().filter(|&&grade| grade > 0).count();
    if positives == 0 {
        return (0.0, 0.0);
    }
    let top = ranked.iter().take(K).copied().collect::<BTreeSet<_>>();
    let recall = query
        .judgments
        .iter()
        .filter(|(id, grade)| **grade > 0 && top.contains(id.as_str()))
        .count() as f64
        / positives as f64;
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
    let mut ideal = query
        .judgments
        .values()
        .copied()
        .filter(|&grade| grade > 0)
        .collect::<Vec<_>>();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let returned = ranked
        .iter()
        .take(10)
        .map(|id| query.judgments.get(*id).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    (recall, dcg(&returned) / dcg(&ideal))
}
