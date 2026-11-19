use super::{FrozenComparison, Plan, milliseconds, quality, validate_queries};
use crate::{
    Query, elastic,
    index::Engine,
    ranking::{RegionalRerank, SearchResult, Strategy},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
pub struct RescueTuning {
    queries: usize,
    selection: &'static str,
    baseline: RescueTrial,
    trials: Vec<RescueTrial>,
    pub selected: usize,
}

#[derive(Serialize)]
struct RescueTrial {
    max_scored: Option<usize>,
    metrics: Metrics,
    global_rescue: super::ScoredPassages,
    regional_or_rescue: super::ScoredPassages,
    budget_hits: usize,
    queries_losing_recall: usize,
    queries_with_changed_candidates: usize,
}

struct DevelopmentResult {
    recall: f64,
    candidates: Vec<String>,
}

pub fn tune_bounded_rescue(
    engine: &mut Engine,
    queries: &[Query],
    source: &FrozenComparison,
) -> Result<RescueTuning> {
    validate_queries(engine, queries)?;
    let config = source.regional_config(engine)?;
    ensure!(
        config.limit == 100 && config.rescue > 0 && config.rescue_max_scored.is_none(),
        "tuning requires an unbounded regional baseline with rescue and exactly 100 results"
    );
    engine.prepare_regional()?;
    for query in queries.iter().take(32) {
        engine.search(query, Strategy::Regional, config)?;
    }
    let (baseline, original) = measure_rescue(engine, queries, config, None)?;
    let corpus = engine.manifest.chunks;
    let budgets = [4_000, 16_000, 64_000, 256_000, corpus]
        .into_iter()
        .filter(|&cap| cap >= config.rescue && cap <= corpus)
        .collect::<std::collections::BTreeSet<_>>();
    let mut trials = Vec::new();
    for cap in budgets {
        let mut trial_config = config.clone();
        trial_config.rescue_max_scored = Some(cap);
        let (trial, _) = measure_rescue(engine, queries, &trial_config, Some(&original))?;
        eprintln!(
            "Dev rescue cap {cap}: Recall@100 {:.4}, p95 {:.2} ms, mean coverage {:.2}%, capped {}, recall losses {}",
            trial.metrics.recall_at_100,
            trial.metrics.p95_ms,
            100.0 * trial.regional_or_rescue.mean_corpus_fraction,
            trial.budget_hits,
            trial.queries_losing_recall
        );
        trials.push(trial);
    }
    let selected = trials.iter().filter(|trial| {
        trial.queries_losing_recall == 0 && trial.regional_or_rescue.mean < baseline.regional_or_rescue.mean
    }).min_by(|a, b| a.regional_or_rescue.mean.total_cmp(&b.regional_or_rescue.mean)
        .then(a.metrics.p95_ms.total_cmp(&b.metrics.p95_ms))
        .then(a.max_scored.cmp(&b.max_scored)))
        .and_then(|trial| trial.max_scored)
        .ok_or_else(|| anyhow::anyhow!("no tested rescue cap reduced scoring coverage with zero development queries losing recall"))?;
    Ok(RescueTuning {
        queries: queries.len(),
        selection: "zero dev queries losing Recall@100, then lowest mean distinct scoring coverage, p95, and cap",
        baseline,
        trials,
        selected,
    })
}

fn measure_rescue(
    engine: &mut Engine,
    queries: &[Query],
    config: &crate::ranking::SearchConfig,
    baseline: Option<&[DevelopmentResult]>,
) -> Result<(RescueTrial, Vec<DevelopmentResult>)> {
    let mut measurements = Measurements::default();
    let mut per_query = Vec::new();
    let mut rescue_counts = Vec::new();
    let mut union_counts = Vec::new();
    let mut budget_hits = 0;
    let mut lost = 0;
    let mut changed = 0;
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let result = engine.search(query, Strategy::Regional, config)?;
        measurements.add(query, &result, milliseconds(start))?;
        let ranked = result
            .hits
            .iter()
            .map(|hit| hit.source.id.as_str())
            .collect::<Vec<_>>();
        let recall = quality(query, &ranked, &result)
            .recall_at_100
            .expect("positive judgments");
        if let Some(original) = baseline {
            lost += usize::from(recall < original[i].recall);
            changed += usize::from(result.candidates != original[i].candidates);
        } else {
            per_query.push(DevelopmentResult {
                recall,
                candidates: result.candidates,
            });
        }
        rescue_counts.push(result.work.global_rescue_chunks_scored.ok_or_else(|| {
            anyhow::anyhow!("global rescue coverage unavailable for {}", query.id)
        })?);
        union_counts.push(
            result
                .work
                .regional_or_rescue_chunks_scored
                .ok_or_else(|| {
                    anyhow::anyhow!("distinct scoring coverage unavailable for {}", query.id)
                })?,
        );
        budget_hits += usize::from(result.work.global_rescue_budget_hit.unwrap_or(false));
    }
    let corpus = engine.manifest.chunks;
    Ok((
        RescueTrial {
            max_scored: config.rescue_max_scored,
            metrics: measurements.finish(),
            global_rescue: super::ScoredPassages::summarize(rescue_counts, corpus),
            regional_or_rescue: super::ScoredPassages::summarize(union_counts, corpus),
            budget_hits,
            queries_losing_recall: lost,
            queries_with_changed_candidates: changed,
        },
        per_query,
    ))
}

#[derive(Serialize)]
pub struct Diagnosis {
    pub queries: usize,
    regional: Metrics,
    elasticsearch: Metrics,
    elastic_only: Losses,
    regional_only: Losses,
    judgments: Vec<Judgment>,
}

#[derive(Default, Serialize)]
struct Metrics {
    candidate_recall: f64,
    recall_at_100: f64,
    ndcg_at_10: f64,
    p95_ms: f64,
}

#[derive(Default, Serialize)]
struct Losses {
    missing_candidates: usize,
    below_cutoff: usize,
    missing_candidates_macro_recall: f64,
    below_cutoff_macro_recall: f64,
}

impl Losses {
    fn add(&mut self, lost: Ranks, share: f64) {
        if lost.candidate.is_some() {
            self.below_cutoff += 1;
            self.below_cutoff_macro_recall += share;
        } else {
            self.missing_candidates += 1;
            self.missing_candidates_macro_recall += share;
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct Ranks {
    candidate: Option<usize>,
    final_rank: Option<usize>,
}

#[derive(Serialize)]
struct Judgment {
    query: String,
    passage: String,
    regional: Ranks,
    elasticsearch: Ranks,
}

#[derive(Default)]
struct Measurements {
    metrics: Metrics,
    durations: Vec<f64>,
}

impl Measurements {
    fn add(&mut self, query: &Query, result: &SearchResult, elapsed: f64) -> Result<()> {
        let ranked = result
            .hits
            .iter()
            .map(|hit| hit.source.id.as_str())
            .collect::<Vec<_>>();
        let quality = quality(query, &ranked, result);
        let Some(candidate_recall) = quality.candidate_recall else {
            anyhow::bail!("query {} has no positive judgments", query.id);
        };
        self.metrics.candidate_recall += candidate_recall;
        self.metrics.recall_at_100 += quality.recall_at_100.unwrap_or_default();
        self.metrics.ndcg_at_10 += quality.ndcg_at_10.unwrap_or_default();
        self.durations.push(elapsed);
        Ok(())
    }

    fn finish(mut self) -> Metrics {
        self.durations.sort_unstable_by(f64::total_cmp);
        let n = self.durations.len() as f64;
        self.metrics.p95_ms = self.durations[(n * 0.95).ceil() as usize - 1];
        self.metrics.candidate_recall /= n;
        self.metrics.recall_at_100 /= n;
        self.metrics.ndcg_at_10 /= n;
        self.metrics
    }
}

fn frozen_plans<'a>(
    engine: &Engine,
    queries: &[Query],
    report: &'a FrozenComparison,
    client: &elastic::Client,
) -> Result<(&'a Plan, &'a Plan)> {
    validate_queries(engine, queries)?;
    ensure!(
        report.index == engine.manifest && report.elasticsearch == client.info()?,
        "index metadata differs from the frozen comparison"
    );
    let plans = [Strategy::Regional, Strategy::Elasticsearch].map(|strategy| {
        report
            .settings
            .iter()
            .find(|plan| plan.strategy == strategy)
            .ok_or_else(|| anyhow::anyhow!("frozen report lacks {strategy:?}"))
    });
    let [regional, elasticsearch] = plans;
    let (regional, elasticsearch) = (regional?, elasticsearch?);
    ensure!(
        regional.config.limit == 100
            && elasticsearch.config.limit == 100
            && elasticsearch.elastic.is_some(),
        "diagnosis requires the frozen top-100 plans"
    );
    Ok((regional, elasticsearch))
}

pub fn diagnose_recall(
    engine: &mut Engine,
    queries: &[Query],
    report: &FrozenComparison,
    client: &elastic::Client,
) -> Result<Diagnosis> {
    let (regional, elasticsearch) = frozen_plans(engine, queries, report, client)?;
    diagnose(engine, queries, client, regional, elasticsearch)
}

fn diagnose(
    engine: &mut Engine,
    queries: &[Query],
    client: &elastic::Client,
    regional: &Plan,
    elasticsearch: &Plan,
) -> Result<Diagnosis> {
    engine.prepare_regional()?;
    for query in queries {
        regional.search(engine, Some(client), query)?;
        elasticsearch.search(engine, Some(client), query)?;
    }
    let mut regional_metrics = Measurements::default();
    let mut elastic_metrics = Measurements::default();
    let mut elastic_only = Losses::default();
    let mut regional_only = Losses::default();
    let mut judgments = Vec::new();
    for (i, query) in queries.iter().enumerate() {
        let mut measure = |plan: &Plan, measurements: &mut Measurements| -> Result<SearchResult> {
            let start = Instant::now();
            let result = plan.search(engine, Some(client), query)?;
            measurements.add(query, &result, milliseconds(start))?;
            Ok(result)
        };
        let (a, b) = if i.is_multiple_of(2) {
            (
                measure(regional, &mut regional_metrics)?,
                measure(elasticsearch, &mut elastic_metrics)?,
            )
        } else {
            let b = measure(elasticsearch, &mut elastic_metrics)?;
            (measure(regional, &mut regional_metrics)?, b)
        };
        let positives = query
            .judgments
            .iter()
            .filter(|(_, grade)| **grade > 0)
            .collect::<Vec<_>>();
        let share = 1.0 / (queries.len() * positives.len()) as f64;
        for (id, _) in positives {
            let locate = |result: &SearchResult| Ranks {
                candidate: result
                    .candidates
                    .iter()
                    .position(|candidate| candidate == id)
                    .map(|n| n + 1),
                final_rank: result
                    .hits
                    .iter()
                    .position(|hit| &hit.source.id == id)
                    .map(|n| n + 1),
            };
            let regional = locate(&a);
            let elasticsearch = locate(&b);
            match (regional.final_rank, elasticsearch.final_rank) {
                (None, Some(_)) => elastic_only.add(regional, share),
                (Some(_), None) => regional_only.add(elasticsearch, share),
                _ => {}
            }
            judgments.push(Judgment {
                query: query.id.clone(),
                passage: id.clone(),
                regional,
                elasticsearch,
            });
        }
    }
    Ok(Diagnosis {
        queries: queries.len(),
        regional: regional_metrics.finish(),
        elasticsearch: elastic_metrics.finish(),
        elastic_only,
        regional_only,
        judgments,
    })
}

#[derive(Serialize)]
pub struct RerankTuning {
    selection: &'static str,
    baseline: Diagnosis,
    trials: Vec<RerankTrial>,
    pub selected: RegionalRerank,
    validation: Diagnosis,
}

#[derive(Serialize)]
struct RerankTrial {
    weights: RegionalRerank,
    metrics: Metrics,
}

pub fn tune_regional_rerank(
    engine: &mut Engine,
    queries: &[Query],
    report: &FrozenComparison,
    client: &elastic::Client,
) -> Result<RerankTuning> {
    let (original, elasticsearch) = frozen_plans(engine, queries, report, client)?;
    eprintln!("Measuring the original regional reranker and frozen Elasticsearch");
    let baseline = diagnose(engine, queries, client, original, elasticsearch)?;
    let mut plan = original.clone();
    let mut trials = Vec::new();
    for coverage in [0.0, 0.08, 0.20, 0.35] {
        for fusion in [0.0, 0.10, 0.25] {
            let weights = RegionalRerank {
                coverage,
                cosine: 1.0 - coverage - fusion,
                fusion,
            };
            plan.config.regional_rerank = Some(weights);
            let mut measurements = Measurements::default();
            for query in queries {
                let start = Instant::now();
                let result = plan.search(engine, Some(client), query)?;
                measurements.add(query, &result, milliseconds(start))?;
            }
            let metrics = measurements.finish();
            ensure!(
                (metrics.candidate_recall - baseline.regional.candidate_recall).abs() < 1e-12,
                "reranking changed development candidate recall"
            );
            eprintln!(
                "Dev coverage {:.2}, cosine {:.2}, fusion {:.2}: recall {:.4}, nDCG {:.4}, p95 {:.2} ms",
                weights.coverage,
                weights.cosine,
                weights.fusion,
                metrics.recall_at_100,
                metrics.ndcg_at_10,
                metrics.p95_ms
            );
            trials.push(RerankTrial { weights, metrics });
        }
    }
    let mut eligible = trials
        .iter()
        .filter(|trial| {
            trial.metrics.recall_at_100 > baseline.regional.recall_at_100
                && trial.metrics.ndcg_at_10 >= baseline.regional.ndcg_at_10
                && trial.metrics.p95_ms < baseline.elasticsearch.p95_ms
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|a, b| {
        b.metrics
            .recall_at_100
            .total_cmp(&a.metrics.recall_at_100)
            .then(b.metrics.ndcg_at_10.total_cmp(&a.metrics.ndcg_at_10))
    });
    for trial in eligible {
        plan.config.regional_rerank = Some(trial.weights);
        eprintln!("Validating the selected weights with interleaved development queries");
        let validation = diagnose(engine, queries, client, &plan, elasticsearch)?;
        if validation.regional.p95_ms < validation.elasticsearch.p95_ms {
            let selected = trial.weights;
            return Ok(RerankTuning {
                selection: "maximize dev Recall@100 with baseline-or-better nDCG@10 and p95 below frozen Elasticsearch, then prefer higher nDCG@10",
                baseline,
                trials,
                selected,
                validation,
            });
        }
    }
    anyhow::bail!("no regional reranker met the development recall, quality, and latency guards")
}
