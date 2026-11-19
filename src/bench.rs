use crate::{
    Query, elastic, graph,
    index::{Engine, Manifest},
    ranking::{SearchConfig, SearchResult, Strategy},
    topology::{MEMBERSHIP_LIMIT, MEMBERSHIP_MARGIN},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
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
    regional_prepare_ms: f64,
    regional_posting_payload_bytes: usize,
    graph_degree: usize,
    graph_construction_width: usize,
    membership_limit: usize,
    membership_margin: f32,
    queries: usize,
    repeats: usize,
    config: SearchConfig,
    settings: Vec<Plan>,
    settings_frozen: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    elasticsearch: Option<elastic::Info>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tuning: Option<Tuning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    regional_minus_elasticsearch: Option<PairedEffects>,
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
    vector_comparisons_mean: Option<f64>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Plan {
    strategy: Strategy,
    config: SearchConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    elastic: Option<elastic::Options>,
}

#[derive(Deserialize)]
pub struct FrozenComparison {
    index: Manifest,
    queries: usize,
    repeats: usize,
    config: SearchConfig,
    settings: Vec<Plan>,
    elasticsearch: elastic::Info,
}

impl FrozenComparison {
    pub fn regional_config(&self, engine: &Engine) -> Result<&SearchConfig> {
        ensure!(
            self.index == engine.manifest,
            "frozen index manifest differs from the current index"
        );
        self.settings
            .iter()
            .find(|plan| plan.strategy == Strategy::Regional)
            .map(|plan| &plan.config)
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional configuration"))
    }
}

impl Plan {
    fn native(strategy: Strategy, config: &SearchConfig) -> Self {
        Self {
            strategy,
            config: config.clone(),
            elastic: None,
        }
    }

    fn search(
        &self,
        engine: &mut Engine,
        client: Option<&elastic::Client>,
        query: &Query,
    ) -> Result<SearchResult> {
        if let Some(options) = self.elastic {
            return client.expect("configured Elasticsearch").search(
                engine,
                query,
                &self.config,
                options,
            );
        }
        engine.search(query, self.strategy, &self.config)
    }
}

#[derive(Debug, Serialize)]
struct Tuning {
    queries: usize,
    selection: &'static str,
    trials: Vec<Trial>,
}

#[derive(Debug, Serialize)]
struct Trial {
    plan: Plan,
    ndcg_at_10: f64,
    recall_at_100: f64,
}

#[derive(Debug, Serialize)]
struct PairedEffects {
    ndcg_at_10: Interval,
    recall_at_100: Interval,
}

#[derive(Debug, Serialize)]
struct Interval {
    mean: f64,
    lower_95: f64,
    upper_95: f64,
}

pub fn evaluate(
    engine: &mut Engine,
    queries: &[Query],
    config: &SearchConfig,
    run: &Run,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(engine, queries, config, run, None, traces)
}

#[derive(Clone, Copy)]
pub struct Development<'a> {
    pub queries: &'a [Query],
    pub elasticsearch: &'a elastic::Client,
}

enum Comparison<'a> {
    Tune(Development<'a>),
    Frozen {
        report: &'a FrozenComparison,
        client: &'a elastic::Client,
    },
}

impl Comparison<'_> {
    fn client(&self) -> &elastic::Client {
        match self {
            Self::Tune(dev) => dev.elasticsearch,
            Self::Frozen { client, .. } => client,
        }
    }
}

pub fn evaluate_comparison(
    engine: &mut Engine,
    queries: &[Query],
    config: &SearchConfig,
    run: &Run,
    development: Development<'_>,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(
        engine,
        queries,
        config,
        run,
        Some(Comparison::Tune(development)),
        traces,
    )
}

pub fn evaluate_frozen(
    engine: &mut Engine,
    queries: &[Query],
    run: &Run,
    report: &FrozenComparison,
    client: &elastic::Client,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(
        engine,
        queries,
        &report.config,
        run,
        Some(Comparison::Frozen { report, client }),
        traces,
    )
}

fn validate_queries(engine: &Engine, queries: &[Query]) -> Result<()> {
    ensure!(!queries.is_empty(), "queries are required");
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
    Ok(())
}

fn evaluate_with(
    engine: &mut Engine,
    queries: &[Query],
    config: &SearchConfig,
    run: &Run,
    comparison: Option<Comparison<'_>>,
    mut traces: impl Write,
) -> Result<Report> {
    config.validate()?;
    ensure!(
        config.limit >= 100,
        "benchmark requires at least 100 result slots"
    );
    ensure!(run.repeats > 0, "repeats are required");
    let mut settings = STRATEGIES
        .map(|strategy| Plan::native(strategy, config))
        .to_vec();
    let client = comparison.as_ref().map(Comparison::client);
    let elasticsearch = client.map(elastic::Client::info).transpose()?;
    if let Some(info) = &elasticsearch {
        ensure!(
            info.documents == engine.manifest.chunks && info.active_primary_shards == 1,
            "Elasticsearch must have one active primary shard and the same document count as Loci"
        );
    }
    let settings_frozen = matches!(&comparison, Some(Comparison::Frozen { .. }));
    if let Some(Comparison::Frozen { report, .. }) = &comparison {
        ensure!(
            report.index == engine.manifest
                && report.queries == queries.len()
                && report.repeats == run.repeats,
            "frozen comparison requires the original Loci manifest, query count, and repeat count"
        );
        ensure!(
            elasticsearch.as_ref() == Some(&report.elasticsearch),
            "Elasticsearch version, index or primary-store statistics differ from the frozen report"
        );
        let expected = STRATEGIES.into_iter().chain([Strategy::Elasticsearch]);
        ensure!(
            report
                .settings
                .iter()
                .map(|plan| plan.strategy)
                .eq(expected),
            "frozen report must contain the four original strategies"
        );
        for plan in &report.settings {
            plan.config.validate()?;
            ensure!(
                plan.config.limit == config.limit
                    && plan.config.depth == config.depth
                    && plan.elastic.is_some() == (plan.strategy == Strategy::Elasticsearch),
                "invalid frozen strategy configuration"
            );
        }
        settings.clone_from(&report.settings);
        eprintln!("Using the original strategy and Elasticsearch settings");
    }
    if let Some(Comparison::Tune(dev)) = &comparison {
        ensure!(
            config.depth <= 8_664,
            "Elasticsearch tuning requires depth <= 8664"
        );
        validate_queries(engine, dev.queries)?;
        let test_ids = queries
            .iter()
            .map(|query| &query.id)
            .collect::<BTreeSet<_>>();
        ensure!(
            !dev.queries.iter().any(|query| test_ids.contains(&query.id)),
            "development and test query IDs overlap"
        );
    }
    let started = Instant::now();
    eprintln!("Building the in-process HNSW control");
    engine.prepare_global();
    let graph_build_ms = milliseconds(started);
    let started = Instant::now();
    engine.prepare_regional()?;
    let regional_prepare_ms = milliseconds(started);
    let regional_posting_payload_bytes = engine.regional_index().payload_bytes();
    let tuning = if let Some(Comparison::Tune(dev)) = &comparison {
        let (plans, tuning) = tune(engine, config, *dev)?;
        settings.truncate(1);
        settings.extend(plans);
        Some(tuning)
    } else {
        None
    };
    validate_queries(engine, queries)?;
    eprintln!(
        "Warming {} strategies over {} evaluation queries",
        settings.len(),
        queries.len()
    );
    // One untimed pass per engine before collecting warm, single-query samples.
    for query in queries {
        for plan in &settings {
            plan.search(engine, client, query)?;
        }
    }
    let mut samples = BTreeMap::<(usize, Option<&str>), Vec<Sample>>::new();
    for repeat in 0..run.repeats {
        eprintln!("Measuring repeat {} of {}", repeat + 1, run.repeats);
        for (position, query) in queries.iter().enumerate() {
            for offset in 0..settings.len() {
                let index = (position + repeat + offset) % settings.len();
                let started = Instant::now();
                let result = settings[index].search(engine, client, query)?;
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
    let global = &settings
        .iter()
        .find(|plan| plan.strategy == Strategy::Global)
        .expect("global plan")
        .config;
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
            engine.nearest(&vector, depth, global.graph_width);
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
    let regional_minus_elasticsearch = elasticsearch.as_ref().and_then(|_| {
        let first_round = |strategy| {
            let index = settings.iter().position(|plan| plan.strategy == strategy)?;
            Some(&samples.get(&(index, None))?[..queries.len()])
        };
        let regional = first_round(Strategy::Regional)?;
        let elastic = first_round(Strategy::Elasticsearch)?;
        Some(PairedEffects {
            ndcg_at_10: paired(regional, elastic, |quality| quality.ndcg_at_10)?,
            recall_at_100: paired(regional, elastic, |quality| quality.recall_at_100)?,
        })
    });
    let results = samples
        .into_iter()
        .map(|((index, group), samples)| {
            summarize(
                settings[index].strategy,
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
        regional_prepare_ms,
        regional_posting_payload_bytes,
        graph_degree: graph::DEGREE,
        graph_construction_width: graph::BUILD_WIDTH,
        membership_limit: MEMBERSHIP_LIMIT,
        membership_margin: MEMBERSHIP_MARGIN,
        queries: queries.len(),
        repeats: run.repeats,
        config: config.clone(),
        settings,
        settings_frozen,
        elasticsearch,
        tuning,
        regional_minus_elasticsearch,
        results,
        exact_reference,
    })
}

fn tune(
    engine: &mut Engine,
    config: &SearchConfig,
    dev: Development<'_>,
) -> Result<(Vec<Plan>, Tuning)> {
    let global = [0, 56, 184, 312, 568, 824]
        .into_iter()
        .map(|extra| {
            Plan::native(
                Strategy::Global,
                &SearchConfig {
                    graph_width: config.depth + extra,
                    ..config.clone()
                },
            )
        })
        .collect::<Vec<_>>();
    let regional = [
        (2, 768),
        (4, 1536),
        (8, 3072),
        (12, 6000),
        (24, 12000),
        (36, 24000),
    ]
    .into_iter()
    .map(|(max_regions, max_chunks)| {
        Plan::native(
            Strategy::Regional,
            &SearchConfig {
                max_regions,
                max_chunks,
                max_postings: max_chunks * 10,
                ..config.clone()
            },
        )
    })
    .collect::<Vec<_>>();
    let external = [elastic::Analyzer::Standard, elastic::Analyzer::English]
        .into_iter()
        .flat_map(|analyzer| {
            [56, 568, 1336].into_iter().map(move |extra| Plan {
                strategy: Strategy::Elasticsearch,
                config: config.clone(),
                elastic: Some(elastic::Options {
                    analyzer,
                    num_candidates: config.depth + extra,
                }),
            })
        })
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    let mut trials = Vec::new();
    for grid in [global, regional, external] {
        let mut best: Option<(f64, f64, Plan)> = None;
        for plan in grid {
            let mut ndcg = 0.0;
            let mut recall = 0.0;
            for query in dev.queries {
                let result = plan.search(engine, Some(dev.elasticsearch), query)?;
                let ranked = result
                    .hits
                    .iter()
                    .map(|hit| hit.source.id.as_str())
                    .collect::<Vec<_>>();
                let quality = quality(query, &ranked, &result);
                ndcg += quality
                    .ndcg_at_10
                    .ok_or_else(|| anyhow::anyhow!("no positive dev judgment for {}", query.id))?;
                recall += quality.recall_at_100.unwrap_or_default();
            }
            ndcg /= dev.queries.len() as f64;
            recall /= dev.queries.len() as f64;
            eprintln!(
                "Dev {:?} trial {} nDCG@10 {ndcg:.4}, Recall@100 {recall:.4}",
                plan.strategy,
                trials.len() + 1
            );
            if best.as_ref().is_none_or(|(high_ndcg, high_recall, _)| {
                (ndcg, recall) > (*high_ndcg, *high_recall)
            }) {
                best = Some((ndcg, recall, plan.clone()));
            }
            trials.push(Trial {
                plan,
                ndcg_at_10: ndcg,
                recall_at_100: recall,
            });
        }
        selected.push(best.expect("nonempty tuning grid").2);
    }
    Ok((
        selected,
        Tuning {
            queries: dev.queries.len(),
            selection: "highest dev nDCG@10, then Recall@100, then earliest trial",
            trials,
        },
    ))
}

fn paired(
    regional: &[Sample],
    elastic: &[Sample],
    field: fn(Quality) -> Option<f64>,
) -> Option<Interval> {
    let differences = regional
        .iter()
        .zip(elastic)
        .filter_map(|(a, b)| Some(field(a.quality)? - field(b.quality)?))
        .collect::<Vec<_>>();
    if differences.is_empty() {
        return None;
    }
    let n = differences.len();
    let mean = differences.iter().sum::<f64>() / n as f64;
    let mut seed = 0x4c6f_6369_5345_4152_u64;
    let mut distribution = (0..2_000)
        .map(|_| {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    differences[(seed % n as u64) as usize]
                })
                .sum::<f64>()
                / n as f64
        })
        .collect::<Vec<_>>();
    distribution.sort_unstable_by(f64::total_cmp);
    Some(Interval {
        mean,
        lower_95: distribution[50],
        upper_95: distribution[1949],
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
        vector_comparisons_mean: (strategy != Strategy::Elasticsearch)
            .then(|| mean(|s| s.comparisons)),
        routing_comparisons_mean: mean(|s| s.routing_comparisons),
        rerank_vector_comparisons_mean: mean(|s| s.rerank_comparisons),
    }
}

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}
