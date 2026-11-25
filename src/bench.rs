use crate::{
    Query, elastic, graph,
    index::{Engine, Manifest},
    ranking::{RegionalRerank, SearchConfig, SearchResult, Strategy},
    topology::{MEMBERSHIP_LIMIT, MEMBERSHIP_MARGIN},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    io::Write,
    time::Instant,
};

mod filtered;
mod pruning;
mod recall;
pub use filtered::{FilterReport, build_elastic_buckets, filter_dev};
pub use pruning::{PruningReport, pruning_dev};
pub use recall::{diagnose_recall, tune_bounded_rescue, tune_regional_rerank};

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
    /// `precomputed` uses the vectors in the query file. `raw_text` encodes each query and adds that time to the sample.
    query_track: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    embed_p50_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embed_p95_ms: Option<f64>,
    config: SearchConfig,
    settings: Vec<Plan>,
    settings_frozen: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_settings_source: Option<Manifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    elasticsearch: Option<elastic::Info>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tuning: Option<Tuning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    regional_minus_elasticsearch: Option<PairedEffects>,
    results: Vec<Summary>,
    regional_scoring: ScoredPassages,
    #[serde(skip_serializing_if = "Option::is_none")]
    global_rescue_scoring: Option<ScoredPassages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    regional_or_rescue_scoring: Option<ScoredPassages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    global_rescue_budget_hits: Option<usize>,
    exact_reference: Vec<Reference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    throughput: Option<Vec<ThroughputRow>>,
    /// How the benchmark reached regional search: `in_process` or `http`.
    regional_transport: &'static str,
}

#[derive(Debug, Serialize)]
struct ScoredPassages {
    mean: f64,
    p50: usize,
    p95: usize,
    min: usize,
    max: usize,
    mean_corpus_fraction: f64,
}

impl ScoredPassages {
    fn summarize(mut counts: Vec<usize>, corpus_size: usize) -> Self {
        counts.sort_unstable();
        let mean = counts.iter().map(|&count| count as f64).sum::<f64>() / counts.len() as f64;
        let percentile = |p: f64| counts[(counts.len() as f64 * p).ceil() as usize - 1];
        Self {
            mean,
            p50: percentile(0.50),
            p95: percentile(0.95),
            min: counts[0],
            max: counts[counts.len() - 1],
            mean_corpus_fraction: mean / corpus_size as f64,
        }
    }
}

#[derive(Debug, Serialize)]
struct Summary {
    strategy: Strategy,
    group: String,
    samples: usize,
    candidate_recall: Option<f64>,
    ndcg_at_10: Option<f64>,
    recall_at_100: Option<f64>,
    /// Fraction of positive judgments present in the top five hits.
    evidence_at_5: Option<f64>,
    all_evidence_at_100: Option<f64>,
    lexical_rescue_relevant_candidates_mean: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
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
    evidence_at_5: Option<f64>,
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
    chunks_scored: usize,
    rescue_chunks_scored: Option<usize>,
    rescue_budget_hit: Option<bool>,
    regional_or_rescue_chunks_scored: Option<usize>,
    routing_comparisons: usize,
    rerank_comparisons: usize,
}

#[derive(Serialize)]
struct Trace<'a> {
    repeat: usize,
    query: &'a str,
    group: &'a str,
    elapsed_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    embed_ms: Option<f64>,
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
    /// Concurrent worker counts timed after the one-query passes.
    pub throughput_widths: Vec<usize>,
    pub throughput_encoder: Option<ThroughputEncoder>,
    /// Send regional searches to `loci bench-serve` over loopback HTTP instead of in-process.
    pub remote: Option<crate::remote::Client>,
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
    pub fn with_bounded_rescue(mut self, cap: usize) -> Result<Self> {
        let plan = self
            .settings
            .iter_mut()
            .find(|plan| plan.strategy.uses_rescue())
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional configuration"))?;
        plan.config.rescue_max_scored = Some(cap);
        plan.config.validate()?;
        Ok(self)
    }

    pub fn with_regional_rescore(mut self, rescore: usize) -> Result<Self> {
        let plan = self
            .settings
            .iter_mut()
            .find(|plan| plan.strategy.uses_rescue())
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional configuration"))?;
        plan.config.regional_rescore = Some(rescore);
        plan.config.validate()?;
        Ok(self)
    }

    pub fn with_routing_entries(mut self, entries: usize) -> Result<Self> {
        let plan = self
            .settings
            .iter_mut()
            .find(|plan| plan.strategy.uses_rescue())
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional configuration"))?;
        plan.config.routing_entries = Some(entries);
        plan.config.validate()?;
        Ok(self)
    }

    pub fn with_graph_candidates(mut self, count: usize) -> Result<Self> {
        let plan = self
            .settings
            .iter_mut()
            .find(|plan| plan.strategy.uses_rescue())
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional configuration"))?;
        plan.config.graph_candidates = Some(count);
        plan.config.validate()?;
        Ok(self)
    }

    fn validate_plans(&self) -> Result<()> {
        let strategies = self
            .settings
            .iter()
            .map(|plan| plan.strategy)
            .collect::<Vec<_>>();
        ensure!(
            strategies
                == [
                    Strategy::Bm25,
                    Strategy::Global,
                    Strategy::Regional,
                    Strategy::Elasticsearch
                ]
                || strategies == [Strategy::Regional, Strategy::Elasticsearch]
                || strategies == [Strategy::Hybrid, Strategy::Elasticsearch],
            "comparison report must contain a supported comparison of strategies"
        );
        for plan in &self.settings {
            plan.config.validate()?;
            ensure!(
                plan.config.limit == self.config.limit
                    && plan.config.depth == self.config.depth
                    && plan.elastic.is_some() == (plan.strategy == Strategy::Elasticsearch),
                "invalid comparison strategy configuration"
            );
        }
        Ok(())
    }

    pub fn with_regional_rerank(mut self, weights: RegionalRerank) -> Result<Self> {
        let plan = self
            .settings
            .iter_mut()
            .find(|plan| plan.strategy.uses_rescue())
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional configuration"))?;
        plan.config.regional_rerank = Some(weights);
        plan.config.validate()?;
        Ok(self)
    }

    /// Strategy of the frozen Loci plan: regional or hybrid.
    pub fn loci_strategy(&self) -> Result<Strategy> {
        self.settings
            .iter()
            .map(|plan| plan.strategy)
            .find(|strategy| strategy.uses_rescue())
            .ok_or_else(|| anyhow::anyhow!("frozen report has no regional or hybrid plan"))
    }

    pub fn regional_config(&self, engine: &Engine) -> Result<&SearchConfig> {
        ensure!(
            self.index == engine.manifest,
            "frozen index manifest differs from the current index"
        );
        self.settings
            .iter()
            .find(|plan| plan.strategy.uses_rescue())
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
        remote: Option<&crate::remote::Client>,
        query: &Query,
    ) -> Result<SearchResult> {
        if self.strategy.uses_rescue()
            && let Some(remote) = remote
        {
            return remote.search(engine, query);
        }
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    excluded_dev_query_ids: Vec<String>,
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

type QueryEncoder<'a> = dyn FnMut(&str) -> Result<Vec<f32>> + 'a;

pub fn evaluate(
    engine: &mut Engine,
    queries: &[Query],
    config: &SearchConfig,
    run: &Run,
    embed: Option<&mut QueryEncoder<'_>>,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(engine, queries, config, run, None, embed, traces)
}

#[derive(Clone, Copy)]
pub struct Development<'a> {
    pub queries: &'a [Query],
    pub elasticsearch: &'a elastic::Client,
}

enum Comparison<'a> {
    Tune {
        development: Development<'a>,
        native_source: Option<&'a FrozenComparison>,
        head_to_head: bool,
    },
    Frozen {
        report: &'a FrozenComparison,
        client: &'a elastic::Client,
    },
}

impl Comparison<'_> {
    fn client(&self) -> &elastic::Client {
        match self {
            Self::Tune { development, .. } => development.elasticsearch,
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
    embed: Option<&mut QueryEncoder<'_>>,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(
        engine,
        queries,
        config,
        run,
        Some(Comparison::Tune {
            development,
            native_source: None,
            head_to_head: false,
        }),
        embed,
        traces,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_transfer(
    engine: &mut Engine,
    queries: &[Query],
    run: &Run,
    report: &FrozenComparison,
    development: Development<'_>,
    head_to_head: bool,
    embed: Option<&mut QueryEncoder<'_>>,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(
        engine,
        queries,
        &report.config,
        run,
        Some(Comparison::Tune {
            development,
            native_source: Some(report),
            head_to_head,
        }),
        embed,
        traces,
    )
}

pub fn evaluate_frozen(
    engine: &mut Engine,
    queries: &[Query],
    run: &Run,
    report: &FrozenComparison,
    client: &elastic::Client,
    embed: Option<&mut QueryEncoder<'_>>,
    traces: impl Write,
) -> Result<Report> {
    evaluate_with(
        engine,
        queries,
        &report.config,
        run,
        Some(Comparison::Frozen { report, client }),
        embed,
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
    mut embed: Option<&mut QueryEncoder<'_>>,
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
    let mut native_settings_source = None;
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
        report.validate_plans()?;
        settings.clone_from(&report.settings);
        eprintln!("Using frozen comparison settings");
    }
    if let Some(Comparison::Tune {
        development: dev,
        native_source,
        head_to_head,
    }) = &comparison
    {
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
        if let Some(source) = native_source {
            source.validate_plans()?;
            settings = source
                .settings
                .iter()
                .filter(|plan| {
                    plan.strategy != Strategy::Elasticsearch
                        && (!head_to_head || plan.strategy == Strategy::Regional)
                })
                .cloned()
                .collect();
            native_settings_source = Some(source.index.clone());
            eprintln!("Carrying over Loci settings and tuning only Elasticsearch");
        }
    }
    let started = Instant::now();
    let graph_build_ms = if settings
        .iter()
        .any(|plan| plan.strategy == Strategy::Global)
    {
        eprintln!("Building the in-process HNSW control");
        engine.prepare_global();
        milliseconds(started)
    } else {
        0.0
    };
    let started = Instant::now();
    // A comparison with only a hybrid Loci plan needs just the lean rescue index.
    if settings
        .iter()
        .any(|plan| plan.strategy == Strategy::Regional)
    {
        engine.prepare_regional()?;
    } else {
        engine.prepare_lexical()?;
    }
    let regional_prepare_ms = milliseconds(started);
    let regional_posting_payload_bytes = engine.regional_index().payload_bytes();
    let tuning = if let Some(Comparison::Tune {
        development,
        native_source,
        ..
    }) = &comparison
    {
        let transfer = native_source.is_some();
        let (plans, tuning) = tune(engine, config, *development, queries, transfer)?;
        if !transfer {
            settings.truncate(1);
        }
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
            let (prepared, _) =
                prepared_query(query, &engine.manifest.embedding, embed.as_deref_mut())?;
            plan.search(engine, client, run.remote.as_ref(), &prepared)?;
        }
    }
    let mut samples = BTreeMap::<(usize, Option<&str>), Vec<Sample>>::new();
    let mut embed_times = Vec::new();
    for repeat in 0..run.repeats {
        eprintln!("Measuring repeat {} of {}", repeat + 1, run.repeats);
        for (position, query) in queries.iter().enumerate() {
            for offset in 0..settings.len() {
                let index = (position + repeat + offset) % settings.len();
                let (prepared, embed_ms) =
                    prepared_query(query, &engine.manifest.embedding, embed.as_deref_mut())?;
                let started = Instant::now();
                let result =
                    settings[index].search(engine, client, run.remote.as_ref(), &prepared)?;
                let elapsed_ms = milliseconds(started) + embed_ms.unwrap_or(0.0);
                if let Some(ms) = embed_ms {
                    embed_times.push(ms);
                }
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
                    chunks_scored: result.work.regional_chunks_scored,
                    rescue_chunks_scored: result.work.global_rescue_chunks_scored,
                    rescue_budget_hit: result.work.global_rescue_budget_hit,
                    regional_or_rescue_chunks_scored: result.work.regional_or_rescue_chunks_scored,
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
                        embed_ms,
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
    let graph_width = settings
        .iter()
        .find(|plan| plan.strategy == Strategy::Global)
        .map(|plan| plan.config.graph_width);
    let mut exact_reference = Vec::new();
    let references = graph_width.map_or(0, |_| run.exact_queries.min(queries.len()));
    for position in 0..references {
        let query = &queries[position * queries.len() / references];
        let (prepared, _) =
            prepared_query(query, &engine.manifest.embedding, embed.as_deref_mut())?;
        let vector = engine.query_vector(&prepared)?;
        let depth = config.depth.min(100);
        let started = Instant::now();
        let exact = engine.exact(&vector, depth);
        let exact_ms = milliseconds(started);
        let started = Instant::now();
        let (approximate, approximate_comparisons) = engine.nearest(
            &vector,
            depth,
            graph_width.expect("global reference enabled"),
        );
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
        let regional = first_round(Strategy::Regional).or_else(|| first_round(Strategy::Hybrid))?;
        let elastic = first_round(Strategy::Elasticsearch)?;
        Some(PairedEffects {
            ndcg_at_10: paired(regional, elastic, |quality| quality.ndcg_at_10)?,
            recall_at_100: paired(regional, elastic, |quality| quality.recall_at_100)?,
        })
    });
    let regional_index = settings
        .iter()
        .position(|plan| plan.strategy.uses_rescue())
        .expect("regional or hybrid plan");
    let regional_samples = &samples[&(regional_index, None)];
    let scoring = |counts| ScoredPassages::summarize(counts, engine.manifest.chunks);
    let regional_scoring = scoring(
        regional_samples
            .iter()
            .map(|sample| sample.chunks_scored)
            .collect(),
    );
    let global_rescue_scoring = regional_samples
        .iter()
        .map(|sample| sample.rescue_chunks_scored)
        .collect::<Option<Vec<_>>>()
        .map(scoring);
    let global_rescue_budget_hits = regional_samples
        .iter()
        .map(|sample| sample.rescue_budget_hit)
        .try_fold(0, |total, reached| {
            reached.map(|hit| total + usize::from(hit))
        });
    let regional_or_rescue_scoring = regional_samples
        .iter()
        .map(|sample| sample.regional_or_rescue_chunks_scored)
        .collect::<Option<Vec<_>>>()
        .map(scoring);
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
    let query_track = if embed_times.is_empty() {
        "precomputed"
    } else {
        "raw_text"
    };
    let embed_p50_ms = (!embed_times.is_empty()).then(|| percentile(&mut embed_times, 0.50));
    let embed_p95_ms = (!embed_times.is_empty()).then(|| percentile(&mut embed_times, 0.95));
    let throughput = if run.throughput_widths.is_empty() {
        None
    } else {
        Some(measure_throughput(engine, queries, &settings, client, run)?)
    };
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
        query_track,
        embed_p50_ms,
        embed_p95_ms,
        config: config.clone(),
        settings,
        settings_frozen,
        native_settings_source,
        elasticsearch,
        tuning,
        regional_minus_elasticsearch,
        results,
        regional_scoring,
        global_rescue_scoring,
        global_rescue_budget_hits,
        regional_or_rescue_scoring,
        exact_reference,
        throughput,
        regional_transport: if run.remote.is_some() {
            "http"
        } else {
            "in_process"
        },
    })
}

#[derive(Debug, Serialize)]
struct ThroughputRow {
    strategy: Strategy,
    width: usize,
    queries: usize,
    elapsed_ms: f64,
    queries_per_second: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

/// How each concurrent worker encodes query text. Every worker opens its own
/// single-threaded encoder before the clock starts.
#[derive(Clone, Debug)]
pub enum ThroughputEncoder {
    Pack {
        model: crate::embedding::ModelPack,
        cache: std::path::PathBuf,
    },
    MiniLm,
}

enum WorkerEncoder {
    Pack(Box<crate::embedding::LocalEncoder>),
    MiniLm(crate::palace::MiniLm),
}

impl WorkerEncoder {
    fn open(encoder: &ThroughputEncoder) -> Result<Self> {
        Ok(match encoder {
            ThroughputEncoder::Pack { model, cache } => {
                Self::Pack(Box::new(model.open_threads(cache, 1)?))
            }
            ThroughputEncoder::MiniLm => {
                let (model, vocab) = crate::palace::pinned_minilm();
                Self::MiniLm(crate::palace::MiniLm::open(&model, &vocab, 1)?)
            }
        })
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Pack(encoder) => encoder.embed_query(text),
            Self::MiniLm(encoder) => encoder
                .embed(&[text])?
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing MiniLM query vector")),
        }
    }
}

/// Time every query at each configured width of concurrent workers over one shared,
/// prepared engine. Each sample covers query encoding (when configured) and the search,
/// including the Elasticsearch request.
fn measure_throughput(
    engine: &mut Engine,
    queries: &[Query],
    plans: &[Plan],
    client: Option<&elastic::Client>,
    run: &Run,
) -> Result<Vec<ThroughputRow>> {
    let mut rows = Vec::new();
    let plans = plans
        .iter()
        .filter(|plan| plan.strategy.uses_rescue() || plan.strategy == Strategy::Elasticsearch)
        .collect::<Vec<_>>();
    for plan in &plans {
        if plan.elastic.is_none() {
            engine.prepare(plan.strategy, &plan.config)?;
        }
    }
    let engine: &Engine = engine;
    let total = queries.len() * run.repeats.max(1);
    for &width in &run.throughput_widths {
        ensure!(width > 0, "throughput width must be positive");
        for plan in &plans {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let ready = std::sync::Barrier::new(width + 1);
            let (started, outcomes) = std::thread::scope(|scope| {
                let workers = (0..width)
                    .map(|_| {
                        scope.spawn(|| -> Result<Vec<f64>> {
                            let encoder = run
                                .throughput_encoder
                                .as_ref()
                                .map(WorkerEncoder::open)
                                .transpose();
                            ready.wait();
                            let mut encoder = encoder?;
                            let mut times = Vec::new();
                            loop {
                                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if index >= total {
                                    return Ok(times);
                                }
                                let began = Instant::now();
                                let source = &queries[index % queries.len()];
                                let encoded;
                                let query = match encoder.as_mut() {
                                    Some(encoder) => {
                                        let mut query = source.clone();
                                        query.vector = encoder.embed(&query.text)?;
                                        encoded = query;
                                        &encoded
                                    }
                                    None => source,
                                };
                                match plan.elastic {
                                    Some(options) => client
                                        .expect("configured Elasticsearch")
                                        .search(engine, query, &plan.config, options)?,
                                    None => match run.remote.as_ref() {
                                        Some(remote) if plan.strategy.uses_rescue() => {
                                            remote.search(engine, query)?
                                        }
                                        _ => engine.search_prepared(
                                            query,
                                            plan.strategy,
                                            &plan.config,
                                        )?,
                                    },
                                };
                                times.push(milliseconds(began));
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                ready.wait();
                let started = Instant::now();
                let outcomes = workers
                    .into_iter()
                    .map(|worker| worker.join().expect("throughput worker panicked"))
                    .collect::<Vec<_>>();
                (started, outcomes)
            });
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            let mut times = Vec::with_capacity(total);
            for outcome in outcomes {
                times.extend(outcome?);
            }
            ensure!(times.len() == total, "throughput run lost samples");
            let row = ThroughputRow {
                strategy: plan.strategy,
                width,
                queries: total,
                elapsed_ms: elapsed * 1000.0,
                queries_per_second: total as f64 / elapsed,
                p50_ms: percentile(&mut times, 0.50),
                p95_ms: percentile(&mut times, 0.95),
                p99_ms: percentile(&mut times, 0.99),
            };
            eprintln!(
                "throughput {:?} width {width}: {:.1} q/s, p95 {:.2} ms",
                row.strategy, row.queries_per_second, row.p95_ms
            );
            rows.push(row);
        }
    }
    Ok(rows)
}

fn tune(
    engine: &mut Engine,
    config: &SearchConfig,
    dev: Development<'_>,
    test_queries: &[Query],
    recall_first: bool,
) -> Result<(Vec<Plan>, Tuning)> {
    let normalized = |query: &Query| {
        query
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let mut queries = Vec::new();
    let mut excluded_dev_query_ids = Vec::new();
    if recall_first {
        let test_texts = test_queries.iter().map(normalized).collect::<BTreeSet<_>>();
        for query in dev.queries {
            if test_texts.contains(&normalized(query)) {
                excluded_dev_query_ids.push(query.id.clone());
            } else {
                queries.push(query);
            }
        }
        if !excluded_dev_query_ids.is_empty() {
            eprintln!(
                "Excluded {} dev queries with matching test text",
                excluded_dev_query_ids.len()
            );
        }
    } else {
        queries.extend(dev.queries);
    }
    ensure!(
        !queries.is_empty(),
        "no development queries remain after excluding test overlap"
    );
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
    let grids = if recall_first {
        vec![external]
    } else {
        vec![global, regional, external]
    };
    for grid in grids {
        let mut best: Option<(f64, f64, Plan)> = None;
        for plan in grid {
            let mut ndcg = 0.0;
            let mut recall = 0.0;
            for &query in &queries {
                let result = plan.search(engine, Some(dev.elasticsearch), None, query)?;
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
            ndcg /= queries.len() as f64;
            recall /= queries.len() as f64;
            eprintln!(
                "Dev {:?} trial {} nDCG@10 {ndcg:.4}, Recall@100 {recall:.4}",
                plan.strategy,
                trials.len() + 1
            );
            let (primary, secondary) = if recall_first {
                (recall, ndcg)
            } else {
                (ndcg, recall)
            };
            if best
                .as_ref()
                .is_none_or(|(best_primary, best_secondary, _)| {
                    (primary, secondary) > (*best_primary, *best_secondary)
                })
            {
                best = Some((primary, secondary, plan.clone()));
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
            queries: queries.len(),
            excluded_dev_query_ids,
            selection: if recall_first {
                "highest dev Recall@100, then nDCG@10, then earliest trial"
            } else {
                "highest dev nDCG@10, then Recall@100, then earliest trial"
            },
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
        evidence_at_5: recall(&ranked.iter().take(5).copied().collect::<BTreeSet<_>>()),
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
        evidence_at_5: quality_mean(|q| q.evidence_at_5),
        all_evidence_at_100: quality_mean(|q| q.all_evidence_at_100),
        lexical_rescue_relevant_candidates_mean: mean(|s| {
            s.quality.lexical_rescue_relevant_candidates
        }),
        p50_ms: percentile(0.50),
        p95_ms: percentile(0.95),
        p99_ms: percentile(0.99),
        regions_expanded_mean: mean(|s| s.regions),
        regional_postings_read_mean: mean(|s| s.postings),
        vector_comparisons_mean: (strategy != Strategy::Elasticsearch)
            .then(|| mean(|s| s.comparisons)),
        routing_comparisons_mean: mean(|s| s.routing_comparisons),
        rerank_vector_comparisons_mean: mean(|s| s.rerank_comparisons),
    }
}

fn prepared_query<'a>(
    query: &'a Query,
    expected: &crate::Embedding,
    embed: Option<&mut QueryEncoder<'_>>,
) -> Result<(Cow<'a, Query>, Option<f64>)> {
    let Some(embed) = embed else {
        return Ok((Cow::Borrowed(query), None));
    };
    let started = Instant::now();
    let vector = embed(&query.text)?;
    let embed_ms = milliseconds(started);
    ensure!(
        query.embedding == *expected,
        "query {} embedding manifest differs from the index",
        query.id
    );
    ensure!(
        vector.len() == expected.dimensions,
        "raw-text embedding for {} has {} dimensions, index has {}",
        query.id,
        vector.len(),
        expected.dimensions
    );
    let mut encoded = query.clone();
    encoded.vector = vector;
    Ok((Cow::Owned(encoded), Some(embed_ms)))
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[((values.len() as f64 * fraction).ceil() as usize) - 1]
}

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;
    use crate::index::{Engine, ingest};
    use crate::ranking::SearchConfig;
    use std::collections::BTreeMap;

    #[test]
    fn raw_text_track_encodes_each_sample_and_rejects_a_short_vector() {
        let directory = tempfile::tempdir().unwrap();
        let db = directory.path().join("fixture.sqlite");
        let (embedding, passages, queries) = fixture::generate(64).unwrap();
        ingest(&db, passages, embedding, 16).unwrap();
        let mut engine = Engine::open(&db).unwrap();
        let config = SearchConfig::default();
        let run = Run {
            fixture: true,
            repeats: 1,
            exact_queries: 0,
            index_bytes: 0,
            ingest_ms: None,
            open_ms: 0.0,
            throughput_widths: Vec::new(),
            throughput_encoder: None,
            remote: None,
        };
        let vectors = queries
            .iter()
            .map(|query| (query.text.clone(), query.vector.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut calls = 0;
        let mut traces = Vec::new();
        let report = evaluate(
            &mut engine,
            &queries,
            &config,
            &run,
            Some(&mut |text: &str| {
                calls += 1;
                Ok(vectors[text].clone())
            }),
            &mut traces,
        )
        .unwrap();
        assert_eq!(report.query_track, "raw_text");
        assert!(report.embed_p50_ms.is_some() && report.embed_p95_ms.is_some());
        assert_eq!(calls, queries.len() * 3 * (1 + run.repeats));
        let text = String::from_utf8(traces).unwrap();
        assert!(text.lines().all(|line| line.contains("\"embed_ms\"")));
        let err = evaluate(
            &mut engine,
            &queries,
            &config,
            &run,
            Some(&mut |_| Ok(vec![1.0])),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("dimensions"), "{err}");
    }
}
