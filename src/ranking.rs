use crate::{
    Query, cosine,
    elastic::Lists,
    graph::Neighbor,
    index::{Engine, Source, frequencies},
    profile::{Phases, Timer},
    regional, terms,
    topology::Route,
};
use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Bm25,
    Exact,
    Global,
    Regional,
    #[value(skip)]
    Elasticsearch,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SearchConfig {
    pub limit: usize,
    pub depth: usize,
    pub graph_width: usize,
    pub max_regions: usize,
    pub max_chunks: usize,
    pub max_postings: usize,
    pub rescue: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rescue_max_scored: Option<usize>,
    /// Scan regions with int8 vectors and re-score this many best candidates exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regional_rescore: Option<usize>,
    /// Best-scoring leaf regions seeded into routing (default 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_entries: Option<usize>,
    /// Add this many nearest passages from the persisted global HNSW graph to
    /// the regional dense list (searched with `graph_width`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_candidates: Option<usize>,
    pub routing_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regional_rerank: Option<RegionalRerank>,
    /// Reranker weights for Elasticsearch candidate lists. Unset keeps the default,
    /// so a control can give Elasticsearch the same reranker as regional search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elastic_rerank: Option<RegionalRerank>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegionalRerank {
    pub coverage: f64,
    pub cosine: f64,
    pub fusion: f64,
}

impl Default for RegionalRerank {
    fn default() -> Self {
        Self {
            coverage: 0.65,
            cosine: 0.25,
            fusion: 0.10,
        }
    }
}

impl RegionalRerank {
    fn validate(self) -> Result<()> {
        let weights = [self.coverage, self.cosine, self.fusion];
        ensure!(
            weights
                .into_iter()
                .all(|weight| (0.0..=1.0).contains(&weight))
                && (weights.into_iter().sum::<f64>() - 1.0).abs() <= 1e-9,
            "regional reranker weights must be in [0, 1] and sum to 1"
        );
        Ok(())
    }

    pub(crate) fn score(self, coverage: f64, cosine: f64, fusion: f64) -> f64 {
        self.coverage * coverage + self.cosine * cosine + self.fusion * fusion * 30.0
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: 100,
            depth: 200,
            graph_width: 256,
            max_regions: 12,
            max_chunks: 6_000,
            max_postings: 60_000,
            rescue: 40,
            rescue_max_scored: None,
            regional_rescore: None,
            routing_entries: None,
            graph_candidates: None,
            routing_ms: 0,
            regional_rerank: None,
            elastic_rerank: None,
        }
    }
}

impl SearchConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.limit > 0 && self.limit <= self.depth,
            "require 0 < limit <= depth"
        );
        ensure!(self.depth <= i64::MAX as usize, "depth too large");
        ensure!(
            self.graph_width > 0
                && self.max_regions > 0
                && self.max_chunks > 0
                && self.max_postings > 0,
            "work budgets must be positive"
        );
        ensure!(
            self.rescue <= self.depth,
            "rescue depth exceeds candidate depth"
        );
        ensure!(
            self.rescue_max_scored
                .is_none_or(|budget| budget >= self.rescue && budget > 0),
            "bounded rescue must allow at least the configured number of rescue hits"
        );
        ensure!(
            self.regional_rescore
                .is_none_or(|rescore| rescore >= self.depth),
            "regional re-scoring must keep at least the candidate depth"
        );
        ensure!(
            self.routing_entries.is_none_or(|entries| entries > 0),
            "routing needs at least one entry region"
        );
        ensure!(
            self.graph_candidates.is_none_or(|count| count > 0),
            "graph candidates must be positive"
        );
        for weights in [self.regional_rerank, self.elastic_rerank]
            .into_iter()
            .flatten()
        {
            weights.validate()?;
        }
        Ok(())
    }
}

#[derive(Default, Debug, Serialize)]
pub struct Work {
    pub regions_expanded: usize,
    pub regional_postings_read: usize,
    pub regional_chunks_scored: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_rescue_chunks_scored: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_rescue_budget_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_or_rescue_chunks_scored: Option<usize>,
    pub vector_comparisons: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescored_vector_comparisons: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_vector_comparisons: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elasticsearch_took_ms: Option<u64>,
    pub routing_comparisons: usize,
    pub global_lexical_returned: usize,
    pub rerank_vector_comparisons: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub strategy: Strategy,
    pub hits: Vec<Hit>,
    pub candidates: Vec<String>,
    pub lexical_rescue_only_candidates: Vec<String>,
    pub expanded: Vec<Route>,
    pub stop_reason: &'static str,
    pub work: Work,
}

#[derive(Debug, Serialize)]
pub struct Hit {
    #[serde(flatten)]
    pub source: Source,
    pub score: f64,
    pub lexical_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub global_lexical_rank: Option<usize>,
    /// Rank among the global HNSW graph's nearest passages, when that source ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_rank: Option<usize>,
    pub routes: Vec<Route>,
}

#[derive(Clone, Copy, Default)]
struct Candidate {
    lexical: Option<usize>,
    vector: Option<usize>,
    rescue: Option<usize>,
}

impl Candidate {
    fn fusion(&self) -> f64 {
        let reciprocal = |rank: usize| 1.0 / (60 + rank) as f64;
        let lexical = self.lexical.into_iter().chain(self.rescue).min();
        lexical.into_iter().chain(self.vector).map(reciprocal).sum()
    }
}

pub(crate) fn search(
    engine: &Engine,
    query: &Query,
    vector: &[f32],
    strategy: Strategy,
    config: &SearchConfig,
) -> Result<SearchResult> {
    search_with(engine, query, vector, strategy, config, None, None, None)
}

/// Regional search restricted to the passages `allow` admits.
pub(crate) fn search_filtered(
    engine: &Engine,
    query: &Query,
    vector: &[f32],
    config: &SearchConfig,
    allow: &[bool],
) -> Result<SearchResult> {
    ensure!(
        allow.len() == engine.ids.len(),
        "the filter must cover every passage"
    );
    search_with(
        engine,
        query,
        vector,
        Strategy::Regional,
        config,
        None,
        Some(allow),
        None,
    )
}

pub(crate) fn profile_regional(
    engine: &Engine,
    query: &Query,
    vector: &[f32],
    config: &SearchConfig,
    phases: &mut Phases,
) -> Result<SearchResult> {
    search_with(
        engine,
        query,
        vector,
        Strategy::Regional,
        config,
        None,
        None,
        Some(phases),
    )
}

pub(crate) fn search_external(
    engine: &Engine,
    query: &Query,
    vector: &[f32],
    config: &SearchConfig,
    lists: Lists,
) -> Result<SearchResult> {
    search_with(
        engine,
        query,
        vector,
        Strategy::Elasticsearch,
        config,
        Some(lists),
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn search_with(
    engine: &Engine,
    query: &Query,
    vector: &[f32],
    strategy: Strategy,
    config: &SearchConfig,
    external: Option<Lists>,
    allow: Option<&[bool]>,
    mut profile: Option<&mut Phases>,
) -> Result<SearchResult> {
    ensure!(
        allow.is_none() || strategy == Strategy::Regional,
        "only regional search accepts a passage filter"
    );
    let query_terms = terms(&query.text).collect::<BTreeSet<_>>();
    let mut work = Work::default();
    let mut expanded = Vec::new();
    let mut routes = regional::Routes::default();
    let mut stop_reason = "complete";
    let mut rescue_scores = None;
    let mut graph_ranks = HashMap::new();
    let timer = Timer::start(&profile, "rescue");
    let global_lexical = match strategy {
        Strategy::Bm25 | Strategy::Global => engine.lexical(&query_terms, config.depth)?,
        Strategy::Regional => {
            let rescue = engine.regional_index().rescue_filtered(
                engine,
                &query_terms,
                config.rescue,
                config.rescue_max_scored,
                allow,
            )?;
            work.global_rescue_chunks_scored = rescue.scores.as_ref().map(|scores| scores.count);
            work.global_rescue_budget_hit = config.rescue_max_scored.map(|_| rescue.budget_hit);
            rescue_scores = rescue.scores;
            rescue.ranked
        }
        Strategy::Exact => Vec::new(),
        Strategy::Elasticsearch => external
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Elasticsearch lists are required"))?
            .lexical
            .clone(),
    };
    timer.finish(&mut profile);
    work.global_lexical_returned = global_lexical.len();
    let hybrid = matches!(
        strategy,
        Strategy::Regional | Strategy::Global | Strategy::Elasticsearch
    );
    let weights = if hybrid {
        coverage_weights(engine, &query_terms)?
    } else {
        BTreeMap::new()
    };
    let mut candidates = BTreeMap::<usize, Candidate>::new();
    for (rank, &id) in global_lexical.iter().enumerate() {
        let candidate = candidates.entry(id).or_default();
        if strategy == Strategy::Regional {
            candidate.rescue = Some(rank + 1);
        } else {
            candidate.lexical = Some(rank + 1);
        }
    }
    let nearest = match strategy {
        Strategy::Bm25 => Vec::new(),
        Strategy::Exact => {
            work.vector_comparisons = engine.ids.len();
            engine.exact(vector, config.depth)
        }
        Strategy::Global => {
            let (nearest, comparisons) = engine.nearest(vector, config.depth, config.graph_width);
            work.vector_comparisons = comparisons;
            nearest
        }
        Strategy::Elasticsearch => {
            let lists = external.expect("external rankings validated");
            work.elasticsearch_took_ms = Some(lists.took_ms);
            lists
                .vector
                .into_iter()
                .map(|id| Neighbor {
                    id,
                    similarity: 0.0,
                })
                .collect()
        }
        Strategy::Regional => {
            let selection = regional::select(
                engine,
                vector,
                config,
                &weights,
                rescue_scores.as_ref(),
                allow,
                &mut work,
                &mut profile,
            );
            stop_reason = selection.stop_reason;
            expanded = selection.expanded;
            routes = selection.routes;
            for (rank, id) in selection.lexical.into_iter().enumerate() {
                candidates.entry(id).or_default().lexical = Some(rank + 1);
            }
            let mut nearest = selection.nearest;
            if let Some(count) = config.graph_candidates {
                let timer = Timer::start(&profile, "graph");
                let (graph, comparisons) = engine.nearest_dense(vector, count, config.graph_width);
                work.graph_vector_comparisons = Some(comparisons);
                let regional = nearest.iter().map(|hit| hit.id).collect::<HashSet<_>>();
                // A filtered search keeps only admitted graph neighbours (post-filtering).
                let graph = graph
                    .into_iter()
                    .filter(|hit| allow.is_none_or(|allow| allow[hit.id]));
                for (rank, hit) in graph.enumerate() {
                    graph_ranks.insert(hit.id, rank + 1);
                    if !regional.contains(&hit.id) {
                        nearest.push(hit);
                    }
                }
                nearest.sort_unstable_by(|a, b| b.cmp(a));
                nearest.truncate(config.depth);
                timer.finish(&mut profile);
            }
            nearest
        }
    };
    for (rank, hit) in nearest.iter().enumerate() {
        candidates.entry(hit.id).or_default().vector = Some(rank + 1);
    }
    let mut fused = candidates.into_iter().collect::<Vec<_>>();
    fused.sort_unstable_by(|(a_id, a), (b_id, b)| {
        b.fusion().total_cmp(&a.fusion()).then(a_id.cmp(b_id))
    });
    let overflow = fused.split_off(fused.len().min(config.depth));
    // Reserve reranker slots for rescue hits even when both local lists agree.
    if strategy == Strategy::Regional {
        let rescue = global_lexical.iter().copied().collect::<BTreeSet<_>>();
        for candidate in overflow.into_iter().filter(|(id, _)| rescue.contains(id)) {
            let replace = fused
                .iter()
                .rposition(|(id, _)| !rescue.contains(id))
                .expect("rescue fits within depth");
            fused.remove(replace);
            fused.push(candidate);
        }
    }
    let candidate_ids = fused
        .iter()
        .map(|(id, _)| engine.ids[*id].clone())
        .collect();
    let lexical_rescue_only_candidates = fused
        .iter()
        .filter(|(_, signal)| {
            signal.rescue.is_some() && signal.lexical.is_none() && signal.vector.is_none()
        })
        .map(|(id, _)| engine.ids[*id].clone())
        .collect();
    if engine.manifest.retrieval_profile == Some(crate::index::RetrievalProfile::Palace)
        && matches!(
            strategy,
            Strategy::Exact | Strategy::Global | Strategy::Regional
        )
    {
        let scored_ids: Vec<usize> = if strategy == Strategy::Exact {
            work.vector_comparisons = engine.ids.len();
            (0..engine.ids.len()).collect()
        } else {
            fused.iter().map(|(id, _)| *id).collect()
        };
        let signals: HashMap<usize, Candidate> = fused.iter().copied().collect();
        let candidate_ids = scored_ids
            .iter()
            .map(|id| engine.ids[*id].clone())
            .collect();
        let timer = Timer::start(&profile, "rerank");
        let explain = HitExplanation {
            strategy,
            signals: &signals,
            graph_ranks: &graph_ranks,
            routes: &routes,
            expanded: &expanded,
        };
        let hits = palace_hits(
            engine,
            query,
            vector,
            config,
            &scored_ids,
            &explain,
            &mut work,
        )?;
        timer.finish(&mut profile);
        return Ok(SearchResult {
            strategy,
            hits,
            candidates: candidate_ids,
            lexical_rescue_only_candidates,
            expanded,
            stop_reason,
            work,
        });
    }
    let total_weight = weights.values().sum::<f64>().max(f64::EPSILON);
    let rerank = match strategy {
        Strategy::Regional => config.regional_rerank.unwrap_or_default(),
        Strategy::Elasticsearch => config.elastic_rerank.unwrap_or_default(),
        _ => RegionalRerank::default(),
    };
    let timer = Timer::start(&profile, "rerank");
    let mut reranked = Vec::with_capacity(fused.len());
    for (id, candidate) in fused {
        let (score, source) = if hybrid {
            let (matched_weight, source) = if strategy == Strategy::Regional {
                let matched_weight = if rerank.coverage > 0.0 {
                    engine.regional_index().coverage(id, &weights)
                } else {
                    0.0
                };
                (matched_weight, None)
            } else {
                let source = engine.source(id)?;
                let document = frequencies(&source.text);
                let matched_weight = weights
                    .iter()
                    .filter(|(term, _)| document.contains_key(*term))
                    .map(|(_, weight)| weight)
                    .sum::<f64>();
                (matched_weight, Some(source))
            };
            let coverage = matched_weight / total_weight;
            let similarity = f64::from(cosine(vector, &engine.vectors[id]));
            work.rerank_vector_comparisons += 1;
            (
                rerank.score(coverage, similarity, candidate.fusion()),
                source,
            )
        } else {
            (candidate.fusion(), None)
        };
        reranked.push((id, candidate, score, source));
    }
    reranked
        .sort_unstable_by(|(a_id, _, a, _), (b_id, _, b, _)| b.total_cmp(a).then(a_id.cmp(b_id)));
    let mut hits = Vec::new();
    for (id, candidate, score, source) in reranked.into_iter().take(config.limit) {
        let global_lexical_rank = if strategy == Strategy::Regional {
            candidate.rescue
        } else {
            candidate.lexical
        };
        hits.push(Hit {
            source: match source {
                Some(source) => source,
                None => engine.source(id)?,
            },
            score,
            lexical_rank: candidate.lexical,
            vector_rank: candidate.vector,
            global_lexical_rank,
            graph_rank: graph_ranks.get(&id).copied(),
            routes: routes.for_hit(engine, id, &expanded),
        });
    }
    timer.finish(&mut profile);
    Ok(SearchResult {
        strategy,
        hits,
        candidates: candidate_ids,
        lexical_rescue_only_candidates,
        expanded,
        stop_reason,
        work,
    })
}

/// IDF weight of each query term, used by the reranker's term coverage.
pub(crate) fn coverage_weights(
    engine: &Engine,
    query_terms: &BTreeSet<String>,
) -> Result<BTreeMap<String, f64>> {
    let mut weights = BTreeMap::new();
    let conn = engine.conn.get()?;
    let mut df = conn.prepare_cached("SELECT df FROM term_stats WHERE term=?")?;
    for term in query_terms {
        let count = df
            .query_row([term], |row| row.get::<_, usize>(0))
            .optional()?
            .unwrap_or(0);
        let idf =
            (1.0 + (engine.ids.len() as f64 - count as f64 + 0.5) / (count as f64 + 0.5)).ln();
        weights.insert(term.clone(), idf);
    }
    Ok(weights)
}

/// What a Palace-ranked hit reports about how retrieval reached it.
struct HitExplanation<'a> {
    strategy: Strategy,
    signals: &'a HashMap<usize, Candidate>,
    graph_ranks: &'a HashMap<usize, usize>,
    routes: &'a regional::Routes,
    expanded: &'a [Route],
}

fn palace_hits(
    engine: &crate::index::Engine,
    query: &Query,
    vector: &[f32],
    config: &SearchConfig,
    scored_ids: &[usize],
    explain: &HitExplanation<'_>,
    work: &mut Work,
) -> Result<Vec<Hit>> {
    let stats = engine
        .palace
        .as_ref()
        .context("palace profile is missing corpus document frequencies")?;
    let owned: Vec<(usize, crate::index::Source)> = scored_ids
        .iter()
        .map(|id| Ok((*id, engine.source(*id)?)))
        .collect::<Result<_>>()?;
    let views: Vec<crate::palace::Passage<'_>> = owned
        .iter()
        .map(|(id, source)| crate::palace::Passage {
            id: &source.id,
            source: &source.document,
            text: &source.text,
            room: metadata_str(&source.metadata, "room"),
            created_at: metadata_str(&source.metadata, "created_at"),
            filed_at: metadata_str(&source.metadata, "filed_at"),
            cosine: f64::from(cosine(vector, &engine.vectors[*id])),
            doc_len: stats.doc_len[*id] as usize,
        })
        .collect();
    work.rerank_vector_comparisons += views.len();
    let corpus = crate::palace::Corpus {
        documents: stats.documents,
        average_length: stats.average_length,
        document_frequency: stats.document_frequency.clone(),
    };
    let ranked = crate::palace::score(
        &query.text,
        &views,
        &corpus,
        jiff::Timestamp::now(),
        &crate::palace::Weights::default(),
        &HashSet::new(),
    );
    drop(views);
    let ids: Vec<usize> = owned.iter().map(|(id, _)| *id).collect();
    let mut sources: Vec<Option<crate::index::Source>> =
        owned.into_iter().map(|(_, source)| Some(source)).collect();
    let mut hits = Vec::new();
    for row in ranked.into_iter().take(config.limit) {
        let id = ids[row.index];
        let source = sources[row.index]
            .take()
            .context("palace hit taken twice")?;
        let signal = explain.signals.get(&id).copied().unwrap_or_default();
        let global_lexical_rank = if explain.strategy == Strategy::Regional {
            signal.rescue
        } else {
            signal.lexical
        };
        hits.push(Hit {
            source,
            score: row.combined,
            lexical_rank: signal.lexical,
            vector_rank: signal.vector,
            global_lexical_rank,
            graph_rank: explain.graph_ranks.get(&id).copied(),
            routes: explain.routes.for_hit(engine, id, explain.expanded),
        });
    }
    Ok(hits)
}

fn metadata_str<'a>(metadata: &'a BTreeMap<String, serde_json::Value>, key: &str) -> &'a str {
    metadata
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or("")
}
