use crate::{
    Query, cosine,
    graph::Neighbor,
    index::{Engine, Source, frequencies},
    terms,
    topology::Route,
};
use anyhow::{Result, ensure};
use clap::ValueEnum;
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Bm25,
    Exact,
    Global,
    Regional,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchConfig {
    pub limit: usize,
    pub depth: usize,
    pub graph_width: usize,
    pub max_regions: usize,
    pub max_chunks: usize,
    pub max_postings: usize,
    pub rescue: usize,
    pub routing_ms: u64,
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
            routing_ms: 0,
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
        Ok(())
    }
}

#[derive(Default, Debug, Serialize)]
pub struct Work {
    pub regions_expanded: usize,
    pub regional_postings_read: usize,
    pub regional_chunks_scored: usize,
    pub vector_comparisons: usize,
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
    pub routes: Vec<Route>,
}

#[derive(Default)]
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
    let query_terms = terms(&query.text).collect::<BTreeSet<_>>();
    let mut work = Work::default();
    let mut expanded = Vec::new();
    let mut routes = HashMap::<usize, Vec<usize>>::new();
    let mut stop_reason = "complete";
    let global_lexical = match strategy {
        Strategy::Bm25 | Strategy::Global => engine.lexical(&query_terms, config.depth)?,
        Strategy::Regional => engine.lexical(&query_terms, config.rescue)?,
        Strategy::Exact => Vec::new(),
    };
    work.global_lexical_returned = global_lexical.len();
    let mut weights = BTreeMap::new();
    if matches!(strategy, Strategy::Regional | Strategy::Global) {
        let mut df = engine
            .conn
            .prepare_cached("SELECT df FROM term_stats WHERE term=?")?;
        for term in &query_terms {
            let count = df
                .query_row([term], |row| row.get::<_, usize>(0))
                .optional()?
                .unwrap_or(0);
            let idf =
                (1.0 + (engine.ids.len() as f64 - count as f64 + 0.5) / (count as f64 + 0.5)).ln();
            weights.insert(term.clone(), idf);
        }
    }
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
        Strategy::Regional => {
            let start = Instant::now();
            let deadline = if config.routing_ms == 0 {
                None
            } else {
                start.checked_add(Duration::from_millis(config.routing_ms))
            };
            let expired = || deadline.is_some_and(|limit| Instant::now() >= limit);
            let mut router = engine
                .topology
                .router(vector, &engine.vectors, &engine.ids, deadline);
            let mut scored = HashMap::<usize, f32>::new();
            let mut local = HashMap::<usize, f64>::new();
            let mut visited_terms = BTreeSet::new();
            let mut postings = engine.conn.prepare_cached(
                "SELECT artifact, frequency FROM postings WHERE region=? AND term=?",
            )?;
            stop_reason = 'routing: loop {
                if expired() {
                    break "routing_time";
                }
                if expanded.len() >= config.max_regions {
                    break "region_budget";
                }
                let Some((region, route)) = router.expand(&engine.topology, vector) else {
                    break "frontier_exhausted";
                };
                let route_id = expanded.len();
                expanded.push(route);
                for (offset, &id) in engine.topology.members[region].iter().enumerate() {
                    if offset % 128 == 0 && expired() {
                        break 'routing "routing_time";
                    }
                    if !scored.contains_key(&id) && scored.len() >= config.max_chunks {
                        continue;
                    }
                    scored
                        .entry(id)
                        .or_insert_with(|| cosine(vector, &engine.vectors[id]));
                    routes.entry(id).or_default().push(route_id);
                }
                for (term_number, (term, &idf)) in weights.iter().enumerate() {
                    let mut rows = postings.query(params![region, term])?;
                    loop {
                        if work.regional_postings_read % 128 == 0 && expired() {
                            break 'routing "routing_time";
                        }
                        if work.regional_postings_read >= config.max_postings {
                            break 'routing "postings_budget";
                        }
                        let Some(row) = rows.next()? else {
                            break;
                        };
                        work.regional_postings_read += 1;
                        let id: usize = row.get(0)?;
                        if scored.contains_key(&id) && visited_terms.insert((id, term_number)) {
                            let tf = row.get::<_, usize>(1)? as f64;
                            let average = engine.manifest.average_length.max(1.0);
                            let length = engine.lengths[id] as f64 / average;
                            *local.entry(id).or_default() +=
                                idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length));
                        }
                    }
                }
                if scored.len() >= config.max_chunks {
                    break "chunk_budget";
                }
            };
            work.regions_expanded = expanded.len();
            work.routing_comparisons = router.comparisons;
            work.regional_chunks_scored = scored.len();
            work.vector_comparisons = scored.len();
            let mut lexical = local.into_iter().collect::<Vec<_>>();
            lexical.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (rank, (id, _)) in lexical.into_iter().take(config.depth).enumerate() {
                candidates.entry(id).or_default().lexical = Some(rank + 1);
            }
            let mut nearest = scored
                .into_iter()
                .map(|(id, similarity)| Neighbor { id, similarity })
                .collect::<Vec<_>>();
            nearest.sort_unstable_by(|a, b| b.cmp(a));
            nearest.truncate(config.depth);
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
    let hybrid = matches!(strategy, Strategy::Global | Strategy::Regional);
    let total_weight = weights.values().sum::<f64>().max(f64::EPSILON);
    let mut reranked = Vec::with_capacity(fused.len());
    for (id, candidate) in fused {
        let (score, source) = if hybrid {
            let source = engine.source(id)?;
            let document = frequencies(&source.text);
            let coverage = weights
                .iter()
                .filter(|(term, _)| document.contains_key(*term))
                .map(|(_, weight)| weight)
                .sum::<f64>()
                / total_weight;
            let similarity = f64::from(cosine(vector, &engine.vectors[id]));
            work.rerank_vector_comparisons += 1;
            (
                0.65 * coverage + 0.25 * similarity + 0.10 * candidate.fusion() * 30.0,
                Some(source),
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
            routes: routes
                .get(&id)
                .into_iter()
                .flatten()
                .map(|&route| expanded[route].clone())
                .collect(),
        });
    }
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
