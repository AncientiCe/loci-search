use crate::{
    cosine,
    graph::Neighbor,
    index::{Engine, frequencies},
    profile::{Phases, Timer},
    ranking::{SearchConfig, Work},
    topology::Route,
};
use anyhow::Result;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Mutex,
    time::{Duration, Instant},
};

pub(crate) mod bound;
mod quantized;
mod wand;

use quantized::Quantized;

// FTS5 truncates longer tokens. Use SQLite for rescue when that can affect matching.
const FTS5_TOKEN_BYTES: usize = 32768;
const UNVISITED: usize = usize::MAX;
/// Leaf entries seeded into routing when the configuration does not set them.
const DEFAULT_ROUTING_ENTRIES: usize = 2;

struct Posting {
    id: u32,
    frequency: u32,
}

#[derive(Default)]
struct TermPostings {
    documents: Vec<Posting>,
    by_region: HashMap<usize, Vec<u32>>,
    max_rescue_impact: f64,
}

pub(crate) struct RegionalIndex {
    postings: HashMap<String, TermPostings>,
    fts5_compatible: bool,
    rescue_norms: Vec<f64>,
    regional_norms: Vec<f64>,
    quantized: Quantized,
    /// One buffer set per in-flight search. Queries do not share or wait on one scratch.
    scratch: Mutex<Vec<Scratch>>,
}

struct Scratch {
    first_visit: Vec<usize>,
    lexical: Vec<f64>,
    seen_terms: Vec<u64>,
}

pub(crate) struct LexicalRescue {
    pub ranked: Vec<usize>,
    pub scores: Option<RescueScores>,
    pub budget_hit: bool,
}

pub(crate) struct RescueScores {
    scored: Scored,
    pub count: usize,
}

/// Passages the rescue scored: dense scores for the full scan, or the ascending
/// ids MaxScore evaluated, which avoids a corpus-sized array per query.
enum Scored {
    Dense(Vec<f64>),
    Sorted(Vec<u32>),
}

impl Default for RescueScores {
    fn default() -> Self {
        Self {
            scored: Scored::Sorted(Vec::new()),
            count: 0,
        }
    }
}

impl RescueScores {
    fn contains(&self, id: usize) -> bool {
        match &self.scored {
            Scored::Dense(scores) => scores.get(id).is_some_and(|&score| score > 0.0),
            Scored::Sorted(ids) => u32::try_from(id).is_ok_and(|id| ids.binary_search(&id).is_ok()),
        }
    }
}

impl RegionalIndex {
    pub fn load(engine: &Engine) -> Result<Self> {
        let mut addresses = vec![Vec::new(); engine.ids.len()];
        for (region, members) in engine.topology.members.iter().enumerate() {
            for &doc in members {
                addresses[doc].push(region);
            }
        }
        let mut postings = HashMap::<String, TermPostings>::new();
        let mut fts5_compatible = true;
        let average = engine.manifest.average_length;
        let rescue_norms = engine
            .lengths
            .iter()
            .map(|&len| {
                1.2 * (0.25 + 0.75 * len as f64 / if average == 0.0 { 1.0 } else { average })
            })
            .collect::<Vec<_>>();
        let conn = engine.conn.get()?;
        let mut stmt = conn.prepare("SELECT rowid, text FROM artifacts ORDER BY rowid")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: usize = row.get(0)?;
            let text = row.get_ref(1)?.as_str()?;
            for (term, frequency) in frequencies(text) {
                fts5_compatible &= term.len() <= FTS5_TOKEN_BYTES;
                let posting = postings.entry(term).or_default();
                let frequency = u32::try_from(frequency)?;
                posting.max_rescue_impact = posting
                    .max_rescue_impact
                    .max(rescue_impact(frequency, rescue_norms[id]));
                let offset = u32::try_from(posting.documents.len())?;
                posting.documents.push(Posting {
                    id: u32::try_from(id)?,
                    frequency,
                });
                for &region in &addresses[id] {
                    posting.by_region.entry(region).or_default().push(offset);
                }
            }
        }
        let regional_norms = engine
            .lengths
            .iter()
            .map(|&len| 1.2 * (0.25 + 0.75 * (len as f64 / average.max(1.0))))
            .collect();
        Ok(Self {
            postings,
            fts5_compatible,
            rescue_norms,
            regional_norms,
            quantized: Quantized::build(&engine.vectors),
            scratch: Mutex::new(vec![Scratch {
                first_visit: vec![UNVISITED; engine.ids.len()],
                lexical: vec![0.0; engine.ids.len()],
                seen_terms: vec![0; engine.ids.len()],
            }]),
        })
    }

    fn checkout(&self, passages: usize) -> Scratch {
        let mut scratch = self
            .scratch
            .lock()
            .expect("regional scratch")
            .pop()
            .unwrap_or_else(|| Scratch {
                first_visit: vec![UNVISITED; passages],
                lexical: vec![0.0; passages],
                seen_terms: vec![0; passages],
            });
        if scratch.first_visit.len() != passages {
            scratch.first_visit.resize(passages, UNVISITED);
            scratch.lexical.resize(passages, 0.0);
            scratch.seen_terms.resize(passages, 0);
        }
        scratch
    }

    fn recycle(&self, scratch: Scratch) {
        self.scratch.lock().expect("regional scratch").push(scratch);
    }

    #[cfg(test)]
    pub fn rescue(
        &self,
        engine: &Engine,
        terms: &BTreeSet<String>,
        limit: usize,
        max_scored: Option<usize>,
    ) -> Result<LexicalRescue> {
        self.rescue_filtered(engine, terms, limit, max_scored, None)
    }

    /// Global lexical rescue over the passages `allow` admits (all when `None`).
    pub fn rescue_filtered(
        &self,
        engine: &Engine,
        terms: &BTreeSet<String>,
        limit: usize,
        max_scored: Option<usize>,
        allow: Option<&[bool]>,
    ) -> Result<LexicalRescue> {
        if terms.is_empty() || limit == 0 {
            return Ok(LexicalRescue {
                ranked: Vec::new(),
                scores: Some(RescueScores::default()),
                budget_hit: false,
            });
        }
        if !self.fts5_compatible || terms.iter().any(|term| term.len() > FTS5_TOKEN_BYTES) {
            anyhow::ensure!(
                max_scored.is_none() && allow.is_none(),
                "bounded rescue requires the in-memory lexical index"
            );
            return Ok(LexicalRescue {
                ranked: engine.lexical(terms, limit)?,
                scores: None,
                budget_hit: false,
            });
        }
        let query_terms = terms
            .iter()
            .filter_map(|term| {
                let posting = self.postings.get(term)?;
                let df = posting.documents.len() as f64;
                let idf = ((self.rescue_norms.len() as f64 - df + 0.5) / (df + 0.5)).ln();
                Some((posting, if idf <= 0.0 { 1e-6 } else { idf }))
            })
            .collect::<Vec<_>>();
        if let Some(budget) = max_scored {
            return Ok(wand::rescue(
                &query_terms,
                &self.rescue_norms,
                limit,
                budget,
                allow,
            ));
        }
        let mut scores = vec![0.0; self.rescue_norms.len()];
        // Reproduce SQLite's OR-of-single-terms BM25 formula, in query term order.
        for (postings, idf) in query_terms {
            for posting in &postings.documents {
                let id = posting.id as usize;
                if allow.is_some_and(|allow| !allow[id]) {
                    continue;
                }
                scores[id] += idf * rescue_impact(posting.frequency, self.rescue_norms[id]);
            }
        }
        let (ranked, count) = top_lexical(&scores, limit);
        Ok(LexicalRescue {
            ranked,
            scores: Some(RescueScores {
                scored: Scored::Dense(scores),
                count,
            }),
            budget_hit: false,
        })
    }

    /// Bytes reserved for posting payloads and length factors, excluding map/string overhead.
    pub fn quantized_bytes(&self) -> usize {
        self.quantized.bytes()
    }

    pub fn payload_bytes(&self) -> usize {
        self.postings
            .values()
            .map(|term| {
                term.documents.capacity() * size_of::<Posting>()
                    + term
                        .by_region
                        .values()
                        .map(|list| list.capacity() * size_of::<u32>())
                        .sum::<usize>()
            })
            .sum::<usize>()
            + (self.rescue_norms.capacity() + self.regional_norms.capacity()) * size_of::<f64>()
    }

    /// Passages containing `term`.
    pub fn document_frequency(&self, term: &str) -> usize {
        self.postings
            .get(term)
            .map_or(0, |postings| postings.documents.len())
    }

    pub fn coverage(&self, id: usize, weights: &BTreeMap<String, f64>) -> f64 {
        let Ok(id) = u32::try_from(id) else {
            return 0.0;
        };
        weights
            .iter()
            .filter_map(|(term, &weight)| {
                let postings = self.postings.get(term)?;
                postings
                    .documents
                    .binary_search_by_key(&id, |posting| posting.id)
                    .ok()
                    .map(|_| weight)
            })
            .sum()
    }
}

fn rescue_impact(frequency: u32, norm: f64) -> f64 {
    let frequency = f64::from(frequency);
    (frequency * 2.2) / (frequency + norm)
}

#[derive(Default)]
pub(crate) struct Routes {
    /// (passage, first visit index) for every scanned passage, sorted by passage.
    first_visit: Vec<(usize, usize)>,
    visits: Vec<Visit>,
}

struct Visit {
    region: usize,
    scanned: usize,
}

impl Routes {
    pub fn for_hit(&self, engine: &Engine, id: usize, expanded: &[Route]) -> Vec<Route> {
        let Ok(position) = self.first_visit.binary_search_by_key(&id, |&(id, _)| id) else {
            return Vec::new();
        };
        let first = self.first_visit[position].1;
        self.visits
            .iter()
            .zip(expanded)
            .skip(first)
            .filter_map(|(visit, route)| {
                engine.topology.members[visit.region][..visit.scanned]
                    .binary_search(&id)
                    .ok()
                    .map(|_| route.clone())
            })
            .collect()
    }
}

pub(crate) struct Selection {
    pub routes: Routes,
    pub expanded: Vec<Route>,
    pub lexical: Vec<usize>,
    pub nearest: Vec<Neighbor>,
    pub stop_reason: &'static str,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn select(
    engine: &Engine,
    vector: &[f32],
    config: &SearchConfig,
    weights: &BTreeMap<String, f64>,
    rescue_scores: Option<&RescueScores>,
    allow: Option<&[bool]>,
    work: &mut Work,
    profile: &mut Option<&mut Phases>,
) -> Selection {
    let index = engine.regional_index();
    let terms = weights
        .iter()
        .map(|(term, &idf)| (index.postings.get(term), idf))
        .collect::<Vec<_>>();
    let mut held = index.checkout(engine.ids.len());
    let Scratch {
        first_visit,
        lexical,
        seen_terms,
    } = &mut held;
    let mut routes = Routes::default();
    let mut expanded = Vec::new();
    let mut nearest = Vec::new();
    let mut dense_only = 0;
    let mut lexical_ids = Vec::new();
    let mut extra_terms = HashSet::new();
    let deadline = (config.routing_ms > 0)
        .then(|| Instant::now().checked_add(Duration::from_millis(config.routing_ms)))
        .flatten();
    let expired = || deadline.is_some_and(|limit| Instant::now() >= limit);
    let timer = Timer::start(profile, "routing");
    let mut router = engine.topology.router(
        vector,
        &engine.vectors,
        &engine.ids,
        deadline,
        config.routing_entries.unwrap_or(DEFAULT_ROUTING_ENTRIES),
    );
    timer.finish(profile);
    let stop_reason = 'routing: loop {
        if expired() {
            break "routing_time";
        }
        if expanded.len() >= config.max_regions {
            break "region_budget";
        }
        let timer = Timer::start(profile, "routing");
        let next = router.expand(&engine.topology, vector);
        timer.finish(profile);
        let Some((region, route)) = next else {
            break "frontier_exhausted";
        };
        let visit_number = expanded.len();
        expanded.push(route);
        routes.visits.push(Visit { region, scanned: 0 });
        let visit = routes.visits.last_mut().expect("visit just recorded");
        let timer = Timer::start(profile, "vectors");
        for (offset, &id) in engine.topology.members[region].iter().enumerate() {
            if offset.is_multiple_of(128) && expired() {
                timer.finish(profile);
                break 'routing "routing_time";
            }
            visit.scanned = offset + 1;
            if allow.is_some_and(|allow| !allow[id]) {
                continue;
            }
            if first_visit[id] == UNVISITED && nearest.len() < config.max_chunks {
                first_visit[id] = visit_number;
                if rescue_scores.is_some_and(|rescue| !rescue.contains(id)) {
                    dense_only += 1;
                }
                let similarity = if config.regional_rescore.is_some() {
                    index.quantized.similarity(vector, id)
                } else {
                    cosine(vector, &engine.vectors[id])
                };
                nearest.push(Neighbor { id, similarity });
            }
        }
        timer.finish(profile);
        let timer = Timer::start(profile, "postings");
        for (number, &(posting, idf)) in terms.iter().enumerate() {
            let mut offsets = posting
                .and_then(|term| term.by_region.get(&region))
                .map_or(&[][..], Vec::as_slice)
                .iter();
            loop {
                if work.regional_postings_read.is_multiple_of(128) && expired() {
                    timer.finish(profile);
                    break 'routing "routing_time";
                }
                if work.regional_postings_read >= config.max_postings {
                    timer.finish(profile);
                    break 'routing "postings_budget";
                }
                let Some(&offset) = offsets.next() else { break };
                work.regional_postings_read += 1;
                let doc = &posting.expect("matched posting").documents[offset as usize];
                let id = doc.id as usize;
                if first_visit[id] == UNVISITED {
                    continue;
                }
                let first_term = if number < u64::BITS as usize {
                    let bit = 1 << number;
                    let first_term = seen_terms[id] & bit == 0;
                    seen_terms[id] |= bit;
                    first_term
                } else {
                    extra_terms.insert((id, number))
                };
                if first_term {
                    if lexical[id] == 0.0 {
                        lexical_ids.push(id);
                    }
                    let tf = f64::from(doc.frequency);
                    lexical[id] += idf * tf * 2.2 / (tf + index.regional_norms[id]);
                }
            }
        }
        timer.finish(profile);
        if nearest.len() >= config.max_chunks {
            break "chunk_budget";
        }
    };
    work.regions_expanded = expanded.len();
    work.routing_comparisons = router.comparisons;
    work.regional_chunks_scored = nearest.len();
    work.vector_comparisons = nearest.len();
    work.regional_or_rescue_chunks_scored = rescue_scores.map(|rescue| rescue.count + dense_only);
    // Every scanned passage is in `nearest` until the cuts below; record and reset.
    routes.first_visit = nearest
        .iter()
        .map(|hit| {
            (
                hit.id,
                std::mem::replace(&mut first_visit[hit.id], UNVISITED),
            )
        })
        .collect();
    routes.first_visit.sort_unstable();
    if let Some(rescore) = config.regional_rescore {
        let timer = Timer::start(profile, "vectors");
        retain_top(&mut nearest, rescore, |a, b| b.cmp(a));
        for neighbor in &mut nearest {
            neighbor.similarity = cosine(vector, &engine.vectors[neighbor.id]);
        }
        work.rescored_vector_comparisons = Some(nearest.len());
        timer.finish(profile);
    }
    let timer = Timer::start(profile, "selection");
    let mut matches = lexical_ids
        .iter()
        .map(|&id| (id, lexical[id]))
        .collect::<Vec<_>>();
    for &id in &lexical_ids {
        lexical[id] = 0.0;
        seen_terms[id] = 0;
    }
    index.recycle(held);
    retain_top(&mut matches, config.depth, |a, b| {
        b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
    });
    let lexical = matches.into_iter().map(|(id, _)| id).collect();
    retain_top(&mut nearest, config.depth, |a, b| b.cmp(a));
    timer.finish(profile);
    Selection {
        routes,
        expanded,
        lexical,
        nearest,
        stop_reason,
    }
}

fn top_lexical(scores: &[f64], depth: usize) -> (Vec<usize>, usize) {
    let mut matches = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    let count = matches.len();
    retain_top(&mut matches, depth, |a, b| {
        b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
    });
    (matches.into_iter().map(|(id, _)| id).collect(), count)
}

fn retain_top<T>(items: &mut Vec<T>, depth: usize, compare: impl Fn(&T, &T) -> Ordering) {
    if depth < items.len() {
        items.select_nth_unstable_by(depth, &compare);
        items.truncate(depth);
    }
    items.sort_unstable_by(compare);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fixture, index, terms};

    #[test]
    fn cached_rescue_matches_fts5_for_ties_and_common_terms() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let db = dir.path().join("rescue.sqlite");
        let (embedding, mut corpus, _) = fixture::generate(128)?;
        for (chunk, passage) in corpus.iter_mut().enumerate() {
            passage.text = match chunk % 4 {
                0 => "common common rare café",
                1 => "common rare café",
                2 => "common École",
                _ => "",
            }
            .into();
            passage.end = passage.text.len() as u64;
        }
        index::ingest(&db, corpus, embedding, 16)?;
        let mut engine = Engine::open(&db)?;
        engine.prepare_regional()?;
        for query in ["common", "common rare", "école CAFÉ", "absent"] {
            let tokens = terms(query).collect::<BTreeSet<_>>();
            let all_matches = engine.lexical(&tokens, 128)?;
            for limit in [1, 40, 128] {
                let rescue = engine
                    .regional_index()
                    .rescue(&engine, &tokens, limit, None)?;
                assert_eq!(rescue.ranked, all_matches[..all_matches.len().min(limit)]);
                assert_eq!(rescue.scores.unwrap().count, all_matches.len());
                let exact = engine
                    .regional_index()
                    .rescue(&engine, &tokens, limit, Some(128))?;
                assert_eq!(exact.ranked, rescue.ranked);
                assert!(!exact.budget_hit);
                let capped = engine
                    .regional_index()
                    .rescue(&engine, &tokens, 1, Some(4))?;
                assert!(capped.scores.unwrap().count <= 4);
            }
        }
        Ok(())
    }

    #[test]
    fn safe_branch_and_bound_returns_the_exhaustive_top_k() -> Result<()> {
        use crate::ranking::{RegionalRerank, coverage_weights};
        let dir = tempfile::tempdir()?;
        let db = dir.path().join("bound.sqlite");
        let (embedding, corpus, queries) = fixture::generate(2_000)?;
        index::ingest(&db, corpus, embedding, 64)?;
        let mut engine = Engine::open(&db)?;
        engine.prepare_regional()?;
        let cones = bound::Cones::build(&engine);
        let mut seen = vec![false; engine.ids.len()];
        let mut pruned_any = false;
        for query in queries.iter().take(16) {
            let vector = engine.query_vector(query)?;
            let weights = coverage_weights(&engine, &terms(&query.text).collect())?;
            for rerank in [
                RegionalRerank {
                    coverage: 0.08,
                    cosine: 0.92,
                    fusion: 0.0,
                },
                RegionalRerank {
                    coverage: 0.0,
                    cosine: 1.0,
                    fusion: 0.0,
                },
            ] {
                let objective = bound::Objective {
                    weights: &weights,
                    rerank,
                };
                let exhaustive = bound::exhaustive(&engine, &vector, &objective, 20, None);
                let pruned = bound::branch_and_bound(
                    &engine, &cones, &vector, &objective, 20, 1.0, &mut seen,
                );
                let ids =
                    |hits: &[bound::Scored]| hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
                assert_eq!(ids(&pruned.top), ids(&exhaustive));
                pruned_any |= pruned.scored < engine.ids.len();
                assert!(seen.iter().all(|&flag| !flag));
            }
        }
        assert!(
            pruned_any,
            "the bound never skipped a passage on the fixture"
        );
        Ok(())
    }

    #[test]
    fn filtered_search_returns_only_admitted_passages() -> Result<()> {
        use crate::ranking::{RegionalRerank, SearchConfig, Strategy, coverage_weights};
        let dir = tempfile::tempdir()?;
        let db = dir.path().join("filtered.sqlite");
        let (embedding, corpus, queries) = fixture::generate(2_000)?;
        index::ingest(&db, corpus, embedding, 64)?;
        let mut engine = Engine::open(&db)?;
        let rerank = RegionalRerank {
            coverage: 0.08,
            cosine: 0.92,
            fusion: 0.0,
        };
        let config = SearchConfig {
            limit: 20,
            depth: 40,
            max_regions: 64,
            max_chunks: 2_000,
            max_postings: 1_000_000,
            rescue: 10,
            rescue_max_scored: Some(2_000),
            regional_rerank: Some(rerank),
            ..SearchConfig::default()
        };
        engine.prepare(Strategy::Regional, &config)?;
        let allow = (0..engine.ids.len())
            .map(|id| id % 7 == 0)
            .collect::<Vec<_>>();
        let admitted = (0..engine.ids.len())
            .filter(|&id| allow[id])
            .collect::<Vec<_>>();
        for query in queries.iter().take(8) {
            let result = engine.search_filtered(query, &config, &allow)?;
            assert!(!result.hits.is_empty());
            for hit in &result.hits {
                let row = engine
                    .ids
                    .iter()
                    .position(|id| *id == hit.source.id)
                    .unwrap();
                assert!(allow[row], "hit {} is outside the filter", hit.source.id);
            }
            let tokens = terms(&query.text).collect::<BTreeSet<_>>();
            let rescue = engine.regional_index().rescue_filtered(
                &engine,
                &tokens,
                10,
                Some(2_000),
                Some(&allow),
            )?;
            assert!(rescue.ranked.iter().all(|&id| allow[id]));
            let vector = engine.query_vector(query)?;
            let weights = coverage_weights(&engine, &tokens)?;
            let objective = bound::Objective {
                weights: &weights,
                rerank,
            };
            let scanned = bound::exhaustive(&engine, &vector, &objective, 20, Some(&allow));
            let listed = bound::exhaustive_over(&engine, &vector, &objective, 20, &admitted);
            let ids = |hits: &[bound::Scored]| hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
            assert_eq!(ids(&scanned), ids(&listed));
        }
        Ok(())
    }

    #[test]
    fn bounded_rescue_skips_subthreshold_passages_and_detects_a_binding_cap() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let db = directory.path().join("bounded.sqlite");
        let (embedding, mut corpus, _) = fixture::generate(128)?;
        corpus.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        for (id, passage) in corpus.iter_mut().enumerate() {
            passage.text = if id == 0 {
                "common distinctive"
            } else {
                "common"
            }
            .into();
            passage.end = passage.text.len() as u64;
        }
        index::ingest(&db, corpus, embedding, 16)?;
        let mut engine = Engine::open(&db)?;
        engine.prepare_regional()?;
        let selective = terms("common distinctive").collect::<BTreeSet<_>>();
        let result = engine
            .regional_index()
            .rescue(&engine, &selective, 1, Some(1))?;
        assert_eq!(result.ranked, engine.lexical(&selective, 1)?);
        assert_eq!(result.scores.unwrap().count, 1);
        assert!(!result.budget_hit);

        let common = terms("common").collect::<BTreeSet<_>>();
        let capped = engine
            .regional_index()
            .rescue(&engine, &common, 1, Some(1))?;
        assert_eq!(capped.scores.unwrap().count, 1);
        assert!(capped.budget_hit);
        Ok(())
    }
}
