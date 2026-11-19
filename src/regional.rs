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
    time::{Duration, Instant},
};

// FTS5 truncates longer tokens. Use SQLite for rescue when that can affect matching.
const FTS5_TOKEN_BYTES: usize = 32768;
const UNVISITED: usize = usize::MAX;

struct Posting {
    id: u32,
    frequency: u32,
}

#[derive(Default)]
struct TermPostings {
    documents: Vec<Posting>,
    by_region: HashMap<usize, Vec<u32>>,
}

pub(crate) struct RegionalIndex {
    postings: HashMap<String, TermPostings>,
    fts5_compatible: bool,
    rescue_norms: Vec<f64>,
    regional_norms: Vec<f64>,
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
        let mut stmt = engine
            .conn
            .prepare("SELECT rowid, text FROM artifacts ORDER BY rowid")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: usize = row.get(0)?;
            let text = row.get_ref(1)?.as_str()?;
            for (term, frequency) in frequencies(text) {
                fts5_compatible &= term.len() <= FTS5_TOKEN_BYTES;
                let posting = postings.entry(term).or_default();
                let offset = u32::try_from(posting.documents.len())?;
                posting.documents.push(Posting {
                    id: u32::try_from(id)?,
                    frequency: u32::try_from(frequency)?,
                });
                for &region in &addresses[id] {
                    posting.by_region.entry(region).or_default().push(offset);
                }
            }
        }
        let average = engine.manifest.average_length;
        let rescue_norms = engine
            .lengths
            .iter()
            .map(|&len| {
                1.2 * (0.25 + 0.75 * len as f64 / if average == 0.0 { 1.0 } else { average })
            })
            .collect();
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
        })
    }

    pub fn rescue(
        &self,
        engine: &Engine,
        terms: &BTreeSet<String>,
        limit: usize,
    ) -> Result<Vec<usize>> {
        if terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        if !self.fts5_compatible || terms.iter().any(|term| term.len() > FTS5_TOKEN_BYTES) {
            return engine.lexical(terms, limit);
        }
        let mut scores = vec![0.0; self.rescue_norms.len()];
        // Reproduce SQLite's OR-of-single-terms BM25 formula, in query term order.
        for term in terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.documents.len() as f64;
            let idf = ((scores.len() as f64 - df + 0.5) / (df + 0.5)).ln();
            let idf = if idf <= 0.0 { 1e-6 } else { idf };
            for posting in &postings.documents {
                let id = posting.id as usize;
                let tf = f64::from(posting.frequency);
                scores[id] += idf * ((tf * 2.2) / (tf + self.rescue_norms[id]));
            }
        }
        Ok(top_lexical(scores, limit))
    }

    /// Bytes reserved for posting payloads and length factors, excluding map/string overhead.
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

#[derive(Default)]
pub(crate) struct Routes {
    first_visit: Vec<usize>,
    visits: Vec<Visit>,
}

struct Visit {
    region: usize,
    scanned: usize,
}

impl Routes {
    pub fn for_hit(&self, engine: &Engine, id: usize, expanded: &[Route]) -> Vec<Route> {
        let first = self.first_visit.get(id).copied().unwrap_or(UNVISITED);
        if first == UNVISITED {
            return Vec::new();
        }
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

pub(crate) fn select(
    engine: &Engine,
    vector: &[f32],
    config: &SearchConfig,
    weights: &BTreeMap<String, f64>,
    work: &mut Work,
    profile: &mut Option<&mut Phases>,
) -> Selection {
    let index = engine.regional_index();
    let terms = weights
        .iter()
        .map(|(term, &idf)| (index.postings.get(term), idf))
        .collect::<Vec<_>>();
    let mut routes = Routes {
        first_visit: vec![UNVISITED; engine.ids.len()],
        visits: Vec::new(),
    };
    let mut expanded = Vec::new();
    let mut nearest = Vec::new();
    let mut lexical = vec![0.0; engine.ids.len()];
    let mut seen_terms = vec![0_u64; engine.ids.len()];
    let mut extra_terms = HashSet::new();
    let deadline = (config.routing_ms > 0)
        .then(|| Instant::now().checked_add(Duration::from_millis(config.routing_ms)))
        .flatten();
    let expired = || deadline.is_some_and(|limit| Instant::now() >= limit);
    let timer = Timer::start(profile, "routing");
    let mut router = engine
        .topology
        .router(vector, &engine.vectors, &engine.ids, deadline);
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
            if routes.first_visit[id] == UNVISITED && nearest.len() < config.max_chunks {
                routes.first_visit[id] = visit_number;
                nearest.push(Neighbor {
                    id,
                    similarity: cosine(vector, &engine.vectors[id]),
                });
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
                if routes.first_visit[id] == UNVISITED {
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
    let timer = Timer::start(profile, "selection");
    let lexical = top_lexical(lexical, config.depth);
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

fn top_lexical(scores: Vec<f64>, depth: usize) -> Vec<usize> {
    let mut matches = scores
        .into_iter()
        .enumerate()
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    retain_top(&mut matches, depth, |a, b| {
        b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
    });
    matches.into_iter().map(|(id, _)| id).collect()
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
            for limit in [1, 40, 128] {
                assert_eq!(
                    engine.regional_index().rescue(&engine, &tokens, limit)?,
                    engine.lexical(&tokens, limit)?
                );
            }
        }
        Ok(())
    }
}
