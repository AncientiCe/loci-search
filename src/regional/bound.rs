//! Exhaustive and branch-and-bound search under the linear regional reranker.
//!
//! The reranker scores a passage as `coverage_w * coverage + cosine_w * cosine`
//! (fusion weight zero). Both terms have exact per-leaf upper bounds: the cosine
//! through the leaf's angular radius around its unit centroid, and the coverage
//! through the query terms that occur anywhere in the leaf. Visiting leaves in
//! descending bound order and stopping once the best remaining bound falls below
//! the current k-th score returns the exhaustive top k.

use crate::{cosine, index::Engine, parallel_map, ranking::RegionalRerank};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap},
};

/// Added to every cosine bound so f32 rounding in the dot products cannot prune a true hit.
const COSINE_SLACK: f64 = 1e-4;

/// Angular extent of every nonempty leaf around its unit centroid.
pub(crate) struct Cones {
    leaves: Vec<usize>,
    radius: Vec<f64>,
}

impl Cones {
    pub fn build(engine: &Engine) -> Self {
        let topology = &engine.topology;
        let leaves = (0..topology.regions.len())
            .filter(|&region| !topology.members[region].is_empty())
            .collect::<Vec<_>>();
        let radius = parallel_map(&leaves, |&leaf| {
            let centroid = &topology.regions[leaf].centroid;
            let nearest_edge = topology.members[leaf]
                .iter()
                .map(|&id| cosine(centroid, &engine.vectors[id]))
                .fold(1.0_f32, f32::min);
            f64::from(nearest_edge).clamp(-1.0, 1.0).acos()
        });
        Self { leaves, radius }
    }

    pub fn leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Leaf radii in degrees, ascending.
    pub fn radii_degrees(&self) -> Vec<f64> {
        let mut degrees = self
            .radius
            .iter()
            .map(|radius| radius.to_degrees())
            .collect::<Vec<_>>();
        degrees.sort_unstable_by(f64::total_cmp);
        degrees
    }
}

/// The final ranking function for one query.
pub(crate) struct Objective<'a> {
    pub weights: &'a BTreeMap<String, f64>,
    pub rerank: RegionalRerank,
}

impl Objective<'_> {
    fn total(&self) -> f64 {
        self.weights.values().sum::<f64>().max(f64::EPSILON)
    }

    /// Exactly the score `ranking::search_with` gives a regional candidate.
    fn score(&self, engine: &Engine, vector: &[f32], id: usize, total: f64) -> f64 {
        let matched = if self.rerank.coverage > 0.0 {
            engine.regional_index().coverage(id, self.weights)
        } else {
            0.0
        };
        let similarity = f64::from(cosine(vector, &engine.vectors[id]));
        self.rerank.score(matched / total, similarity, 0.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Scored {
    pub id: usize,
    pub score: f64,
}

/// Higher score first, then lower passage id, matching the reranker's sort.
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.id.cmp(&self.id))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Scored {}

/// Score every passage and return the best `k`, best first.
pub(crate) fn exhaustive(
    engine: &Engine,
    vector: &[f32],
    objective: &Objective<'_>,
    k: usize,
) -> Vec<Scored> {
    let index = engine.regional_index();
    let total = objective.total();
    // Sum matched weights in term order, as `RegionalIndex::coverage` does.
    let mut matched = vec![0.0_f64; engine.ids.len()];
    if objective.rerank.coverage > 0.0 {
        for (term, &weight) in objective.weights {
            if let Some(postings) = index.postings.get(term) {
                for posting in &postings.documents {
                    matched[posting.id as usize] += weight;
                }
            }
        }
    }
    let mut scored = engine
        .vectors
        .iter()
        .zip(matched)
        .enumerate()
        .map(|(id, (candidate, matched))| Scored {
            id,
            score: objective.rerank.score(
                matched / total,
                f64::from(cosine(vector, candidate)),
                0.0,
            ),
        })
        .collect::<Vec<_>>();
    if k < scored.len() {
        scored.select_nth_unstable_by(k, |a, b| b.cmp(a));
        scored.truncate(k);
    }
    scored.sort_unstable_by(|a, b| b.cmp(a));
    scored
}

pub(crate) struct Pruned {
    pub top: Vec<Scored>,
    pub scored: usize,
    pub regions: usize,
    /// Leaves whose cosine bound was the trivial 1.0 because the query lies inside the cone.
    pub inside_cone: usize,
    pub stop_reason: &'static str,
}

/// Visit leaves by descending score bound until none can beat the k-th score.
///
/// `shrink` scales every leaf radius. At 1.0 the bound is safe and the result equals
/// [`exhaustive`]; below 1.0 the bound is optimistic and may miss true hits.
pub(crate) fn branch_and_bound(
    engine: &Engine,
    cones: &Cones,
    vector: &[f32],
    objective: &Objective<'_>,
    k: usize,
    shrink: f64,
    seen: &mut [bool],
) -> Pruned {
    let index = engine.regional_index();
    let topology = &engine.topology;
    let total = objective.total();
    let terms = objective
        .weights
        .iter()
        .map(|(term, &weight)| (index.postings.get(term), weight))
        .collect::<Vec<_>>();
    let mut inside_cone = 0;
    let mut order = cones
        .leaves
        .iter()
        .zip(&cones.radius)
        .map(|(&leaf, &radius)| {
            let centroid = &topology.regions[leaf].centroid;
            let angle = f64::from(cosine(vector, centroid)).clamp(-1.0, 1.0).acos();
            let gap = angle - shrink * radius;
            let cosine_bound = if gap <= 0.0 {
                inside_cone += 1;
                1.0
            } else {
                gap.cos()
            } + COSINE_SLACK;
            let coverage_bound = if objective.rerank.coverage > 0.0 {
                terms
                    .iter()
                    .filter(|(postings, _)| {
                        postings.is_some_and(|postings| postings.by_region.contains_key(&leaf))
                    })
                    .map(|(_, weight)| weight)
                    .sum::<f64>()
                    / total
            } else {
                0.0
            };
            (
                objective.rerank.score(coverage_bound, cosine_bound, 0.0),
                leaf,
            )
        })
        .collect::<Vec<_>>();
    order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut best = BinaryHeap::<Reverse<Scored>>::with_capacity(k + 1);
    let mut touched = Vec::new();
    let mut regions = 0;
    let mut stop_reason = "frontier_exhausted";
    for (bound, leaf) in order {
        if best.len() == k && best.peek().is_some_and(|kth| bound < kth.0.score) {
            stop_reason = "bound";
            break;
        }
        regions += 1;
        for &id in &topology.members[leaf] {
            if seen[id] {
                continue;
            }
            seen[id] = true;
            touched.push(id);
            let candidate = Scored {
                id,
                score: objective.score(engine, vector, id, total),
            };
            if best.len() < k {
                best.push(Reverse(candidate));
            } else if best.peek().is_some_and(|kth| candidate > kth.0) {
                best.pop();
                best.push(Reverse(candidate));
            }
        }
    }
    let scored = touched.len();
    for id in touched {
        seen[id] = false;
    }
    let mut top = best.into_iter().map(|entry| entry.0).collect::<Vec<_>>();
    top.sort_unstable_by(|a, b| b.cmp(a));
    Pruned {
        top,
        scored,
        regions,
        inside_cone,
        stop_reason,
    }
}
