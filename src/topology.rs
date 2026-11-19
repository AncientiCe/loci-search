use crate::{cosine, graph::Neighbor, normalize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::time::Instant;

pub const MEMBERSHIP_LIMIT: usize = 3;
pub const MEMBERSHIP_MARGIN: f32 = 0.10;
const EDGE_LIMIT: usize = 4;

#[derive(Debug, Deserialize, Serialize)]
pub struct Region {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub centroid: Vec<f32>,
    pub landmark: usize,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Edge {
    pub target: usize,
    pub weight: f32,
    pub shared_memberships: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub region: usize,
    pub via: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Route {
    pub entry_landmark: String,
    pub path: Vec<Step>,
}

pub struct Topology {
    pub regions: Vec<Region>,
    pub members: Vec<Vec<usize>>,
}

impl Topology {
    pub fn build(vectors: &[Vec<f32>], leaf_size: usize) -> Self {
        let mut regions = Vec::new();
        partition(
            (0..vectors.len()).collect(),
            None,
            vectors,
            leaf_size,
            &mut regions,
        );
        let leaves = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.children.is_empty())
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        let mut members = vec![Vec::new(); regions.len()];
        let mut overlaps = BTreeMap::<(usize, usize), usize>::new();
        for (id, vector) in vectors.iter().enumerate() {
            let mut matches = leaves
                .iter()
                .map(|&leaf| Neighbor {
                    id: leaf,
                    similarity: cosine(vector, &regions[leaf].centroid),
                })
                .collect::<Vec<_>>();
            matches.sort_unstable_by(|a, b| b.cmp(a));
            let cutoff = matches[0].similarity - MEMBERSHIP_MARGIN;
            let addresses = matches
                .iter()
                .take(MEMBERSHIP_LIMIT)
                .take_while(|hit| hit.similarity >= cutoff)
                .map(|hit| hit.id)
                .collect::<Vec<_>>();
            for (position, &leaf) in addresses.iter().enumerate() {
                members[leaf].push(id);
                for &other in &addresses[..position] {
                    *overlaps
                        .entry((leaf.min(other), leaf.max(other)))
                        .or_default() += 1;
                }
            }
        }
        for ((left, right), shared_memberships) in overlaps {
            let union = members[left].len() + members[right].len() - shared_memberships;
            let weight = shared_memberships as f32 / union as f32;
            regions[left].edges.push(Edge {
                target: right,
                weight,
                shared_memberships,
            });
            regions[right].edges.push(Edge {
                target: left,
                weight,
                shared_memberships,
            });
        }
        for region in &mut regions {
            region.edges.sort_unstable_by(|a, b| {
                b.weight.total_cmp(&a.weight).then(a.target.cmp(&b.target))
            });
            region.edges.truncate(EDGE_LIMIT);
        }
        Self { regions, members }
    }

    pub fn router(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        ids: &[String],
        deadline: Option<Instant>,
    ) -> Router {
        let mut entries = BinaryHeap::new();
        for (id, node) in self
            .regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.children.is_empty())
        {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                break;
            }
            let similarity =
                (cosine(query, &node.centroid) + cosine(query, &vectors[node.landmark])) / 2.0;
            entries.push(Neighbor { id, similarity });
        }
        let seed_comparisons = 2 * entries.len();
        let mut pending = BinaryHeap::new();
        let mut routes = Vec::new();
        for item in entries.into_sorted_vec().into_iter().rev().take(2) {
            let landmark = self.regions[item.id].landmark;
            let route = Route {
                entry_landmark: ids[landmark].clone(),
                path: vec![Step {
                    region: item.id,
                    via: "entry",
                }],
            };
            routes.push((item.id, route));
            pending.push(Neighbor {
                id: routes.len() - 1,
                similarity: item.similarity,
            });
        }
        Router {
            pending,
            routes,
            visited: HashSet::new(),
            comparisons: seed_comparisons,
        }
    }
}

pub struct Router {
    pending: BinaryHeap<Neighbor>,
    routes: Vec<(usize, Route)>,
    visited: HashSet<usize>,
    pub comparisons: usize,
}

impl Router {
    pub fn expand(&mut self, topology: &Topology, query: &[f32]) -> Option<(usize, Route)> {
        while let Some(candidate) = self.pending.pop() {
            let (id, route) = self.routes[candidate.id].clone();
            if !self.visited.insert(id) {
                continue;
            }
            let node = &topology.regions[id];
            let hierarchy = node
                .parent
                .into_iter()
                .chain(node.children.iter().copied())
                .map(|target| (target, "hierarchy", 0.0));
            let edges = node
                .edges
                .iter()
                .map(|edge| (edge.target, "shared_membership", edge.weight));
            for (target, via, weight) in hierarchy.chain(edges) {
                if self.visited.contains(&target) {
                    continue;
                }
                let similarity = cosine(query, &topology.regions[target].centroid) + weight * 0.15;
                self.comparisons += 1;
                let mut next = route.clone();
                next.path.push(Step {
                    region: target,
                    via,
                });
                self.routes.push((target, next));
                self.pending.push(Neighbor {
                    id: self.routes.len() - 1,
                    similarity,
                });
            }
            return Some((id, route));
        }
        None
    }
}

fn partition(
    ids: Vec<usize>,
    parent: Option<usize>,
    vectors: &[Vec<f32>],
    leaf_size: usize,
    regions: &mut Vec<Region>,
) -> usize {
    let center = centroid(&ids, vectors);
    let landmark = closest(&ids, &center, vectors);
    let id = regions.len();
    regions.push(Region {
        parent,
        children: Vec::new(),
        centroid: center,
        landmark,
        edges: Vec::new(),
    });
    if ids.len() > leaf_size {
        let mut left_center = vectors[landmark].clone();
        let farthest = *ids
            .iter()
            .min_by(|&&a, &&b| {
                cosine(&left_center, &vectors[a]).total_cmp(&cosine(&left_center, &vectors[b]))
            })
            .expect("nonempty region");
        let mut right_center = vectors[farthest].clone();
        let mut left = Vec::new();
        let mut right = Vec::new();
        for iteration in 0..6 {
            left.clear();
            right.clear();
            for &member in &ids {
                let difference = cosine(&left_center, &vectors[member])
                    - cosine(&right_center, &vectors[member]);
                if difference > 0.0 || (difference == 0.0 && left.len() <= right.len()) {
                    left.push(member);
                } else {
                    right.push(member);
                }
            }
            // Keep highly imbalanced or degenerate partitions from producing a deep chain.
            if left.len().min(right.len()) < (ids.len() / 16).max(1) {
                let mut projected = ids
                    .iter()
                    .map(|&member| Neighbor {
                        id: member,
                        similarity: cosine(&left_center, &vectors[member])
                            - cosine(&right_center, &vectors[member]),
                    })
                    .collect::<Vec<_>>();
                projected.sort_unstable_by(|a, b| b.cmp(a));
                let (a, b) = projected.split_at(projected.len() / 2);
                left = a.iter().map(|member| member.id).collect();
                right = b.iter().map(|member| member.id).collect();
                break;
            }
            if iteration < 5 {
                left_center = centroid(&left, vectors);
                right_center = centroid(&right, vectors);
            }
        }
        let left = partition(left, Some(id), vectors, leaf_size, regions);
        let right = partition(right, Some(id), vectors, leaf_size, regions);
        regions[id].children = vec![left, right];
    }
    id
}

fn centroid(ids: &[usize], vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut mean = vec![0.0; vectors[0].len()];
    for &id in ids {
        for (sum, &value) in mean.iter_mut().zip(&vectors[id]) {
            *sum += value;
        }
    }
    normalize(&mean, mean.len()).unwrap_or_else(|_| vectors[ids[0]].clone())
}

fn closest(ids: &[usize], center: &[f32], vectors: &[Vec<f32>]) -> usize {
    ids.iter()
        .map(|&id| Neighbor {
            id,
            similarity: cosine(center, &vectors[id]),
        })
        .max()
        .expect("nonempty region")
        .id
}
