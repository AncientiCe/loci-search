use crate::cosine;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

pub const DEGREE: usize = 16;
pub const BUILD_WIDTH: usize = 80;
pub const MAX_LEVEL: usize = 12;

#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub id: usize,
    pub similarity: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then_with(|| other.id.cmp(&self.id))
    }
}

pub struct Graph {
    links: Vec<Vec<Vec<usize>>>,
    entry: Option<usize>,
}

impl Graph {
    pub fn build(vectors: &[Vec<f32>]) -> Self {
        let mut graph = Self {
            links: Vec::with_capacity(vectors.len()),
            entry: None,
        };
        let mut seed = 0x4c6f_6369_5345_4152_u64;
        for id in 0..vectors.len() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let level = (seed.trailing_zeros() as usize / 3).min(MAX_LEVEL);
            graph.insert(id, level, vectors);
        }
        graph
    }

    fn insert(&mut self, id: usize, level: usize, vectors: &[Vec<f32>]) {
        self.links.push(vec![Vec::new(); level + 1]);
        let Some(entry) = self.entry else {
            self.entry = Some(id);
            return;
        };
        let top = self.links[entry].len() - 1;
        let mut comparisons = 0;
        let mut cursor = entry;
        for layer in ((level + 1)..=top).rev() {
            cursor = self.layer(cursor, layer, 1, vectors, &vectors[id], &mut comparisons)[0].id;
        }
        for layer in (0..=level.min(top)).rev() {
            let neighbors = self.layer(
                cursor,
                layer,
                BUILD_WIDTH,
                vectors,
                &vectors[id],
                &mut comparisons,
            );
            cursor = neighbors[0].id;
            let degree = if layer == 0 { 2 * DEGREE } else { DEGREE };
            let selected = diversify(neighbors, degree, vectors);
            for &neighbor in &selected {
                let links = &mut self.links[neighbor][layer];
                links.push(id);
                if links.len() > degree {
                    let mut candidates = links
                        .iter()
                        .map(|&target| Neighbor {
                            id: target,
                            similarity: cosine(&vectors[neighbor], &vectors[target]),
                        })
                        .collect::<Vec<_>>();
                    candidates.sort_unstable_by(|a, b| b.cmp(a));
                    *links = diversify(candidates, degree, vectors);
                }
            }
            self.links[id][layer] = selected;
        }
        if level > top {
            self.entry = Some(id);
        }
    }

    pub fn search(
        &self,
        vectors: &[Vec<f32>],
        query: &[f32],
        depth: usize,
        width: usize,
    ) -> (Vec<Neighbor>, usize) {
        let Some(mut cursor) = self.entry else {
            return (Vec::new(), 0);
        };
        let mut comparisons = 0;
        for layer in (1..self.links[cursor].len()).rev() {
            cursor = self.layer(cursor, layer, 1, vectors, query, &mut comparisons)[0].id;
        }
        let mut result = self.layer(
            cursor,
            0,
            width.max(depth),
            vectors,
            query,
            &mut comparisons,
        );
        result.truncate(depth);
        (result, comparisons)
    }

    fn layer(
        &self,
        entry: usize,
        layer: usize,
        width: usize,
        vectors: &[Vec<f32>],
        query: &[f32],
        comparisons: &mut usize,
    ) -> Vec<Neighbor> {
        let start = Neighbor {
            id: entry,
            similarity: cosine(query, &vectors[entry]),
        };
        *comparisons += 1;
        let mut pending = BinaryHeap::from([start]);
        let mut best = BinaryHeap::from([Reverse(start)]);
        let mut visited = HashSet::from([entry]);
        while let Some(current) = pending.pop() {
            if best.len() >= width && current < best.peek().expect("entry").0 {
                break;
            }
            for &id in &self.links[current.id][layer] {
                if !visited.insert(id) {
                    continue;
                }
                let next = Neighbor {
                    id,
                    similarity: cosine(query, &vectors[id]),
                };
                *comparisons += 1;
                if best.len() < width || next > best.peek().expect("entry").0 {
                    pending.push(next);
                    best.push(Reverse(next));
                    if best.len() > width {
                        best.pop();
                    }
                }
            }
        }
        best.into_sorted_vec()
            .into_iter()
            .map(|item| item.0)
            .collect()
    }
}

fn diversify(candidates: Vec<Neighbor>, degree: usize, vectors: &[Vec<f32>]) -> Vec<usize> {
    let mut chosen = Vec::<usize>::with_capacity(degree);
    let mut pruned = Vec::new();
    for candidate in candidates {
        if chosen.len() == degree {
            break;
        }
        if chosen
            .iter()
            .all(|&id| cosine(&vectors[id], &vectors[candidate.id]) < candidate.similarity)
        {
            chosen.push(candidate.id);
        } else {
            pruned.push(candidate.id);
        }
    }
    let remaining = degree - chosen.len();
    chosen.extend(pruned.into_iter().take(remaining));
    chosen
}
