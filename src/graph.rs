use crate::{cosine, parallel_map};
use anyhow::{Result, ensure};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashSet},
    fs::File,
    io::{BufReader, BufWriter, ErrorKind, Read, Write},
    path::Path,
};

const FILE_MAGIC: &[u8; 8] = b"LOCIHNSW";
const FILE_VERSION: u32 = 1;
/// Nodes inserted one at a time before batched insertion starts.
const WARM_START: usize = 1024;
/// Upper bound on nodes inserted against the same read-only graph snapshot.
const MAX_BATCH: usize = 2048;

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
        for (id, level) in levels(vectors.len()).into_iter().enumerate() {
            graph.insert(id, level, vectors);
        }
        graph
    }

    /// Build with batched insertion, searching each batch in parallel.
    ///
    /// Every node in a batch searches the same read-only graph, so batch members do
    /// not link to each other directly; links are then applied in id order and
    /// overfull neighbor lists are pruned in parallel. The result depends only on
    /// the batch schedule, never on the number of threads.
    pub fn build_batched(vectors: &[Vec<f32>]) -> Self {
        let levels = levels(vectors.len());
        let mut graph = Self {
            links: Vec::with_capacity(vectors.len()),
            entry: None,
        };
        let warm = vectors.len().min(WARM_START);
        for (id, &level) in levels.iter().enumerate().take(warm) {
            graph.insert(id, level, vectors);
        }
        let mut start = warm;
        while start < vectors.len() {
            let end = (start + (start / 16).clamp(1, MAX_BATCH)).min(vectors.len());
            let ids = (start..end).collect::<Vec<_>>();
            let plans = parallel_map(&ids, |&id| graph.plan(levels[id], &vectors[id], vectors));
            let mut backlinks = BTreeMap::<(usize, usize), Vec<usize>>::new();
            for (&id, plan) in ids.iter().zip(plans) {
                let mut links = vec![Vec::new(); levels[id] + 1];
                for (layer, selected) in plan.into_iter().enumerate() {
                    for &target in &selected {
                        backlinks.entry((target, layer)).or_default().push(id);
                    }
                    links[layer] = selected;
                }
                graph.links.push(links);
            }
            for &id in &ids {
                let entry = graph.entry.expect("warm start sets an entry");
                if levels[id] + 1 > graph.links[entry].len() {
                    graph.entry = Some(id);
                }
            }
            let tasks = backlinks.into_iter().collect::<Vec<_>>();
            let updated = parallel_map(&tasks, |((target, layer), added)| {
                let degree = if *layer == 0 { 2 * DEGREE } else { DEGREE };
                let mut links = graph.links[*target][*layer].clone();
                links.extend(added);
                if links.len() > degree {
                    let mut candidates = links
                        .iter()
                        .map(|&other| Neighbor {
                            id: other,
                            similarity: cosine(&vectors[*target], &vectors[other]),
                        })
                        .collect::<Vec<_>>();
                    candidates.sort_unstable_by(|a, b| b.cmp(a));
                    links = diversify(candidates, degree, vectors);
                }
                links
            });
            for (((target, layer), _), links) in tasks.iter().zip(updated) {
                graph.links[*target][*layer] = links;
            }
            start = end;
        }
        graph
    }

    /// Neighbors a new node would link to on each of its layers, without mutating.
    fn plan(&self, level: usize, query: &[f32], vectors: &[Vec<f32>]) -> Vec<Vec<usize>> {
        let entry = self.entry.expect("planning needs an entry");
        let top = self.links[entry].len() - 1;
        let mut comparisons = 0;
        let mut cursor = entry;
        for layer in ((level + 1)..=top).rev() {
            cursor = self.layer(cursor, layer, 1, vectors, query, &mut comparisons)[0].id;
        }
        let mut plan = vec![Vec::new(); level.min(top) + 1];
        for layer in (0..=level.min(top)).rev() {
            let neighbors = self.layer(cursor, layer, BUILD_WIDTH, vectors, query, &mut comparisons);
            cursor = neighbors[0].id;
            let degree = if layer == 0 { 2 * DEGREE } else { DEGREE };
            plan[layer] = diversify(neighbors, degree, vectors);
        }
        plan
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn save(&self, path: &Path, manifest: &str) -> Result<()> {
        let temporary = path.with_extension("hnsw.tmp");
        let mut out = BufWriter::new(File::create(&temporary)?);
        out.write_all(FILE_MAGIC)?;
        out.write_all(&FILE_VERSION.to_le_bytes())?;
        out.write_all(&(manifest.len() as u64).to_le_bytes())?;
        out.write_all(manifest.as_bytes())?;
        out.write_all(&(self.links.len() as u64).to_le_bytes())?;
        out.write_all(&self.entry.map_or(u64::MAX, |id| id as u64).to_le_bytes())?;
        for layers in &self.links {
            out.write_all(&[u8::try_from(layers.len())?])?;
            for links in layers {
                out.write_all(&u16::try_from(links.len())?.to_le_bytes())?;
                for &id in links {
                    out.write_all(&u32::try_from(id)?.to_le_bytes())?;
                }
            }
        }
        out.into_inner()?.sync_all()?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Load a saved graph, or `None` when absent or built for a different index.
    pub fn load(path: &Path, manifest: &str) -> Result<Option<Self>> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut input = BufReader::new(file);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        ensure!(&magic == FILE_MAGIC, "{} is not a Loci graph", path.display());
        if u32::from_le_bytes(read_array(&mut input)?) != FILE_VERSION {
            return Ok(None);
        }
        let stored_len = usize::try_from(u64::from_le_bytes(read_array(&mut input)?))?;
        let mut stored = vec![0; stored_len];
        input.read_exact(&mut stored)?;
        if stored != manifest.as_bytes() {
            return Ok(None);
        }
        let nodes = usize::try_from(u64::from_le_bytes(read_array(&mut input)?))?;
        let entry = u64::from_le_bytes(read_array(&mut input)?);
        let entry = (entry != u64::MAX).then(|| usize::try_from(entry)).transpose()?;
        let mut links = Vec::with_capacity(nodes);
        for _ in 0..nodes {
            let [layers] = read_array::<1>(&mut input)?;
            let mut node = Vec::with_capacity(usize::from(layers));
            for _ in 0..layers {
                let count = usize::from(u16::from_le_bytes(read_array(&mut input)?));
                let mut layer = Vec::with_capacity(count);
                for _ in 0..count {
                    let id = usize::try_from(u32::from_le_bytes(read_array(&mut input)?))?;
                    ensure!(id < nodes, "graph link out of range");
                    layer.push(id);
                }
                node.push(layer);
            }
            links.push(node);
        }
        ensure!(
            entry.is_none_or(|id| id < nodes),
            "graph entry out of range"
        );
        Ok(Some(Self { links, entry }))
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

/// Deterministic node levels shared by the sequential and batched builds.
fn levels(nodes: usize) -> Vec<usize> {
    let mut seed = 0x4c6f_6369_5345_4152_u64;
    (0..nodes)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed.trailing_zeros() as usize / 3).min(MAX_LEVEL)
        })
        .collect()
}

fn read_array<const N: usize>(input: &mut impl Read) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vectors(count: usize, dimensions: usize, mut seed: u64) -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                let vector = (0..dimensions)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .collect::<Vec<_>>();
                let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                vector.into_iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn batched_build_finds_most_exact_neighbors_and_round_trips() -> Result<()> {
        let vectors = unit_vectors(6000, 32, 0x9e37_79b9_7f4a_7c15);
        let queries = unit_vectors(40, 32, 0xd1b5_4a32_d192_ed03);
        let graph = Graph::build_batched(&vectors);
        assert_eq!(graph.len(), vectors.len());
        let mut found = 0;
        for query in &queries {
            let mut exact = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| Neighbor {
                    id,
                    similarity: cosine(query, vector),
                })
                .collect::<Vec<_>>();
            exact.sort_unstable_by(|a, b| b.cmp(a));
            let truth = exact[..10].iter().map(|hit| hit.id).collect::<HashSet<_>>();
            let (hits, _) = graph.search(&vectors, query, 10, 128);
            found += hits.iter().filter(|hit| truth.contains(&hit.id)).count();
        }
        let recall = found as f64 / (10 * queries.len()) as f64;
        assert!(recall > 0.9, "batched graph recall@10 was {recall}");

        let directory = tempfile::tempdir()?;
        let path = directory.path().join("index.sqlite.hnsw");
        graph.save(&path, "manifest-a")?;
        let loaded = Graph::load(&path, "manifest-a")?.expect("matching manifest loads");
        assert_eq!(loaded.links, graph.links);
        assert_eq!(loaded.entry, graph.entry);
        assert!(Graph::load(&path, "manifest-b")?.is_none());
        assert!(Graph::load(&directory.path().join("missing.hnsw"), "manifest-a")?.is_none());
        Ok(())
    }
}
