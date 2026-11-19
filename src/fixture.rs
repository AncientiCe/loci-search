use crate::{Embedding, Normalization, Passage, Query};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

pub const MIN_CHUNKS: usize = 64;
const DIMENSIONS: usize = 64;
const TOPICS: [&str; 16] = [
    "runtime executor tasks",
    "ownership lifetimes borrowing",
    "storage checkpoint disk",
    "replication quorum consensus",
    "stream cursor offsets",
    "network gateway transport",
    "billing invoice refunds",
    "identity credentials login",
    "metrics dashboard telemetry",
    "jobs scheduler workers",
    "backup snapshot recovery",
    "compiler optimization inference",
    "search indexing tokenizer",
    "cache eviction expiry",
    "deployment rollout releases",
    "database migration schema",
];
const DISTRACTORS: [&str; 8] = [
    "routine review completed",
    "operational baseline recorded",
    "retries under ordinary load",
    "inspect the service configuration",
    "diagnostic notes for operators",
    "recorded test environment",
    "verify status and latency",
    "follow up after maintenance",
];

pub fn generate(chunks: usize) -> Result<(Embedding, Vec<Passage>, Vec<Query>)> {
    ensure!(
        chunks >= MIN_CHUNKS,
        "fixture requires at least {MIN_CHUNKS} chunks"
    );
    let embedding = Embedding {
        model_revision: "loci-synthetic-v1".into(),
        dimensions: DIMENSIONS,
        normalization: Normalization::L2,
    };
    let mut passages = Vec::with_capacity(chunks);
    let mut queries = Vec::new();
    let mut rng = Rng(0x447c_a372_364b_5c0a);
    for index in 0..8 {
        let topic = index * 2;
        let other = (topic + 5) % TOPICS.len();
        let misplaced = (topic + 8) % TOPICS.len();
        let incident = format!("NOVA{index:03}X");
        let label = format!("witness{index:03}");
        let a = rng.signature(topic, index);
        let b = rng.signature(other, index);
        let between = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a + b) * 0.5)
            .collect::<Vec<_>>();
        let texts = [
            format!(
                "{incident} means an expired lease after a gateway rotation. Rotate the lease holder."
            ),
            format!(
                "{} {label} evidence is pinned until the outstanding operation is released.",
                TOPICS[topic]
            ),
            format!(
                "{} {label} correlates a stalled operation with a retained handle.",
                TOPICS[other]
            ),
            format!(
                "{} {} ownership crosses the integration boundary during {label}.",
                TOPICS[topic], TOPICS[other]
            ),
        ];
        let first = passages.len();
        for (offset, (text, vector)) in texts
            .into_iter()
            .zip([rng.topic(misplaced), a.clone(), b.clone(), between.clone()])
            .enumerate()
        {
            passages.push(passage(first + offset, text, vector));
        }
        let examples = [
            (
                "identifier",
                incident,
                a.clone(),
                vec![(first, 3)],
                Vec::new(),
            ),
            (
                "paraphrase",
                "What keeps proof around while work remains active?".into(),
                a,
                vec![(first + 1, 3)],
                Vec::new(),
            ),
            (
                "cross_domain",
                format!("{} {} integration boundary", TOPICS[topic], TOPICS[other]),
                between.clone(),
                vec![(first + 3, 3), (first + 1, 1), (first + 2, 1)],
                Vec::new(),
            ),
            (
                "multi_hop",
                format!("{label} retained handle outstanding operation"),
                between,
                vec![(first + 1, 3), (first + 2, 3), (first + 3, 1)],
                vec![first + 1, first + 2],
            ),
        ];
        for (group, text, vector, grades, required) in examples {
            queries.push(Query {
                id: format!("{group}-{index:02}"),
                text,
                vector,
                embedding: embedding.clone(),
                group: group.into(),
                judgments: grades
                    .into_iter()
                    .map(|(id, grade)| (chunk_id(id), grade))
                    .collect(),
                required: required.into_iter().map(chunk_id).collect(),
            });
        }
    }
    for index in passages.len()..chunks {
        let topic = rng.next() as usize % TOPICS.len();
        let activity = DISTRACTORS[rng.next() as usize % DISTRACTORS.len()];
        let text = format!("{} case {index}: {activity}.", TOPICS[topic]);
        passages.push(passage(index, text, rng.topic(topic)));
    }
    Ok((embedding, passages, queries))
}

fn passage(index: usize, text: String, vector: Vec<f32>) -> Passage {
    Passage {
        id: chunk_id(index),
        document: format!("fixture/{index:09}.txt"),
        revision: "1".into(),
        start: 0,
        end: text.len() as u64,
        text,
        metadata: BTreeMap::new(),
        vector,
    }
}

fn chunk_id(id: usize) -> String {
    format!("chunk-{id:09}")
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn topic(&mut self, topic: usize) -> Vec<f32> {
        let mut vector = (0..DIMENSIONS)
            .map(|_| ((self.next() >> 40) as f32 / 16_777_216.0 - 0.5) * 0.18)
            .collect::<Vec<_>>();
        vector[topic] += 1.4;
        vector
    }

    fn signature(&mut self, topic: usize, example: usize) -> Vec<f32> {
        let mut vector = self.topic(topic);
        vector[TOPICS.len() + example] += 0.82;
        vector
    }
}
