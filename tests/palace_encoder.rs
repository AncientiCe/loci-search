//! MiniLM int8 cosine check against the archived exchange dump.
//! Skips when the pinned ONNX files or the dump are absent.

use loci_search::palace::{MiniLm, pinned_minilm};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CHUNKS: &str = r"D:\Dev\Projects\mempalace-rs\data\out\longmemeval\recall_all_research\minilm-l6-v2-q8-exchange\longmemeval_cases.jsonl";

#[derive(Deserialize)]
struct DumpCase {
    question_id: String,
    chunks: Vec<DumpChunk>,
}

#[derive(Deserialize)]
struct DumpChunk {
    text: String,
    cosine: f64,
}

#[test]
fn pinned_minilm_matches_dumped_cosines_with_ort_rc11_sidecar() {
    let (model, vocab) = pinned_minilm();
    if !model.exists() || !vocab.exists() || !Path::new(CHUNKS).exists() {
        eprintln!("pinned MiniLM or the exchange dump is absent; skipping the cosine sample");
        return;
    }
    let mut lines = BufReader::new(File::open(CHUNKS).unwrap()).lines();
    let case: DumpCase = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(case.question_id, "e47becba");
    let question = "What degree did I graduate with?";
    let sample: Vec<&DumpChunk> = case
        .chunks
        .iter()
        .filter(|chunk| chunk.cosine > 0.0)
        .take(6)
        .collect();
    assert!(sample.len() >= 4, "need several nonzero dumped cosines");
    let mut texts = vec![question];
    texts.extend(sample.iter().map(|chunk| chunk.text.as_str()));
    let encoder = MiniLm::open(&model, &vocab, 2).unwrap();
    let vectors = encoder.embed(&texts).unwrap();
    assert!(vectors.iter().all(|vector| vector.len() == 384));
    let query = &vectors[0];
    for (chunk, vector) in sample.iter().zip(vectors.iter().skip(1)) {
        let dot = query
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let rounded = (f64::from(dot) * 1000.0).round() / 1000.0;
        assert!(
            (rounded - chunk.cosine).abs() < 1e-9,
            "cosine {rounded} != dumped {} for ort rc.15",
            chunk.cosine
        );
    }
    eprintln!(
        "ort rc.11 sidecar matched {} dumped MiniLM cosines for {}",
        sample.len(),
        case.question_id
    );
}
