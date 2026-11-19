pub mod bench;
pub mod fixture;
mod graph;
pub mod index;
pub mod ranking;
mod topology;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Embedding {
    pub model_revision: String,
    pub dimensions: usize,
    pub normalization: Normalization,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    L2,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub document: String,
    pub revision: String,
    pub start: u64,
    pub end: u64,
    pub text: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub vector: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub embedding: Embedding,
    pub vector: Vec<f32>,
    #[serde(default = "default_group")]
    pub group: String,
    #[serde(default)]
    pub judgments: BTreeMap<String, u8>,
    #[serde(default)]
    pub required: Vec<String>,
}

fn default_group() -> String {
    "general".into()
}

fn normalize(vector: &[f32], dimensions: usize) -> Result<Vec<f32>> {
    ensure!(
        dimensions > 0 && vector.len() == dimensions,
        "vector dimension mismatch"
    );
    let norm = vector
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "vector must be finite and nonzero"
    );
    Ok(vector
        .iter()
        .map(|&value| (f64::from(value) / norm) as f32)
        .collect())
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn terms(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
}
