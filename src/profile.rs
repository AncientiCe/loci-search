use crate::{Query, index::Engine, ranking::SearchConfig};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::time::Instant;

#[derive(Default)]
pub(crate) struct Phases {
    pub rescue: f64,
    pub routing: f64,
    pub vectors: f64,
    pub postings: f64,
    pub selection: f64,
    pub rerank: f64,
}

pub(crate) struct Timer(Option<(&'static str, Instant)>);

impl Timer {
    pub(crate) fn start(profile: &Option<&mut Phases>, label: &'static str) -> Self {
        Self(profile.is_some().then(|| (label, Instant::now())))
    }

    pub(crate) fn finish(self, profile: &mut Option<&mut Phases>) {
        if let Some((label, start)) = self.0 {
            let profile = profile.as_deref_mut().expect("active profile");
            let field = match label {
                "rescue" => &mut profile.rescue,
                "routing" => &mut profile.routing,
                "vectors" => &mut profile.vectors,
                "postings" => &mut profile.postings,
                "selection" => &mut profile.selection,
                "rerank" => &mut profile.rerank,
                _ => unreachable!(),
            };
            *field += start.elapsed().as_secs_f64() * 1000.0;
        }
    }
}

#[derive(Serialize)]
pub struct Report {
    queries: usize,
    config: SearchConfig,
    preparation_ms: f64,
    posting_payload_bytes: usize,
    quantized_vector_bytes: usize,
    phases: Vec<Summary>,
    signatures: Vec<Signature>,
}

#[derive(Serialize)]
struct Summary {
    phase: &'static str,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
}

#[derive(Serialize)]
struct Signature {
    query: String,
    ranked: Vec<String>,
    rescue_ranks: Vec<Option<usize>>,
    route_regions: Vec<Vec<Vec<usize>>>,
}

pub fn regional(engine: &mut Engine, queries: &[Query], config: &SearchConfig) -> Result<Report> {
    ensure!(!queries.is_empty(), "queries are required");
    let started = Instant::now();
    engine.prepare_regional()?;
    let preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
    let posting_payload_bytes = engine.regional_index().payload_bytes();
    let quantized_vector_bytes = engine.regional_index().quantized_bytes();
    for query in queries.iter().take(32) {
        engine.search(query, crate::ranking::Strategy::Regional, config)?;
    }
    let mut samples: [Vec<f64>; 8] = Default::default();
    let mut signatures = Vec::new();
    for query in queries {
        let start = Instant::now();
        let mut profile = Phases::default();
        let result = engine.profile_regional(query, config, &mut profile)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let measured = profile.rescue
            + profile.routing
            + profile.vectors
            + profile.postings
            + profile.selection
            + profile.rerank;
        let values = [
            profile.rescue,
            profile.routing,
            profile.vectors,
            profile.postings,
            profile.selection,
            profile.rerank,
            (elapsed - measured).max(0.0),
            elapsed,
        ];
        for (series, sample) in samples.iter_mut().zip(values) {
            series.push(sample);
        }
        signatures.push(Signature {
            query: query.id.clone(),
            rescue_ranks: result
                .hits
                .iter()
                .map(|hit| hit.global_lexical_rank)
                .collect(),
            route_regions: result
                .hits
                .iter()
                .map(|hit| {
                    hit.routes
                        .iter()
                        .map(|route| route.path.iter().map(|step| step.region).collect())
                        .collect()
                })
                .collect(),
            ranked: result.hits.into_iter().map(|hit| hit.source.id).collect(),
        });
    }
    let names = [
        "global_lexical_rescue",
        "routing",
        "vectors_and_memberships",
        "regional_postings",
        "regional_top_k",
        "rerank_and_hits",
        "other",
        "total",
    ];
    let phases = names
        .into_iter()
        .zip(samples)
        .map(|(phase, mut values)| {
            values.sort_unstable_by(f64::total_cmp);
            let percentile = |p: f64| values[(values.len() as f64 * p).ceil() as usize - 1];
            Summary {
                phase,
                mean_ms: values.iter().sum::<f64>() / values.len() as f64,
                p50_ms: percentile(0.50),
                p95_ms: percentile(0.95),
            }
        })
        .collect();
    Ok(Report {
        queries: queries.len(),
        config: config.clone(),
        preparation_ms,
        posting_payload_bytes,
        quantized_vector_bytes,
        phases,
        signatures,
    })
}
