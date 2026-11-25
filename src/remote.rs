//! Benchmark transport: regional search over loopback HTTP.
//!
//! `loci bench-serve` holds one prepared engine and answers `POST /search`.
//! The benchmark client sends the query text and vector, exactly as it does for
//! Elasticsearch, and materializes the returned passages from its own index, as
//! the Elasticsearch path does. Both engines then pay an HTTP and JSON round trip.

use crate::{
    Query,
    index::Engine,
    ranking::{Hit, SearchConfig, SearchResult, Strategy, Work},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use ureq::{Agent, http::Uri};

#[derive(Serialize, Deserialize)]
struct Request {
    text: String,
    vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Response {
    hits: Vec<(String, f64)>,
    candidates: Vec<String>,
}

/// Serve regional search with `config` on `bind` until the process stops.
pub fn serve(mut engine: Engine, config: SearchConfig, bind: &str) -> Result<()> {
    engine.prepare(Strategy::Regional, &config)?;
    let state = Arc::new((engine, config));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let app = axum::Router::new()
            .route(
                "/search",
                axum::routing::post(
                    |axum::extract::State(state): axum::extract::State<
                        Arc<(Engine, SearchConfig)>,
                    >,
                     axum::Json(request): axum::Json<Request>| async move {
                        let answer = tokio::task::spawn_blocking(move || -> Result<Response> {
                            let (engine, config) = &*state;
                            let query = Query {
                                id: "http".into(),
                                text: request.text,
                                embedding: engine.manifest.embedding.clone(),
                                vector: request.vector,
                                group: "general".into(),
                                judgments: Default::default(),
                                required: Vec::new(),
                            };
                            let result =
                                engine.search_prepared(&query, Strategy::Regional, config)?;
                            Ok(Response {
                                hits: result
                                    .hits
                                    .into_iter()
                                    .map(|hit| (hit.source.id, hit.score))
                                    .collect(),
                                candidates: result.candidates,
                            })
                        })
                        .await;
                        match answer {
                            Ok(Ok(response)) => Ok(axum::Json(response)),
                            Ok(Err(error)) => {
                                Err((axum::http::StatusCode::BAD_REQUEST, format!("{error:#}")))
                            }
                            Err(error) => Err((
                                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                                error.to_string(),
                            )),
                        }
                    },
                ),
            )
            .with_state(state);
        let listener = tokio::net::TcpListener::bind(bind).await?;
        eprintln!("Serving regional search on {bind}");
        axum::serve(listener, app).await?;
        Ok(())
    })
}

pub struct Client {
    agent: Agent,
    url: String,
}

impl Client {
    pub fn new(base: &str) -> Result<Self> {
        let uri: Uri = base.parse()?;
        ensure!(
            uri.scheme_str() == Some("http")
                && matches!(uri.host(), Some("127.0.0.1" | "localhost" | "[::1]"))
                && uri.path() == "/"
                && uri.query().is_none(),
            "the benchmark requires a Loci endpoint on loopback"
        );
        let agent = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(60)))
            .proxy(None)
            .build()
            .into();
        Ok(Self {
            agent,
            url: format!("{}/search", base.trim_end_matches('/')),
        })
    }

    pub(crate) fn search(&self, engine: &Engine, query: &Query) -> Result<SearchResult> {
        let vector = engine.query_vector(query)?;
        let response: Response = self
            .agent
            .post(&self.url)
            .send_json(Request {
                text: query.text.clone(),
                vector,
            })
            .with_context(|| format!("Loci query {}", query.id))?
            .body_mut()
            .read_json()?;
        let hits = response
            .hits
            .into_iter()
            .map(|(id, score)| {
                let row = engine
                    .ids
                    .binary_search(&id)
                    .map_err(|_| anyhow::anyhow!("Loci returned unknown passage {id}"))?;
                Ok(Hit {
                    source: engine.source(row)?,
                    score,
                    lexical_rank: None,
                    vector_rank: None,
                    global_lexical_rank: None,
                    graph_rank: None,
                    routes: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SearchResult {
            strategy: Strategy::Regional,
            hits,
            candidates: response.candidates,
            lexical_rescue_only_candidates: Vec::new(),
            expanded: Vec::new(),
            stop_reason: "remote",
            work: Work::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fixture, index::ingest};

    #[test]
    fn http_search_returns_the_in_process_hits_and_scores() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let db = directory.path().join("remote.sqlite");
        let (embedding, passages, queries) = fixture::generate(256)?;
        ingest(&db, passages, embedding, 16)?;
        let config = SearchConfig::default();
        let port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let served = Engine::open(&db)?;
        let server_config = config.clone();
        std::thread::spawn(move || serve(served, server_config, &format!("127.0.0.1:{port}")));
        let client = Client::new(&format!("http://127.0.0.1:{port}/"))?;
        let mut engine = Engine::open(&db)?;
        engine.prepare(Strategy::Regional, &config)?;
        let mut ready = false;
        for _ in 0..200 {
            if client.search(&engine, &queries[0]).is_ok() {
                ready = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(ready, "bench server did not start");
        // JSON may move a score by its last bit; order and ids must match exactly.
        let hits = |result: SearchResult| {
            result
                .hits
                .into_iter()
                .map(|hit| (hit.source.id, hit.score))
                .collect::<Vec<_>>()
        };
        for query in queries.iter().take(8) {
            let local = engine.search_prepared(query, Strategy::Regional, &config)?;
            let remote = client.search(&engine, query)?;
            assert_eq!(remote.candidates, local.candidates);
            let (remote, local) = (hits(remote), hits(local));
            assert_eq!(remote.len(), local.len());
            for ((remote_id, remote_score), (local_id, local_score)) in remote.iter().zip(&local) {
                assert_eq!(remote_id, local_id);
                assert!((remote_score - local_score).abs() < 1e-12);
            }
        }
        Ok(())
    }
}
