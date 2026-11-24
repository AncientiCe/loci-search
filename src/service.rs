mod files;
mod semantic;

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use clap::ValueEnum;
pub use files::{backup, import, restore};
use loci_search::embedding::ModelPack;
use loci_search::store::{Catalog, Mutation, SearchRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    net::SocketAddr,
    path::Path as FsPath,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::Semaphore;

const BODY_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Credential {
    token: String,
    indexes: Vec<String>,
    #[serde(default)]
    write: bool,
    #[serde(default)]
    principals: Vec<String>,
}

#[derive(Clone)]
struct App(Arc<Inner>);

struct Inner {
    catalog: Catalog,
    semantic: semantic::RegionalService,
    credentials: Option<Vec<Credential>>,
    workers: Arc<Semaphore>,
    searches: AtomicU64,
    writes: AtomicU64,
}

struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

fn failure(error: anyhow::Error) -> ApiError {
    let msg = error.to_string();
    if msg.contains("stale generation")
        || msg.contains("stale cursor")
        || msg.contains("conflicting settings")
        || msg.contains("conflicting content")
        || msg.contains("already used for a different batch")
    {
        ApiError(StatusCode::CONFLICT, msg)
    } else if msg.contains("unknown index") || msg.contains("index does not exist") {
        ApiError(StatusCode::NOT_FOUND, msg)
    } else if error.downcast_ref::<rusqlite::Error>().is_some()
        || error.downcast_ref::<std::io::Error>().is_some()
    {
        eprintln!("Loci storage or model operation failed: {error:#}");
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage or model operation failed".into(),
        )
    } else {
        invalid(msg)
    }
}

impl App {
    fn identity<'a>(&'a self, headers: &HeaderMap) -> Result<Option<&'a Credential>, ApiError> {
        let Some(keys) = &self.0.credentials else {
            return Ok(None);
        };
        let bearer = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        keys.iter()
            .find(|key| Some(key.token.as_str()) == bearer)
            .map(Some)
            .ok_or_else(|| {
                ApiError(
                    StatusCode::UNAUTHORIZED,
                    "valid bearer credential required".into(),
                )
            })
    }

    fn authorize(
        &self,
        headers: &HeaderMap,
        index: &str,
        write: bool,
    ) -> Result<Vec<String>, ApiError> {
        if !valid_index(index) {
            return Err(invalid("invalid index name"));
        }
        let Some(key) = self.identity(headers)? else {
            return Ok(Vec::new());
        };
        if (write && !key.write) || !key.indexes.iter().any(|v| v == "*" || v == index) {
            return Err(ApiError(StatusCode::FORBIDDEN, "insufficient scope".into()));
        }
        Ok(key.principals.clone())
    }

    async fn blocking<F>(&self, f: F) -> Result<Json<Value>, ApiError>
    where
        F: FnOnce(&Inner) -> Result<Value> + Send + 'static,
    {
        let permit =
            self.0.workers.clone().try_acquire_owned().map_err(|_| {
                ApiError(StatusCode::TOO_MANY_REQUESTS, "worker queue is full".into())
            })?;
        let state = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            f(&state)
        })
        .await
        .map_err(|_| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "worker stopped".into()))?
        .map(Json)
        .map_err(failure)
    }
}

pub(super) fn valid_index(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

pub fn serve(bind: SocketAddr, data: &FsPath, credentials: Option<&FsPath>) -> Result<()> {
    listen(bind, data, credentials, None)
}

fn listen(
    bind: SocketAddr,
    data: &FsPath,
    credentials: Option<&FsPath>,
    bound: Option<std::sync::mpsc::Sender<SocketAddr>>,
) -> Result<()> {
    ensure!(
        bind.ip().is_loopback() || credentials.is_some(),
        "external binds require --credentials"
    );
    let credentials: Option<Vec<Credential>> = credentials
        .map(|p| -> Result<_> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?;
    if let Some(keys) = &credentials {
        let mut unique = std::collections::HashSet::new();
        ensure!(
            !keys.is_empty()
                && keys.iter().all(|key| key.token.len() >= 24
                    && unique.insert(key.token.as_str())
                    && !key.indexes.is_empty()
                    && key.indexes.iter().all(|i| i == "*" || valid_index(i))),
            "invalid credential configuration"
        );
    }
    fs::create_dir_all(data)?;
    let state = App(Arc::new(Inner {
        catalog: Catalog::open(data.join("catalog.sqlite"))?,
        semantic: semantic::RegionalService::new(&data.join("models"))?,
        credentials,
        workers: Arc::new(Semaphore::new(
            std::thread::available_parallelism()
                .map_or(4, usize::from)
                .clamp(4, 32),
        )),
        searches: AtomicU64::new(0),
        writes: AtomicU64::new(0),
    }));
    let routes = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("service/console.html")) }),
        )
        .route(
            "/openapi.json",
            get(|| async {
                (
                    [("content-type", "application/json")],
                    include_str!("../docs/openapi.json"),
                )
            }),
        )
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/metrics", get(metrics))
        .route("/v1/indexes", get(indexes))
        .route("/v1/indexes/{name}", axum::routing::put(create))
        .route("/v1/indexes/{name}/bulk", post(bulk))
        .route("/v1/indexes/{name}/search", post(search))
        .route("/v1/indexes/{name}/suggest", post(suggest))
        .route("/v1/indexes/{name}/expire", post(expire))
        .route("/v1/indexes/{name}/partitions", get(partitions))
        .route("/v1/indexes/{name}/model", axum::routing::put(upgrade))
        .route("/v1/indexes/{name}/documents/{id}", get(document))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state.clone());
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let listener = tokio::net::TcpListener::bind(bind).await?;
            let addr = listener.local_addr()?;
            eprintln!("Listening on http://{addr}");
            if let Some(bound) = bound {
                let _ = bound.send(addr);
            }
            tokio::spawn(async move {
                let mut timer = tokio::time::interval(std::time::Duration::from_secs(60));
                timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    timer.tick().await;
                    let app = state.0.clone();
                    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?
                            .as_secs();
                        app.catalog.maintain(i64::try_from(now)?)?;
                        Ok(())
                    })
                    .await;
                    if let Err(error) = result.unwrap_or_else(|error| Err(error.into())) {
                        eprintln!("catalog maintenance: {error}");
                    }
                }
            });
            axum::serve(listener, routes).await?;
            Ok(())
        })
}

type Reply = Result<Json<Value>, ApiError>;

async fn metrics(State(app): State<App>, headers: HeaderMap) -> Reply {
    if let Some(key) = app.identity(&headers)?
        && !(key.write && key.indexes.iter().any(|i| i == "*"))
    {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "server-wide metrics require administrative scope".into(),
        ));
    }
    Ok(Json(
        json!({"searches":app.0.searches.load(Ordering::Relaxed),"bulk_requests":app.0.writes.load(Ordering::Relaxed)}),
    ))
}

async fn indexes(State(app): State<App>, headers: HeaderMap) -> Reply {
    let names = app.identity(&headers)?.map(|k| k.indexes.clone());
    app.blocking(move |s| {
        let mut indexes = s.catalog.list_indexes()?;
        if let Some(names) = names
            && !names.iter().any(|i| i == "*")
        {
            indexes.retain(|i| names.contains(i));
        }
        Ok(json!({"indexes":indexes}))
    })
    .await
}

async fn create(
    State(app): State<App>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(settings): Json<Value>,
) -> Reply {
    app.authorize(&headers, &name, true)?;
    app.blocking(move |s| {
        semantic::validate_settings(&settings)?;
        let created = s.catalog.create_index_with_settings(&name, &settings)?;
        Ok(json!({"name":name,"created":created}))
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Bulk {
    operations: Vec<Mutation>,
}

async fn bulk(
    State(app): State<App>,
    Path(name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Reply {
    app.authorize(&headers, &name, true)?;
    let key = headers
        .get("idempotency-key")
        .map(|v| v.to_str().map(String::from))
        .transpose()
        .map_err(|_| invalid("invalid idempotency key"))?;
    let ndjson = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(';').next() == Some("application/x-ndjson"));
    app.blocking(move |s| {
        let mutations = if ndjson {
            std::str::from_utf8(&body)?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<serde_json::Result<Vec<Mutation>>>()?
        } else {
            serde_json::from_slice::<Bulk>(&body)?.operations
        };
        ensure!(mutations.len() <= 10_000, "bulk limit is 10000 operations");
        let settings = s.catalog.snapshot(&name)?.settings()?;
        let result = if let Some(model) = settings
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| *model != "lexical")
        {
            let pack = ModelPack::from_str(model, false).map_err(anyhow::Error::msg)?;
            let prepared = s.semantic.encode_changes(pack, &mutations)?;
            s.catalog.bulk_encoded(
                &name,
                &mutations,
                key.as_deref(),
                model,
                &pack.manifest(),
                &prepared,
            )?
        } else {
            s.catalog.bulk_lexical(&name, &mutations, key.as_deref())?
        };
        s.writes.fetch_add(1, Ordering::Relaxed);
        Ok(serde_json::to_value(result)?)
    })
    .await
}

async fn document(
    State(app): State<App>,
    Path((name, id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Reply {
    let principals = app.authorize(&headers, &name, false)?;
    let value = app
        .blocking(move |s| {
            Ok(serde_json::to_value(
                s.catalog.snapshot(&name)?.document(&id, &principals)?,
            )?)
        })
        .await?;
    if value.0.is_null() {
        return Err(ApiError(StatusCode::NOT_FOUND, "document not found".into()));
    }
    Ok(value)
}

async fn search(
    State(app): State<App>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Reply {
    let principals = app.authorize(&headers, &name, false)?;
    app.blocking(move |s| {
        let obj = body
            .as_object_mut()
            .context("search body must be an object")?;
        let mode = obj
            .remove("mode")
            .map(serde_json::from_value::<String>)
            .transpose()?;
        let generation = obj
            .remove("generation")
            .map(serde_json::from_value::<u64>)
            .transpose()?;
        let snap = match generation {
            Some(g) => s.catalog.snapshot_at(&name, g)?,
            None => s.catalog.snapshot(&name)?,
        };
        let settings = snap.settings()?;
        let semantic = mode.as_deref() == Some("regional")
            || (mode.is_none()
                && settings
                    .get("model")
                    .and_then(Value::as_str)
                    .is_some_and(|m| m != "lexical")
                && obj.contains_key("text")
                && obj
                    .keys()
                    .all(|k| matches!(k.as_str(), "text" | "limit" | "filter" | "budget")));
        let result = if semantic {
            s.semantic
                .search(&name, &snap, &settings, body, &principals)?
        } else {
            ensure!(
                mode.is_none() || mode.as_deref() == Some("exact"),
                "unsupported search mode"
            );
            let request: SearchRequest = serde_json::from_value(body)?;
            drop(snap);
            if let Some(generation) = generation {
                let pinned = s.catalog.snapshot_at(&name, generation)?;
                serde_json::to_value(pinned.search(&request, &principals)?)?
            } else {
                serde_json::to_value(s.catalog.search(&name, &request, &principals)?)?
            }
        };
        s.searches.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expire {
    now: i64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Upgrade {
    model: String,
}

async fn upgrade(
    State(app): State<App>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(req): Json<Upgrade>,
) -> Reply {
    app.authorize(&headers, &name, true)?;
    app.blocking(move |s| {
        let pack = ModelPack::from_str(&req.model, false).map_err(anyhow::Error::msg)?;
        let snap = s.catalog.snapshot(&name)?;
        let generation = snap.generation;
        let docs = snap.documents_for_model_upgrade()?;
        drop(snap);
        let prepared = s.semantic.prepare(pack, &docs)?;
        let next =
            s.catalog
                .publish_model(&name, generation, pack.id(), &pack.manifest(), &prepared)?;
        Ok(json!({"generation": next, "model": pack.id(), "documents": docs.len()}))
    })
    .await
}
async fn expire(
    State(app): State<App>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Expire>,
) -> Reply {
    app.authorize(&headers, &name, true)?;
    app.blocking(move |s| {
        Ok(serde_json::to_value(
            s.catalog.expire_stream(&name, body.now)?,
        )?)
    })
    .await
}
async fn partitions(State(app): State<App>, Path(name): Path<String>, headers: HeaderMap) -> Reply {
    app.authorize(&headers, &name, true)?;
    app.blocking(move |s| Ok(json!({"partitions":s.catalog.stream_partitions(&name)?})))
        .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suggest {
    prefix: String,
    #[serde(default = "suggest_limit")]
    limit: usize,
}
fn suggest_limit() -> usize {
    10
}

async fn suggest(
    State(app): State<App>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(req): Json<Suggest>,
) -> Reply {
    let principals = app.authorize(&headers, &name, false)?;
    app.blocking(move |s| {
        let snap=s.catalog.snapshot(&name)?;
        Ok(json!({"generation":snap.generation,"suggestions":snap.suggest(&req.prefix,req.limit,&principals)?}))
    }).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::TransactionBehavior;
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    fn code(result: std::result::Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> u16 {
        match result {
            Ok(response) => response.status().as_u16(),
            Err(ureq::Error::StatusCode(status)) => status,
            Err(error) => panic!("{error}"),
        }
    }

    #[test]
    fn a_full_worker_queue_rejects_the_overflow_and_keeps_the_document() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().to_path_buf();
        let (sender, receiver) = mpsc::channel();
        let data_for_server = data.clone();
        std::thread::spawn(move || {
            listen(
                "127.0.0.1:0".parse().unwrap(),
                &data_for_server,
                None,
                Some(sender),
            )
            .unwrap();
        });
        let addr = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        let base = format!("http://{addr}");
        let agent = ureq::Agent::new_with_defaults();
        assert_eq!(code(agent.get(format!("{base}/health")).call()), 200);
        assert_eq!(
            code(
                agent
                    .put(format!("{base}/v1/indexes/logs"))
                    .send_json(json!({"model": "lexical"}))
            ),
            200
        );
        assert_eq!(
            code(agent
                .post(format!("{base}/v1/indexes/logs/bulk"))
                .send_json(json!({"operations":[{"op":"upsert","document":{"id":"kept","revision":"1","text":"kept event","fields":{"time":1}}}]}))),
            200
        );

        let mut db = rusqlite::Connection::open(data.join("catalog.sqlite")).unwrap();
        let held = db
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        let statuses = Arc::new(Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            for _ in 0..64 {
                let statuses = Arc::clone(&statuses);
                let url = format!("{base}/v1/indexes/logs/search");
                scope.spawn(move || {
                    let status = code(
                        ureq::Agent::new_with_defaults()
                            .post(&url)
                            .send_json(json!({"text": "kept", "limit": 1})),
                    );
                    statuses.lock().unwrap().push(status);
                });
            }
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline && !statuses.lock().unwrap().contains(&429) {
                std::thread::sleep(Duration::from_millis(20));
            }
            drop(held);
        });
        let statuses = statuses.lock().unwrap().clone();
        assert!(
            statuses.contains(&429),
            "expected a rejected overflow, saw {statuses:?}"
        );

        let mut response = agent
            .post(format!("{base}/v1/indexes/logs/search"))
            .send_json(json!({"text": "kept", "limit": 1}))
            .unwrap();
        let page: Value = response.body_mut().read_json().unwrap();
        assert_eq!(page["total"], json!(1));
        assert_eq!(page["hits"][0]["id"], json!("kept"));
    }
}
