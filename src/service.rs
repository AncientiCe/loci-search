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
use loci_search::generation::{self, TextChange, TextUpdate};
use loci_search::ranking::Strategy;
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
    data: std::path::PathBuf,
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
    } else if msg.contains("unknown index")
        || msg.contains("index does not exist")
        || msg.contains("no published generation")
    {
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
    let data = data.to_path_buf();
    fs::create_dir_all(&data)?;
    let state = App(Arc::new(Inner {
        catalog: Catalog::open(data.join("catalog.sqlite"))?,
        semantic: semantic::RegionalService::new(&data.join("models"))?,
        data,
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
        .route("/v1/indexes/{name}/rebuild", post(rebuild))
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
        let mut settings = settings;
        generation::apply_new_text_defaults(&mut settings);
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
        if generation::is_palace(&settings) {
            let changes = mutations.iter().map(text_change).collect::<Vec<_>>();
            let mut durable = generation::open_index(&s.data, &name)?;
            let published = generation::commit_and_publish(&mut durable, &changes)?;
            s.writes.fetch_add(1, Ordering::Relaxed);
            return Ok(serde_json::to_value(published)?);
        }
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
        let settings = s.catalog.snapshot(&name)?.settings()?;
        if generation::is_palace(&settings) {
            let request: PalaceSearch = serde_json::from_value(body)?;
            let page = generation::search_text(
                &generation::index_root(&s.data, &name),
                request.generation.as_deref(),
                &request.text,
                request.limit,
                palace_hold,
            )?;
            s.searches.fetch_add(1, Ordering::Relaxed);
            return Ok(serde_json::to_value(page)?);
        }
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
        let semantic = matches!(mode.as_deref(), Some("regional" | "hybrid"))
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
            // Hybrid is the default; a request that sends a regional `budget` keeps regional search.
            let strategy = match mode.as_deref() {
                Some("regional") => Strategy::Regional,
                Some(_) => Strategy::Hybrid,
                None if obj.contains_key("budget") => Strategy::Regional,
                None => Strategy::Hybrid,
            };
            s.semantic
                .search(&name, &snap, &settings, body, &principals, strategy)?
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

async fn rebuild(State(app): State<App>, Path(name): Path<String>, headers: HeaderMap) -> Reply {
    app.authorize(&headers, &name, true)?;
    app.blocking(move |s| {
        let settings = s.catalog.snapshot(&name)?.settings()?;
        ensure!(
            generation::is_palace(&settings),
            "rebuild publishes a palace generation"
        );
        let mut durable = generation::open_index(&s.data, &name)?;
        let published = generation::publish_live(&mut durable)?;
        Ok(serde_json::to_value(published)?)
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PalaceSearch {
    text: String,
    #[serde(default = "palace_limit")]
    limit: usize,
    #[serde(default)]
    generation: Option<String>,
}

fn palace_limit() -> usize {
    5
}

fn text_change(operation: &Mutation) -> TextChange {
    match operation {
        Mutation::Upsert { document } => TextChange::Upsert(TextUpdate {
            id: document.id.clone(),
            revision: document.revision.clone(),
            text: document.text.clone(),
            metadata: Value::Object(document.fields.clone().into_iter().collect()),
        }),
        Mutation::Delete { id } => TextChange::Delete { id: id.clone() },
    }
}

fn palace_hold() {
    #[cfg(test)]
    hold_open_reader();
}

#[cfg(test)]
struct OpenHold {
    entered: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

#[cfg(test)]
static PALACE_HOLD: std::sync::Mutex<Option<OpenHold>> = std::sync::Mutex::new(None);

#[cfg(test)]
fn hold_open_reader() {
    let hold = PALACE_HOLD.lock().expect("palace hold").take();
    if let Some(hold) = hold {
        let _ = hold.entered.send(());
        let _ = hold
            .release
            .recv_timeout(std::time::Duration::from_secs(60));
    }
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

    fn copy_dir(from: &FsPath, to: &FsPath) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn text_search_on_a_model_index_defaults_to_hybrid() {
        // Opt-in: point LOCI_MODEL_CACHE at a models directory with bge-small-en-v1.5.
        let Ok(models) = std::env::var("LOCI_MODEL_CACHE") else {
            eprintln!("skipped: set LOCI_MODEL_CACHE to a models directory with bge-small-en-v1.5");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().to_path_buf();
        copy_dir(FsPath::new(&models), &data.join("models"));
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
        assert_eq!(
            code(
                agent
                    .put(format!("{base}/v1/indexes/manuals"))
                    .send_json(json!({"model": "bge-small-en-v1.5"}))
            ),
            200
        );
        let documents = [
            (
                "vault",
                "The encrypted vault rotates its master key every ninety days.",
            ),
            (
                "lamp",
                "Lamp LMP-201 ships with a warm white bulb and a fabric shade.",
            ),
            (
                "router",
                "Reset the router by holding the recessed button for ten seconds.",
            ),
            (
                "invoice",
                "Invoices are payable within thirty days of the billing date.",
            ),
            (
                "garden",
                "Water tomato seedlings in the morning and keep the soil moist.",
            ),
        ];
        let operations = documents
            .iter()
            .map(|(id, text)| json!({"op":"upsert","document":{"id":id,"revision":"1","text":text,"fields":{}}}))
            .collect::<Vec<_>>();
        assert_eq!(
            code(
                agent
                    .post(format!("{base}/v1/indexes/manuals/bulk"))
                    .send_json(json!({ "operations": operations }))
            ),
            200
        );
        let search = |body: Value| {
            agent
                .post(format!("{base}/v1/indexes/manuals/search"))
                .send_json(body)
        };
        let page: Value = search(json!({"text": "how often is the vault key rotated", "limit": 3}))
            .unwrap()
            .body_mut()
            .read_json()
            .unwrap();
        assert_eq!(page["strategy"], "hybrid");
        assert_eq!(page["hits"][0]["document"], "vault");
        let identifier: Value = search(json!({"text": "LMP-201", "limit": 1}))
            .unwrap()
            .body_mut()
            .read_json()
            .unwrap();
        assert_eq!(identifier["hits"][0]["document"], "lamp");
        let regional: Value =
            search(json!({"text": "router reset", "limit": 2, "mode": "regional"}))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap();
        assert_eq!(regional["strategy"], "regional");
        assert_eq!(regional["hits"][0]["document"], "router");
        // A budget without a mode keeps regional search for existing callers.
        let budgeted: Value =
            search(json!({"text": "router reset", "limit": 2, "budget": {"regions": 4}}))
                .unwrap()
                .body_mut()
                .read_json()
                .unwrap();
        assert_eq!(budgeted["strategy"], "regional");
        assert_eq!(
            code(search(
                json!({"text": "router", "mode": "hybrid", "budget": {"regions": 4}})
            )),
            400
        );
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

    #[test]
    fn a_fresh_catalog_highlights_an_imported_note_within_ten_minutes() {
        const QUOTE: &str = "Restore the encrypted vault from its backup before the next rotation.";
        let started = Instant::now();
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let note = dir.path().join("recovery.md");
        std::fs::write(
            &note,
            format!("Operators keep one copy offline.\n\n{QUOTE}\n"),
        )
        .unwrap();
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
        assert_eq!(
            code(
                agent
                    .put(format!("{base}/v1/indexes/manuals"))
                    .send_json(json!({"model": "lexical"}))
            ),
            200
        );
        import(&base, "manuals", std::slice::from_ref(&note)).unwrap();
        let mut response = agent
            .post(format!("{base}/v1/indexes/manuals/search"))
            .send_json(json!({"text": "encrypted vault", "limit": 1, "highlight": true}))
            .unwrap();
        let page: Value = response.body_mut().read_json().unwrap();
        let hit = &page["hits"][0];
        let text = hit["text"].as_str().unwrap();
        assert!(text.contains(QUOTE));
        let spans = page["highlights"][hit["id"].as_str().unwrap()]
            .as_array()
            .unwrap();
        assert!(!spans.is_empty());
        for span in spans {
            let start = span["start"].as_u64().unwrap() as usize;
            let end = span["end"].as_u64().unwrap() as usize;
            assert!(QUOTE.contains(&text[start..end]), "{}", &text[start..end]);
        }
        assert!(started.elapsed() < Duration::from_secs(600));
    }

    #[test]
    fn architecture_query_survives_without_postgres_and_elasticsearch() {
        if std::env::var("LOCI_POSTGRES").ok().as_deref() != Some("1") {
            eprintln!(
                "skipped: set LOCI_POSTGRES=1 and LOCI_POSTGRES_URL to run the palace generation test"
            );
            return;
        }
        let url = std::env::var("LOCI_POSTGRES_URL").expect("LOCI_POSTGRES_URL");
        let running = std::process::Command::new("docker")
            .args([
                "inspect",
                "-f",
                "{{.State.Running}}",
                "loci-search-postgres-1",
            ])
            .output()
            .expect("docker inspect");
        assert!(
            running.status.success() && String::from_utf8_lossy(&running.stdout).trim() == "true",
            "loci-search-postgres-1 is not running"
        );
        let index = format!("n{}", std::process::id());
        let schema = loci_search::generation::schema_for_index(&index).unwrap();
        drop_schema(&url, &schema);
        let _schema = SchemaDrop {
            url: url.clone(),
            schema,
        };

        let dir = tempfile::tempdir().unwrap();
        let base = serve_at(dir.path());
        let agent = ureq::Agent::new_with_defaults();
        let settings = json!({
            "model": "minilm-l6-v2-q8",
            "retrieval_profile": "palace"
        });
        assert_eq!(
            code(
                agent
                    .put(format!("{base}/v1/indexes/{index}"))
                    .send_json(&settings)
            ),
            200
        );
        assert_eq!(
            code(
                agent
                    .put(format!("{base}/v1/indexes/logs"))
                    .send_json(json!({"model":"lexical"}))
            ),
            200
        );
        assert_eq!(
            code(agent.post(format!("{base}/v1/indexes/logs/bulk")).send_json(json!({
                "operations":[{"op":"upsert","document":{"id":"kept","revision":"1","text":"kept event","fields":{"time":1}}}]
            }))),
            200
        );
        let corpus = corpus_body("1");
        assert_eq!(
            code(
                agent
                    .post(format!("{base}/v1/indexes/{index}/bulk"))
                    .send_json(&corpus)
            ),
            200
        );
        let first = palace_hits(&agent, &base, &index, None);
        assert!(
            hit_rows(&first)
                .iter()
                .any(|hit| hit["citation"].as_str().unwrap().contains("north cabinet")),
            "{first}"
        );
        let generation = first_generation(&agent, &base, &index);
        let hnsw = dir
            .path()
            .join("indexes")
            .join(&index)
            .join("generations")
            .join(&generation)
            .join("retrieval.sqlite.hnsw");
        assert!(
            hnsw.is_file(),
            "published generation includes the hnsw sidecar"
        );

        let paused = pause_postgres_and_elasticsearch();
        let during = palace_hits(&agent, &base, &index, None);
        assert_eq!(
            during, first,
            "query changed while PostgreSQL and Elasticsearch were stopped"
        );
        let mut lexical = agent
            .post(format!("{base}/v1/indexes/logs/search"))
            .send_json(json!({"text":"kept","limit":1}))
            .unwrap();
        let lexical: Value = lexical.body_mut().read_json().unwrap();
        assert_eq!(lexical["hits"][0]["id"], json!("kept"));
        drop(paused);
        wait_postgres(&url);

        assert_eq!(
            code(
                agent
                    .post(format!("{base}/v1/indexes/{index}/bulk"))
                    .send_json(&corpus)
            ),
            200
        );
        assert_eq!(
            palace_hits(&agent, &base, &index, None),
            first,
            "retry changed the hits"
        );

        let root = dir.path().join("indexes").join(&index);
        let partial = loci_search::durable::Generations::open(&root)
            .unwrap()
            .create_staging()
            .unwrap();
        std::fs::write(partial.join("manifest.json"), b"{").unwrap();
        assert_eq!(
            code(
                agent
                    .post(format!("{base}/v1/indexes/{index}/rebuild"))
                    .send_json(json!({}))
            ),
            200
        );
        assert!(
            partial.is_dir(),
            "a partial staging directory stays unpublished"
        );
        assert_eq!(palace_hits(&agent, &base, &index, None), first);

        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(
            code(
                agent
                    .post(format!("{base}/v1/indexes/{index}/search"))
                    .send_json(json!({"text":"brass telescope"}))
            ),
            404
        );

        let clean = tempfile::tempdir().unwrap();
        let rebuilt = serve_at(clean.path());
        assert_eq!(
            code(
                agent
                    .put(format!("{rebuilt}/v1/indexes/{index}"))
                    .send_json(&settings)
            ),
            200
        );
        assert_eq!(
            code(
                agent
                    .post(format!("{rebuilt}/v1/indexes/{index}/rebuild"))
                    .send_json(json!({}))
            ),
            200
        );
        assert_eq!(palace_hits(&agent, &rebuilt, &index, None), first);

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *super::PALACE_HOLD.lock().unwrap() = Some(super::OpenHold {
            entered: entered_tx,
            release: release_rx,
        });
        let search_base = rebuilt.clone();
        let search_index = index.clone();
        let inflight = std::thread::spawn(move || palace_page(&search_base, &search_index, None));
        entered_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("search did not open the generation");
        let mut operations = corpus_body("2")["operations"].as_array().unwrap().clone();
        operations[0] = json!({"op":"upsert","document":{"id":"alpha","revision":"2","text":"The cedar archive sold the brass telescope and stored only empty oak drawers in the north cabinet."}});
        assert_eq!(
            code(
                agent
                    .post(format!("{rebuilt}/v1/indexes/{index}/bulk"))
                    .send_json(json!({"operations": operations}))
            ),
            200
        );
        release_tx.send(()).unwrap();
        let held_page = inflight.join().unwrap();
        let held = held_page["hits"].clone();
        assert_eq!(
            held, first,
            "an open reader changed during document replacement"
        );
        assert!(hit_rows(&held).iter().any(|hit| {
            let citation = hit["citation"].as_str().unwrap();
            citation.contains("north cabinet") && !citation.contains("oak drawers")
        }));
        let latest = palace_hits(&agent, &rebuilt, &index, None);
        assert!(
            hit_rows(&latest)
                .iter()
                .any(|hit| hit["citation"].as_str().unwrap().contains("oak drawers"))
        );
        assert_ne!(latest, held);
        let pinned = palace_hits(&agent, &rebuilt, &index, held_page["generation"].as_str());
        assert_eq!(pinned, held);
    }

    fn serve_at(data: &std::path::Path) -> String {
        let (sender, receiver) = mpsc::channel();
        let data = data.to_path_buf();
        std::thread::spawn(move || {
            listen("127.0.0.1:0".parse().unwrap(), &data, None, Some(sender)).unwrap();
        });
        let addr = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        format!("http://{addr}")
    }

    fn corpus_body(revision: &str) -> Value {
        json!({"operations":[
            {"op":"upsert","document":{"id":"alpha","revision":revision,"text":"The cedar archive keeps the brass telescope in the north cabinet beside the river maps and tide tables."}},
            {"op":"upsert","document":{"id":"beta","revision":revision,"text":"Moth survey notes record silver wing dust on the lantern glass after midnight in the east marsh."}},
            {"op":"upsert","document":{"id":"gamma","revision":revision,"text":"The kiln log says the cobalt glaze cracked when the west wind reached the drying yard at dawn."}}
        ]})
    }

    fn palace_page(base: &str, index: &str, generation: Option<&str>) -> Value {
        let mut request = json!({"text": "brass telescope", "limit": 5});
        if let Some(generation) = generation {
            request["generation"] = json!(generation);
        }
        let mut response = ureq::Agent::new_with_defaults()
            .post(format!("{base}/v1/indexes/{index}/search"))
            .send_json(&request)
            .unwrap();
        response.body_mut().read_json().unwrap()
    }

    fn hit_rows(value: &Value) -> &[Value] {
        value.as_array().map(Vec::as_slice).unwrap_or(&[])
    }

    fn palace_hits(
        agent: &ureq::Agent,
        base: &str,
        index: &str,
        generation: Option<&str>,
    ) -> Value {
        let _ = agent;
        palace_page(base, index, generation)["hits"].clone()
    }

    fn first_generation(agent: &ureq::Agent, base: &str, index: &str) -> String {
        let _ = agent;
        palace_page(base, index, None)["generation"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn drop_schema(url: &str, schema: &str) {
        let mut client = postgres::Client::connect(url, postgres::NoTls).unwrap();
        client
            .batch_execute(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE"))
            .unwrap();
    }

    fn wait_postgres(url: &str) {
        let started = Instant::now();
        loop {
            if postgres::Client::connect(url, postgres::NoTls).is_ok() {
                return;
            }
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "PostgreSQL did not accept connections again"
            );
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    struct SchemaDrop {
        url: String,
        schema: String,
    }

    impl Drop for SchemaDrop {
        fn drop(&mut self) {
            if let Ok(mut client) = postgres::Client::connect(&self.url, postgres::NoTls) {
                let _ =
                    client.batch_execute(&format!("DROP SCHEMA IF EXISTS {} CASCADE", self.schema));
            }
        }
    }

    struct Paused {
        commands: Vec<String>,
    }

    impl Drop for Paused {
        fn drop(&mut self) {
            let _ = std::process::Command::new("docker")
                .args(["start", "loci-search-postgres-1"])
                .status();
            for command in &self.commands {
                let _ = std::process::Command::new("powershell")
                    .args([
                        "-NoProfile",
                        "-Command",
                        "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $env:LOCI_RESTART } | Out-Null",
                    ])
                    .env("LOCI_RESTART", command)
                    .status();
            }
        }
    }

    fn pause_postgres_and_elasticsearch() -> Paused {
        let listing = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "$needle = 'elasticsearch' + '.bat'; Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -like ('*' + $needle + '*') } | ForEach-Object { $_.ProcessId.ToString() + \"`t\" + $_.CommandLine }",
            ])
            .output()
            .expect("list elasticsearch processes");
        let text = String::from_utf8_lossy(&listing.stdout);
        let mut commands = Vec::new();
        for line in text.lines() {
            let Some((pid, command)) = line.split_once('\t') else {
                continue;
            };
            if command.is_empty() {
                continue;
            }
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", pid.trim(), "/T", "/F"])
                .status();
            if !commands.iter().any(|saved: &String| saved == command) {
                commands.push(command.to_string());
            }
        }
        let stopped = std::process::Command::new("docker")
            .args(["stop", "-t", "30", "loci-search-postgres-1"])
            .status()
            .expect("docker stop");
        assert!(stopped.success(), "failed to stop loci-search-postgres-1");
        Paused { commands }
    }
}
