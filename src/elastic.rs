use crate::{
    Query,
    index::{Engine, RetrievalProfile},
    ranking::{self, Hit, SearchConfig, SearchResult, Strategy},
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;
use ureq::{Agent, http::Uri};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Options {
    pub analyzer: Analyzer,
    pub num_candidates: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Analyzer {
    Standard,
    English,
}

pub struct Client {
    agent: Agent,
    base: String,
    index: String,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct Info {
    pub version: String,
    pub index: String,
    pub documents: usize,
    pub active_primary_shards: usize,
    pub index_bytes: u64,
}

pub(crate) struct Lists {
    pub lexical: Vec<usize>,
    pub vector: Vec<usize>,
    pub took_ms: u64,
}

#[derive(Deserialize)]
struct MultiSearch {
    took: u64,
    responses: Vec<Response>,
}

#[derive(Deserialize)]
struct Response {
    took: Option<u64>,
    error: Option<Value>,
    timed_out: Option<bool>,
    #[serde(rename = "_shards")]
    shards: Option<Shards>,
    hits: Option<Hits>,
}

#[derive(Deserialize)]
struct Shards {
    failed: usize,
}

#[derive(Deserialize)]
struct Hits {
    hits: Vec<EsHit>,
}

#[derive(Deserialize)]
struct EsHit {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "_score")]
    score: Option<f64>,
}

impl Client {
    pub fn new(base: &str, index: &str) -> Result<Self> {
        let uri: Uri = base.parse()?;
        ensure!(
            uri.scheme_str() == Some("http")
                && matches!(uri.host(), Some("127.0.0.1" | "localhost" | "[::1]"))
                && uri.path() == "/"
                && uri.query().is_none(),
            "the benchmark requires an HTTP Elasticsearch endpoint on loopback"
        );
        ensure!(
            !index.is_empty()
                && index.bytes().all(|byte| byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'-' | b'_')),
            "invalid Elasticsearch index name"
        );
        let agent = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(60)))
            .proxy(None)
            .build()
            .into();
        Ok(Self {
            agent,
            base: base.trim_end_matches('/').into(),
            index: index.into(),
        })
    }

    pub fn index_name(&self) -> &str {
        &self.index
    }

    pub(crate) fn info(&self) -> Result<Info> {
        let root = self.get("")?;
        let count = self.get(&format!("/{}/_count", self.index))?;
        let stats = self.get(&format!("/{}/_stats/store", self.index))?;
        let health = self.get(&format!("/_cluster/health/{}", self.index))?;
        Ok(Info {
            version: root["version"]["number"]
                .as_str()
                .context("missing Elasticsearch version")?
                .into(),
            index: self.index.clone(),
            documents: count["count"]
                .as_u64()
                .context("missing Elasticsearch count")?
                .try_into()?,
            active_primary_shards: health["active_primary_shards"]
                .as_u64()
                .context("missing Elasticsearch primary shard count")?
                .try_into()?,
            index_bytes: stats["_all"]["primaries"]["store"]["size_in_bytes"]
                .as_u64()
                .context("missing Elasticsearch primary store size")?,
        })
    }

    /// Create this index from the Loci passages and vectors, adding an integer
    /// `bucket` field from `bucket_of(passage id)`, then merge it to one segment.
    /// The mapping matches the comparison indexes built by `tools/fiqa.py`.
    pub(crate) fn build_bucketed(
        &self,
        engine: &Engine,
        bucket_of: impl Fn(&str) -> u32 + Sync,
    ) -> Result<()> {
        let long: Agent = Agent::config_builder()
            .timeout_global(None)
            .proxy(None)
            .build()
            .into();
        let url = |route: &str| format!("{}/{}{route}", self.base, self.index);
        long.put(url(""))
            .header("Content-Type", "application/json")
            .send_json(json!({
                "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
                "mappings": {"dynamic": "strict", "_source": {"excludes": ["embedding"]}, "properties": {
                    "text": {"type": "text", "analyzer": "standard", "fields": {
                        "english": {"type": "text", "analyzer": "english"}}},
                    "embedding": {"type": "dense_vector", "dims": engine.manifest.embedding.dimensions,
                        "similarity": "cosine",
                        "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100}},
                    "passage_id": {"type": "keyword"},
                    "bucket": {"type": "integer"}
                }}
            }))
            .with_context(|| format!("create index {}", self.index))?;
        let conn = engine.conn.get()?;
        let mut statement = conn.prepare("SELECT rowid, id, text FROM artifacts ORDER BY rowid")?;
        let mut rows = statement.query([])?;
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<u8>>(8);
        let receiver = std::sync::Mutex::new(receiver);
        std::thread::scope(|scope| -> Result<()> {
            let workers = (0..4)
                .map(|_| {
                    scope.spawn(|| -> Result<()> {
                        loop {
                            let Ok(body) = receiver.lock().expect("bulk queue").recv() else {
                                return Ok(());
                            };
                            let mut response = long
                                .post(url("/_bulk"))
                                .header("Content-Type", "application/x-ndjson")
                                .send(&body[..])?;
                            let result: Value = response.body_mut().read_json()?;
                            ensure!(
                                result["errors"] == Value::Bool(false),
                                "bulk indexing failed: {}",
                                result["items"]
                                    .as_array()
                                    .and_then(|items| items
                                        .iter()
                                        .find(|item| item["index"]["error"].is_object()))
                                    .unwrap_or(&Value::Null)
                            );
                        }
                    })
                })
                .collect::<Vec<_>>();
            let mut batch = Vec::new();
            let mut count = 0;
            while let Some(row) = rows.next()? {
                let rowid: usize = row.get(0)?;
                let id: String = row.get(1)?;
                let text: String = row.get(2)?;
                serde_json::to_writer(&mut batch, &json!({"index": {"_id": id}}))?;
                batch.push(b'\n');
                serde_json::to_writer(
                    &mut batch,
                    &json!({"passage_id": id, "text": text, "bucket": bucket_of(&id),
                        "embedding": engine.vectors[rowid]}),
                )?;
                batch.push(b'\n');
                count += 1;
                if count % 400 == 0 {
                    sender.send(std::mem::take(&mut batch))?;
                    if count % 50_000 == 0 {
                        eprintln!("Submitted {count} passages");
                    }
                }
            }
            if !batch.is_empty() {
                sender.send(batch)?;
            }
            drop(sender);
            for worker in workers {
                worker.join().expect("bulk worker panicked")?;
            }
            Ok(())
        })?;
        eprintln!("Refreshing and merging to one segment");
        long.post(url("/_refresh")).send_empty()?;
        long.post(url("/_forcemerge?max_num_segments=1"))
            .send_empty()?;
        long.put(url("/_settings"))
            .header("Content-Type", "application/json")
            .send_json(json!({"index": {"refresh_interval": "1s"}}))?;
        long.post(url("/_refresh")).send_empty()?;
        Ok(())
    }

    fn get(&self, route: &str) -> Result<Value> {
        Ok(self
            .agent
            .get(format!("{}{route}", self.base))
            .call()?
            .body_mut()
            .read_json()?)
    }

    pub(crate) fn search(
        &self,
        engine: &Engine,
        query: &Query,
        config: &SearchConfig,
        options: Options,
    ) -> Result<SearchResult> {
        let vector = engine.query_vector(query)?;
        if engine.manifest.retrieval_profile == Some(RetrievalProfile::Palace) {
            return self.native_hybrid(engine, query, &vector, config, options);
        }
        let values = self.rankings(engine, query, &vector, config.depth, options, None)?;
        ranking::search_external(engine, query, &vector, config, values)
    }

    /// BM25 and kNN lists restricted by an Elasticsearch filter clause, reranked by Loci.
    pub(crate) fn search_filtered(
        &self,
        engine: &Engine,
        query: &Query,
        config: &SearchConfig,
        options: Options,
        filter: &Value,
    ) -> Result<SearchResult> {
        let vector = engine.query_vector(query)?;
        let values = self.rankings(engine, query, &vector, config.depth, options, Some(filter))?;
        ranking::search_external(engine, query, &vector, config, values)
    }

    /// Elasticsearch combines BM25 and kNN itself. The hit order and scores are
    /// Elasticsearch's; Loci does not rescore them.
    fn native_hybrid(
        &self,
        engine: &Engine,
        query: &Query,
        vector: &[f32],
        config: &SearchConfig,
        options: Options,
    ) -> Result<SearchResult> {
        let field = match options.analyzer {
            Analyzer::Standard => "text",
            Analyzer::English => "text.english",
        };
        let body = json!({
            "size": config.depth,
            "track_total_hits": false,
            "_source": false,
            "query": {"match": {field: &query.text}},
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": config.depth,
                "num_candidates": options.num_candidates
            }
        });
        let mut response = self
            .agent
            .post(format!("{}/{}/_search", self.base, self.index))
            .header("Content-Type", "application/json")
            .send_json(&body)
            .with_context(|| format!("Elasticsearch hybrid query {}", query.id))?;
        let result: Response = response.body_mut().read_json()?;
        if let Some(error) = result.error {
            bail!("Elasticsearch query {} failed: {error}", query.id);
        }
        ensure!(
            result.timed_out == Some(false)
                && result.shards.is_some_and(|shards| shards.failed == 0),
            "incomplete Elasticsearch response for {}",
            query.id
        );
        let took = result.took;
        let returned = result.hits.context("missing Elasticsearch hits")?.hits;
        let mut hits = Vec::new();
        let mut candidates = Vec::new();
        for hit in returned.into_iter().take(config.limit) {
            let index = engine.ids.binary_search(&hit.id).map_err(|_| {
                anyhow::anyhow!("Elasticsearch returned unknown passage {}", hit.id)
            })?;
            candidates.push(hit.id);
            hits.push(Hit {
                source: engine.source(index)?,
                score: hit.score.unwrap_or(0.0),
                lexical_rank: None,
                vector_rank: None,
                global_lexical_rank: None,
                graph_rank: None,
                routes: Vec::new(),
            });
        }
        Ok(SearchResult {
            strategy: Strategy::Elasticsearch,
            hits,
            candidates,
            lexical_rescue_only_candidates: Vec::new(),
            expanded: Vec::new(),
            stop_reason: "elasticsearch_hybrid",
            work: ranking::Work {
                elasticsearch_took_ms: took,
                ..ranking::Work::default()
            },
        })
    }

    fn rankings(
        &self,
        engine: &Engine,
        query: &Query,
        vector: &[f32],
        depth: usize,
        options: Options,
        filter: Option<&Value>,
    ) -> Result<Lists> {
        let field = match options.analyzer {
            Analyzer::Standard => "text",
            Analyzer::English => "text.english",
        };
        let matched = json!({"match": {field: &query.text}});
        let lexical_query = match filter {
            Some(filter) => json!({"bool": {"must": matched, "filter": filter}}),
            None => matched,
        };
        let mut knn = json!({"field": "embedding", "query_vector": vector, "k": depth,
            "num_candidates": options.num_candidates});
        if let Some(filter) = filter {
            knn["filter"] = filter.clone();
        }
        let lexical = json!({
            "size": depth, "track_total_hits": false, "_source": false,
            "query": lexical_query,
            "sort": [{"_score": "desc"}, {"passage_id": "asc"}]
        });
        let dense = json!({
            "size": depth, "track_total_hits": false, "_source": false,
            "knn": knn,
            "sort": [{"_score": "desc"}, {"passage_id": "asc"}]
        });
        let body = format!("{{}}\n{lexical}\n{{}}\n{dense}\n");
        let mut response = self
            .agent
            .post(format!("{}/{}/_msearch", self.base, self.index))
            .header("Content-Type", "application/x-ndjson")
            .send(body.as_bytes())
            .with_context(|| format!("Elasticsearch query {}", query.id))?;
        let result: MultiSearch = response.body_mut().read_json()?;
        let [lexical, vector]: [Response; 2] = result
            .responses
            .try_into()
            .ok()
            .context("expected two Elasticsearch search results")?;
        let ids = |response: Response| -> Result<Vec<usize>> {
            if let Some(error) = response.error {
                bail!("Elasticsearch query {} failed: {error}", query.id);
            }
            ensure!(
                response.timed_out == Some(false)
                    && response.shards.is_some_and(|shards| shards.failed == 0),
                "incomplete Elasticsearch response for {}",
                query.id
            );
            response
                .hits
                .context("missing Elasticsearch hits")?
                .hits
                .into_iter()
                .map(|hit| {
                    engine.ids.binary_search(&hit.id).map_err(|_| {
                        anyhow::anyhow!("Elasticsearch returned unknown passage {}", hit.id)
                    })
                })
                .collect()
        };
        Ok(Lists {
            lexical: ids(lexical)?,
            vector: ids(vector)?,
            took_ms: result.took,
        })
    }
}
