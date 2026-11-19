use crate::{
    Query,
    index::Engine,
    ranking::{self, SearchConfig, SearchResult},
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
    hits: Vec<Hit>,
}

#[derive(Deserialize)]
struct Hit {
    #[serde(rename = "_id")]
    id: String,
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
        let values = self.rankings(engine, query, &vector, config.depth, options)?;
        ranking::search_external(engine, query, &vector, config, values)
    }

    fn rankings(
        &self,
        engine: &Engine,
        query: &Query,
        vector: &[f32],
        depth: usize,
        options: Options,
    ) -> Result<Lists> {
        let field = match options.analyzer {
            Analyzer::Standard => "text",
            Analyzer::English => "text.english",
        };
        let lexical = json!({
            "size": depth, "track_total_hits": false, "_source": false,
            "query": {"match": {field: &query.text}},
            "sort": [{"_score": "desc"}, {"passage_id": "asc"}]
        });
        let dense = json!({
            "size": depth, "track_total_hits": false, "_source": false,
            "knn": {"field": "embedding", "query_vector": vector, "k": depth,
                "num_candidates": options.num_candidates},
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
