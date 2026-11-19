# Loci Search

Loci Search is a single-node Rust retrieval experiment. It stores each passage and embedding once in SQLite, assigns passages to up to three semantic regions, and routes queries through a fixed hierarchy and shared-membership edges. Each hit identifies its source document, revision, and span. Explanations include the region paths and ranking signals that surfaced it.

## Reproduce the 100,000-chunk benchmark

```sh
cargo run --locked --release -- bench --fixture --out runs/fixture-100k
```

The command builds a deterministic 100,000-passage synthetic corpus and runs 32 labeled queries across identifier, paraphrase, cross-domain, and multi-hop cases. It performs one warm-up pass, then reports three measured passes by default. Use a fresh output directory for each run.

- `report.json` records the embedding manifest, work budgets, index size, build timings, candidate recall, nDCG@10, Recall@100, all-evidence recall, and p50/p95 latency. It also includes a sampled HNSW comparison with exact cosine.
- `traces.jsonl` has a record per query, strategy, and repeat, including ranked IDs, reranker candidates, region paths, lexical rescue, and work counters.
- `fixture.sqlite` holds the generated index. The HNSW control builds its deterministic in-memory graph when opened for global hybrid search. The benchmark measures that build separately from warm query latency.

The benchmark compares [SQLite FTS5 BM25](https://www.sqlite.org/fts5.html), global BM25 plus HNSW, and budgeted regional BM25 plus cosine. The hybrids share reciprocal rank fusion and a reranker that uses query-term coverage and exact cosine. The regional engine reserves candidate slots for global lexical rescue. Budget parameters include `--max-regions`, `--max-chunks`, `--max-postings`, `--rescue`, and `--routing-ms`. The time budget applies to routing and is checked cooperatively. SQLite global lexical work is represented by returned results and elapsed time because FTS5 does not expose postings visited.

The built-in fixture checks behavior and reproducibility. Use a held-out judged corpus and a tuned external control for a quality or latency decision. This benchmark currently measures warm sequential latency. Elasticsearch, cold and concurrent loads, and adaptive topology experiments are outside this baseline.

## Compare against Elasticsearch on FiQA

The comparison uses the full [BEIR FiQA corpus](https://huggingface.co/datasets/BeIR/fiqa), its official development and test judgments, and the published [Cohere Embed English v3 vectors](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3). Keep the inputs and generated outputs in `runs/fiqa`.

| Save as | Download |
| --- | --- |
| `fiqa.zip` | [BEIR dataset archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) |
| `corpus.parquet` | [Corpus embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/fiqa/corpus/0000.parquet) |
| `dev.parquet` | [Development query embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/fiqa/queries/dev.parquet) |
| `test.parquet` | [Test query embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/fiqa/queries/test.parquet) |

```sh
python -m pip install --target runs/fiqa/python-deps pyarrow==23.0.1
python tools/fiqa.py prepare runs/fiqa --deps runs/fiqa/python-deps
cargo run --locked --release -- ingest --db runs/fiqa/loci.sqlite --passages runs/fiqa/passages.jsonl --embedding runs/fiqa/embedding.json
```

Start a dedicated [Elasticsearch 8.19.25 instance](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.19.25-windows-x86_64.zip) bound to loopback on port 19200. The measured run used a 2 GiB JVM heap and four active JVM processors. For this isolated benchmark, run the bundled launcher with `-Ediscovery.type=single-node -Expack.security.enabled=false -Expack.ml.enabled=false -Enetwork.host=127.0.0.1 -Ehttp.port=19200 -Etransport.port=19300`. Keep the endpoint on loopback.

```sh
python tools/fiqa.py index runs/fiqa
cargo run --locked --release -- bench --db runs/fiqa/loci.sqlite --queries runs/fiqa/test.jsonl --dev runs/fiqa/dev.jsonl --elastic-url http://127.0.0.1:19200 --elastic-index loci-fiqa --out runs/fiqa/evaluation
```

The run selects settings on the 500 development queries and freezes them before scoring the 648 test queries. Each hybrid gets six trials. Loci tunes HNSW width or regional work limits. Elasticsearch tunes the lexical analyzer and HNSW candidate count. Selection maximizes development nDCG@10, then Recall@100, then prefers the earliest grid entry on an exact tie. All hybrids use the same 200-candidate fusion and reranking pipeline. Each measured round interleaves engines, includes the Elasticsearch loopback request, and uses one query at a time after warm-up. The report records the grid, selected settings, Elasticsearch version and index size, and paired query-level bootstrap intervals for regional minus Elasticsearch quality.

### Held-out run

These results use all 57,638 passages, three measured passes over the 648 test queries, and an AMD Ryzen 7 7800X3D running Windows. Elasticsearch selected the English analyzer and 256 HNSW candidates. Loci regional selected limits of 36 regions, 24,000 scored chunks, and 240,000 postings per query.

| Strategy | nDCG@10 | Recall@100 | Warm p95 |
| --- | ---: | ---: | ---: |
| SQLite BM25 | 0.2348 | 0.5187 | 81.75 ms |
| Loci global hybrid | 0.2859 | 0.6305 | 88.24 ms |
| Loci regional hybrid | 0.2844 | 0.6212 | 153.97 ms |
| Elasticsearch hybrid | 0.2910 | 0.6498 | 20.98 ms |

Regional minus Elasticsearch was −0.0066 nDCG@10 (95% paired bootstrap interval −0.0114 to −0.0019) and −0.0286 Recall@100 (−0.0408 to −0.0178). Regional search averaged 20,220 candidate vector comparisons per query, versus 2,541 for the in-process global hybrid.

## Use an existing corpus

```sh
cargo run --locked --release -- fixture --chunks 128 --out runs/example-data
cargo run --locked --release -- ingest --db runs/example.sqlite --passages runs/example-data/passages.jsonl --embedding runs/example-data/embedding.json
cargo run --locked --release -- search --db runs/example.sqlite --query runs/example-data/query.json --limit 5 --explain
cargo run --locked --release -- bench --db runs/example.sqlite --queries runs/example-data/queries.jsonl --out runs/example-bench
```

Use the same formats for custom embeddings. Each passage line has a stable `id`, `document`, `revision`, `start`, `end`, `text`, and numeric `vector`; `metadata` is an optional JSON object. Source offsets are supplied by the caller. An embedding manifest has `model_revision`, `dimensions`, and `normalization` set to `"l2"`. A query has `id`, `text`, `embedding` (the complete manifest), and `vector`. Evaluation also accepts `group`, a `judgments` object mapping chunk IDs to integer relevance grades, and `required` chunk IDs for all-evidence recall. Grade zero denotes nonrelevant.

Ingest replaces the complete index snapshot atomically. Use unique chunk IDs and the same embedding model revision for passages and queries. The engine normalizes finite nonzero vectors at both boundaries.
