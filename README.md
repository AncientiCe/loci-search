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

## Use an existing corpus

```sh
cargo run --locked --release -- fixture --chunks 128 --out runs/example-data
cargo run --locked --release -- ingest --db runs/example.sqlite --passages runs/example-data/passages.jsonl --embedding runs/example-data/embedding.json
cargo run --locked --release -- search --db runs/example.sqlite --query runs/example-data/query.json --limit 5 --explain
cargo run --locked --release -- bench --db runs/example.sqlite --queries runs/example-data/queries.jsonl --out runs/example-bench
```

Use the same formats for custom embeddings. Each passage line has a stable `id`, `document`, `revision`, `start`, `end`, `text`, and numeric `vector`; `metadata` is an optional JSON object. Source offsets are supplied by the caller. An embedding manifest has `model_revision`, `dimensions`, and `normalization` set to `"l2"`. A query has `id`, `text`, `embedding` (the complete manifest), and `vector`. Evaluation also accepts `group`, a `judgments` object mapping chunk IDs to integer relevance grades, and `required` chunk IDs for all-evidence recall. Grade zero denotes nonrelevant.

Ingest replaces the complete index snapshot atomically. Use unique chunk IDs and the same embedding model revision for passages and queries. The engine normalizes finite nonzero vectors at both boundaries.
