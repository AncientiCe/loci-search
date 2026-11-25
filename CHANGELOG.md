# Changelog

## Unreleased

- `Strategy::Hybrid` searches with the bounded global rescue, the persisted HNSW graph, and the reranker, without the regional scan. It loads a lean index without per-region posting lists, int8 vectors, or scan scratch; a later regional search upgrades it. On one million HotpotQA passages a hybrid `bench-serve` was ready in 12.1 s at 2.49 GB, against 24.8 s and 4.51 GB for regional search with budgets of 1, at identical quality. Over HTTP on Linux, with no paging, its raw-text p95 was 25.35 ms against Elasticsearch's 33.78 ms (−25%), and 13.00 ms against 20.31 ms with precomputed vectors.

## [0.3.0]

With its regional scan switched off, Loci beats Elasticsearch on speed at equal quality. Loci searches its HNSW graph (150 candidates, width 128), runs a bounded global keyword search (60 hits), and reranks with 8% term coverage and 92% cosine. On Linux, with the same one million HotpotQA passages and `bge-small-en-v1.5` vectors, 320 held-out queries, Loci's reranker on both engines, and both engines answering over loopback HTTP, Loci reached 90.78% Recall@100 and 0.822 nDCG@10. Elasticsearch 8.19 reached 90.63% and 0.823; the differences are not significant. Loci's p95 was 13.4 ms against 19.5 ms with precomputed vectors, and about 20% lower with query encoding included (25.7 against 32.4 ms, and 26.2 against 32.4 ms in a second run). At 16 concurrent clients Loci served 377 raw-text queries per second against 260, and 1,358 against 560 with precomputed vectors. This is one corpus, and swap in the benchmark VM touched both raw-text runs. In this release the faster configuration is a benchmark setting; `loci serve` still uses the regional search. See the [benchmark history](https://github.com/AncientiCe/loci-search/blob/main/benchmarks/README.md#regions-off-graph-rescue-and-reranker-only).

- `loci bench-serve` holds one engine prepared with a frozen regional plan and answers `POST /search` over HTTP. `loci bench --loci-url` sends regional searches there, so Loci and Elasticsearch both pay an HTTP round trip. Reports record `regional_transport`.
- The reranker reads term coverage from the in-memory postings for every strategy once they are loaded, instead of re-reading passage text. Scores are unchanged; Elasticsearch candidates are no longer charged for the extra reads.
- `loci pruning-dev` scores the whole corpus with the final ranking function and measures how much of the top 100 each plan finds and how many passages it scores. Its branch-and-bound over regions skipped no passage: each region spans about 52° around its centroid, so the topology cannot rule a region out. Plans without the regional scan matched its quality in about half the time.
- Regional search, global rescue, and Elasticsearch comparisons accept a passage filter. `loci filter-dev` compares them with an exhaustive filtered top 100, and `loci elastic-buckets` builds a matching Elasticsearch index. Filtering gave Loci no advantage: below 1% of the corpus a brute-force scan won, and at matched quality Elasticsearch was as fast or faster.
- `--no-bound` runs the `pruning-dev` plan sweep without branch-and-bound.

## [0.2.0]

Loci now searches from query text with a local model and serves a local HTTP catalog. The release also records an honest comparison with Elasticsearch. With the same `bge-small-en-v1.5` vectors on Linux, one million HotpotQA passages, and query embedding counted on both sides, Loci reached 91.09% Recall@100 and 0.822 nDCG@10 at 37.5 ms p95. Elasticsearch 8.19 reached 90.31% and 0.790 at 34.3 ms. At 16 concurrent clients, Loci served 282 queries per second and Elasticsearch served 227. When Elasticsearch candidates used the same reranker, recall and nDCG@10 were tied on HotpotQA and FiQA. Loci matches Elasticsearch on this workload but does not beat it. See the [benchmark history](https://github.com/AncientiCe/loci-search/blob/main/benchmarks/README.md#raw-text-verdict-against-elasticsearch).

- `loci bench --throughput-width 1,4,8,16` times concurrent queries on one shared index and reports queries per second with p50, p95, and p99.
- A comparison can rerank Elasticsearch candidates with the same weights as regional search through `elastic_rerank`.
- One opened index serves concurrent searches from a pool of read-only connections pinned to its generation.
- Local models run directly on ONNX Runtime and tokenize with `kitoken` instead of `fastembed`. `bge-small-en-v1.5` vectors match the previous encoder to within 4e-8, and the embedding manifest is unchanged, so existing indexes stay valid.
- Linux release binaries are built on Ubuntu 24.04 and need glibc 2.39 or newer.
- Timestamps parse with `jiff` instead of the unmaintained `chrono`. `cookie` and `url` are updated for RUSTSEC-2026-0594 and RUSTSEC-2026-0592.
- Answer `429` when every worker is busy, without starting that request. A request that already holds a worker still commits or rolls back as one generation.
- Updating one document replaces its text and vector together and leaves every other document's vector in place. Deleting it removes that vector.
- The fixture scorecard records each workload's query latency, catalog disk, process working set, and the time to restart and find the saved document again.
- A fresh data directory can import one Markdown note and highlight a sentence from it. The Rust test fails if that takes ten minutes or longer.
- Each in-flight regional search keeps its own scratch buffers, and a saved graph is rebuilt when the vectors change.
- Runnable Rust examples cover knowledge, catalog, records, and logs.
- Operator notes record the catalog file, the worker cap, and the published million-passage graph size.
- `loci bench --embed-model` encodes each query from its text and adds that time to the sample. The model must match the index.
- `loci embed` writes packed vectors for an installed local model and can resume a partial file.
- An interrupted model upgrade leaves the published generation and its vectors in place.
- A full day of one event per minute can be appended while searches stay exact, including a late event in an earlier hour.
- A document can set `embed` to false on a model index. It stays in lexical search and is left out of embedding and model rebuilds.
- A long lexical question borrows up to four distinctive terms from its best hit so a second document can enter the top results. Short identifier queries stay exact.
- Record one sequential pass of the four workload fixtures, and rehearse import, query comparison, backup, and restore on a disposable index.
- Say when an exact scan stops at its match cap, and keep private documents out of snippets, suggestions, and totals.
- Skip dense vectors and extra analyzers during import, and report them. NFCorpus's 3633 documents and five lexical totals matched Elasticsearch.
- Record catalog autocomplete latency and how soon a write can be searched.
- Append 2000 log events while searches run. This pass kept an exact count at 3763 events per second.
- Serve a local HTTP catalog with exact filters, permissions, cursors, and event retention.
- Embed new writes with a pinned local model and switch that model together with every stored vector.
- Reclaim expired cursors, expire automatic event streams, and compact the lexical index on a one-minute maintenance pass.

## [0.1.0]

First public preview of Loci Search, a local Rust CLI and library for explainable hybrid retrieval over precomputed embeddings.

- Build SQLite indexes from JSONL snapshots and search with BM25, global hybrid retrieval, or semantic region routing.
- Trace each hit back to its document, revision, and span. Inspect routing, lexical rescue, and optional HNSW candidate ranks.
- Evaluate retrieval quality and warm sequential latency with a deterministic fixture or with judged corpora. See the [benchmark history](https://github.com/AncientiCe/loci-search/blob/main/benchmarks/README.md) for measured results and configurations.

This preview replaces the index on ingest and uses caller-provided embeddings for passages and queries.
