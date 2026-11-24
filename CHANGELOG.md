# Changelog

## Unreleased

- Answer `429` when every worker is busy, without starting that request. A request that already holds a worker still commits or rolls back as one generation.
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
