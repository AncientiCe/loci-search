# Changelog

## Unreleased

## [0.1.0]

First public preview of Loci Search, a local Rust CLI and library for explainable hybrid retrieval over precomputed embeddings.

- Build SQLite indexes from JSONL snapshots and search with BM25, global hybrid retrieval, or semantic region routing.
- Trace each hit back to its document, revision, and span. Inspect routing, lexical rescue, and optional HNSW candidate ranks.
- Evaluate retrieval quality and warm sequential latency with a deterministic fixture or with judged corpora. See the [benchmark history](https://github.com/AncientiCe/loci-search/blob/main/benchmarks/README.md) for measured results and configurations.

This preview replaces the index on ingest and uses caller-provided embeddings for passages and queries.
