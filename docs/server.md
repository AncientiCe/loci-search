# Running the service

`loci serve` keeps a document catalog in a data directory and answers the HTTP API in [openapi.json](openapi.json). The retrieval CLI still accepts caller-supplied vectors. This process can also embed text with a pinned local model.

```sh
loci serve --data loci-data
```

The default bind is `127.0.0.1:7707`. A non-loopback bind requires `--credentials`. The file is a JSON array of bearer tokens. Each token is at least 24 characters and lists the indexes it can use (`*` for every index), whether it can write, and the reader principals applied to its searches.

```json
[{
  "token": "replace-with-a-long-random-string",
  "indexes": ["*"],
  "write": true,
  "principals": ["staff"]
}]
```

Create an index with `PUT /v1/indexes/{name}`. `"model": "lexical"` stores text only. `"bge-small-en-v1.5"` or `"multilingual-e5-small"` embeds each write unless that document sets `"embed": false`. A skipped document remains in lexical search and is left out of the vector rebuild. Install the pack before the first embedded write:

```sh
loci model-install bge-small-en-v1.5 --data loci-data
loci model-install multilingual-e5-small --data loci-data --source /path/to/verified-pack
```

`PUT /v1/indexes/{name}/model` rebuilds every embedded document vector and publishes the new model in one generation. Documents with `"embed": false` stay lexical. A reader opened on the previous generation keeps that generation. An upgrade that stops before commit leaves that generation and its vectors in place. A bulk write that started against the old model is rejected. A later bulk write replaces only the documents in that batch: their text and vectors change together, and every other document keeps the vector it already has. Deleting a document removes its vector.

When every worker is already busy, the service answers `429` and does not start that request. A request that already holds a worker still commits or rolls back as one generation.

The service runs maintenance about once a minute. It deletes expired cursor pages, merges lexical segments, and expires events on indexes whose stream policy has `"automatic": true`. `POST /v1/indexes/{name}/expire` applies the same retention cutoff at a caller-supplied timestamp, which is how the log fixture checks a fixed day.

```sh
loci import --index manuals notes.md records.jsonl
loci backup --data loci-data --out loci-data.sqlite
loci restore --input loci-data.sqlite --data loci-data-restored
```

Import sends batches to a running server. The named index must already exist. It reads `LOCI_TOKEN` when the server requires a bearer token. Backup writes a new SQLite snapshot and refuses to replace an existing file. Restore checks that snapshot, then copies it into an empty data directory.

The console at `/` and the clients under `clients/` call this API. Runnable Rust examples, one per workload:

```sh
cargo run --locked --example knowledge
cargo run --locked --example catalog
cargo run --locked --example records
cargo run --locked --example logs
```

`python tools/rehearse_cutover.py --binary target/debug/loci` imports two documents, checks an ordered query, restores a backup into an empty directory, and checks that a second restore refuses to overwrite it. That script stays in Python because it replays an Elasticsearch export. Product checks run in Rust: `cargo test --bin loci a_fresh_catalog_highlights_an_imported_note_within_ten_minutes` starts from an empty data directory, imports one Markdown note, and checks that search highlights a sentence from that note within ten minutes. `benchmarks/workloads/live.py --scorecard` records fixture disk, working set, and restart time.

The supported release is the latest version published from an annotated tag. `main` can move ahead of that release. An exact search that stops at `match_limit` sets `exact` to false and `limit_reason` to `match_limit` instead of presenting a partial total as complete. The default cap is 100000 authorized matches.
