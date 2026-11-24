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

Create an index with `PUT /v1/indexes/{name}`. `"model": "lexical"` stores text only. `"bge-small-en-v1.5"` or `"multilingual-e5-small"` embeds each write. Install the pack before the first embedded write:

```sh
loci model-install bge-small-en-v1.5 --data loci-data
loci model-install multilingual-e5-small --data loci-data --source /path/to/verified-pack
```

`PUT /v1/indexes/{name}/model` rebuilds every document vector and publishes the new model in one generation. A reader opened on the previous generation keeps that generation. A bulk write that started against the old model is rejected.

The service runs maintenance about once a minute. It deletes expired cursor pages, merges lexical segments, and expires events on indexes whose stream policy has `"automatic": true`. `POST /v1/indexes/{name}/expire` applies the same retention cutoff at a caller-supplied timestamp, which is how the log fixture checks a fixed day.

```sh
loci import --index manuals notes.md records.jsonl
loci backup --data loci-data --out loci-data.sqlite
loci restore --input loci-data.sqlite --data loci-data-restored
```

Import sends batches to a running server. The named index must already exist. It reads `LOCI_TOKEN` when the server requires a bearer token. Backup writes a new SQLite snapshot and refuses to replace an existing file. Restore checks that snapshot, then copies it into an empty data directory.

The console at `/` and the clients under `clients/` call this API. The embedded example is `cargo run --locked --example embedded`.

`python tools/rehearse_cutover.py --binary target/debug/loci` imports two documents, checks an ordered query, restores a backup into an empty directory, and checks that a second restore refuses to overwrite it.
