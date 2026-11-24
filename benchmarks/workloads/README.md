# Search workload fixtures

Four small, hand-auditable datasets describe what users should find as documents change. They run offline with Python 3.9+ and its standard library. From the repository root, run

```sh
python benchmarks/workloads/validate.py
python benchmarks/workloads/validate.py --workload knowledge
```

The first command checks 19 cases. For structured queries, the validator applies the writes to an independent in-memory reference and compares the results with the frozen JSON expectations. For text searches, it checks the human-authored judgments and the evidence coordinates. This run provides no engine relevance, throughput, latency or pilot measurements.

| Fixture | Frozen behaviors |
| --- | --- |
| Knowledge | A paraphrase must find the recovery manual in the top three. The exact identifier must rank first despite being filed under Billing. A two-source question must surface both required passages with verifiable citations in the top three. |
| Catalog | An exact SKU must rank first. A misspelled product description must find the intended item in the top two. Stock and price changes must update the available IDs, ascending order, total and category facets. Prices are integer cents. |
| Business records | Group membership limits both results and facet counts. A numeric range includes its lower endpoint and excludes its upper endpoint. A second page continues the original result set after two writes. A fresh query shows the current order. Empty groups grant no access. |
| Logs | Hourly appends span a UTC day. Exact trace lookup, five-minute buckets including empty intervals, a late arrival and inclusive retention at the cutoff have frozen results. Reopening the process must preserve the retained events and bucket counts. |

## Grade an engine

The supplied live adapter starts a fresh service on a temporary loopback port, grants each record reader a separate credential, replays all writes and queries over HTTP, and reopens the same data directory at the log restart barrier. It prints a line per case and returns a failing status for a missing behavior. Build the current service from the repository root, then run

```sh
cargo build --locked --bin loci
python -B benchmarks/workloads/live.py --binary target/debug/loci --capture engine-results.jsonl
python benchmarks/workloads/validate.py --actual engine-results.jsonl
```

On Windows pass `target/debug/loci.exe`. The default run uses lexical retrieval. To also exercise Loci’s regional route, supply local assets for a pinned model. The service verifies the model in the temporary directory. This script makes no model downloads.

```sh
python -B benchmarks/workloads/live.py --binary target/debug/loci --workload knowledge --model bge-small-en-v1.5 --model-source /path/to/pinned/model
```

Use `--workload catalog`, `records`, `knowledge` or `logs` for a single case group. Each capture path must be new. A lexical knowledge hit cites the complete returned document. A regional hit uses the passage coordinates supplied by the engine. The fixture IDs for source passages map to the original source URIs. The record index grants access to explicitly named groups. The log index uses hourly rollover and a one-day retention period, with manual expiration so the replay uses the same date every time.

The Windows development build passed all four catalog cases, all five record cases and all seven log cases in a live replay. On the three knowledge queries, lexical retrieval passed two and omitted the release note from the two-source question. Both BGE-small-en-v1.5 and multilingual-E5-small passed all three through regional search. These are functional results on the checked-in fixtures. The lexical command above returns a failing status when it reproduces that missing passage.

## Compare another adapter

Export the ordered operations for an adapter and grade its captured results. PowerShell 5 users can use `Out-File -Encoding utf8` when saving UTF-8 output.

```sh
python benchmarks/workloads/validate.py --emit-requests
python benchmarks/workloads/validate.py --actual engine-results.jsonl
python benchmarks/workloads/validate.py --workload records --actual records-results.jsonl
```

Each emitted JSON line names the workload. Apply `reset` to a fresh, disposable index with the supplied documents. Apply `upsert` as a full replacement, `patch` as a partial update and `expire_before` as removal of every event strictly earlier than the supplied timestamp. The generator expands the 24 log appends into individual writes. Times are seconds since 2025-01-01 00:00 UTC. A `restart` operation requires stopping and reopening the actual engine against the same storage. Wait for each write to become searchable before continuing. Use the service's actual access controls for `principal`.

Only query operations have an `id`. Each `search` carries free text and a top-k limit. Each `select` carries structured predicates, facets, ordering, a page or a histogram. Apply numeric and timestamp ranges as half-open intervals and use the document ID as an ascending tiebreaker. A `save_cursor` value names the opaque cursor obtained from the engine. `resume` reuses that cursor with the same principal and filter. Facets and totals cover every visible match before pagination. Histogram buckets cover the entire requested interval, including zeros.

Emit one result row per query, with the workload and case ID. Capture the engine's values directly. The grader fails on missing, duplicate or unexpected cases, incorrect order, facets or totals, invalid evidence and missing judged hits. Ranked cases permit any order within the stated top k unless k is one. Every returned knowledge hit must carry a source URI, revision, UTF-8 byte interval and an exact quote from that interval. Judged hits must cover the labeled phrase. For example, one ranked result and one structured result have these shapes.

```jsonl
{"workload":"knowledge","id":"misplaced-id","result":{"hits":[{"id":"misc","uri":"tickets/138.md","revision":"r7","start":27,"end":34,"quote":"VLC-019"}]}}
{"workload":"records","id":"no-permission","result":{"ids":[],"count":0,"facets":{}}}
```

To compare engine versions, retain the fixture version, commit, adapter command, index configuration, model fingerprint and raw capture with the result. Report latency distributions and data volume separately from these tiny functional fixtures. A pilot needs its own representative data, user judgments and operating measurements.
