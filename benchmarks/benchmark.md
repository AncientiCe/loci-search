# Benchmark results

Loci Search 0.4.0 returns the same retrieval quality as Elasticsearch 8.19 and answers faster. On two public corpora, with both engines answering over HTTP on the same Linux machine, Recall@100 and nDCG@10 were statistically tied, Loci's p95 latency was 25% to 72% lower, and Loci served 1.45 to 5.4 times as many queries per second.

This page lists only the current results. Earlier configurations, including the runs where Loci lost or tied, are in the [benchmark history](benchmark_history.md).

![p95 latency at equal quality. HotpotQA 1M raw text: Loci 25.4 ms, Elasticsearch 33.8 ms. HotpotQA 1M precomputed vectors: Loci 13.0 ms, Elasticsearch 20.3 ms. FiQA: Loci 4.2 ms, Elasticsearch 15.1 ms.](../docs/images/latency.svg)

![Recall@100 and nDCG@10 for Loci and Elasticsearch on HotpotQA 1M and FiQA. Every difference is within its 95% interval of zero.](../docs/images/quality.svg)

![Queries per second at 1, 4, 8, and 16 concurrent clients. HotpotQA 1M raw text: Loci 377, Elasticsearch 260 at 16 clients. FiQA: Loci 2,577, Elasticsearch 473 at 16 clients.](../docs/images/throughput.svg)

## How the comparison was run

| Condition | Setting |
| --- | --- |
| Machine | One Docker Desktop Linux VM, 16 CPUs, 19.5 GiB. Both engines and the benchmark client ran in the same VM and network namespace. |
| Transport | Both engines answered over loopback HTTP. Loci ran `loci bench-serve`; the client sent each engine the same query text and vector and read the returned passages from its own copy of the index. |
| Loci | 0.4.0 hybrid search: bounded global keyword search, the persisted HNSW graph, and the reranker. No regional scan. |
| Elasticsearch | 8.19.25, one shard, English analyzer, kNN with 256 candidates, a 4 GiB heap locked in memory with `bootstrap.memory_lock`. |
| Embeddings | Identical vectors in both engines: `bge-small-en-v1.5` for HotpotQA, Cohere English v3 for FiQA. |
| Reranking | The same reranker on both engines' candidates: 92% cosine similarity, 8% IDF-weighted term coverage. Quality differences therefore come from candidate generation. |
| Queries | Held-out queries only, three passes each. Settings were chosen on separate development queries and frozen before the held-out run. |
| Paging | Swap counters were read before and after each run. Neither run swapped memory back in. |

### How the Loci settings were chosen

On the development queries of each corpus, Loci swept 150, 200, and 300 graph candidates, graph widths of 128 and 256, and 40 or 60 keyword hits. A rule fixed before the run picked the fastest p95 within 0.2 points of the best development Recall@100. Both corpora chose 150 graph candidates at width 128; HotpotQA chose 60 keyword hits and FiQA 40. `loci serve` uses 150, 128, and 60 (`SearchConfig::hybrid`).

## HotpotQA, one million passages

320 held-out questions from BEIR HotpotQA, three passes each, over one million passages embedded with `bge-small-en-v1.5`.

| Track | Engine | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw text, query encoding included | Loci | 90.78% | 0.8219 | 16.86 ms | 25.35 ms | 29.89 ms |
| Raw text, query encoding included | Elasticsearch | 90.63% | 0.8225 | 23.83 ms | 33.78 ms | 38.69 ms |
| Precomputed vectors | Loci | 90.78% | 0.8219 | 6.73 ms | 13.00 ms | 16.17 ms |
| Precomputed vectors | Elasticsearch | 90.63% | 0.8225 | 13.33 ms | 20.31 ms | 25.32 ms |

Loci minus Elasticsearch, paired by query: nDCG@10 −0.0006 (95% interval −0.0042 to +0.0025), Recall@100 +0.16 points (−0.63 to +0.94). Query encoding took 9.2 ms at p50 and is included for both engines in the raw-text track.

| Concurrent clients | Loci, raw text | Elasticsearch, raw text | Loci, vectors | Elasticsearch, vectors |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50.4 | 37.1 | 141.2 | 71.0 |
| 4 | 192.5 | 140.7 | 562.9 | 275.5 |
| 8 | 310.9 | 212.6 | 1,025.0 | 464.8 |
| 16 | 376.9 | 259.9 | 1,390.9 | 565.4 |

## FiQA

648 official test questions from BEIR FiQA, three passes each, over 57,638 passages with Cohere English v3 1,024-dimension vectors. Only the precomputed-vector track can run, because the Cohere model is not available locally.

| Engine | Candidate recall | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Loci | 79.07% | 74.00% | 0.4252 | 3.36 ms | 4.20 ms | 4.59 ms |
| Elasticsearch | 77.77% | 74.21% | 0.4261 | 13.99 ms | 15.06 ms | 16.89 ms |

Loci minus Elasticsearch, paired by query: nDCG@10 −0.0009 (95% interval −0.0046 to +0.0018), Recall@100 −0.21 points (−0.84 to +0.42).

| Concurrent clients | Loci | Elasticsearch |
| ---: | ---: | ---: |
| 1 | 334.9 | 72.2 |
| 4 | 1,279.5 | 274.4 |
| 8 | 2,205.6 | 418.4 |
| 16 | 2,576.5 | 473.4 |

## Memory and startup

On the HotpotQA index, the hybrid benchmark server held 2.6 GiB on Linux. On Windows, a hybrid server was ready to answer in 12.1 seconds with a 2.49 GB working set, against 24.8 seconds and 4.51 GB for regional search on the same index. Elasticsearch ran with a fixed 4 GiB heap.

## What these results do not show

- Two corpora on one machine. Other data, larger corpora, and other hardware may behave differently.
- Quality is a tie, not a win. Both engines' candidates go through the same reranker. With the older 65% coverage, 25% cosine reranker weights, Elasticsearch's HotpotQA candidates reached 0.790 nDCG@10; the cosine-heavy reranker lifts either engine.
- The raw-text track includes about 9 ms of query encoding on both sides, which narrows the relative difference.
- The comparison covers retrieval only. It does not cover indexing speed, aggregations, or Elasticsearch features that Loci does not have.

## Reproduce

The [reproduction guide](reproduce.md) covers the data and index builds. The held-out runs used these commands inside the Linux VM, with Elasticsearch listening on port 9200 in the same network namespace:

```sh
loci bench-serve --db /data/loci.sqlite --frozen-report /data/frozen-hybrid.json --bind 127.0.0.1:7711
loci bench --db /data/loci.sqlite --queries /data/test.jsonl --frozen-report /data/frozen-hybrid.json \
  --embed-model bge-small-en-v1.5 --model-cache /data/models \
  --elastic-url http://127.0.0.1:9200/ --elastic-index loci-hotpotqa-bge \
  --loci-url http://127.0.0.1:7711/ --throughput-width 1,4,8,16 --repeats 3 --out /data/out/http-hybrid-raw
```

Leave out `--embed-model` and `--model-cache` for the precomputed-vector track. Each report records its settings, the index and Elasticsearch statistics it was checked against, paired confidence intervals, and throughput rows.
