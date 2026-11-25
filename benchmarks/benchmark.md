# Benchmark results

Loci Search returns the same retrieval quality as Elasticsearch 8.19 and answers faster. On two public corpora, with both engines answering over HTTP on the same Linux machine, Recall@100 and nDCG@10 were statistically tied, Loci's p95 latency was 44% to 75% lower, and Loci served 1.6 to 6.6 times as many queries per second at 16 concurrent clients.

This page lists only the current results, measured with the hybrid search settings that `loci serve` uses. Earlier configurations, including the runs where Loci lost or tied, are in the [benchmark history](benchmark_history.md).

![p95 latency at equal quality. HotpotQA 1M raw text: Loci 19.2 ms, Elasticsearch 34.3 ms. HotpotQA 1M precomputed vectors: Loci 6.9 ms, Elasticsearch 20.4 ms. FiQA: Loci 3.9 ms, Elasticsearch 15.7 ms.](../docs/images/latency.svg)

![Recall@100 and nDCG@10 for Loci and Elasticsearch on HotpotQA 1M and FiQA. Every difference is within its 95% interval of zero.](../docs/images/quality.svg)

![Queries per second at 1, 4, 8, and 16 concurrent clients. HotpotQA 1M raw text: Loci 429, Elasticsearch 276 at 16 clients. FiQA: Loci 2,629, Elasticsearch 396 at 16 clients.](../docs/images/throughput.svg)

## How the comparison was run

| Condition | Setting |
| --- | --- |
| Machine | One Docker Desktop Linux VM, 16 CPUs, 19.5 GiB. Both engines and the benchmark client ran in the same VM and network namespace. |
| Transport | Both engines answered over loopback HTTP. Loci ran `loci bench-serve`; the client sent each engine the same query text and vector and read the returned passages from its own copy of the index. |
| Loci | Hybrid search with the `loci serve` settings (`SearchConfig::hybrid`): a keyword search for 60 passages over the 8 rarest query terms, 150 HNSW graph neighbours at width 128, and the reranker. No regional scan. |
| Elasticsearch | 8.19.25, one shard, English analyzer over all query terms, kNN with 256 candidates, a 4 GiB heap locked in memory with `bootstrap.memory_lock`. |
| Embeddings | Identical vectors in both engines: `bge-small-en-v1.5` for HotpotQA, Cohere English v3 for FiQA. |
| Reranking | The same reranker on both engines' candidates: 92% cosine similarity and 8% IDF-weighted term coverage over all query terms, or 65% coverage, 25% cosine, and 10% rank fusion for identifier-like queries such as `LMP-201`. Quality differences therefore come from candidate generation. |
| Queries | Held-out queries only, three passes each. Settings were chosen on separate development queries and frozen before the held-out run. |
| Paging | Swap counters were read before and after each run. No memory was swapped back in during any run. |

### How the Loci settings were chosen

On the development queries of each corpus, Loci swept 150, 200, and 300 graph candidates, graph widths of 128 and 256, and 40 or 60 keyword hits. A rule fixed before the run picked the fastest p95 within 0.2 points of the best development Recall@100. Both corpora chose 150 graph candidates at width 128.

Profiling then showed that the keyword search on long questions set the latency tail: questions with 17 or more distinct terms spent 6.3 ms on average there. Running the keyword search on only the rarest query terms, by the same rule, chose 8 terms. On HotpotQA development queries that cut p95 from 13.8 ms to 4.9 ms, with Recall@100 0.2 points lower and the same nDCG@10; FiQA development quality did not change. The reranker still scores coverage over every query term.

Queries that look like identifiers (at most four words, one of them a code such as `LMP-201`, `ERR_CONNECTION_RESET`, or a number of three or more digits) rerank with keyword-heavy weights, so an exact code still ranks first when its embedding is poor. No HotpotQA query matches that rule; one FiQA test query does (`Tax: 1099 paper form`).

## HotpotQA, one million passages

320 held-out questions from BEIR HotpotQA, three passes each, over one million passages embedded with `bge-small-en-v1.5`.

| Track | Engine | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw text, query encoding included | Loci | 91.09% | 0.8236 | 15.74 ms | 19.18 ms | 21.29 ms |
| Raw text, query encoding included | Elasticsearch | 90.63% | 0.8225 | 24.63 ms | 34.33 ms | 39.23 ms |
| Precomputed vectors | Loci | 91.09% | 0.8236 | 5.47 ms | 6.89 ms | 7.47 ms |
| Precomputed vectors | Elasticsearch | 90.63% | 0.8225 | 13.69 ms | 20.36 ms | 25.61 ms |

Loci minus Elasticsearch, paired by query: nDCG@10 +0.0011 (95% interval −0.0035 to +0.0060), Recall@100 +0.47 points (−0.31 to +1.25). Query encoding took 9.5 ms at p50 and is included for both engines in the raw-text track.

| Concurrent clients | Loci, raw text | Elasticsearch, raw text | Loci, vectors | Elasticsearch, vectors |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54.0 | 36.3 | 206.1 | 70.0 |
| 4 | 211.6 | 137.9 | 821.1 | 280.9 |
| 8 | 343.2 | 221.6 | 1,507.4 | 449.7 |
| 16 | 429.2 | 275.6 | 1,875.5 | 406.6 |

## FiQA

648 official test questions from BEIR FiQA, three passes each, over 57,638 passages with Cohere English v3 1,024-dimension vectors. Only the precomputed-vector track can run, because the Cohere model is not available locally.

| Engine | Candidate recall | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Loci | 79.24% | 74.00% | 0.4242 | 3.11 ms | 3.91 ms | 4.35 ms |
| Elasticsearch | 77.77% | 74.21% | 0.4253 | 14.29 ms | 15.67 ms | 17.41 ms |

Loci minus Elasticsearch, paired by query: nDCG@10 −0.0011 (95% interval −0.0049 to +0.0014), Recall@100 −0.21 points (−0.84 to +0.42).

| Concurrent clients | Loci | Elasticsearch |
| ---: | ---: | ---: |
| 1 | 373.6 | 69.8 |
| 4 | 1,340.3 | 259.9 |
| 8 | 2,134.5 | 397.4 |
| 16 | 2,629.3 | 396.1 |

## Memory and startup

On the HotpotQA index, the hybrid benchmark server held 2.6 GiB on Linux. On Windows, a hybrid server was ready to answer in 12.1 seconds with a 2.49 GB working set, against 24.8 seconds and 4.51 GB for regional search on the same index. Elasticsearch ran with a fixed 4 GiB heap.

## What these results do not show

- Two corpora on one machine. Other data, larger corpora, and other hardware may behave differently.
- Quality is a tie, not a win. Both engines' candidates go through the same reranker. With the older 65% coverage, 25% cosine reranker weights, Elasticsearch's HotpotQA candidates reached 0.790 nDCG@10; the cosine-heavy reranker lifts either engine.
- Loci's keyword search reads only the 8 rarest terms of a long question, while Elasticsearch's BM25 reads all of them. On these corpora that cost no measurable quality; on queries whose meaning sits in common words it could.
- The raw-text track includes about 9.5 ms of query encoding on both sides, which narrows the relative difference.
- Elasticsearch's throughput at 16 clients varies between runs. Across the recorded HTTP runs it ranged from 258 to 276 raw-text queries per second and 407 to 565 vector queries per second on HotpotQA, and 396 to 473 on FiQA. Loci led at every concurrency in every run.
- The comparison covers retrieval only. It does not cover indexing speed, aggregations, or Elasticsearch features that Loci does not have.

## Reproduce

The [reproduction guide](reproduce.md) covers the data and index builds. The held-out runs used these commands inside the Linux VM, with Elasticsearch listening on port 9200 in the same network namespace:

```sh
loci bench-serve --db /data/loci.sqlite --frozen-report /data/frozen-serve8.json --bind 127.0.0.1:7711
loci bench --db /data/loci.sqlite --queries /data/test.jsonl --frozen-report /data/frozen-serve8.json \
  --embed-model bge-small-en-v1.5 --model-cache /data/models \
  --elastic-url http://127.0.0.1:9200/ --elastic-index loci-hotpotqa-bge \
  --loci-url http://127.0.0.1:7711/ --throughput-width 1,4,8,16 --repeats 3 --out /data/out/serve8-raw
```

The frozen plan's hybrid settings are those of `SearchConfig::hybrid`: `graph_candidates` 150, `graph_width` 128, `rescue` 60, `rescue_max_terms` 8, `regional_rerank`, and `identifier_rerank`, which the Elasticsearch plan also sets. Leave out `--embed-model` and `--model-cache` for the precomputed-vector track. Each report records its settings, the index and Elasticsearch statistics it was checked against, paired confidence intervals, and throughput rows.
