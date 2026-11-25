# Benchmark history

Every recorded comparison, newest first, including the runs where Loci lost or tied. For the results that describe the current release, see [Benchmark results](benchmark.md).

The latency columns summarize all recorded warm sequential queries. The [machine-readable history](history.json) includes per-pass timings, dev trials, index preparation, and exact settings. See the [reproduction instructions](reproduce.md) for the corpus and launch steps.

## Raw-text verdict against Elasticsearch

These runs asked whether Loci beats Elasticsearch when both receive query text, not precomputed vectors. Every sample includes encoding the query with `bge-small-en-v1.5`, about 9 ms at p50. Both engines searched the same one million HotpotQA passages and vectors and returned 100 hits for each of the 320 held-out queries, three passes each.

Loci used the settings that had won on the earlier precomputed-vector runs: a 12,000-passage regional budget, 100 HNSW candidates, int8 scanning with 400 exact re-scores, bounded MaxScore rescue, and the 8% coverage, 92% cosine reranker. Both the 12,000 and 24,000 budgets reached 92.2% development Recall@100, so the faster one was frozen. Elasticsearch 8.19.25 ran its English analyzer with 256 kNN candidates, which matched its development recall, a 4 GiB heap, and 16 processors.

On Linux, in a Docker Desktop VM with 16 CPUs and 16 GiB, both engines ran on the same VM:

| Strategy | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Loci regional | 91.09% | 0.8223 | 26.26 ms | 37.49 ms | 42.35 ms |
| Elasticsearch | 90.31% | 0.7900 | 24.67 ms | 34.25 ms | 39.23 ms |

Queries per second with concurrent clients on one shared, prepared engine. Each worker encodes its own queries:

| Clients | Loci | Elasticsearch |
| ---: | ---: | ---: |
| 1 | 33.0 | 35.8 |
| 4 | 124.3 | 136.7 |
| 8 | 219.5 | 200.2 |
| 16 | 281.6 | 227.2 |

On Windows, Elasticsearch stopped at about 104 hybrid queries per second while using under 60% of the CPU, and Loci reached three times its throughput. The same Elasticsearch index on Linux served 547 hybrid queries per second when queried directly, so the Windows lead does not hold.

The 0.032 nDCG@10 lead in the Linux run came from the reranker. That run reranked Elasticsearch candidates with the default 65% coverage, 25% cosine, and 10% fusion weights. With `elastic_rerank` set to Loci's weights, Elasticsearch matched Loci:

| Corpus | Loci Recall@100 / nDCG@10 | Elasticsearch, same reranker | Loci − Elasticsearch nDCG@10 (95% interval) |
| --- | --- | --- | --- |
| HotpotQA 1M, raw text | 91.09% / 0.8223 | 90.62% / 0.8225 | −0.0002 (−0.0029 to +0.0025) |
| FiQA | 73.64% / 0.4266 | 74.21% / 0.4261 | +0.0005 (−0.0022 to +0.0036) |

The earlier FiQA result, 8.67 percentage points of Recall@100 above Elasticsearch, therefore measured the reranker, not the engine. The NFCorpus control did not run because the Elasticsearch index statistics no longer matched its frozen report.

With the same embeddings and reranker, Loci matches Elasticsearch on recall and top-10 quality and does not beat it on latency or throughput. The cosine-heavy reranker over fused BM25 and kNN candidates helped both engines.

## Regions off: graph, rescue, and reranker only

The pruning experiment below showed that the regional scan costs time without adding recall. With regions switched off (one region, one chunk, one posting), Loci searches only the persisted HNSW graph (150 candidates, width 128), bounded global rescue (60 hits), and the 8% coverage, 92% cosine reranker. Those settings were chosen on the 250 development queries by a rule fixed before the run: the fastest p95 within 0.2 percentage points of the best Recall@100.

The held-out run repeated the E1 setup on Linux: both engines in one Docker Desktop VM with 16 CPUs and 16 GiB, the same one million HotpotQA passages and `bge-small-en-v1.5` vectors, 320 held-out queries with three passes each, and Elasticsearch 8.19.25 with its English analyzer, 256 kNN candidates, a 4 GiB heap, and Loci's reranker. Both engines read reranker coverage from Loci's in-memory postings.

| Track | Engine | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw text, query encoding included | Loci | 90.78% | 0.8219 | 17.13 ms | 25.71 ms | 30.74 ms |
| Raw text, query encoding included | Elasticsearch | 90.63% | 0.8225 | 24.18 ms | 34.54 ms | 40.82 ms |
| Precomputed vectors | Loci | 90.78% | 0.8219 | 5.83 ms | 11.82 ms | 14.92 ms |
| Precomputed vectors | Elasticsearch | 90.63% | 0.8225 | 13.49 ms | 20.69 ms | 24.81 ms |

Loci minus Elasticsearch: nDCG@10 −0.0006 (95% interval −0.0042 to +0.0025), Recall@100 +0.16 points (−0.63 to +0.94). Query encoding took 9.3 ms at p50.

Queries per second with concurrent clients:

| Clients | Loci, raw text | Elasticsearch, raw text | Loci, vectors | Elasticsearch, vectors |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 49.5 | 35.5 | 158.3 | 70.4 |
| 4 | 181.7 | 132.6 | 597.5 | 267.9 |
| 8 | 319.7 | 198.8 | 1,060.3 | 453.4 |
| 16 | 374.7 | 234.9 | 1,656.4 | 548.4 |

At equal quality, Loci's raw-text p95 is 25.6% lower than Elasticsearch's, which meets the 20% target, and Loci serves more queries per second at every concurrency. Loci ran in-process in that run, while Elasticsearch answered over loopback HTTP.

### Both engines over HTTP

`loci bench-serve` held the same prepared engine and answered over loopback HTTP; `loci bench --loci-url` sent it the query text and vector exactly as it sends them to Elasticsearch, and read the returned passages from its own index, as the Elasticsearch path does. The Docker VM had 19.5 GiB. Quality was identical to the in-process run.

| Track | Loci p50 / p95 | Elasticsearch p50 / p95 | Loci p95 vs Elasticsearch |
| --- | --- | --- | ---: |
| Precomputed vectors | 7.12 / 13.39 ms | 13.14 / 19.49 ms | −31.3% |
| Raw text, run A | 17.75 / 25.65 ms | 23.34 / 32.40 ms | −20.8% |
| Raw text, run B, Elasticsearch heap locked | 17.85 / 26.22 ms | 23.59 / 32.37 ms | −19.0% |

Queries per second at 16 clients: raw text 377.3 and 367.3 for Loci against 259.9 and 258.5 for Elasticsearch; precomputed vectors 1,357.8 against 560.4. Loci led at every concurrency in every run.

The VM paged memory to swap during both raw-text runs although memory was available. In run A it paged 1.8 GB of Elasticsearch's Java heap; in run B, with `bootstrap.memory_lock`, it paged about 200 MB of the Loci server and part of the benchmark client instead. The precomputed-vector run saw almost no swap-ins.

### Hybrid search over HTTP, without swap

`Strategy::Hybrid` runs the same graph, rescue, and reranker without the regional scan and loads a lean index without per-region postings or int8 vectors. The hybrid server held 2.6 GiB against 4.4 GiB for regional search with budgets of 1, so this run swapped out 70 pages and swapped none back in. Elasticsearch kept its heap locked. Quality was identical to the runs above.

| Track | Loci hybrid p50 / p95 / p99 | Elasticsearch p50 / p95 / p99 | Loci p95 vs Elasticsearch |
| --- | --- | --- | ---: |
| Raw text, query encoding included | 16.86 / 25.35 / 29.89 ms | 23.83 / 33.78 / 38.69 ms | −25.0% |
| Precomputed vectors | 6.73 / 13.00 / 16.17 ms | 13.33 / 20.31 / 25.32 ms | −36.0% |

| Clients | Loci, raw text | Elasticsearch, raw text | Loci, vectors | Elasticsearch, vectors |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50.4 | 37.1 | 141.2 | 71.0 |
| 4 | 192.5 | 140.7 | 562.9 | 275.5 |
| 8 | 310.9 | 212.6 | 1,025.0 | 464.8 |
| 16 | 376.9 | 259.9 | 1,390.9 | 565.4 |

With both engines over HTTP, equal quality, and no paging, Loci's raw-text p95 is 25% lower than Elasticsearch's, which meets the 20% target, and its engine-only p95 is 36% lower. Loci serves 1.45 times the raw-text queries per second at 16 clients and 2.5 times with precomputed vectors.

### FiQA, hybrid search over HTTP

The same procedure on FiQA: 57,638 passages with Cohere English v3 1,024-dimension vectors, so only the precomputed-vector track can run. Hybrid settings were chosen on the 500 development queries by the same rule and frozen at 150 graph candidates, width 128, and 60 → 40 rescue hits; on development queries hybrid search reached 74.66% Recall@100 against 74.75% for an exhaustive scan and 74.18% for regional search. The 648 official test queries then ran three times on Linux with both engines over HTTP, Elasticsearch's frozen E2 settings (English analyzer, 256 kNN candidates, Loci's reranker), its heap locked, and no swap.

| Engine | Candidate recall | Recall@100 | nDCG@10 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Loci hybrid | 79.07% | 74.00% | 0.4252 | 3.36 ms | 4.20 ms | 4.59 ms |
| Elasticsearch | 77.77% | 74.21% | 0.4261 | 13.99 ms | 15.06 ms | 16.89 ms |

Loci minus Elasticsearch: nDCG@10 −0.0009 (95% interval −0.0046 to +0.0018), Recall@100 −0.21 points (−0.84 to +0.42).

| Clients | Loci hybrid | Elasticsearch |
| ---: | ---: | ---: |
| 1 | 334.9 | 72.2 |
| 4 | 1,279.5 | 274.4 |
| 8 | 2,205.6 | 418.4 |
| 16 | 2,576.5 | 473.4 |

On FiQA, at equal quality, Loci's p95 is 72% lower and it serves 5.4 times the queries per second at 16 clients. Both corpora now show the same result: a tie on quality and a clear lead on latency and throughput.

### FiQA with identifier weights, as `loci serve` runs it

Hybrid search's 92% cosine reranker could drop an exact code whose embedding pointed elsewhere: on the synthetic fixture, `NOVA000X` fell out of the top five. `SearchConfig::hybrid` now reranks identifier-like queries with 65% coverage, 25% cosine, and 10% fusion. No HotpotQA query matches the rule, so the HotpotQA results above are unchanged. One FiQA test query matches (`Tax: 1099 paper form`), so FiQA ran again with the rule on both engines, the same settings otherwise, and no swap.

| Engine | Candidate recall | Recall@100 | nDCG@10 | p50 | p95 | p99 | Queries per second at 16 clients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Loci hybrid | 79.07% | 74.00% | 0.4242 | 3.67 ms | 4.73 ms | 5.30 ms | 2,727.3 |
| Elasticsearch | 77.77% | 74.21% | 0.4253 | 14.53 ms | 17.02 ms | 20.39 ms | 472.5 |

Loci minus Elasticsearch: nDCG@10 −0.0011 (95% interval −0.0049 to +0.0014), Recall@100 −0.21 points (−0.84 to +0.42). The identifier weights lowered nDCG@10 on that one query for both engines, and the comparison is unchanged: tied quality, 72% lower p95, and 5.8 times the queries per second.

## Can the topology rule regions out?

`loci pruning-dev` scores all one million HotpotQA passages with the regional reranker (92% cosine, 8% term coverage) and compares each plan's top 100 with that exhaustive top 100. The 250 development queries used precomputed vectors on Windows, so the timings compare rows only.

| Plan | Overlap with exhaustive top 100 | Recall@100 | Passages scored per query |
| --- | ---: | ---: | ---: |
| Exhaustive scan | 100% | 93.00% | 1,000,000 |
| Frozen regional plan | 85.54% | 92.20% | 12,327 |
| Global rescue capped at 1% of the corpus | 85.54% | 92.20% | 12,327 |
| No global rescue | 87.24% | 88.60% | 11,939 |
| No HNSW graph | 62.56% | 89.40% | 12,327 |
| Regions only | 58.64% | 57.40% | 11,939 |
| Branch-and-bound, safe radius | 100% | 93.00% | 1,000,000 |
| Branch-and-bound, radius × 0.1 | 99.99% | 93.00% | 702,088 |
| Branch-and-bound, centroid only | 99.57% | 93.00% | 287,761 |

Each of the 1,388 leaves spans 50.5° to 54.4° around its centroid at the 5th and 95th percentiles. A cone that wide gives nearly every leaf a cosine bound above the 100th-best score, so the safe bound skipped no passage. Even scoring leaves by centroid similarity alone, which gives up the guarantee, had to score 29% of the corpus before it could stop. The topology routes queries usefully, but it cannot certify that a region holds no top-100 passage. The HNSW graph and global rescue supply most of the regional plan's recall.

## Filtered search

`loci filter-dev` restricts both engines to passages whose id hashes into the first buckets of 1,000, a filter unrelated to topic. Elasticsearch ran from `loci elastic-buckets`, an index built from Loci's own passages and vectors with an integer `bucket` field, and its candidates were reranked with Loci's reranker. Each cell gives overlap with the exhaustive filtered top 100, then p95 latency. The 250 development queries ran on Windows, where Elasticsearch had 4 processors, so the latencies are indicative only.

| Admitted | Brute force over admitted | Loci, region budget × 1 / × 2 / × 4 | Elasticsearch, 200 / 400 / 1536 kNN candidates |
| --- | --- | --- | --- |
| 0.1% | 100% / 3.6 ms | 96.7% / 76.1 ms (× 1) | 99.9% / 23.9 ms (1536) |
| 1% | 100% / 22.9 ms | 98.3% / 75.0 ms (× 1) | 98.5% / 30.3 ms (1536) |
| 10% | 100% / 203 ms | 90.5% / 29.5 · 93.9% / 47.8 · 95.4% / 86.5 ms | 92.5% / 33.6 · 94.2% / 35.5 · 95.5% / 75.4 ms |
| 50% | 100% / 847 ms | 82.9% / 19.3 · 87.4% / 26.6 · 90.3% / 40.8 ms | 87.7% / 39.2 · 90.4% / 37.7 · 92.4% / 39.0 ms |

Below 1% admitted, scoring the admitted passages directly is fastest and exact. At 10%, Elasticsearch with 400 candidates reaches 94.2% overlap at 35.5 ms, where Loci needs a doubled budget and 47.8 ms. At 50%, Loci is faster only at the low-quality end and cannot reach Elasticsearch's 92.4%. Filters do not give Loci an advantage: Elasticsearch's slowdown at 10% came from its 1,536-candidate setting, and fewer candidates kept its quality.

## Query latency and recall

| Run | Passages | Queries × passes | Strategy | Recall@100 | Warm p50 | Warm p95 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Synthetic 100K, initial | 100,000 | 32 × 3 | SQLite BM25 | 58.33% | 0.12 ms | 8.76 ms |
| Synthetic 100K, initial | 100,000 | 32 × 3 | Loci global | 76.04% | 1.52 ms | 9.47 ms |
| Synthetic 100K, initial | 100,000 | 32 × 3 | Loci regional | 92.71% | 2.62 ms | 18.35 ms |
| Synthetic 100K, verified | 100,000 | 32 × 3 | SQLite BM25 | 58.33% | 0.13 ms | 9.62 ms |
| Synthetic 100K, verified | 100,000 | 32 × 3 | Loci global | 76.04% | 1.83 ms | 10.08 ms |
| Synthetic 100K, verified | 100,000 | 32 × 3 | Loci regional | 92.71% | 2.82 ms | 19.97 ms |
| CLI fixture smoke | 128 | 32 × 1 | SQLite BM25 | 75.00% | 0.15 ms | 0.38 ms |
| CLI fixture smoke | 128 | 32 × 1 | Loci global | 100.00% | 2.37 ms | 2.60 ms |
| CLI fixture smoke | 128 | 32 × 1 | Loci regional | 100.00% | 1.89 ms | 2.39 ms |
| FiQA, tuned baseline | 57,638 | 648 × 3 | SQLite BM25 | 51.87% | 44.23 ms | 81.75 ms |
| FiQA, tuned baseline | 57,638 | 648 × 3 | Loci global | 63.05% | 50.66 ms | 88.24 ms |
| FiQA, tuned baseline | 57,638 | 648 × 3 | Loci regional | 62.12% | 90.03 ms | 153.97 ms |
| FiQA, tuned baseline | 57,638 | 648 × 3 | Elasticsearch | 64.98% | 18.01 ms | 20.98 ms |
| FiQA, optimized regional engine | 57,638 | 648 × 3 | SQLite BM25 | 51.87% | 42.32 ms | 77.32 ms |
| FiQA, optimized regional engine | 57,638 | 648 × 3 | Loci global | 63.05% | 48.64 ms | 85.06 ms |
| FiQA, optimized regional engine | 57,638 | 648 × 3 | Loci regional | 62.12% | 11.01 ms | 13.49 ms |
| FiQA, optimized regional engine | 57,638 | 648 × 3 | Elasticsearch | 64.98% | 17.95 ms | 21.34 ms |
| FiQA, dev-selected reranker | 57,638 | 648 × 3 | SQLite BM25 | 51.87% | 44.57 ms | 82.85 ms |
| FiQA, dev-selected reranker | 57,638 | 648 × 3 | Loci global | 63.05% | 52.05 ms | 90.94 ms |
| FiQA, dev-selected reranker | 57,638 | 648 × 3 | Loci regional | 73.64% | 11.66 ms | 14.33 ms |
| FiQA, dev-selected reranker | 57,638 | 648 × 3 | Elasticsearch | 64.98% | 18.83 ms | 21.21 ms |
| FiQA harness synthetic smoke | 128 | 32 × 1 | SQLite BM25 | 75.00% | 0.04 ms | 0.09 ms |
| FiQA harness synthetic smoke | 128 | 32 × 1 | Loci global | 100.00% | 0.36 ms | 0.43 ms |
| FiQA harness synthetic smoke | 128 | 32 × 1 | Loci regional | 100.00% | 0.34 ms | 0.46 ms |
| NFCorpus, frozen FiQA transfer | 3,633 | 323 × 3 | SQLite BM25 | 23.42% | 0.71 ms | 3.38 ms |
| NFCorpus, frozen FiQA transfer | 3,633 | 323 × 3 | Loci global | 32.15% | 8.06 ms | 11.53 ms |
| NFCorpus, frozen FiQA transfer | 3,633 | 323 × 3 | Loci regional | 35.41% | 1.23 ms | 1.85 ms |
| NFCorpus, frozen FiQA transfer | 3,633 | 323 × 3 | Elasticsearch | 32.99% | 16.71 ms | 20.70 ms |
| NFCorpus harness synthetic smoke | 128 | 32 × 1 | SQLite BM25 | 75.00% | 0.04 ms | 0.09 ms |
| NFCorpus harness synthetic smoke | 128 | 32 × 1 | Loci global | 100.00% | 0.38 ms | 0.46 ms |
| NFCorpus harness synthetic smoke | 128 | 32 × 1 | Loci regional | 100.00% | 0.15 ms | 0.18 ms |
| HotpotQA 1M, frozen FiQA transfer | 1,000,000 | 320 × 3 | Loci regional | 86.25% | 35.08 ms | 44.26 ms |
| HotpotQA 1M, frozen FiQA transfer | 1,000,000 | 320 × 3 | Elasticsearch | 90.00% | 20.64 ms | 30.38 ms |
| HotpotQA 1M, MaxScore rescue | 1,000,000 | 320 × 3 | Loci regional | 86.25% | 27.65 ms | 39.07 ms |
| HotpotQA 1M, MaxScore rescue | 1,000,000 | 320 × 3 | Elasticsearch | 90.00% | 21.24 ms | 32.78 ms |
| HotpotQA 1M, int8 scan | 1,000,000 | 320 × 3 | Loci regional | 86.25% | 24.32 ms | 34.17 ms |
| HotpotQA 1M, int8 scan | 1,000,000 | 320 × 3 | Elasticsearch | 90.00% | 22.72 ms | 33.32 ms |
| HotpotQA 1M, HNSW candidates | 1,000,000 | 320 × 3 | Loci regional | 89.53% | 33.03 ms | 47.29 ms |
| HotpotQA 1M, HNSW candidates | 1,000,000 | 320 × 3 | Elasticsearch | 90.00% | 25.57 ms | 37.33 ms |
| HotpotQA 1M, 12k budget and HNSW, fresh Elasticsearch | 1,000,000 | 320 × 3 | Loci regional | 89.38% | 19.31 ms | 31.01 ms |
| HotpotQA 1M, 12k budget and HNSW, fresh Elasticsearch | 1,000,000 | 320 × 3 | Elasticsearch | 90.00% | 27.45 ms | 41.20 ms |
| HotpotQA 1M, BGE raw text | 1,000,000 | 320 × 3 | SQLite BM25 | 84.53% | 780.97 ms | 1255.60 ms |
| HotpotQA 1M, BGE raw text | 1,000,000 | 320 × 3 | Loci global | 90.47% | 791.24 ms | 1279.39 ms |
| HotpotQA 1M, BGE raw text | 1,000,000 | 320 × 3 | Loci regional | 89.06% | 39.54 ms | 50.66 ms |
| HotpotQA 1M, BGE raw text | 1,000,000 | 320 × 3 | Elasticsearch | 90.47% | 31.29 ms | 37.17 ms |

The BGE raw-text rows use the same `bge-small-en-v1.5` vectors on both engines and include query embedding time. Embedding itself was 8.64 ms at p50 and 12.17 ms at p95. On the held-out queries, regional search was 0.017 nDCG@10 and 1.41 percentage points of Recall@100 behind Elasticsearch, and its p95 was higher. Global search matched Elasticsearch recall and stayed within 0.003 nDCG@10, with a p95 of 1.28 s because lexical rescue scored most of the corpus.

## Loci preparation timings

| Run | Ingest | Open | Global HNSW build | Regional cache |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 100K, initial | 5685.58 ms | 91.61 ms | 52333.03 ms | — |
| Synthetic 100K, verified | 5448.29 ms | 87.72 ms | 52923.18 ms | — |
| CLI fixture smoke | — | 1.39 ms | 286.00 ms | — |
| FiQA, tuned baseline | — | 330.36 ms | 74342.37 ms | — |
| FiQA, optimized regional engine | — | 339.25 ms | 69703.97 ms | 2417.88 ms |
| FiQA, dev-selected reranker | — | 318.72 ms | 91495.28 ms | 4934.36 ms |
| FiQA harness synthetic smoke | 6.69 ms | 1.19 ms | 10.57 ms | — |
| NFCorpus, frozen FiQA transfer | — | 23.96 ms | 2226.12 ms | 181.84 ms |
| NFCorpus harness synthetic smoke | 6.61 ms | 1.18 ms | 10.56 ms | 0.32 ms |
| HotpotQA 1M, frozen FiQA transfer | — | 68256.07 ms | — | 20237.52 ms |
| HotpotQA 1M, MaxScore rescue | — | 5782.56 ms | — | 21426.41 ms |
| HotpotQA 1M, int8 scan | — | 18910.61 ms | — | 21436.15 ms |

An em dash denotes a timing absent from the recorded report. Fixtures and smoke runs use synthetic inputs. The HotpotQA head-to-head omitted the global control. The MaxScore run opened the index with a warm operating-system file cache.

## Recorded index builds

| Corpus and engine | Build time |
| --- | ---: |
| FiQA Loci | 45.92 s |
| FiQA Elasticsearch | 106.51 s |
| NFCorpus Elasticsearch | 6.07 s |
| HotpotQA 1M Loci | 442.44 s |
| HotpotQA 1M Elasticsearch | — |

The HotpotQA Elasticsearch index finished with eight idle segments. Its asynchronous task record failed to persist, leaving the total build time absent.

## FiQA development profiles

Measured over 500 development queries.

| Phase | Before mean | After mean | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: |
| global lexical rescue | 44.26 ms | 0.46 ms | 84.70 ms | 0.67 ms |
| routing | 0.17 ms | 0.12 ms | 0.25 ms | 0.15 ms |
| vectors and memberships | 12.35 ms | 6.40 ms | 16.23 ms | 7.74 ms |
| regional postings | 27.36 ms | 1.12 ms | 52.82 ms | 2.13 ms |
| regional top k | 0.96 ms | 0.28 ms | 1.27 ms | 0.37 ms |
| rerank and hits | 7.01 ms | 1.00 ms | 8.99 ms | 1.28 ms |
| other | 0.87 ms | 0.17 ms | 1.36 ms | 0.22 ms |
| total | 92.98 ms | 9.55 ms | 163.39 ms | 11.52 ms |

## FiQA development reranking

| Check | Loci regional p95 | Elasticsearch p95 |
| --- | ---: | ---: |
| initial recall audit | 13.42 ms | 19.53 ms |
| weight sweep baseline | 14.42 ms | 22.61 ms |
| selected reranker validation | 13.72 ms | 21.92 ms |

The machine-readable history contains the twelve timed regional reranker trials. Earlier analyzer and candidate sweeps recorded development quality and the selected configuration.

## HotpotQA 1M global lexical rescue

Measured over 250 development queries with frozen regional settings. WAND and MaxScore used a 256,000-passage cap, which no query reached. All three produced identical rankings, rescue ranks, and routes.

| Rescue | Rescue mean | Rescue p95 | Query mean | Query p95 |
| --- | ---: | ---: | ---: | ---: |
| dense scan | 11.91 ms | 15.52 ms | 35.95 ms | 43.27 ms |
| WAND | 40.40 ms | 110.35 ms | 66.20 ms | 139.23 ms |
| MaxScore | 4.62 ms | 10.91 ms | 28.56 ms | 36.84 ms |

On the 320 held-out queries, MaxScore scored a mean of 494 rescue passages instead of 894,306. All 960 regional traces matched the original run on rankings, candidates, and scores.

A `-C target-cpu=native` build (AVX2 and FMA) cut the vectors and memberships phase from 12.98 ms to 10.80 ms mean but query p95 only from 36.84 ms to 35.98 ms, with identical signatures. The phase reads about 98 MB of scattered float32 vectors per query and is memory-bound, so the default build stays portable.

## HotpotQA 1M int8 regional scan

Measured over 250 development queries with MaxScore rescue. The int8 scan scores passages against one byte per dimension plus a per-vector scale, then re-scores the best candidates with exact float32 cosine. The int8 copy takes 1.03 GB.

| Vector scan | Vectors mean | Query mean | Query p95 | Recall@100 | Rankings identical to exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact float32 | 12.98 ms | 28.56 ms | 36.84 ms | 88.40% | — |
| int8, re-score 200 | 8.74 ms | 23.72 ms | 33.08 ms | 88.40% | 248 of 250 |
| int8, re-score 400 | 8.99 ms | 24.11 ms | 35.33 ms | 88.40% | 250 of 250 |
| int8, re-score 800 | 8.97 ms | 23.68 ms | 32.34 ms | 88.40% | 250 of 250 |
| int8, re-score 1600 | 11.14 ms | 30.21 ms | 41.35 ms | 88.40% | 250 of 250 |

The held-out run uses 400, the smallest depth with identical development rankings. Single-pass p95 varies by about 2 ms between runs.

On the 320 held-out queries with re-score depth 400, Loci regional p95 fell from 39.07 ms to 34.17 ms against 33.32 ms for Elasticsearch in the same interleaved run. Recall@100 stayed at 86.25%, and all 960 regional traces matched the MaxScore run on rankings and candidates.

## HotpotQA 1M recall diagnosis

On 250 development queries, Elasticsearch returned 18 judged passages in its top 100 that never entered the Loci regional candidate pool; the reverse happened twice. Of the 53 judged passages missing from regional candidates, about 28 rank in the exact global dense top 200 and 9 are the single nearest passage, while most rank beyond 40 in BM25. They are routing misses: the 1M topology has 365 regions with 8,192-passage leaves, and the 24,000-passage budget expands about 2.7 of them.

Raising the budget recovers recall, with roughly linear cost (MaxScore rescue, int8 re-score 400, regions and postings scaled with chunks):

| Chunk budget | Recall@100 | Query mean | Query p95 |
| ---: | ---: | ---: | ---: |
| 24,000 | 88.40% | 28.96 ms | 42.00 ms |
| 48,000 | 90.60% | 38.09 ms | 49.26 ms |
| 96,000 | 91.40% | 69.94 ms | 88.14 ms |
| 192,000 | 91.60% | 178.27 ms | 233.39 ms |

Frozen Elasticsearch reached 91.60% development Recall@100 in the audit.

Finer topologies at the same 24,000-passage budget, re-ingested from the same passages and vectors:

| Leaf size | Regions | Recall@100 | Routing mean | Query mean | Query p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 365 | 88.40% | 0.20 ms | 28.96 ms | 42.00 ms |
| 2,048 | 1,503 | 89.80% | 1.22 ms | 49.04 ms | 64.97 ms |
| 1,024 | 2,981 | 89.80% | 1.62 ms | 32.82 ms | 43.85 ms |

The leaf-2,048 profile ran directly after that index was written and its cache preparation took twice as long, so its latency is suspect.

Leaf 1,024 with larger budgets (the 24,000 row is a rerun of the earlier profile, which overwrote its file; recall matched and timing moved by about 5 ms):

| Leaf size | Chunk budget | Recall@100 | Query mean | Query p95 |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 24,000 | 89.80% | 27.57 ms | 36.34 ms |
| 1,024 | 36,000 | 90.60% | 36.19 ms | 47.70 ms |
| 1,024 | 48,000 | 90.80% | 41.29 ms | 52.18 ms |

Leaf 1,024 at 36,000 matches the coarse index at 48,000 (90.60%) slightly faster, and returns diminish beyond it. Neither topology reaches the 91.60% Elasticsearch development recall below 96,000 scanned passages.

Seeding more leaf entries into routing (leaf 1,024, 24,000 chunks) did not recover the misses: Recall@100 was 89.80% with 2 entries and 90.00% with 8, 24, or 64, with p95 between 34.97 ms and 50.20 ms across single-pass runs. The remaining misses lie in leaves whose centroids score low for the query even when the passage itself is among its nearest neighbors, so region probing alone does not reach them cheaply.

## HotpotQA 1M global HNSW candidates

`loci graph-build` built a batched HNSW graph over all 1,000,000 vectors in 369 seconds on 16 threads and saved it as a 140 MB file next to the index. Regional search then merges the graph's nearest passages into its dense list before candidate selection.

| Development setting | Recall@100 | Graph mean | Query mean | Query p95 |
| --- | ---: | ---: | ---: | ---: |
| no graph | 88.40% | — | 28.96 ms | 42.00 ms |
| 100 graph candidates | 91.20% | 3.53 ms | 33.07 ms | 46.72 ms |
| 200 graph candidates | 91.20% | 3.78 ms | 35.05 ms | 46.61 ms |

The held-out run froze 100 candidates. Loci regional reached 89.53% Recall@100 and 0.8005 nDCG@10 against 90.00% and 0.7930 for Elasticsearch. The paired 95% interval for the Recall@100 difference was −1.88 to +0.94 points and for nDCG@10 −0.0116 to +0.0262. Loci p95 was 47.29 ms against 37.33 ms for Elasticsearch in the same run.

## HotpotQA 1M latency with HNSW candidates

Reusing corpus-sized scratch buffers across queries, instead of allocating four 8 MB arrays per query, cut the development mean from 33.07 ms to 28.18 ms and p95 from 46.72 ms to 41.54 ms with identical signatures. With the graph supplying global nearest neighbors, a smaller regional budget then improved both latency and recall on development queries:

| Regional chunks | Graph width | Recall@100 | Graph mean | Query mean | Query p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24,000 | 256 | 91.20% | 3.56 ms | 28.18 ms | 41.54 ms |
| 24,000 | 128 | 90.80% | 2.46 ms | 32.30 ms | 59.02 ms |
| 12,000 | 256 | 91.80% | 3.81 ms | 22.20 ms | 36.80 ms |
| 12,000 | 128 | 91.40% | 1.99 ms | 17.77 ms | 29.88 ms |
| 8,000 | 128 | 91.40% | 2.79 ms | 22.99 ms | 43.22 ms |

Selection used development recall first and froze 12,000 chunks with width 256. Single-pass timings vary by several milliseconds.

The held-out run was repeated three times with the same frozen settings. All 960 regional traces matched across runs. In the first two, Elasticsearch p95 reached 75 to 77 ms, roughly double its earlier runs, probably because its heap had been paged out while Loci used more memory. After restarting Elasticsearch, Loci regional reached 89.38% Recall@100, 0.8005 nDCG@10, 19.31 ms p50 and 31.01 ms p95, against 90.00%, 0.7930, 27.45 ms and 41.20 ms. The paired Recall@100 difference was −0.63 points (95% interval −2.03 to +0.78). Earlier runs measured Elasticsearch p95 between 30.38 ms and 37.33 ms on this machine.
