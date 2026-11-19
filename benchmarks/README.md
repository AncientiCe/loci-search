# Benchmark history

The latency columns summarize all recorded warm sequential queries. The [machine-readable history](history.json) includes per-pass timings, dev trials, index preparation, and exact settings. See the [reproduction instructions](../README.md) for the corpus and launch steps.

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
