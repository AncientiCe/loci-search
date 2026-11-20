# Reproduce the benchmarks

Run these commands from the repository root. See the [project quick start](../README.md#quick-start) for a smaller synthetic dataset and the [benchmark history](README.md) for recorded results.

## Reproduce the 100,000-chunk benchmark

```sh
cargo run --locked --release -- bench --fixture --out runs/fixture-100k
```

The command builds a deterministic 100,000-passage synthetic corpus and runs 32 labeled queries across identifier, paraphrase, cross-domain, and multi-hop cases. It performs one warm-up pass, then reports three measured passes by default. Use a fresh output directory for each run.

- `report.json` records the embedding manifest, work budgets, index size, build timings, candidate recall, nDCG@10, Recall@100, all-evidence recall, and p50/p95 latency. It also includes a sampled HNSW comparison with exact cosine.
- `traces.jsonl` has a record per query, strategy, and repeat, including ranked IDs, reranker candidates, region paths, lexical rescue, and work counters.
- `fixture.sqlite` holds the generated index. The HNSW control builds its deterministic in-memory graph when opened for global hybrid search. The benchmark measures that build separately from warm query latency.

The benchmark compares [SQLite FTS5 BM25](https://www.sqlite.org/fts5.html), global BM25 plus HNSW, and budgeted regional BM25 plus cosine. The hybrids share reciprocal rank fusion and a reranker that uses query-term coverage and exact cosine. The regional engine reserves candidate slots for global lexical rescue. Budget parameters include `--max-regions`, `--max-chunks`, `--max-postings`, `--rescue`, and `--routing-ms`. The time budget applies to routing and is checked cooperatively. SQLite global lexical work is represented by returned results and elapsed time because FTS5 does not expose postings visited.

The built-in fixture checks behavior and reproducibility. Use a held-out judged corpus and a tuned external control for a quality or latency decision. This benchmark measures warm sequential latency. The recorded results below correspond to earlier engine and dependency versions. See the [benchmark history](README.md) and Git history for follow-up runs.

## External benchmark data

The corpus archives, relevance judgments, and embeddings used for the external benchmarks are downloaded separately and kept under `runs/`. Check the upstream terms before using or redistributing these files. The repository's MIT license covers the project code and documentation. See the dataset cards for [FiQA](https://huggingface.co/datasets/BeIR/fiqa), [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus), [HotpotQA](https://huggingface.co/datasets/BeIR/hotpotqa), and [Cohere BEIR embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3).

## Compare against Elasticsearch on FiQA

The comparison uses the full [BEIR FiQA corpus](https://huggingface.co/datasets/BeIR/fiqa), its official development and test judgments, and the published [Cohere Embed English v3 vectors](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3). Keep the inputs and generated outputs in `runs/fiqa`.

| Save as | Download |
| --- | --- |
| `fiqa.zip` | [BEIR dataset archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) |
| `corpus.parquet` | [Corpus embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/fiqa/corpus/0000.parquet) |
| `dev.parquet` | [Development query embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/fiqa/queries/dev.parquet) |
| `test.parquet` | [Test query embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/fiqa/queries/test.parquet) |

```sh
python -m pip install --target runs/fiqa/python-deps pyarrow==23.0.1
python tools/fiqa.py prepare runs/fiqa --deps runs/fiqa/python-deps
cargo run --locked --release -- ingest --db runs/fiqa/loci.sqlite --passages runs/fiqa/passages.jsonl --embedding runs/fiqa/embedding.json
```

Start a dedicated [Elasticsearch 8.19.25 instance](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.19.25-windows-x86_64.zip) bound to loopback on port 19200. The measured run used a 2 GiB JVM heap and four active JVM processors. For this isolated benchmark, run the bundled launcher with `-Ediscovery.type=single-node -Expack.security.enabled=false -Expack.ml.enabled=false -Enetwork.host=127.0.0.1 -Ehttp.port=19200 -Etransport.port=19300`. Keep the endpoint on loopback.

```sh
python tools/fiqa.py index runs/fiqa
cargo run --locked --release -- bench --db runs/fiqa/loci.sqlite --queries runs/fiqa/test.jsonl --dev runs/fiqa/dev.jsonl --elastic-url http://127.0.0.1:19200 --elastic-index loci-fiqa --out runs/fiqa/evaluation
```

The run selects settings on the 500 development queries and freezes them before scoring the 648 test queries. Each hybrid gets six trials. Loci tunes HNSW width or regional work limits. Elasticsearch tunes the lexical analyzer and HNSW candidate count. Selection maximizes development nDCG@10, then Recall@100, then prefers the earliest grid entry on an exact tie. All hybrids use the same 200-candidate fusion and reranking pipeline. Each measured round interleaves engines, includes the Elasticsearch loopback request, and uses one query at a time after warm-up. The report records the grid, selected settings, Elasticsearch version and index size, and paired query-level bootstrap intervals for regional minus Elasticsearch quality.

### Held-out run

These results use all 57,638 passages, three measured passes over the 648 test queries, and an AMD Ryzen 7 7800X3D running Windows. Elasticsearch selected the English analyzer and 256 HNSW candidates. Loci regional selected limits of 36 regions, 24,000 scored chunks, and 240,000 postings per query.

| Strategy | nDCG@10 | Recall@100 | Warm p95 |
| --- | ---: | ---: | ---: |
| SQLite BM25 | 0.2348 | 0.5187 | 81.75 ms |
| Loci global hybrid | 0.2859 | 0.6305 | 88.24 ms |
| Loci regional hybrid | 0.2844 | 0.6212 | 153.97 ms |
| Elasticsearch hybrid | 0.2910 | 0.6498 | 20.98 ms |

Regional minus Elasticsearch was −0.0066 nDCG@10 (95% paired bootstrap interval −0.0114 to −0.0019) and −0.0286 Recall@100 (−0.0408 to −0.0178). Regional search averaged 20,220 candidate vector comparisons per query, versus 2,541 for the in-process global hybrid.

### Profile and rerun with frozen settings

```sh
cargo run --locked --release -- profile --db runs/fiqa/loci.sqlite --queries runs/fiqa/dev.jsonl --frozen-report runs/fiqa/evaluation/report.json --out runs/fiqa/profile-after.json
cargo run --locked --release -- bench --db runs/fiqa/loci.sqlite --queries runs/fiqa/test.jsonl --frozen-report runs/fiqa/evaluation/report.json --elastic-url http://127.0.0.1:19200 --elastic-index loci-fiqa --out runs/fiqa/evaluation-optimized
```

The profile on the 500 development queries put 77% of mean regional latency in global lexical rescue and regional postings. The optimized engine builds an in-memory cache from the open SQLite snapshot, shares canonical posting payloads across regions, and scores lexical rescue from the same BM25 inputs as FTS5. It selects only the top candidates, reconstructs paths for returned hits, and looks up reranker term coverage in the cache. SQLite indexes remain compatible. Regional startup now includes cache preparation, which took approximately 2.4 seconds and reserved 138 MiB for posting payloads and length factors on FiQA, plus lookup overhead.

| Development profile | Original mean | Optimized mean |
| --- | ---: | ---: |
| Global lexical rescue | 44.26 ms | 0.46 ms |
| Regional postings | 27.36 ms | 1.12 ms |
| Vectors and memberships | 12.35 ms | 6.40 ms |
| Rerank and result assembly | 7.01 ms | 1.00 ms |

The profiler recorded identical ranked IDs, rescue ranks, and result paths across all 500 development queries. The frozen rerun reuses each strategy's selected settings and checks the original index statistics and evaluation dimensions.

On the same 648 held-out queries, across three warm sequential passes:

| Strategy | nDCG@10 | Recall@100 | Original p95 | Rerun p95 |
| --- | ---: | ---: | ---: | ---: |
| SQLite BM25 | 0.2348 | 0.5187 | 81.75 ms | 77.32 ms |
| Loci global hybrid | 0.2859 | 0.6305 | 88.24 ms | 85.06 ms |
| Loci regional hybrid | 0.2844 | 0.6212 | 153.97 ms | 13.49 ms |
| Frozen Elasticsearch hybrid | 0.2910 | 0.6498 | 20.98 ms | 21.34 ms |

All 7,776 paired traces matched on ranked IDs, scores, candidate IDs, and retrieval paths. Loci regional cut p95 by 91% from the original run and finished 37% below Elasticsearch p95 in the rerun. Elasticsearch retained its relevance lead.

### Recover recall at the reranker cutoff

On the 500 development queries, candidate recall at 200 was 0.7711 for Loci regional and 0.7736 for frozen Elasticsearch. After reranking to 100, recall was 0.6141 and 0.6529. Of the 51 judged passages returned only by Elasticsearch at 100, 41 were already in Loci's candidate pool. The development sweep therefore kept the candidate pool, regional work limits, and lexical rescue fixed and changed only the regional reranker.

The sweep tried 12 weight combinations. It selected highest development Recall@100 among trials that maintained the original regional nDCG@10 and stayed below frozen Elasticsearch p95, then checked the choice again with interleaved searches. The selected FiQA preset scores candidates with 92% cosine similarity and 8% term coverage. It recovered 40 of the 41 cutoff misses. Development Recall@100 rose to 0.7418 and development p95 was 13.72 ms versus 21.92 ms for frozen Elasticsearch in the validation pass.

```sh
cargo run --locked --release -- recall-dev --db runs/fiqa/loci.sqlite --dev runs/fiqa/dev.jsonl --frozen-report runs/fiqa/evaluation-optimized/report.json --elastic-url http://127.0.0.1:19200 --elastic-index loci-fiqa --out runs/fiqa/recall-dev-tuning.json --tune-rerank runs/fiqa/regional-rerank.json
cargo run --locked --release -- bench --db runs/fiqa/loci.sqlite --queries runs/fiqa/test.jsonl --frozen-report runs/fiqa/evaluation-optimized/report.json --regional-rerank benchmarks/fiqa-regional-rerank.json --elastic-url http://127.0.0.1:19200 --elastic-index loci-fiqa --out runs/fiqa/evaluation-recall
```

Use `loci search --regional-rerank benchmarks/fiqa-regional-rerank.json` with a FiQA query and index to apply the preset outside the benchmark.

On the same 648 held-out queries and three warm sequential passes, the reranker raised Loci regional Recall@100 by 11.52 percentage points. It finished 8.67 percentage points above frozen Elasticsearch, with p95 32% lower.

| Strategy | Candidate recall@200 | nDCG@10 | Recall@100 | Warm p95 |
| --- | ---: | ---: | ---: | ---: |
| SQLite BM25 | 0.5867 | 0.2348 | 0.5187 | 82.85 ms |
| Loci global hybrid | 0.7688 | 0.2859 | 0.6305 | 90.94 ms |
| Loci regional, dev-selected reranker | 0.7681 | 0.4266 | 0.7364 | 14.33 ms |
| Frozen Elasticsearch hybrid | 0.7777 | 0.2910 | 0.6498 | 21.21 ms |

All 7,776 paired traces matched on candidate IDs and retrieval work. The Elasticsearch, SQLite BM25, and Loci global traces also matched on rankings and scores. The paired 95% bootstrap interval for the regional minus Elasticsearch Recall@100 difference was +6.79 to +10.64 percentage points.

## Transfer the FiQA settings to NFCorpus

Use the full [BEIR NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) and the same published embedding model used for FiQA. Save these files in `runs/nfcorpus`.

| Save as | Download |
| --- | --- |
| `nfcorpus.zip` | [BEIR archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) |
| `corpus.parquet` | [Corpus embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/nfcorpus/corpus/0000.parquet) |
| `dev.parquet` | [Development query embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/nfcorpus/queries/dev.parquet) |
| `test.parquet` | [Test query embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/f018922a46a51388348a307ec2fef28019a33026/nfcorpus/queries/test.parquet) |

```sh
python tools/fiqa.py prepare runs/nfcorpus --dataset nfcorpus --deps runs/fiqa/python-deps
cargo run --locked --release -- ingest --db runs/nfcorpus/loci.sqlite --passages runs/nfcorpus/passages.jsonl --embedding runs/nfcorpus/embedding.json
python tools/fiqa.py index runs/nfcorpus --index loci-nfcorpus
cargo run --locked --release -- bench --db runs/nfcorpus/loci.sqlite --queries runs/nfcorpus/test.jsonl --dev runs/nfcorpus/dev.jsonl --transfer-report runs/fiqa/evaluation-recall/report.json --elastic-url http://127.0.0.1:19200 --elastic-index loci-nfcorpus --out runs/nfcorpus/evaluation
```

The transfer copies all three Loci configurations from the FiQA report, including the 92% cosine and 8% term coverage reranker. It tunes only Elasticsearch over six analyzer and HNSW configurations, prioritizing development Recall@100 and using nDCG@10 to break ties. One development query shared normalized text with a test query, so tuning used the remaining 323 development queries. Elasticsearch selected the English analyzer and 256 HNSW candidates. The held-out benchmark used all 3,633 passages and all 323 official test queries across three warm sequential passes.

| Strategy | nDCG@10 | Recall@100 | Warm p95 |
| --- | ---: | ---: | ---: |
| SQLite BM25 | 0.3091 | 0.2342 | 3.38 ms |
| Loci global hybrid | 0.3520 | 0.3215 | 11.53 ms |
| Loci regional, frozen FiQA reranker | 0.3983 | 0.3541 | 1.85 ms |
| Elasticsearch hybrid, tuned on NFCorpus dev | 0.3557 | 0.3299 | 20.70 ms |

Regional recall was 2.42 percentage points higher, with a paired 95% bootstrap interval from +1.30 to +3.52 points. Its p95 was 91% lower in this end-to-end comparison, which includes Elasticsearch's loopback request and the original Elasticsearch reranker. Loci scored every passage for each query under the transferred routing budget, so these measurements describe a small-corpus run.

## Run the million-passage HotpotQA comparison

The snapshot uses [BEIR HotpotQA](https://huggingface.co/datasets/BeIR/hotpotqa) and the [pinned Cohere embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/tree/f018922a46a51388348a307ec2fef28019a33026/hotpotqa). Save the [BEIR archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip) as `runs/hotpotqa-1m/hotpotqa.zip`. Save the six corpus Parquet shards under `runs/hotpotqa-1m/parquet/` and the development and test query Parquet files as `dev.parquet` and `test.parquet` in `runs/hotpotqa-1m`.

The snapshot chooses 250 dev and 320 test queries by hash from the official splits, keeps their normalized texts disjoint, and includes every judged positive. It fills the remaining slots with a seeded sample of real passages from the same source corpus. The preparation verifies all input checksums and writes the selected query IDs and sampling recipe to `snapshot.json`.

```sh
python -m pip install --target runs/hotpotqa-1m/python-deps numpy==2.4.2 orjson==3.11.7
python -B tools/hotpotqa.py runs/hotpotqa-1m --deps runs/fiqa/python-deps --deps runs/hotpotqa-1m/python-deps
cargo run --locked --release -- ingest --db runs/hotpotqa-1m/loci.sqlite --passages runs/hotpotqa-1m/passages.jsonl --embedding runs/hotpotqa-1m/embedding.json --vectors runs/hotpotqa-1m/vectors.f32 --compact --leaf-size 8192
python -B tools/fiqa.py index runs/hotpotqa-1m --index loci-hotpotqa-1m --vectors runs/hotpotqa-1m/vectors.f32 --deps runs/hotpotqa-1m/python-deps --workers 8 --max-segments 8
cargo run --locked --release -- bench --db runs/hotpotqa-1m/loci.sqlite --queries runs/hotpotqa-1m/test.jsonl --dev runs/hotpotqa-1m/dev.jsonl --transfer-report runs/fiqa/evaluation-recall/report.json --head-to-head --elastic-url http://127.0.0.1:19200 --elastic-index loci-hotpotqa-1m --out runs/hotpotqa-1m/evaluation
```

Both engines use the same packed float32 vectors and passage texts. Loci uses larger target leaves for the million-passage topology and the transferred FiQA reranker and per-query work limits. The compact layout avoids a redundant on-disk copy of regional postings and reconstructs them from canonical text and memberships. The head-to-head run builds only the controls needed for Loci regional and Elasticsearch. Elasticsearch uses one shard and float HNSW and tunes its analyzer and candidate count on dev by Recall@100, then nDCG@10. Warm p95 is end-to-end and includes the Elasticsearch loopback request.

Elasticsearch selected the standard analyzer and 768 HNSW candidates. Its final index had eight segments and no active merges. The bulk build used a 4 GiB heap and eight active JVM processors. Tuning and timing used the previous benchmark settings of a 2 GiB heap and four active processors.

| Strategy | Recall@100 | nDCG@10 | Warm p50 | Warm p95 |
| --- | ---: | ---: | ---: | ---: |
| Loci regional, frozen FiQA reranker | 86.25% | 0.7896 | 35.08 ms | 44.26 ms |
| Tuned Elasticsearch hybrid | 90.00% | 0.7930 | 20.64 ms | 30.38 ms |

The three warm sequential passes covered all 320 sampled test queries. Loci scored exactly 24,000 passages with regional cosine on every query, or 2.40% of the corpus. Global lexical rescue scored a mean of 894,306 passages. The deduplicated mean across those two scoring paths was 895,391 passages, or 89.54% of the corpus. The paired 95% bootstrap interval for regional minus Elasticsearch Recall@100 was −5.63 to −1.88 percentage points.
