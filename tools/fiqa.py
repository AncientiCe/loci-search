"""Prepare BEIR passages and index the same inputs into Elasticsearch."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import sys
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile


DIMENSIONS = 1024
DATASETS = {
    "fiqa": {
        "archive_md5": "17918ed23cd04fb15047f73e6c3bd9d9",
        "passages": 57_638,
        "queries": (500, 648),
        "embedding_files": {
            "corpus.parquet": "9b862c5f1d023435f71ebeb8f71942a8caf3b90b1b07b14b435e7a31623c2a52",
            "dev.parquet": "729e3d1055cf679358d95d0631403ac978c1242d49b145a5e25ad4ce9e91f323",
            "test.parquet": "d745fb2470b1d683c7897cef86b1f1ce73741332a9fb9426027556fa07e9d2fa",
        },
    },
    "nfcorpus": {
        "archive_md5": "a89dba18a62ef92f7d323ec890a0d38d",
        "passages": 3_633,
        "queries": (324, 323),
        "embedding_files": {
            "corpus.parquet": "34703a489f88e1c0d7328d78b7c274c937cb0e4c807fcbfcb20aee5fe823687b",
            "dev.parquet": "0eb57d19a83d4413eefa4b70facdc6f3b4d3b7fcb68c916caf7c2150d22f6a89",
            "test.parquet": "39e6a8d051752b31e018dfa386b0ef044e748ca8881873009b0a02859f35b636",
        },
    },
}


def digest(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def rows(archive, name):
    with archive.open(name) as stream:
        for line in stream:
            yield json.loads(line)


def write_json(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(directory, dependencies, dataset="fiqa"):
    if dependencies:
        sys.path.insert(0, str(dependencies.resolve()))
    import pyarrow.parquet as pq

    metadata = DATASETS[dataset]
    archive_path = directory / f"{dataset}.zip"
    if digest(archive_path, "md5") != metadata["archive_md5"]:
        raise ValueError(f"BEIR {dataset} archive checksum differs from the published checksum")
    embedding_files = metadata["embedding_files"]
    for name, checksum in embedding_files.items():
        if digest(directory / name, "sha256") != checksum:
            raise ValueError(f"Unexpected SHA-256 for {name}")
    embedding = {
        "model_revision": f"CohereLabs/beir-embed-english-v3:{dataset}:" + embedding_files["corpus.parquet"],
        "dimensions": DIMENSIONS,
        "normalization": "l2",
    }

    with ZipFile(archive_path) as archive:
        documents = {row["_id"]: row for row in rows(archive, f"{dataset}/corpus.jsonl")}
        queries = {row["_id"]: row["text"] for row in rows(archive, f"{dataset}/queries.jsonl")}
        data = pq.ParquetFile(directory / "corpus.parquet")
        seen = set()
        with (directory / "passages.partial").open("w", encoding="utf-8", newline="\n") as out:
            for batch in data.iter_batches(batch_size=512, columns=["_id", "emb"]):
                for row in batch.to_pylist():
                    identifier = str(row["_id"])
                    if identifier in seen or len(row["emb"]) != DIMENSIONS:
                        raise ValueError(f"Invalid document embedding for {identifier}")
                    seen.add(identifier)
                    document = documents[identifier]
                    text = "\n\n".join(value for value in (document["title"], document["text"]) if value)
                    write_json(out, {
                        "id": identifier, "document": identifier, "revision": f"beir-{dataset}",
                        "start": 0, "end": len(text.encode("utf-8")), "text": text, "vector": row["emb"],
                    })
        if seen != documents.keys() or len(seen) != metadata["passages"]:
            raise ValueError(f"Corpus and embeddings differ from the full {metadata['passages']:,}-document snapshot")
        (directory / "passages.partial").replace(directory / "passages.jsonl")
        splits = {}
        for split in ("dev", "test"):
            judgments = defaultdict(dict)
            with archive.open(f"{dataset}/qrels/{split}.tsv") as binary:
                for row in csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8"), delimiter="\t"):
                    identifier = row["corpus-id"]
                    if identifier not in seen:
                        raise ValueError(f"Judgment references an out-of-corpus ID {identifier}")
                    judgments[row["query-id"]][identifier] = int(row["score"])
            vectors = {}
            for batch in pq.ParquetFile(directory / f"{split}.parquet").iter_batches(columns=["_id", "emb"]):
                for row in batch.to_pylist():
                    identifier = str(row["_id"])
                    if identifier in vectors or len(row["emb"]) != DIMENSIONS:
                        raise ValueError(f"Invalid query embedding for {identifier}")
                    vectors[identifier] = row["emb"]
            if vectors.keys() != judgments.keys():
                raise ValueError(f"{split} query IDs differ between the published embeddings and judgments")
            with (directory / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as out:
                for identifier in sorted(judgments):
                    write_json(out, {
                        "id": identifier, "text": queries[identifier], "vector": vectors[identifier],
                        "embedding": embedding, "judgments": judgments[identifier],
                    })
            splits[split] = set(judgments)
        if splits["dev"] & splits["test"] or (len(splits["dev"]), len(splits["test"])) != metadata["queries"]:
            raise ValueError(f"Expected independent development and test splits for {dataset}")
    (directory / "embedding.json").write_text(json.dumps(embedding) + "\n", encoding="utf-8")
    print(f"Prepared {len(seen):,} passages, {len(splits['dev'])} dev queries and {len(splits['test'])} test queries")


def request(base, route, body=None, method="POST", *, timeout=180):
    headers = {}
    if isinstance(body, bytes):
        headers["Content-Type"] = "application/x-ndjson"
    elif body is not None:
        body = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    with urlopen(Request(base + route, data=body, headers=headers, method=method), timeout=timeout) as response:
        return json.load(response)


def elastic_index(directory, base, index, vectors=None, workers=1, max_segments=1):
    if workers < 1 or max_segments < 1:
        raise ValueError("Workers and maximum segments must be positive")
    started = perf_counter()
    manifest = directory / "embedding.json"
    dimensions = json.loads(manifest.read_text(encoding="utf-8"))["dimensions"] if manifest.exists() else DIMENSIONS
    mapping = {"dynamic": "strict", "properties": {
        "text": {"type": "text", "analyzer": "standard", "fields": {
            "english": {"type": "text", "analyzer": "english"},
        }},
        "embedding": {
            "type": "dense_vector", "dims": dimensions, "similarity": "cosine",
            "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
        },
        "passage_id": {"type": "keyword"},
    }}
    if vectors is not None:
        import numpy as np
        import orjson
        # Search uses Lucene's vectors and returns IDs, so avoid another vector copy in _source.
        mapping["_source"] = {"excludes": ["embedding"]}
    request(base, "/" + index, {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": mapping,
    }, method="PUT")

    def send(batch):
        for attempt in range(5):
            try:
                result = request(base, "/" + index + "/_bulk", batch)
            except HTTPError as error:
                if error.code not in (429, 502, 503, 504) or attempt == 4:
                    raise
            except (URLError, TimeoutError):
                if attempt == 4:
                    raise
            else:
                if not result["errors"]:
                    return
                failed = [item["index"] for item in result["items"] if "error" in item["index"]]
                if attempt == 4 or any(item["status"] not in (429, 502, 503, 504) for item in failed):
                    raise ValueError(failed[0])
            sleep(2**attempt)

    count = 0
    pending = deque()
    batch = []
    packed = vectors.open("rb") if vectors is not None else nullcontext()
    with ThreadPoolExecutor(max_workers=workers) as pool, (directory / "passages.jsonl").open(encoding="utf-8") as stream, packed as vector_stream:
        for line in stream:
            passage = json.loads(line)
            identifier = passage["id"]
            header = json.dumps({"index": {"_id": identifier}}).encode()
            document = {"passage_id": identifier, "text": passage["text"]}
            if vector_stream is not None:
                data = vector_stream.read(dimensions * 4)
                if len(data) != dimensions * 4 or "vector" in passage:
                    raise ValueError(f"Invalid packed vector for {identifier}")
                document["embedding"] = np.frombuffer(data, dtype="<f4")
                body = orjson.dumps(document, option=orjson.OPT_SERIALIZE_NUMPY)
            else:
                document["embedding"] = passage["vector"]
                body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
            batch.extend((header, body))
            count += 1
            if len(batch) == 800:
                pending.append(pool.submit(send, b"\n".join(batch) + b"\n"))
                batch.clear()
                if len(pending) >= workers * 2:
                    pending.popleft().result()
                if count % 20_000 == 0:
                    print(f"Submitted {count:,} passages", flush=True)
        if vector_stream is not None and vector_stream.read(1):
            raise ValueError("Packed vector file contains excess data")
        if batch:
            pending.append(pool.submit(send, b"\n".join(batch) + b"\n"))
        for batch_result in pending:
            batch_result.result()
    request(base, "/" + index + "/_refresh")
    print(f"Waiting for Elasticsearch merge to at most {max_segments} segments", flush=True)
    # Wait on the request itself. Asynchronous task results need a separate system shard.
    merge = request(base, f"/{index}/_forcemerge?max_num_segments={max_segments}", timeout=3600)
    if merge["_shards"]["failed"]:
        raise ValueError(merge)
    request(base, "/" + index + "/_refresh")
    total = request(base, "/" + index + "/_count", method="GET")["count"]
    if count != total:
        raise ValueError(f"Indexed {total} passages, expected {count}")
    primary = request(base, "/" + index + "/_stats/store,segments", method="GET")["_all"]["primaries"]
    if primary["segments"]["count"] > max_segments:
        raise ValueError("Elasticsearch still has more segments than requested")
    report = {
        "documents": count, "build_seconds": perf_counter() - started,
        "index_bytes": primary["store"]["size_in_bytes"], "segments": primary["segments"]["count"],
    }
    (directory / "elastic-build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--deps", type=Path)
    prep.add_argument("--dataset", choices=DATASETS, default="fiqa")
    index = actions.add_parser("index")
    index.add_argument("directory", type=Path)
    index.add_argument("--url", default="http://127.0.0.1:19200")
    index.add_argument("--index", default="loci-fiqa")
    index.add_argument("--vectors", type=Path)
    index.add_argument("--deps", type=Path, action="append", default=[])
    index.add_argument("--workers", type=int, default=1)
    index.add_argument("--max-segments", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.deps, args.dataset)
    else:
        for folder in args.deps:
            sys.path.insert(0, str(folder.resolve()))
        elastic_index(args.directory, args.url.rstrip("/"), args.index, args.vectors, args.workers, args.max_segments)


if __name__ == "__main__":
    main()
