"""Prepare BEIR FiQA and index exactly the same passages into Elasticsearch."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import sys
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile


DIMENSIONS = 1024
FILES = {
    "corpus.parquet": "9b862c5f1d023435f71ebeb8f71942a8caf3b90b1b07b14b435e7a31623c2a52",
    "dev.parquet": "729e3d1055cf679358d95d0631403ac978c1242d49b145a5e25ad4ce9e91f323",
    "test.parquet": "d745fb2470b1d683c7897cef86b1f1ce73741332a9fb9426027556fa07e9d2fa",
}
EMBEDDING = {
    "model_revision": "CohereLabs/beir-embed-english-v3:fiqa:" + FILES["corpus.parquet"],
    "dimensions": DIMENSIONS,
    "normalization": "l2",
}


def digest(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def rows(archive, name):
    with archive.open("fiqa/" + name) as stream:
        for line in stream:
            yield json.loads(line)


def write_json(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(directory, dependencies):
    if dependencies:
        sys.path.insert(0, str(dependencies.resolve()))
    import pyarrow.parquet as pq

    if digest(directory / "fiqa.zip", "md5") != "17918ed23cd04fb15047f73e6c3bd9d9":
        raise ValueError("BEIR FiQA archive checksum differs from the published checksum")
    for name, checksum in FILES.items():
        if digest(directory / name, "sha256") != checksum:
            raise ValueError(f"Unexpected SHA-256 for {name}")

    with ZipFile(directory / "fiqa.zip") as archive:
        documents = {row["_id"]: row for row in rows(archive, "corpus.jsonl")}
        queries = {row["_id"]: row["text"] for row in rows(archive, "queries.jsonl")}
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
                        "id": identifier, "document": identifier, "revision": "beir-fiqa",
                        "start": 0, "end": len(text.encode("utf-8")), "text": text, "vector": row["emb"],
                    })
        if seen != documents.keys() or len(seen) != 57638:
            raise ValueError("Corpus and embedding IDs do not match the full 57,638-document snapshot")
        (directory / "passages.partial").replace(directory / "passages.jsonl")
        splits = {}
        for split in ("dev", "test"):
            judgments = defaultdict(dict)
            with archive.open(f"fiqa/qrels/{split}.tsv") as binary:
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
                        "embedding": EMBEDDING, "judgments": judgments[identifier],
                    })
            splits[split] = set(judgments)
        if splits["dev"] & splits["test"] or len(splits["test"]) != 648:
            raise ValueError("Expected independent development and 648-query test splits")
    (directory / "embedding.json").write_text(json.dumps(EMBEDDING) + "\n", encoding="utf-8")
    print(f"Prepared {len(seen):,} passages, {len(splits['dev'])} dev queries and {len(splits['test'])} test queries")


def request(base, route, body=None, method="POST"):
    headers = {}
    if isinstance(body, bytes):
        headers["Content-Type"] = "application/x-ndjson"
    elif body is not None:
        body = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    with urlopen(Request(base + route, data=body, headers=headers, method=method), timeout=180) as response:
        return json.load(response)


def elastic_index(directory, base, index):
    started = perf_counter()
    request(base, "/" + index, {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"dynamic": "strict", "properties": {
            "text": {"type": "text", "analyzer": "standard", "fields": {
                "english": {"type": "text", "analyzer": "english"},
            }},
            "embedding": {
                "type": "dense_vector", "dims": DIMENSIONS, "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
            "passage_id": {"type": "keyword"},
        }},
    }, method="PUT")
    count = 0

    def send(batch):
        result = request(base, "/" + index + "/_bulk", ("\n".join(batch) + "\n").encode())
        if result["errors"]:
            error = next(item["index"] for item in result["items"] if "error" in item["index"])
            raise ValueError(error)

    batch = []
    with (directory / "passages.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            passage = json.loads(line)
            identifier = passage["id"]
            batch.extend((
                json.dumps({"index": {"_id": identifier}}),
                json.dumps({"passage_id": identifier, "text": passage["text"], "embedding": passage["vector"]},
                           ensure_ascii=False, separators=(",", ":")),
            ))
            count += 1
            if len(batch) == 600:
                send(batch)
                batch.clear()
                if count % 6000 == 0:
                    print(f"Indexed {count:,} passages", flush=True)
    if batch:
        send(batch)
    request(base, "/" + index + "/_refresh")
    request(base, "/" + index + "/_forcemerge?max_num_segments=1")
    total = request(base, "/" + index + "/_count", method="GET")["count"]
    if count != total:
        raise ValueError(f"Indexed {total} passages, expected {count}")
    index_bytes = request(base, "/" + index + "/_stats/store", method="GET")["_all"]["primaries"]["store"]["size_in_bytes"]
    report = {"documents": count, "build_seconds": perf_counter() - started, "index_bytes": index_bytes}
    (directory / "elastic-build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--deps", type=Path)
    index = actions.add_parser("index")
    index.add_argument("directory", type=Path)
    index.add_argument("--url", default="http://127.0.0.1:19200")
    index.add_argument("--index", default="loci-fiqa")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.deps)
    else:
        elastic_index(args.directory, args.url.rstrip("/"), args.index)


if __name__ == "__main__":
    main()
