"""Build the 1M BEIR HotpotQA snapshot from the pinned Cohere Parquet files."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import unicodedata
from zipfile import ZipFile

from fiqa import DIMENSIONS, digest, rows, write_json


CORPUS_SIZE = 1_000_000
SOURCE_SIZE = 5_233_329
SEED = 20261119
QUERY_COUNTS = {"dev": 250, "test": 320}
RELEASE = "f018922a46a51388348a307ec2fef28019a33026"
ARCHIVE_MD5 = "f412724f78b0d91183a0e86805e16114"
SHARDS = {
    "parquet/0000.parquet": "6462fc6e8349973a744bfd55e297d4f9f7008810cb0f9de918c6360893d624f7",
    "parquet/0001.parquet": "177e1ddf749554d9bed2b684fa4a4c224f2af574a5af0ec573eea72a10afa0bd",
    "parquet/0002.parquet": "fa91f45cd0658187b820605b027961db9943d84b709c4ff42233e73b3cda224d",
    "parquet/0003.parquet": "93004a92651cd257ff61b8f44be2517c2b0277fb5fb32656ea8dc6acfc7a18fc",
    "parquet/0004.parquet": "4487d87e34fb28f450dd41d6f359fb55afd392d1d4a5c252ab49263d74d9f9b1",
    "parquet/0005.parquet": "48d3181044f29bf4ffecd519851de14f0b3ce60e7c67db5b50889b3747d510f4",
}
QUERIES = {
    "dev.parquet": "3bf94e909f9c2d069e422dcba5d34d03a2c8d99530aa0cbfb81c1029cf2cb1b4",
    "test.parquet": "02a1770ca110d5b7a9c7e2be7bd509f17c4ef0ccc2723f13db6af6fac44a5870",
}
EMBEDDING = {
    "model_revision": f"CohereLabs/beir-embed-english-v3:hotpotqa-1m:{RELEASE}",
    "dimensions": DIMENSIONS,
    "normalization": "l2",
}


def query_sample(archive):
    query_text = {row["_id"]: row["text"] for row in rows(archive, "hotpotqa/queries.jsonl")}
    selected = {}
    seen_text = set()
    # Keep the test query when a development query has the same normalized text.
    for split in ("test", "dev"):
        qrels = defaultdict(dict)
        with archive.open(f"hotpotqa/qrels/{split}.tsv") as binary:
            for row in csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8"), delimiter="\t"):
                grade = int(row["score"])
                if grade > 0:
                    qrels[row["query-id"]][row["corpus-id"]] = grade
        priority = lambda qid: hashlib.sha256(f"loci-hotpotqa-1m/v1/{split}/{qid}".encode()).digest()
        split_queries = {}
        for qid in sorted(qrels, key=priority):
            text = " ".join(unicodedata.normalize("NFKC", query_text[qid]).casefold().split())
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            split_queries[qid] = {
                "id": qid, "text": query_text[qid], "embedding": EMBEDDING,
                "judgments": qrels[qid], "required": sorted(qrels[qid]),
            }
            if len(split_queries) == QUERY_COUNTS[split]:
                break
        if len(split_queries) != QUERY_COUNTS[split]:
            raise ValueError(f"Could not choose {QUERY_COUNTS[split]} independent {split} queries")
        selected[split] = split_queries
    if selected["dev"].keys() & selected["test"].keys():
        raise ValueError("Development and test query IDs overlap")
    return selected


def sample_rows(shards, protected, np, pq):
    positions = {}
    offset = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=8192, columns=["_id"]):
            for row, identifier in enumerate(batch.column(0).to_pylist()):
                if identifier in protected:
                    if identifier in positions:
                        raise ValueError(f"Duplicate judged passage {identifier}")
                    positions[identifier] = offset + row
            offset += batch.num_rows
    if offset != SOURCE_SIZE or positions.keys() != protected:
        raise ValueError("Source passage count or protected judgments differ from HotpotQA")
    mask = np.zeros(SOURCE_SIZE, dtype=bool)
    mask[list(positions.values())] = True
    # Draw enough distinct rows to fill the snapshot even if every protected row is drawn.
    drawn = np.random.Generator(np.random.PCG64(SEED)).choice(
        SOURCE_SIZE, size=CORPUS_SIZE, replace=False,
    )
    needed = CORPUS_SIZE - len(protected)
    mask[drawn[~mask[drawn]][:needed]] = True
    if np.count_nonzero(mask) != CORPUS_SIZE:
        raise ValueError("Snapshot selection did not produce one million rows")
    return mask


def write_snapshot(directory, shards, mask, protected, np, pq):
    offset = 0
    seen = set()
    passage_output = directory / "passages.partial"
    vector_output = directory / "vectors.partial"
    with passage_output.open("w", encoding="utf-8", newline="\n") as passages, vector_output.open("wb") as vectors:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=4096, columns=["_id", "title", "text", "emb"],
            ):
                chosen = np.flatnonzero(mask[offset:offset + batch.num_rows])
                offset += batch.num_rows
                if chosen.size == 0:
                    continue
                subset = batch.take(chosen)
                flat = subset.column(3).flatten().to_numpy(zero_copy_only=False)
                if len(flat) != len(chosen) * DIMENSIONS:
                    raise ValueError(f"Invalid embedding dimensions in {shard.name}")
                vectors.write(flat.astype("<f4", copy=False).tobytes())
                ids, titles, bodies = (subset.column(i).to_pylist() for i in range(3))
                for identifier, title, body in zip(ids, titles, bodies, strict=True):
                    if identifier in seen:
                        raise ValueError(f"Duplicate passage {identifier}")
                    seen.add(identifier)
                    text = "\n\n".join(value for value in (title, body) if value)
                    write_json(passages, {
                        "id": identifier, "document": identifier, "revision": "beir-hotpotqa",
                        "start": 0, "end": len(text.encode("utf-8")), "text": text,
                    })
            print(f"Prepared {len(seen):,} selected passages after {shard.name}", flush=True)
    if len(seen) != CORPUS_SIZE or not protected.issubset(seen):
        raise ValueError("Incomplete snapshot or missing judged positives")
    passage_output.replace(directory / "passages.jsonl")
    vector_output.replace(directory / "vectors.f32")


def prepare(directory, dependencies):
    for folder in dependencies:
        sys.path.insert(0, str(folder.resolve()))
    import numpy as np
    import pyarrow.parquet as pq

    archive_path = directory / "hotpotqa.zip"
    if digest(archive_path, "md5") != ARCHIVE_MD5:
        raise ValueError("HotpotQA BEIR archive differs from the pinned checksum")
    for relative, checksum in (SHARDS | QUERIES).items():
        if digest(directory / relative, "sha256") != checksum:
            raise ValueError(f"Unexpected SHA-256 for {relative}")
    with ZipFile(archive_path) as archive:
        selected = query_sample(archive)
    for split, queries in selected.items():
        embeddings = pq.read_table(directory / f"{split}.parquet", columns=["_id", "emb"])
        embeddings = {row["_id"]: row["emb"] for row in embeddings.to_pylist() if row["_id"] in queries}
        if embeddings.keys() != queries.keys() or any(len(vector) != DIMENSIONS for vector in embeddings.values()):
            raise ValueError(f"Missing or malformed published {split} embeddings")
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as out:
            for identifier, query in queries.items():
                write_json(out, {**query, "vector": embeddings[identifier]})
    protected = {identifier for split in selected.values() for query in split.values() for identifier in query["judgments"]}
    shards = [directory / relative for relative in SHARDS]
    mask = sample_rows(shards, protected, np, pq)
    write_snapshot(directory, shards, mask, protected, np, pq)
    with (directory / "embedding.json").open("w", encoding="utf-8") as out:
        write_json(out, EMBEDDING)
    summary = {
        "documents": CORPUS_SIZE, "source_documents": SOURCE_SIZE, "protected_documents": len(protected),
        "corpus_seed": SEED, "embedding_release": RELEASE, "archive_md5": ARCHIVE_MD5,
        "embedding_sha256": SHARDS | QUERIES,
        "query_ids": {split: list(queries) for split, queries in selected.items()},
    }
    (directory / "snapshot.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--deps", type=Path, action="append", default=[])
    args = parser.parse_args()
    prepare(args.directory, args.deps)
