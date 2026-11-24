"""Migrate the local NFCorpus index and compare lexical counts with Elasticsearch.

The snapshot stays under runs/, which is gitignored. From the repository root:

    python tools/rehearse_nfcorpus.py --binary target/debug/loci.exe
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import migrate_search as migration
from rehearse_cutover import Service


QUERIES = ["zinc", "cancer", "vitamin", "diet", "infection"]


def es_total(query):
    body = json.dumps({"query": {"match": {"text": query}}, "size": 10, "track_total_hits": True}).encode()
    request = Request("http://127.0.0.1:19200/loci-nfcorpus/_search", data=body,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    total = payload["hits"]["total"]["value"]
    ids = [hit["_id"] for hit in payload["hits"]["hits"]]
    return total, ids


def run(binary):
    out = Path("runs") / "nfcorpus-migrate"
    if not (out / "documents.ndjson").exists():
        status = migration.main([
            "export", "--source-url", "http://127.0.0.1:19200", "--index", "loci-nfcorpus", "--out", str(out),
        ])
        if status != 0:
            raise RuntimeError(f"export exited {status}")
    mapping = json.loads((out / "mapping.json").read_text(encoding="utf-8"))
    parsed = migration.Mapping(mapping, "text")
    data = Path(tempfile.mkdtemp(prefix="loci-nfcorpus-"))
    service = Service(binary, data / "data")
    service.start()
    try:
        imported = migration.main([
            "import", "--input", str(out / "documents.ndjson"), "--mapping", str(out / "mapping.json"),
            "--index", "nfcorpus", "--text-field", "text", "--target-url", service.url, "--public",
            "--source-index", "loci-nfcorpus",
        ])
        if imported != 0:
            raise RuntimeError(f"import exited {imported}")
        page = service.request("POST", "/v1/indexes/nfcorpus/search", {"text": "", "limit": 1})
        if page["total"] != 3633 or page.get("exact", True) is not True:
            raise RuntimeError(f"document count diverged: {page['total']} exact={page.get('exact')}")
        comparisons = []
        for query in QUERIES:
            source_total, source_ids = es_total(query)
            translated = migration.translate({"query": {"match": {"text": query}}, "size": 10}, parsed)
            found = service.request("POST", "/v1/indexes/nfcorpus/search", translated)
            target_ids = [hit["id"] for hit in found["hits"]]
            comparisons.append({
                "query": query,
                "elasticsearch_total": source_total,
                "loci_total": found["total"],
                "overlap_at_10": len(set(source_ids) & set(target_ids)),
            })
    finally:
        service.stop()
    report = {
        "index": "loci-nfcorpus",
        "documents": 3633,
        "omitted": parsed.omitted,
        "queries": comparisons,
        "note": "Lexical replay. Elasticsearch uses its English analyzer and vectors; Loci searches the stored text. Totals are not required to match.",
    }
    print(json.dumps(report, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    try:
        return run(args.binary.resolve())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"NFCorpus rehearsal failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
