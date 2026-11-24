"""Replay the four offline fixtures and optionally grade a search engine's JSONL."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
WORKLOADS = ("knowledge", "catalog", "records", "logs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_capture(path):
    results = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["workload"], row["id"])
            require(key not in results, f"duplicate capture {key} at line {number}")
            results[key] = row["result"]
    return results


def expand_series(spec):
    for i in range(spec["count"]):
        yield {
            "id": f'{spec["prefix"]}{i:02}',
            "time": spec["start"] + i * spec["seconds_between"],
            "trace": spec["trace"],
        }


def update(state, step):
    if "upsert" in step:
        doc = step["upsert"]
        state[doc["id"]] = dict(doc)
    elif "patch" in step:
        doc = step["patch"]
        state[doc["id"]].update(doc)
    elif "expire_before" in step:
        for key in [k for k, d in state.items() if d["time"] < step["expire_before"]]:
            del state[key]
    elif step != {"restart": True}:
        raise ValueError(f"unknown operation {step}")
    # The reference truth survives a restart. An engine adapter must actually reopen.


def matches(doc, filters):
    for field, predicates in filters.items():
        value = doc[field]
        for op, target in predicates.items():
            if op == "eq":
                accept = value == target
            elif op == "gte":
                accept = value >= target
            elif op == "lt":
                accept = value < target
            else:
                raise ValueError(f"unknown filter operator {op}")
            if not accept:
                return False
    return True


def select(state, query, cursors):
    if "resume" in query:
        principal, old_query, rows = cursors[query["resume"]]
        require(principal == query["principal"], "fixture cursor principal differs")
        offset = old_query["limit"]
        query = old_query
    else:
        rows = [
            d.copy() for d in state.values()
            if ("principal" not in query or query["principal"] in d["groups"])
            and matches(d, query.get("filter", {}))
        ]
        offset = 0
        if "histogram" in query:
            h = query["histogram"]
            counts = Counter(
                h["start"] + ((d["time"] - h["start"]) // h["width"]) * h["width"]
                for d in rows if h["start"] <= d["time"] < h["end"]
            )
            return {"buckets": [[b, counts[b]] for b in range(h["start"], h["end"], h["width"])]}
        rows.sort(key=lambda d: (d[query["sort"]], d["id"]))
        if "save_cursor" in query:
            cursors[query["save_cursor"]] = (query["principal"], query, rows)
    limit = query.get("limit", len(rows))
    result = {"ids": [d["id"] for d in rows[offset:offset + limit]], "count": len(rows)}
    if "facet" in query:
        result["facets"] = dict(Counter(d[query["facet"]] for d in rows))
    return result


def grade_ranked(documents, step, result, knowledge):
    hits = result["hits"]
    ids = [h["id"] for h in hits]
    require(len(ids) == len(set(ids)), "duplicate ranked ID")
    require(len(ids) <= step["search"]["k"], "returned too many hits")
    require(set(ids) <= documents.keys(), "unknown ranked ID")
    require(step["judged"].keys() <= set(ids), f"missing judged target in top {step['search']['k']}: {ids}")
    if not knowledge:
        return
    for hit in hits:
        doc = documents[hit["id"]]
        require(hit["revision"] == doc["revision"] and hit["uri"] == doc["uri"], "wrong citation origin")
        data = doc["text"].encode("utf-8")
        start, end = hit["start"], hit["end"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(data), "invalid byte range")
        quote = data[start:end].decode("utf-8")
        require(quote == hit["quote"] and bool(quote.strip()), "citation quote differs from source")
        require(step["judged"].get(hit["id"], "") in quote, "citation misses required evidence")


def reference_ranked(state, step, knowledge):
    # Human-authored qrels define relevance. This validates the qrels and citation
    # coordinates. It deliberately does not implement a search or score a model.
    hits = []
    for key, evidence in step["judged"].items():
        hit = {"id": key}
        doc = state[key]
        if knowledge:
            data, expected = doc["text"].encode("utf-8"), evidence.encode("utf-8")
            start = data.index(expected)
            hit.update(uri=doc["uri"], revision=doc["revision"], quote=evidence,
                       start=start, end=start + len(expected))
        hits.append(hit)
    return {"hits": hits}


def run(fixture, capture, emit):
    name = fixture["workload"]
    require(fixture["version"] == 1, "unsupported fixture version")
    state = {d["id"]: dict(d) for d in fixture["documents"]}
    require(len(state) == len(fixture["documents"]), "duplicate initial document ID")
    if emit:
        print(json.dumps({"workload": name, "reset": fixture["documents"]}, ensure_ascii=False))
    cursors = {}
    seen = set()
    for step in fixture["steps"]:
        # Expand repeatable inputs so adapters need only a handful of operations.
        if "append_series" in step:
            inputs = ({"upsert": doc} for doc in expand_series(step["append_series"]))
        else:
            inputs = [step]
        for action in inputs:
            if emit:
                request = {k: v for k, v in action.items() if k not in ("expect", "judged")}
                print(json.dumps({"workload": name, **request}, ensure_ascii=False))
            if "id" not in action:
                update(state, action)
                continue
            case = action["id"]
            require(case not in seen, f"duplicate fixture query {case}")
            seen.add(case)
            try:
                if "select" in action:
                    truth = select(state, action["select"], cursors)
                    require(truth == action["expect"], f"reference {truth} differs from frozen {action['expect']}")
                else:
                    truth = reference_ranked(state, action, name == "knowledge")
                    grade_ranked(state, action, truth, name == "knowledge")
                if capture is None:
                    continue
                result = capture.pop((name, case))
                if "select" in action:
                    # JSON types matter (Python normally treats True as equal to 1).
                    require(json.dumps(result, sort_keys=True) == json.dumps(truth, sort_keys=True),
                            f"expected {truth}, got {result}")
                else:
                    grade_ranked(state, action, result, name == "knowledge")
            except (KeyError, TypeError, UnicodeError, ValueError) as error:
                raise ValueError(f"{name}/{case}: {error}") from error
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--actual", type=Path, help="grade actual engine responses in JSONL format")
    modes.add_argument("--emit-requests", action="store_true", help="print ordered adapter input as JSONL")
    parser.add_argument("--workload", choices=WORKLOADS, help="check a single workload")
    args = parser.parse_args()
    try:
        capture = read_capture(args.actual) if args.actual is not None else None
        total = 0
        for name in ([args.workload] if args.workload else WORKLOADS):
            fixture = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
            require(fixture["workload"] == name, "workload name differs from filename")
            total += run(fixture, capture, args.emit_requests)
        require(not capture, f"unexpected captured cases: {list(capture or {})}")
        if not args.emit_requests:
            mode = "engine response checks" if capture is not None else "offline fixture checks"
            print(f"{total} {mode} passed")
        return 0
    except (KeyError, TypeError, UnicodeError, ValueError, OSError) as error:
        print(f"Workload validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
