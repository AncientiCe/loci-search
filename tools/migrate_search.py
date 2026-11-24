"""Move a supported Elasticsearch/OpenSearch index to Loci and compare saved searches."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def check(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def load(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        return json.load(file)


def only(value, allowed, where):
    check(isinstance(value, dict), f"{where}: expected an object")
    unknown = sorted(value.keys() - set(allowed))
    check(not unknown, f"{where}: unsupported keys {unknown}")


def one(value, where):
    check(isinstance(value, dict) and len(value) == 1, f"{where}: expected one key")
    return next(iter(value.items()))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Endpoint:
    def __init__(self, base, authorization=None):
        url = urlsplit(base)
        check(url.scheme in ("http", "https") and url.hostname and url.username is None
              and not url.query and not url.fragment, "invalid server URL")
        self.base = base.rstrip("/")
        self.auth = authorization
        self.opener = build_opener(NoRedirect)

    def request(self, method, path, body=None, key=None):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.auth:
            headers["Authorization"] = self.auth
        if key:
            headers["Idempotency-Key"] = key
        req = Request(self.base + path, data=None if body is None else encoded(body),
                      headers=headers, method=method)
        with self.opener.open(req, timeout=60) as response:
            return json.load(response)


def target(args):
    token = os.environ.get("LOCI_TOKEN")
    return Endpoint(args.target_url, f"Bearer {token}" if token else None)


def source(args):
    return Endpoint(args.source_url, os.environ.get("ES_AUTHORIZATION"))


def source_index(value):
    check(re.fullmatch(r"[\w.\-]+", value, re.ASCII) and value not in (".", "..", "_all"),
          "specify one concrete source index")
    return quote(value, safe="")


def target_index(value):
    check(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value), "invalid destination index")
    return "/v1/indexes/" + value


def complete(response):
    check(response.get("timed_out") is False and response.get("_shards", {}).get("failed") == 0,
          "search returned partial or timed-out results")


def total(hits):
    value = hits["total"]
    if isinstance(value, dict):
        check(value.get("relation") == "eq", "source returned an inexact total")
        value = value["value"]
    check(type(value) is int and value >= 0, "invalid hit count")
    return value


def export_index(args):
    es = source(args)
    index = source_index(args.index)
    output = args.out
    check(not output.exists(), f"{output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        base = Path(temporary) / "snapshot"
        base.mkdir()
        mapping = es.request("GET", f"/{index}/_mapping")
        check(set(mapping) == {args.index}, "source must identify one concrete index")
        (base / "mapping.json").write_bytes(encoded(mapping))
        body = {"size": args.page_size, "sort": ["_doc"], "query": {"match_all": {}},
                "version": True, "track_total_hits": True}
        scroll = None
        count = 0
        try:
            page = es.request("POST", f"/{index}/_search?scroll=2m", body)
            expected = total(page["hits"])
            with (base / "documents.ndjson").open("wb") as docs:
                while True:
                    scroll = page.get("_scroll_id", scroll)
                    complete(page)
                    rows = page["hits"]["hits"]
                    if not rows:
                        break
                    check(scroll, "source did not return a scroll cursor")
                    for hit in rows:
                        check(hit.get("_index") == args.index and isinstance(hit.get("_source"), dict),
                              "source hit lacked the selected index or an object _source")
                        docs.write(encoded({k: hit[k] for k in ("_index", "_id", "_version", "_source")}) + b"\n")
                        count += 1
                    page = es.request("POST", "/_search/scroll", {"scroll": "2m", "scroll_id": scroll})
        finally:
            if scroll:
                try:
                    es.request("DELETE", "/_search/scroll", {"scroll_id": [scroll]})
                except (OSError, ValueError) as exc:
                    print(f"Scroll cleanup: {exc}", file=sys.stderr)
        check(count == expected, f"incomplete snapshot: expected {expected}, received {count}")
        base.rename(output)
    print(json.dumps({"exported": count, "snapshot": str(output)}))
    return 0


TYPES = {"text": "text", "keyword": "keyword", "byte": "integer", "short": "integer",
         "integer": "integer", "long": "integer", "float": "number", "double": "number",
         "half_float": "number", "boolean": "boolean", "date": "date"}


class Mapping:
    def __init__(self, raw, text_field, arrays=()):
        if "mappings" in raw:
            raw = raw["mappings"]
        elif "properties" not in raw:
            _, item = one(raw, "mapping export")
            raw = item["mappings"]
        only(raw, ["properties", "_meta"], "mapping")
        properties = raw.get("properties", {})
        check(isinstance(properties, dict), "mapping.properties must be an object")
        self.types = {}
        self.aliases = {}
        self.arrays = set(arrays)
        self.text_field = text_field
        for name, field in properties.items():
            only(field, ["type", "fields"], f"mapping.{name}")
            kind = field.get("type")
            check(kind in TYPES, f"mapping.{name}: unsupported type {kind!r}")
            self.types[name] = TYPES[kind]
            for subname, subfield in field.get("fields", {}).items():
                check(kind == "text" and subfield == {"type": "keyword"},
                      f"mapping.{name}.{subname}: unsupported multi-field settings")
                self.aliases[name + "." + subname] = name
        check(self.arrays <= self.types.keys(), "array field is not mapped")
        check(text_field is None or self.types.get(text_field) in ("text", "keyword"),
              "the selected text field must be a mapped text or keyword field")

    def settings(self, args):
        return {"model": args.model, "default_visibility": "private" if args.readers_field else "public",
                "mappings": {key: val + ("[]" if key in self.arrays else "")
                             for key, val in self.types.items()}}

    def field(self, name, *, exact=False):
        if name == "_id":
            return "id"
        key = self.aliases.get(name, name)
        check(key in self.types, f"unknown mapped field {name}")
        check(not exact or name in self.aliases or self.types[key] != "text",
              f"exact semantics unavailable for analyzed text field {name}; use its keyword subfield")
        return "fields." + key

    def value(self, name, value):
        kind = self.types[self.aliases.get(name, name)]
        if kind == "date":
            if type(value) is int:
                stamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=value)
            else:
                check(isinstance(value, str) and re.fullmatch(
                      r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|[+-]\d\d:\d\d)", value),
                      f"{name}: expected epoch milliseconds or a fully specified RFC3339 date")
                stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return stamp.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        checks = {"keyword": isinstance(value, str), "text": isinstance(value, str),
                  "boolean": type(value) is bool,
                  "integer": type(value) is int and -(2**63) <= value < 2**63,
                  "number": type(value) in (float, int) and math.isfinite(value)}
        check(checks[kind], f"{name}: expected {kind}")
        return value

    def doc(self, hit, args):
        ident = hit.get("_id")
        check(isinstance(ident, str) and ident.strip(), "missing string _id")
        if args.source_index:
            check(hit.get("_index") == args.source_index, "unexpected or absent _index")
        if hit.get("_delete") is True:
            return {"op": "delete", "id": ident}
        record = hit.get("_source")
        check(isinstance(record, dict), "expected object _source")
        check(not record.keys() - self.types.keys(), "unmapped source fields: " +
              ", ".join(sorted(record.keys() - self.types.keys())))
        readers = []
        if args.readers_field:
            readers = record.get(args.readers_field)
            if isinstance(readers, str):
                readers = [readers]
            check(isinstance(readers, list) and readers and
                  all(isinstance(p, str) and p.strip() for p in readers), "missing or invalid document readers")
        fields = {}
        for field, value in record.items():
            if value is None:
                continue
            if field in self.arrays:
                items = value if isinstance(value, list) else [value]
                items = [self.value(field, v) for v in items if v is not None]
                if items:
                    fields[field] = items
            else:
                check(not isinstance(value, list), f"{field}: declare repeated values using --array-field")
                fields[field] = self.value(field, value)
        text = fields.get(self.text_field, "")
        if isinstance(text, list):
            text = "\n".join(text)
        rev = hit.get("_version")
        if rev is None:
            rev = hashlib.sha256(encoded(record)).hexdigest()
        check(isinstance(rev, (str, int)) and type(rev) is not bool and str(rev).strip(), "invalid revision")
        return {"op": "upsert", "document": {"id": ident, "revision": str(rev), "text": text,
                "fields": fields, "readers": readers}}


def filt(query, mapping):
    kind, body = one(query, "filter")
    if kind == "bool":
        only(body, {"filter", "must", "must_not", "should", "minimum_should_match"}, "bool filter")
        clauses = []
        for key in ("must", "filter", "must_not"):
            for child in as_list(body.get(key, [])):
                value = filt(child, mapping)
                clauses.append({"op": "not", "filter": value} if key == "must_not" else value)
        if "should" in body:
            minimum = body.get("minimum_should_match", 0 if "must" in body or "filter" in body else 1)
            check(minimum in [1, "1"], "optional or multi-match bool.should is unsupported")
            options = [filt(q, mapping) for q in as_list(body["should"])]
            check(options, "empty bool.should")
            clauses.append({"op": "or", "filters": options})
        else:
            check("minimum_should_match" not in body, "minimum_should_match requires should")
        return {"op": "and", "filters": clauses}
    if kind == "ids":
        only(body, ["values"], "ids")
        name, values = "_id", body.get("values")
        kind = "terms"
    elif kind == "exists":
        only(body, ["field"], "exists")
        return {"op": "exists", "field": mapping.field(body["field"])}
    elif kind in ("term", "terms", "range", "prefix"):
        name, values = one(body, kind)
    else:
        raise ValueError(f"unsupported filter {kind}")
    key = mapping.field(name, exact=True)

    def literal(value):
        if name == "_id":
            check(isinstance(value, str), "_id must be a string")
            return value
        return mapping.value(name, value)

    if kind == "terms":
        check(isinstance(values, list) and values, "terms requires a nonempty array")
        return {"op": "or", "filters": [{"op": "eq", "field": key, "value": literal(v)} for v in values]}
    if kind == "range":
        only(values, ["gt", "gte", "lt", "lte"], "range")
        check(values, "range requires at least one bound")
        return {"op": "range", "field": key, **{k: literal(v) for k, v in values.items()}}
    if isinstance(values, dict):
        only(values, ["value"], kind)
        values = values["value"]
    if kind == "prefix":
        check(isinstance(values, str), "prefix expects a string")
    return {"op": "eq" if kind == "term" else "prefix", "field": key, "value": literal(values)}


def as_list(value):
    return value if isinstance(value, list) else [value]


def translate(dsl, mapping):
    only(dsl, ["query", "size", "from", "sort", "aggs", "aggregations", "track_total_hits"], "query body")
    out = {"mode": "exact"}
    for source_key, dest_key in [("from", "offset"), ("size", "limit")]:
        if source_key in dsl:
            n = dsl[source_key]
            check(type(n) is int and n >= 0, f"invalid {source_key}")
            out[dest_key] = n
    check(dsl.get("track_total_hits", True) is True, "only exact hit counts are supported")
    query = dsl.get("query", {"match_all": {}})
    kind, body = one(query, "query")
    if kind == "bool" and "must" in body:
        clauses = as_list(body["must"])
        matches = [clause for clause in clauses if isinstance(clause, dict) and "match" in clause]
        if matches:
            check(len(matches) == 1 and "should" not in body, "unsupported combination of scored match clauses")
            rest = dict(body, must=[clause for clause in clauses if clause not in matches])
            out["filter"] = filt({"bool": rest}, mapping)
            kind, body = one(matches[0], "match query")
    if kind == "match":
        name, value = one(body, "match")
        check(name == mapping.text_field, "match requires the imported text field")
        if isinstance(value, dict):
            only(value, ["query", "operator"], "match")
            check(value.get("operator", "or") == "or", "only match operator 'or' is supported")
            value = value["query"]
        check(isinstance(value, str) and value.strip(), "match requires nonempty text")
        out["text"] = value
    elif kind == "match_all":
        only(body, [], "match_all")
    else:
        out["filter"] = filt({kind: body}, mapping)
    for sort in as_list(dsl.get("sort", [])):
        name, direction = (sort, "asc") if isinstance(sort, str) else one(sort, "sort")
        if isinstance(direction, dict):
            only(direction, ["order"], "sort")
            direction = direction["order"]
        check(direction in ("asc", "desc"), "invalid sort order")
        check(mapping.aliases.get(name, name) not in mapping.arrays, "array sort differs between engines")
        out.setdefault("sort", []).append({"field": mapping.field(name, exact=True), "direction": direction})
    check(not ("aggs" in dsl and "aggregations" in dsl), "duplicate aggregation sections")
    aggs = dsl.get("aggs", dsl.get("aggregations", {}))
    check(isinstance(aggs, dict), "aggregations must be an object")
    for alias, spec in aggs.items():
        kind, opts = one(spec, "aggregation")
        check(kind in ("sum", "min", "max", "avg"), f"unsupported aggregation {kind}")
        only(opts, ["field"], kind)
        name = opts["field"]
        check(mapping.types.get(name) in ("integer", "number"), "metric requires a numeric field")
        out.setdefault("aggregations", {})[alias] = {"kind": kind, "field": mapping.field(name)}
    return out


def rows(path, format_name):
    with path.open(encoding="utf-8-sig") as file:
        lines = ((n, line) for n, line in enumerate(file, 1) if line.strip())
        for n, line in lines:
            if format_name == "hits":
                try:
                    item = json.loads(line)
                    check(isinstance(item, dict), "expected a hit object")
                    # Only the bulk action parser may emit deletes.
                    check("_delete" not in item, "unexpected hit metadata _delete")
                    yield n, item, None
                except ValueError as exc:
                    yield n, {}, str(exc)
                continue
            action, meta = one(json.loads(line), f"bulk action at line {n}")
            check(action in ("index", "delete", "create", "update"), f"line {n}: unknown action, cannot recover framing")
            content = None if action == "delete" else next(lines, (None, None))[1]
            check(action == "delete" or content is not None, f"line {n}: missing bulk document")
            try:
                only(meta, ["_id", "_index", "_version", "version"], "bulk action metadata")
                check(action in ("index", "delete"), f"{action} preserves source-only write conditions and cannot be imported")
                hit = {"_id": meta.get("_id"), "_index": meta.get("_index")}
                if action == "delete":
                    hit["_delete"] = True
                else:
                    hit.update(_source=json.loads(content), _version=meta.get("version", meta.get("_version")))
                yield n, hit, None
            except ValueError as exc:
                yield n, {}, str(exc)


def import_index(args):
    mapping = Mapping(load(args.mapping), args.text_field, args.array_field)
    if args.readers_field:
        check(mapping.types.get(args.readers_field) in ("text", "keyword"), "readers field must be a mapped string")
    dest = target(args)
    prefix = target_index(args.index)
    if not args.dry_run:
        with args.input.open("rb") as input_file:
            snapshot_id = args.run_id or hashlib.file_digest(input_file, "sha256").hexdigest()
        dest.request("PUT", prefix, mapping.settings(args))
    counts = {"accepted": 0, "failed": 0}

    def report(line, ident, error=None):
        counts["failed" if error else "accepted"] += 1
        print(json.dumps({"line": line, "id": ident, "status": "error" if error else
              ("valid" if args.dry_run else "applied"), **({"error": error} if error else {})}, ensure_ascii=True))

    def write(batch):
        if not batch:
            return
        if not args.dry_run:
            payload = {"operations": [item[2] for item in batch]}
            key = hashlib.sha256(encoded([snapshot_id, [entry[0] for entry in batch], payload])).hexdigest()
            for attempt in range(3):
                try:
                    dest.request("POST", prefix + "/bulk", payload, key)
                    break
                except HTTPError as exc:
                    try:
                        if exc.code in (400, 409, 413, 422):
                            if len(batch) > 1:
                                mid = len(batch) // 2
                                write(batch[:mid])
                                write(batch[mid:])
                            else:
                                detail = exc.read(2048).decode(errors="replace")
                                report(*batch[0][:2], f"HTTP {exc.code}: {detail}")
                            return
                        if exc.code not in (429, 502, 503, 504) or attempt == 2:
                            raise
                    finally:
                        exc.close()
                except (URLError, TimeoutError):
                    if attempt == 2:
                        raise
                time.sleep(2 ** attempt)
        for line, ident, _ in batch:
            report(line, ident)

    batch = []
    for line, hit, error in rows(args.input, args.format):
        try:
            check(not error, error)
            operation = mapping.doc(hit, args)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            write(batch)
            batch.clear()
            report(line, hit.get("_id"), str(exc))
            continue
        batch.append((line, hit["_id"], operation))
        if len(batch) >= args.batch_size:
            write(batch)
            batch.clear()
    write(batch)
    print(json.dumps(counts), file=sys.stderr)
    return int(counts["failed"] > 0)


def replay(args):
    mapping = Mapping(load(args.mapping), args.text_field, args.array_field)
    prefix = target_index(args.index)
    es, loci = source(args), target(args)
    drift = 0
    with args.queries.open(encoding="utf-8-sig") as file:
        for n, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                dsl = json.loads(line)
                native = translate(dsl, mapping)
                original = es.request("POST", f"/{source_index(args.source_index)}/_search",
                                      dict(dsl, track_total_hits=True))
                complete(original)
                found = loci.request("POST", prefix + "/search", native)
                a = [v["_id"] for v in original["hits"]["hits"]]
                b = [v["id"] for v in found["hits"]]
                expected = total(original["hits"])
                metrics = {k: v.get("value") for k, v in original.get("aggregations", {}).items()}
                actual = found.get("aggregations", {})
                same_metrics = metrics.keys() == actual.keys() and all(
                    v == actual[k] or isinstance(v, (int, float)) and isinstance(actual[k], (int, float))
                    and math.isclose(v, actual[k], rel_tol=1e-9, abs_tol=1e-12)
                    for k, v in metrics.items())
                same = a == b and expected == found["total"] and same_metrics
                result = {"line": n, "source_total": expected, "target_total": found["total"],
                          "source_ids": a, "target_ids": b, "metrics_match": same_metrics,
                          "source_metrics": metrics, "target_metrics": actual,
                          "same": same, "overlap": len(set(a) & set(b))}
            except (KeyError, ValueError, TypeError, OSError, OverflowError) as exc:
                result = {"line": n, "same": False, "error": str(exc)}
            drift += not result["same"]
            print(json.dumps(result))
    return int(drift > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="save one source index mapping and document snapshot")
    export.add_argument("--index", required=True)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--page-size", type=int, default=500)
    imp = sub.add_parser("import", help="validate or apply exported hits or bulk NDJSON")
    imp.add_argument("--input", required=True, type=Path)
    imp.add_argument("--format", choices=["hits", "bulk"], default="hits")
    imp.add_argument("--source-index", help="require this source _index on each record")
    imp.add_argument("--model", choices=["lexical", "bge-small-en-v1.5", "multilingual-e5-small"], default="lexical")
    imp.add_argument("--batch-size", type=int, default=100)
    imp.add_argument("--dry-run", action="store_true")
    imp.add_argument("--run-id", help="explicit identity for an intentional new replay of the same input")
    access = imp.add_mutually_exclusive_group(required=True)
    access.add_argument("--public", action="store_true", help="make the migrated documents readable to index readers")
    access.add_argument("--readers-field", help="source field containing authorized principal(s)")
    query = sub.add_parser("translate", help="print the Loci equivalent of a single search body")
    query.add_argument("--query", required=True, type=Path)
    rep = sub.add_parser("replay", help="compare saved query bodies on both engines")
    rep.add_argument("--source-index", required=True)
    rep.add_argument("--queries", required=True, type=Path)
    for cmd in (imp, query, rep):
        cmd.add_argument("--mapping", required=True, type=Path)
        cmd.add_argument("--text-field", help="source field supplying the full-text body")
        cmd.add_argument("--array-field", action="append", default=[], help="repeat for each array-valued source field")
    for cmd in (rep, export):
        cmd.add_argument("--source-url", required=True)
    for cmd in (imp, rep):
        cmd.add_argument("--target-url", default="http://127.0.0.1:7707")
        cmd.add_argument("--index", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            check(1 <= args.page_size <= 10000, "page size must be 1..10000")
            return export_index(args)
        if args.command == "import":
            check(1 <= args.batch_size <= 10000, "batch size must be 1..10000")
            return import_index(args)
        if args.command == "replay":
            return replay(args)
        print(json.dumps(translate(load(args.query), Mapping(load(args.mapping), args.text_field, args.array_field)), indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
