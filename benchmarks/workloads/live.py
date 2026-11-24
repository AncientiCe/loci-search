"""Run the frozen workload queries against a temporary local Loci service."""

import argparse
from collections import deque
import json
from pathlib import Path
from queue import Empty, Queue
import secrets
import subprocess
import sys
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import validate


DAY_START = 1735689600
DAY = 86400


class Server:
    def __init__(self, binary, directory):
        self.binary = str(binary.resolve())
        self.data = directory / "data"
        self.credentials = directory / "credentials.json"
        self.process = None
        self.tokens = {name: secrets.token_urlsafe(36) for name in ("admin", "sales", "ops", "guest")}
        self.credentials.write_text(json.dumps([
            {"token": token, "indexes": ["*"] if name == "admin" else ["records"],
             "write": name == "admin", "principals": [] if name == "admin" else [name]}
            for name, token in self.tokens.items()
        ]), encoding="utf-8")

    def start(self):
        self.process = subprocess.Popen(
            [self.binary, "serve", "--bind", "127.0.0.1:0", "--data", str(self.data),
             "--credentials", str(self.credentials)], stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ready = Queue()

        def read_log(pipe):
            recent = deque(maxlen=8)
            with pipe:
                for line in pipe:
                    line = line.strip()
                    if line.startswith("Listening on http://127.0.0.1:"):
                        ready.put(line.removeprefix("Listening on "))
                    recent.append(line)
            ready.put(RuntimeError("service exited: " + " ".join(recent)))

        threading.Thread(target=read_log, args=(self.process.stderr,), daemon=True).start()
        try:
            value = ready.get(timeout=30)
        except Empty as error:
            raise RuntimeError("service did not report its local address within 30 seconds") from error
        if isinstance(value, Exception):
            raise value
        self.url = value

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None

    def api(self, method, path, body=None, principal="admin"):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(self.url + path, data=data, method=method, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.tokens[principal],
        })
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error


class Adapter:
    def __init__(self, server, name, model):
        self.server = server
        self.name = name
        self.model = model if name == "knowledge" else "lexical"
        self.index = "/v1/indexes/" + name
        self.documents = {}
        self.versions = {}
        self.by_uri = {}
        self.cursors = {}

    def create(self, docs):
        settings = {"model": self.model}
        if self.name == "records":
            settings["default_visibility"] = "private"
        if self.name == "logs":
            settings["stream"] = {"timestamp": "time", "rollover_seconds": 3600,
                                  "retention_seconds": DAY, "automatic": False}
        self.server.api("PUT", self.index, settings)
        self.write(docs)

    def write(self, docs):
        ops = []
        for raw in docs:
            key = raw["id"]
            self.documents[key] = dict(raw)
            self.versions[key] = self.versions.get(key, 0) + 1
            fields = {k: v for k, v in raw.items() if k not in ("id", "text", "revision", "groups")}
            if self.name == "logs":
                fields["time"] += DAY_START
            text = raw.get("text", raw.get("name", key))
            uri = raw.get("uri", key)
            self.by_uri[uri] = key
            ops.append({"op": "upsert", "document": {
                "id": uri, "text": text, "readers": raw.get("groups", []), "fields": fields,
                "revision": raw.get("revision", str(self.versions[key])),
            }})
        if ops:
            self.server.api("POST", self.index + "/bulk", {"operations": ops})

    def mutate(self, step):
        if "upsert" in step:
            self.write([step["upsert"]])
        elif "patch" in step:
            patch = step["patch"]
            self.write([{**self.documents[patch["id"]], **patch}])
        elif "append_series" in step:
            for doc in validate.expand_series(step["append_series"]):
                self.write([doc])
        elif "expire_before" in step:
            self.server.api("POST", self.index + "/expire", {"now": DAY_START + step["expire_before"] + DAY})
        elif step == {"restart": True}:
            self.server.stop()
            self.server.start()
        else:
            raise ValueError(f"unknown fixture operation {step}")

    def ranked(self, step):
        query = step["search"]
        body = {"text": query["text"], "limit": query["k"]}
        if self.name == "catalog" and step["id"] == "typo":
            body["fuzzy"] = True
        response = self.server.api("POST", self.index + "/search", body)
        if self.name != "knowledge":
            return {"hits": [{"id": h["id"]} for h in response["hits"]]}
        result = []
        for h in response["hits"]:
            # Regional hits cite passage coordinates. Exact lexical hits return a
            # complete document, whose citation covers its complete returned body.
            uri = h["document"] if self.model != "lexical" else h["id"]
            if self.model != "lexical":
                start, end = h["start"], h["end"]
            else:
                start, end = 0, len(h["text"].encode("utf-8"))
            result.append({"id": self.by_uri[uri], "uri": uri, "revision": h["revision"],
                           "start": start, "end": end, "quote": h["text"]})
        return {"hits": result}

    def structured(self, q):
        if "resume" in q:
            cursor, old_query = self.cursors[q["resume"]]
            body = {"cursor": cursor}
            facet = old_query.get("facet")
        else:
            body = {"limit": q.get("limit", 200)}
            if "sort" in q:
                body["sort"] = [{"field": q["sort"], "direction": "asc"}]
            facet = q.get("facet")
            if facet:
                body["facets"] = [facet]
            filters = []
            for field, constraints in q.get("filter", {}).items():
                if "eq" in constraints:
                    filters.append({"op": "eq", "field": field, "value": constraints["eq"]})
                else:
                    shift = DAY_START if self.name == "logs" and field == "time" else 0
                    filters.append({"op": "range", "field": field, **{k: v + shift for k, v in constraints.items()}})
            if filters:
                body["filter"] = filters[0] if len(filters) == 1 else {"op": "and", "filters": filters}
            if "histogram" in q:
                spec = q["histogram"]
                body["aggregations"] = {"buckets": {"kind": "date_histogram", "field": "time",
                    "start": spec["start"] + DAY_START, "end": spec["end"] + DAY_START, "width": spec["width"]}}
        page = self.server.api("POST", self.index + "/search", body, q.get("principal", "admin"))
        if "histogram" in q:
            return {"buckets": [[t - DAY_START, count] for t, count in page["aggregations"]["buckets"]]}
        if "save_cursor" in q:
            self.cursors[q["save_cursor"]] = (page.get("next_cursor"), q)
        result = {"ids": [h["id"] for h in page["hits"]], "count": page["total"]}
        if facet:
            result["facets"] = {b["value"]: b["count"] for b in page["facets"][facet]}
        return result


def run_workload(server, name, model, stream):
    fixture = json.loads((validate.ROOT / f"{name}.json").read_text(encoding="utf-8"))
    adapter = Adapter(server, name, model)
    successes = failures = 0
    try:
        adapter.create(fixture["documents"])
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL {name} setup: {error}", file=sys.stderr)
        return 0, sum("id" in s for s in fixture["steps"])
    blocked = None
    for step in fixture["steps"]:
        if "id" not in step:
            try:
                adapter.mutate(step)
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                blocked = f"setup operation {next(iter(step))} failed: {error}"
            continue
        case = step["id"]
        result = None
        try:
            if blocked:
                raise RuntimeError(blocked)
            if "search" in step:
                result = adapter.ranked(step)
                validate.grade_ranked(adapter.documents, step, result, name == "knowledge")
            else:
                result = adapter.structured(step["select"])
                validate.require(json.dumps(result, sort_keys=True) == json.dumps(step["expect"], sort_keys=True),
                                 f"expected {step['expect']}, received {result}")
        except (KeyError, TypeError, OSError, RuntimeError, ValueError) as error:
            if result is None:
                result = {"error": str(error)}
            print(f"FAIL {name}/{case}: {error}", file=sys.stderr)
            failures += 1
        else:
            print(f"PASS {name}/{case}")
            successes += 1
        if stream:
            print(json.dumps({"workload": name, "id": case, "result": result}), file=stream)
    return successes, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="built loci executable")
    parser.add_argument("--workload", choices=validate.WORKLOADS)
    parser.add_argument("--capture", type=Path, help="new JSONL output file")
    parser.add_argument("--model", choices=("lexical", "bge-small-en-v1.5", "multilingual-e5-small"), default="lexical")
    parser.add_argument("--model-source", type=Path, help="local pinned model assets for a semantic run")
    args = parser.parse_args()
    if (args.model == "lexical") != (args.model_source is None):
        parser.error("semantic runs require --model-source and a supported --model")
    try:
        with tempfile.TemporaryDirectory(prefix="loci-workloads-") as temporary:
            server = Server(args.binary, Path(temporary))
            output = args.capture.open("x", encoding="utf-8") if args.capture else None
            try:
                if args.model_source:
                    subprocess.run([server.binary, "model-install", args.model, "--data", str(server.data),
                                    "--source", str(args.model_source)], check=True)
                server.start()
                passed = failed = 0
                for name in ([args.workload] if args.workload else validate.WORKLOADS):
                    ok, bad = run_workload(server, name, args.model, output)
                    passed += ok
                    failed += bad
                print(f"{passed} live cases passed, {failed} failed; knowledge model={args.model}")
                return 1 if failed else 0
            finally:
                server.stop()
                if output:
                    output.close()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Live run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
