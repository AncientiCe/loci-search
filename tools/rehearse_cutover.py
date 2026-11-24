"""Import a tiny exported index, compare one ordered query, then restore its backup.

This is a disposable rehearsal. It does not contact Elasticsearch and it does not
change the published version. From the repository root:

    python tools/rehearse_cutover.py --binary target/debug/loci.exe
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import migrate_search as migration


class Service:
    def __init__(self, binary, data):
        self.binary = str(binary)
        self.data = data
        self.process = None
        self.url = None

    def start(self):
        self.process = subprocess.Popen(
            [self.binary, "serve", "--bind", "127.0.0.1:0", "--data", str(self.data)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
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
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            error.close()
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def run(binary):
    with tempfile.TemporaryDirectory(prefix="loci-cutover-") as temporary:
        root = Path(temporary)
        mapping = root / "mapping.json"
        documents = root / "documents.ndjson"
        query = root / "query.json"
        data = root / "data"
        restored = root / "restored"
        backup = root / "catalog.sqlite"
        mapping.write_text(json.dumps({"properties": {
            "body": {"type": "text"},
            "price": {"type": "integer"},
        }}), encoding="utf-8")
        rows = [
            {"_id": "bright", "_index": "inventory", "_version": 1, "_source": {"body": "brass lamp", "price": 12}},
            {"_id": "dim", "_index": "inventory", "_version": 1, "_source": {"body": "brass lamp shade", "price": 3}},
        ]
        documents.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        query.write_text(json.dumps({"query": {"match": {"body": "lamp"}}, "sort": [{"price": "desc"}]}), encoding="utf-8")

        service = Service(binary, data)
        service.start()
        try:
            status = migration.main([
                "import", "--input", str(documents), "--mapping", str(mapping), "--index", "inventory",
                "--text-field", "body", "--target-url", service.url, "--public",
            ])
            if status != 0:
                raise RuntimeError(f"import exited {status}")
            translated = migration.translate(
                json.loads(query.read_text(encoding="utf-8")),
                migration.Mapping(json.loads(mapping.read_text(encoding="utf-8")), "body"),
            )
            page = service.request("POST", "/v1/indexes/inventory/search", translated)
            ids = [hit["id"] for hit in page["hits"]]
            if ids != ["bright", "dim"]:
                raise RuntimeError(f"ordered query returned {ids}")
            if page["total"] != 2:
                raise RuntimeError(f"expected 2 matches, received {page['total']}")
        finally:
            service.stop()

        backup_status = subprocess.run(
            [str(binary), "backup", "--data", str(data), "--out", str(backup)],
            check=False,
        )
        if backup_status.returncode != 0:
            raise RuntimeError("backup failed")
        restore_status = subprocess.run(
            [str(binary), "restore", "--input", str(backup), "--data", str(restored)],
            check=False,
        )
        if restore_status.returncode != 0:
            raise RuntimeError("restore into an empty directory failed")
        rejected = subprocess.run(
            [str(binary), "restore", "--input", str(backup), "--data", str(restored)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if rejected.returncode == 0 or "empty destination" not in rejected.stderr:
            raise RuntimeError("restore replaced a directory that already held a catalog")

        restored_service = Service(binary, restored)
        restored_service.start()
        try:
            document = restored_service.request("GET", "/v1/indexes/inventory/documents/bright")
            if document.get("text") != "brass lamp" or document.get("revision") != "1":
                raise RuntimeError(f"restored document diverged: {document}")
            page = restored_service.request("POST", "/v1/indexes/inventory/search", translated)
            if [hit["id"] for hit in page["hits"]] != ["bright", "dim"]:
                raise RuntimeError("restored catalog changed query order")
        finally:
            restored_service.stop()
    print("Cutover rehearsal passed: import, ordered query, backup, restore, and rejected overwrite.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    try:
        return run(args.binary.resolve())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Cutover rehearsal failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
