"""Time a fresh catalog from an empty directory to one highlighted passage.

The clock starts with the built binary. It does not include compiling Loci.
The run fails if that path takes ten minutes or longer. From the repository root:

    python tools/rehearse_install.py --binary target/debug/loci.exe
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from urllib.error import HTTPError
from urllib.request import Request, urlopen

QUOTE = "Restore the encrypted vault from its backup before the next rotation."
LIMIT_SECONDS = 600


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
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def request(self, method, path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def run(binary):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="loci-install-") as temporary:
        root = Path(temporary)
        note = root / "recovery.md"
        note.write_text(
            "Operators keep one copy offline.\n\n" + QUOTE + "\n",
            encoding="utf-8",
        )
        service = Service(binary, root / "data")
        try:
            service.start()
            service.request("PUT", "/v1/indexes/manuals", {"model": "lexical"})
            subprocess.run(
                [str(binary), "import", "--index", "manuals", "--url", service.url, str(note)],
                check=True,
            )
            page = service.request(
                "POST",
                "/v1/indexes/manuals/search",
                {"text": "encrypted vault", "limit": 1, "highlight": True},
            )
        finally:
            service.stop()
    elapsed = time.perf_counter() - started
    if elapsed >= LIMIT_SECONDS:
        raise RuntimeError(f"install to a cited passage took {elapsed:.3f}s")
    hits = page.get("hits") or []
    if not hits or QUOTE not in hits[0]["text"]:
        raise RuntimeError(f"search did not return the imported note: {page}")
    spans = page.get("highlights", {}).get(hits[0]["id"], [])
    quoted = [hits[0]["text"][item["start"]:item["end"]] for item in spans]
    if not quoted or any(piece not in QUOTE for piece in quoted):
        raise RuntimeError(f"highlight did not cite the imported note: {quoted}")
    print(f"cited passage in {elapsed:.3f} seconds")
    print(f"{hits[0]['id']} highlights {quoted}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.binary.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, KeyError) as error:
        print(f"Install rehearsal failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
