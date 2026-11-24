"""Regression tests for the grading boundary, independent of the search service."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate


class GraderTests(unittest.TestCase):
    def test_frozen_truth_and_adapter_stream(self):
        for name in validate.WORKLOADS:
            fixture = json.loads((validate.ROOT / f"{name}.json").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cases = validate.run(fixture, None, emit=True)
            requests = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len([req for req in requests if "id" in req]), cases)
            self.assertEqual(requests[0]["reset"], fixture["documents"])
            self.assertTrue(all("expect" not in req and "judged" not in req for req in requests))
            if name == "logs":
                self.assertEqual(sum("upsert" in req for req in requests), 27)
                self.assertIn({"workload": "logs", "restart": True}, requests)

    def test_citations_and_missing_evidence(self):
        data = {"a": {"uri": "docs/a", "revision": "1", "text": "café closed"}}
        step = {"search": {"k": 1}, "judged": {"a": "café"}}
        result = validate.reference_ranked(data, step, True)
        self.assertEqual(result["hits"][0]["end"], 5)
        validate.grade_ranked(data, step, result, True)
        result["hits"][0]["end"] = 4
        with self.assertRaises(ValueError):
            validate.grade_ranked(data, step, result, True)
        with self.assertRaisesRegex(ValueError, "missing judged"):
            validate.grade_ranked(data, step, {"hits": []}, True)

    def test_duplicate_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "result.jsonl"
            row = json.dumps({"workload": "logs", "id": "trace", "result": {}}) + "\n"
            capture.write_text(row * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate capture"):
                validate.read_capture(capture)

    def test_pinned_page_and_acl_grade(self):
        fixture = json.loads((validate.ROOT / "records.json").read_text(encoding="utf-8"))
        expected = {("records", s["id"]): s["expect"] for s in fixture["steps"] if "id" in s}
        validate.run(fixture, expected.copy(), emit=False)
        expected["records", "pinned-next"] = {"ids": ["b2", "b5"], "count": 4}
        with self.assertRaisesRegex(ValueError, "records/pinned-next"):
            validate.run(fixture, expected, emit=False)


if __name__ == "__main__":
    unittest.main()
