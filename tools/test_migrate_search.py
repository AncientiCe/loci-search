import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import migrate_search as m


MAPPING = {"inventory": {"mappings": {"properties": {
    "body": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
    "price": {"type": "double"}, "updated": {"type": "date"},
    "groups": {"type": "keyword"}, "sku": {"type": "keyword"}
}}}}


def hit(ident, **fields):
    return {"_id": ident, "_index": "inventory", "_version": 3, "_source": fields}


class Migration(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.mapping = self.base / "mapping.json"
        self.docs = self.base / "documents.ndjson"
        self.mapping.write_text(json.dumps(MAPPING), encoding="utf-8")
        self.codec = m.Mapping(MAPPING, "body", ["groups"])

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = m.main(list(map(str, args)))
        return status, [json.loads(x) for x in stdout.getvalue().splitlines()], stderr.getvalue()

    def import_args(self):
        return ["import", "--input", self.docs, "--mapping", self.mapping, "--index", "inventory",
                "--text-field", "body", "--array-field", "groups"]

    def write_hits(self, records):
        self.docs.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_dry_run_validates_data_and_never_contacts_servers(self):
        self.write_hits([hit("a", body="hello", groups="engineering", updated=1000),
                         hit("b", body="second"), hit("c", groups=[""])])
        with patch.object(m.Endpoint, "request", side_effect=AssertionError("HTTP request")):
            code, rows, _ = self.run_cli(*self.import_args(), "--dry-run", "--readers-field", "groups")
        self.assertEqual(1, code)
        self.assertEqual(["valid", "error", "error"], [r["status"] for r in rows])
        self.assertEqual("b", rows[1]["id"])

    def test_overrides_and_unrecognized_features_are_explicit(self):
        for bad in [{"type": "nested"}, {"type": "keyword", "normalizer": "lowercase"},
                    {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 8}}}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.Mapping({"properties": {"body": bad}}, "body")
        with self.assertRaises(ValueError):
            m.translate({"query": {"term": {"body": "two words"}}}, self.codec)
        with self.assertRaises(ValueError):
            m.translate({"query": {"script": {"source": "return true"}}}, self.codec)

    def test_filter_text_arrays_date_sort_and_metrics(self):
        query = {"query": {"bool": {"must": {"match": {"body": "repair"}},
                 "filter": [{"term": {"body.keyword": "repair"}}, {"terms": {"groups": ["a", "b"]}},
                 {"range": {"updated": {"gte": 1000}}}]}}, "sort": [{"price": "desc"}],
                 "aggs": {"meanPrice": {"avg": {"field": "price"}}}}
        translated = m.translate(query, self.codec)
        self.assertEqual("repair", translated["text"])
        filters = translated["filter"]["filters"]
        self.assertEqual("fields.body", filters[0]["field"])
        self.assertEqual("or", filters[1]["op"])
        self.assertEqual("1970-01-01T00:00:01.000000Z", filters[2]["gte"])
        self.assertEqual({"field": "fields.price", "direction": "desc"}, translated["sort"][0])
        self.assertEqual("avg", translated["aggregations"]["meanPrice"]["kind"])

    def test_bulk_item_isolation_and_repeatable_receipts(self):
        self.write_hits([hit("a", body="first"), hit("bad", body="conflict"), hit("c", body="last")])
        applied = {}
        def server(method, path, body, key=None):
            if method == "PUT":
                return {}
            identifiers = [op["document"]["id"] for op in body["operations"]]
            if "bad" in identifiers:
                raise HTTPError("local", 409, "conflict", {}, io.BytesIO(b"conflicting input"))
            applied.setdefault(key, []).append(identifiers)
            return {"generation": 1}
        with patch.object(m.Endpoint, "request", side_effect=server):
            first = self.run_cli(*self.import_args(), "--public")
            second = self.run_cli(*self.import_args(), "--public")
        self.assertEqual(first, second)
        self.assertEqual([r["status"] for r in first[1]], ["applied", "error", "applied"])
        self.assertEqual(sorted(applied.values()), [[["a"], ["a"]], [["c"], ["c"]]])

    def test_bulk_actions_keep_input_order_and_distinct_receipts(self):
        index = json.dumps({"index": {"_id": "a", "_index": "inventory"}})
        delete = json.dumps({"delete": {"_id": "a", "_index": "inventory"}})
        record = json.dumps({"body": "same"})
        self.docs.write_text("\n".join([index, record, delete, index, record, delete]), encoding="utf-8")
        events = []
        def server(method, path, body, key=None):
            if method == "POST":
                events.append((key, body["operations"][0]["op"]))
            return {}
        with patch.object(m.Endpoint, "request", side_effect=server):
            result = self.run_cli(*self.import_args(), "--format", "bulk", "--batch-size", "1", "--public")
        self.assertEqual(0, result[0])
        self.assertEqual([op for _, op in events], ["upsert", "delete", "upsert", "delete"])
        self.assertEqual(4, len(set(key for key, _ in events)))

    def test_scroll_export_is_complete_or_unpublished(self):
        def page(rows, timed_out=False):
            return {"timed_out": timed_out, "_shards": {"failed": 0}, "_scroll_id": "s1",
                    "hits": {"total": {"value": 1, "relation": "eq"}, "hits": rows}}
        for failed in [False, True]:
            folder = self.base / str(failed)
            calls = [MAPPING, page([hit("a", body="abc")]), page([], failed), {}]
            with patch.object(m.Endpoint, "request", side_effect=calls) as requests:
                status, _, _ = self.run_cli("export", "--source-url", "http://localhost:9200",
                                            "--index", "inventory", "--out", folder)
            self.assertEqual("DELETE", requests.call_args.args[0])
            self.assertEqual(not failed, folder.exists())
            self.assertEqual(2 if failed else 0, status)
            if not failed:
                self.assertEqual("a", json.loads((folder / "documents.ndjson").read_text())["_id"])

    def test_replay_catches_order_count_metric_and_unsupported_queries(self):
        self.docs.write_text('{"sort":["_id"]}\n{"query":{"script":{}}}\n', encoding="utf-8")
        results = [{"timed_out": False, "_shards": {"failed": 0}, "hits": {
                    "total": {"value": 2, "relation": "eq"}, "hits": [{"_id": "a"}, {"_id": "b"}]}},
                   {"total": 2, "hits": [{"id": "b"}, {"id": "a"}], "aggregations": {}}]
        with patch.object(m.Endpoint, "request", side_effect=results):
            code, rows, _ = self.run_cli("replay", "--source-url", "http://localhost:9200",
                "--source-index", "inventory", "--index", "inventory", "--queries", self.docs, "--mapping", self.mapping)
        self.assertEqual(1, code)
        self.assertFalse(rows[0]["same"])
        self.assertEqual(2, rows[0]["overlap"])
        self.assertIn("unsupported", rows[1]["error"])


if __name__ == "__main__":
    unittest.main()
