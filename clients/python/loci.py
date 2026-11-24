"""Synchronous Loci HTTP client using the Python standard library."""

import json
from urllib.parse import quote
from urllib.request import Request, urlopen


class Loci:
    def __init__(self, base_url="http://127.0.0.1:7707", token=None, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method, path, body=None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, allow_nan=False).encode("utf-8")
        req = Request(self.base_url + path, data=data, headers=headers, method=method)
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _index(name):
        return "/v1/indexes/" + quote(name, safe="")

    def create_index(self, name, settings=None):
        return self._request("PUT", self._index(name), settings or {})

    def bulk(self, name, operations):
        return self._request("POST", self._index(name) + "/bulk", {"operations": list(operations)})

    def search(self, name, **query):
        return self._request("POST", self._index(name) + "/search", query)

    def upgrade_model(self, name, model):
        return self._request("PUT", self._index(name) + "/model", {"model": model})
