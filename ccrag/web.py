import hashlib
import json
import os
import re
import time

from .common import Doc
from .llm import _words


class TavilyWeb:
    """Tavily search with an on-disk cache (saves your free-tier credits on re-runs)."""

    def __init__(self, api_key, cache_dir=".cache"):
        from tavily import TavilyClient

        self.client = TavilyClient(api_key=api_key)
        os.makedirs(cache_dir, exist_ok=True)
        self._path = os.path.join(cache_dir, "web_cache.jsonl")
        self._cache = {}
        if os.path.exists(self._path):
            with open(self._path) as f:
                for line in f:
                    r = json.loads(line)
                    self._cache[r["k"]] = r
        self.calls, self.latency = 0, 0.0

    def search(self, query, k=4):
        key = hashlib.sha256(f"{query}|{k}".encode()).hexdigest()
        self.calls += 1
        hit = self._cache.get(key)
        if hit is None:
            t = time.time()
            res = self.client.search(query=query, max_results=k, search_depth="basic")
            docs = [{"id": r["url"], "text": f'{r.get("title", "")}. {r.get("content", "")}'} for r in res.get("results", [])]
            hit = {"k": key, "docs": docs, "latency": time.time() - t}
            self._cache[key] = hit
            with open(self._path, "a") as f:
                f.write(json.dumps(hit) + "\n")
        self.latency += hit["latency"]
        return [Doc(id=d["id"], text=d["text"], source="web") for d in hit["docs"]]

    def snapshot(self):
        return self.calls, self.latency


class MockWeb:
    """Pretends the 'web' contains the whole corpus (smoke tests only)."""

    def __init__(self, corpus):
        self.items = [(d, _words(d.text)) for d in corpus]
        self.calls, self.latency = 0, 0.0

    def search(self, query, k=4):
        self.calls += 1
        self.latency += 0.05
        qw = _words(query)
        top = sorted(self.items, key=lambda x: -len(qw & x[1]))[:k]
        return [Doc(id="web:" + d.id, text=d.text, source="web") for d, _ in top]

    def snapshot(self):
        return self.calls, self.latency
