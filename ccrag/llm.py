import hashlib
import json
import os
import re
import time

_STOP = set("what which who whom whose when where why does did that this with from have has were was are the and for into about their there than then they them".split())


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3 and w not in _STOP}


class LLM:
    """OpenAI-compatible chat model with an on-disk cache.

    The cache stores the ORIGINAL latency of each call, so re-running an experiment
    is free but still reports realistic latency numbers.
    """

    def __init__(self, model, base_url, api_key, cache_dir=".cache", max_tokens=64, min_interval=0.0):
        from langchain_openai import ChatOpenAI

        self.client = ChatOpenAI(model=model, base_url=base_url, api_key=api_key or "none",
                                 temperature=0, max_tokens=max_tokens, max_retries=2, timeout=90)
        os.makedirs(cache_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        endpoint = hashlib.sha256(f"{base_url}|{max_tokens}".encode()).hexdigest()[:16]
        self._path = os.path.join(cache_dir, f"llm_{safe}_{endpoint}.jsonl")
        self._cache = {}
        if os.path.exists(self._path):
            with open(self._path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self._cache[r["k"]] = r
                    except (json.JSONDecodeError, KeyError):
                        continue  # A previous run may have stopped mid-write.
        self.min_interval = min_interval
        self.calls, self.latency = 0, 0.0

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        self.calls += 1
        hit = self._cache.get(key)
        if hit is None:
            t = time.time()
            text = self.client.invoke(prompt).content.strip()
            hit = {"k": key, "text": text, "latency": time.time() - t}
            self._cache[key] = hit
            with open(self._path, "a") as f:
                f.write(json.dumps(hit) + "\n")
            if self.min_interval:
                time.sleep(self.min_interval)
        self.latency += hit["latency"]
        return hit["text"]

    def snapshot(self):
        return self.calls, self.latency


class MockLLM:
    """Deterministic lexical stand-in used ONLY to test the plumbing (no API needed)."""

    def __init__(self):
        self.calls, self.latency = 0, 0.0

    def snapshot(self):
        return self.calls, self.latency

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.latency += 0.01
        task = re.search(r"### TASK: (\w+)", prompt).group(1)
        question = prompt.split("Question:")[-1].split("\n")[0].strip()
        if task == "REWRITE":
            return question.rstrip("?") + " facts"
        ctx = prompt.split("Context:")[1].split("Question:")[0]
        qw = _words(question)
        if task in ("SUFFICIENCY", "CHUNK"):
            if not qw:
                return "0"
            return str(int(100 * len(qw & _words(ctx)) / len(qw)))
        sents = [s for s in re.split(r"(?<=[.!?])\s+|\n", ctx) if s.strip()]
        if not sents:
            return "unknown"
        best = max(sents, key=lambda s: len(qw & _words(s)))
        toks = re.findall(r"[A-Za-z0-9]+", best)
        return toks[-1] if toks else "unknown"
