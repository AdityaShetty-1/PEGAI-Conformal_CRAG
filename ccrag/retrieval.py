import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from langchain_core.embeddings import Embeddings

from .common import Doc


class HashEmbeddings(Embeddings):
    """Bag-of-words hashing embeddings. Offline stand-in for smoke tests only."""

    def __init__(self, dim=512):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class OllamaEmbeddings(Embeddings):
    def __init__(self, model, base_url):
        import requests

        self.model = model
        self.url = base_url.rstrip("/") + "/api/embed"
        self.session = requests.Session()

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), 64):
            response = self.session.post(self.url, json={
                "model": self.model, "input": texts[start:start + 64],
            }, timeout=120)
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return vectors

    @lru_cache(maxsize=4096)
    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings(name, mock=False, ollama_url="http://127.0.0.1:11434"):
    if mock:
        return HashEmbeddings()
    if name.startswith("ollama:"):
        return OllamaEmbeddings(name.removeprefix("ollama:"), ollama_url)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=name, encode_kwargs={"normalize_embeddings": True})


class Retriever:
    def __init__(self, corpus, embeddings, index_dir):
        from langchain_community.vectorstores import FAISS

        texts = [d.text for d in corpus]
        if not texts:
            raise ValueError("The document corpus is empty.")
        path = Path(index_dir) / "vectors.npy"
        if path.exists():
            vectors = np.load(path, allow_pickle=False)
        else:
            vectors = np.asarray(embeddings.embed_documents(texts), dtype=np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                np.save(tmp, vectors, allow_pickle=False)
            os.replace(tmp.name, path)
        self.vs = FAISS.from_embeddings(zip(texts, vectors), embeddings,
                                        metadatas=[{"id": d.id} for d in corpus])

    def retrieve(self, query, k=4):
        return [Doc(id=str(r.metadata["id"]), text=r.page_content) for r in self.vs.similarity_search(query, k=k)]
