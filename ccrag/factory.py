import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .data import load_dataset_by_name
from .llm import LLM, MockLLM
from .retrieval import Retriever, get_embeddings
from .web import MockWeb, TavilyWeb

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


@dataclass
class Env:
    llm: object
    retriever: object
    web: object
    corpus: list
    questions: list
    stale_pool: list
    profile: dict


def build(dataset="mini_wiki", mock=False, cache_dir=".cache"):
    corpus, questions = load_dataset_by_name(dataset)
    embed_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    endpoint = urlparse(base_url)
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Use a plain LLM_BASE_URL and put credentials in LLM_API_KEY.")
    embed_id = ["hash-512"] if mock else [embed_name, ollama_url if embed_name.startswith("ollama:") else ""]
    index_tag = hashlib.sha256(json.dumps([embed_id, [(d.id, d.text) for d in corpus]],
                                          ensure_ascii=False).encode()).hexdigest()
    retriever = Retriever(corpus, get_embeddings(embed_name, mock, ollama_url),
                          os.path.join(cache_dir, f"index_{index_tag}"))
    if mock:
        llm, web = MockLLM(), MockWeb(corpus)
    else:
        llm = LLM(model, base_url, os.getenv("LLM_API_KEY", ""), cache_dir,
                  min_interval=float(os.getenv("LLM_MIN_INTERVAL", "0")))
        key = os.getenv("TAVILY_API_KEY", "")
        web = TavilyWeb(key, cache_dir) if key and not key.startswith("your_") else None
        if web is None:
            print("[info] Live web fallback is unavailable without TAVILY_API_KEY.")
    stale_pool = sorted({a for q in questions for a in q.answers})
    profile = {"dataset": dataset, "mock": mock, "index": index_tag,
               "generator": "mock" if mock else model,
               "endpoint": None if mock else hashlib.sha256(base_url.encode()).hexdigest(),
               "prompts": hashlib.sha256(Path(__file__).with_name("prompts.py").read_bytes()).hexdigest()}
    return Env(llm, retriever, web, corpus, questions, stale_pool, profile)
