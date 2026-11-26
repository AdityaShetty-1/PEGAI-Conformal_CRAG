"""Web fallback (open-domain mode only).

Default implementation models an independent web search by re-retrieving
from a *clean, uncorrupted* copy of the corpus: a real web search isn't
subject to whatever noise was injected into the local retrieval index, which
is exactly why it can rescue a query that local noise has degraded. Closed-
domain mode disables this entirely (see pipeline.py), reflecting the
proposal's data-governance caveat.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Document
from .retriever import Retriever


@dataclass
class WebFallback:
    clean_retriever: Retriever

    def search(self, query: str, k: int = 2) -> list[Document]:
        return self.clean_retriever.retrieve(query, k)


class TavilyWebFallback:
    """Real web search via the Tavily API, matching the proposal's intended
    stack. Not used by default -- this repo's experiments run with no
    network access or paid API budget. Provide a TAVILY_API_KEY and
    implement `search()` to switch a live deployment over to it; the call
    site in router.py only depends on the `search(query, k) -> list[Document]`
    signature, so no other code changes are needed.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, k: int = 2) -> list[Document]:  # pragma: no cover - stub
        raise NotImplementedError(
            "Wire up a Tavily client call here (tavily-python's TavilyClient.search), "
            "wrap each hit's text into a Document, and return the top k."
        )
