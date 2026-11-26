"""Pluggable answer generator.

`ExtractiveReader` is a deterministic, zero-cost stand-in for a real LLM: it
picks the context sentence(s) with the highest lexical overlap with the
question and returns them verbatim. This makes answer correctness a
transparent, reproducible function of *what was actually retrieved* -- which
is exactly the variable the noise-injection experiments manipulate. Point a
real LLM call at the same `answer(question, context_docs)` signature to
switch a live run over to it; nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Document

_SENT_SPLIT = re.compile(r"(?<=[.])\s+")
_WORD = re.compile(r"[A-Za-z']+")
_PROPER_NOUN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")

STOPWORDS = set(
    # Generic English function words only. Domain nouns such as "capital",
    # "landmark", "located", "designed" are deliberately *not* here: they
    # are exactly the type-signal that tells a grader whether retrieved
    # text addresses what the question is asking about (e.g. "landmark"
    # appearing only in capital-city docs, not country docs, is what makes
    # a 2-hop HotpotQA-style question detectably under-answered by a
    # country doc alone). Stripping them collapses that signal to noise.
    "a an the of is are was were in on at to for and or by with what who "
    "which does do about has have it its this that these those".split()
)


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def overlap_score(question_tokens: set[str], sentence: str) -> int:
    toks = set(tokenize(sentence)) - STOPWORDS
    return len(toks & question_tokens)


@dataclass
class ExtractiveReader:
    def answer(self, question: str, context_docs: list[Document]) -> tuple[str, list[str]]:
        q_tokens = set(tokenize(question)) - STOPWORDS
        candidates: list[tuple[int, str]] = []
        for doc in context_docs:
            for sent in sentences(doc.text):
                candidates.append((overlap_score(q_tokens, sent), sent))
        if not candidates:
            return "", []
        candidates.sort(key=lambda x: -x[0])
        best_score = candidates[0][0]
        if best_score == 0:
            # Nothing in context is relevant -- still produce *a* best-effort
            # guess (the single most topically related sentence) so that
            # baselines without an abstain path have something to say. This
            # is exactly the hallucination failure mode the proposal targets.
            top_sents = [candidates[0][1]]
        else:
            top_sents = [s for score, s in candidates if score == best_score][:2]
        return " ".join(top_sents), top_sents

    def extract_rewrite_entities(self, context_docs: list[Document], exclude_tokens: set[str]) -> list[str]:
        """Entities mentioned in context but absent from the current query --
        candidates for building a repaired query in the rewrite-and-re-retrieve
        step."""
        entities: list[str] = []
        for doc in context_docs:
            for m in _PROPER_NOUN.finditer(doc.text):
                phrase = m.group(1)
                toks = set(tokenize(phrase))
                if toks & exclude_tokens:
                    continue
                if phrase not in entities:
                    entities.append(phrase)
        return entities


def is_correct(answer: str, question_gold_aliases: list[str]) -> bool:
    if not answer:
        return False
    low = answer.lower()
    return any(alias.lower() in low for alias in question_gold_aliases)
