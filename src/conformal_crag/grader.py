"""Sufficiency graders.

Each grader looks only at the question and the retrieved context -- never at
the gold answer -- and outputs a sufficiency score s in [0, 1], matching the
proposal's "grader score s" in the method diagram. Three variants proxy the
three real grader families named in RQ3 (prompted LLM, fine-tuned small
model, purpose-trained sufficiency autorater) with different, deterministic
feature weightings and noise so the ablation in experiments/run_experiments.py
has something real to compare. Swap any of these for an actual model call
behind the same `score(question, context_docs) -> float` signature.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .data import Document
from .llm import STOPWORDS, overlap_score, sentences, tokenize


def _stable_noise(key: str, scale: float) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return (frac - 0.5) * 2 * scale


def _features(question: str, context_docs: list[Document]) -> dict[str, float]:
    q_tokens = set(tokenize(question)) - STOPWORDS
    if not q_tokens:
        return {"lexical_overlap": 0.0, "best_sentence_overlap": 0.0, "doc_count_bonus": 0.0}
    context_tokens: set[str] = set()
    best_sent_overlap = 0
    for doc in context_docs:
        context_tokens |= set(tokenize(doc.text)) - STOPWORDS
        for sent in sentences(doc.text):
            best_sent_overlap = max(best_sent_overlap, overlap_score(q_tokens, sent))
    lexical_overlap = len(q_tokens & context_tokens) / len(q_tokens)
    best_sentence_overlap = best_sent_overlap / len(q_tokens)
    doc_count_bonus = min(len(context_docs), 3) / 3.0
    return {
        "lexical_overlap": min(lexical_overlap, 1.0),
        "best_sentence_overlap": min(best_sentence_overlap, 1.0),
        "doc_count_bonus": doc_count_bonus,
    }


@dataclass
class SufficiencyGrader:
    variant: str = "autorater_sim"  # "prompted_llm_sim" | "finetuned_small_sim" | "autorater_sim"
    noise_scale: float = 0.05
    seed: int = 0

    def score(self, question: str, context_docs: list[Document]) -> float:
        f = _features(question, context_docs)
        if self.variant == "prompted_llm_sim":
            # Careful "reader": weighs the single best supporting sentence heavily.
            raw = 0.75 * f["best_sentence_overlap"] + 0.20 * f["lexical_overlap"] + 0.05 * f["doc_count_bonus"]
        elif self.variant == "finetuned_small_sim":
            # Cruder, cheaper model: leans on crude bag-of-words overlap, noisier.
            raw = 0.35 * f["best_sentence_overlap"] + 0.55 * f["lexical_overlap"] + 0.10 * f["doc_count_bonus"]
        elif self.variant == "autorater_sim":
            # Purpose-trained sufficiency classifier (Joren et al.-style): sharper
            # decision boundary around the "is there one sentence that nails it" signal.
            base = 0.6 * f["best_sentence_overlap"] + 0.3 * f["lexical_overlap"] + 0.1 * f["doc_count_bonus"]
            raw = base ** 0.7  # push mid-range scores apart -> sharper boundary
        else:
            raise ValueError(f"unknown grader variant: {self.variant}")
        key = f"{self.variant}:{self.seed}:{question}:{len(context_docs)}"
        noisy = raw + _stable_noise(key, self.noise_scale)
        return max(0.0, min(1.0, noisy))
