"""Controlled retrieval corruption (applied AFTER retrieval, to local docs only; web results are never corrupted).

  irrelevant : with prob p the retrieved list is replaced by random non-gold passages (transient retriever failure;
               a re-retrieval after query rewriting gets an independent draw)
  missing    : with prob p the gold passage is absent from the index (persistent: rewriting cannot recover it)
  partial    : with prob p the gold passage loses every sentence containing the answer (persistent)
  outdated   : with prob p the answer inside the gold passage is replaced by a stale value (persistent)
"""
import random
import re

from .common import Doc, is_correct

MODES = ("irrelevant", "missing", "partial", "outdated")
PERSISTENT = {"missing", "partial", "outdated"}


def parse_spec(s):
    if s == "none":
        return ("none", 0.0)
    mode, _, value = s.partition(":")
    if mode not in MODES or not value:
        raise ValueError(f"Invalid noise specification: {s!r}. Use none or MODE:PROBABILITY.")
    p = float(value)
    if not 0 <= p <= 1:
        raise ValueError(f"Noise probability must be between 0 and 1: {s!r}")
    return mode, p


def spec_label(spec):
    return "none" if spec[0] == "none" else f"{spec[0]}@{spec[1]}"


def corrupt(docs, q, spec, pool, stale_pool, stage, seed=0):
    mode, p = spec
    if mode == "none" or p <= 0:
        return docs
    rng = random.Random(f"{seed}|{q.id}|{mode}|{p}|{'' if mode in PERSISTENT else stage}")
    if rng.random() >= p:
        return docs
    gold = set(q.gold_ids)
    if mode == "irrelevant":
        excluded = gold or {d.id for d in docs}
        alternatives = [d for d in pool if d.id not in excluded]
        return rng.sample(alternatives, min(len(docs), len(alternatives)))
    if mode == "missing":
        return [d for d in docs if d.id not in gold]
    if mode == "partial":
        out = []
        for d in docs:
            if d.id in gold:
                sents = [s for s in re.split(r"(?<=[.!?])\s+", d.text)
                         if not is_correct(s, q.answers)]
                if sents:
                    out.append(Doc(d.id, " ".join(sents)))
            else:
                out.append(d)
        return out
    if mode == "outdated":
        alternatives = [a for a in stale_pool if not is_correct(a, q.answers)]
        stale = rng.choice(alternatives) if alternatives else "[stale value]"
        aliases = "|".join(re.escape(a) for a in sorted(filter(None, q.answers), key=len, reverse=True))
        if not aliases:
            return docs
        pattern = re.compile(r"(?<!\w)(?:" + aliases + r")(?!\w)", re.I)
        out = []
        for d in docs:
            if d.id in gold:
                out.append(Doc(d.id, pattern.sub(lambda _: stale, d.text)))
            else:
                out.append(d)
        return out
    return docs
