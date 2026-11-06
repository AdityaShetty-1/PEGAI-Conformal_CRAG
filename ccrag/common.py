import re
import string
from dataclasses import dataclass, field


@dataclass
class Doc:
    id: str
    text: str
    source: str = "local"  # local | web


@dataclass
class Question:
    id: str
    text: str
    answers: list
    gold_ids: list = field(default_factory=list)


_ART = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(_ART.sub(" ", s).split())


def is_correct(pred: str, answers: list) -> bool:
    """Match a complete normalized gold answer, allowing surrounding words."""
    p = normalize(pred)
    if not p:
        return False
    for a in answers:
        a = normalize(a)
        if a and f" {a} " in f" {p} ":
            return True
    return False
