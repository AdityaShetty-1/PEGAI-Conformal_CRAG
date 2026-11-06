import json
import os
import random
import re

from .common import Doc, Question, normalize


def attach_gold(corpus, questions):
    """Gold passage = passage that contains an answer string. Drops questions with no gold
    (and trivial yes/no or very short answers, which match everything)."""
    norm_corpus = [(d.id, normalize(d.text)) for d in corpus]
    unique = {}
    for q in questions:
        key = " ".join(q.text.casefold().split())
        if key in unique:
            existing = unique[key]
            existing.answers = list(dict.fromkeys(existing.answers + q.answers))
        else:
            unique[key] = q
    kept = []
    for q in unique.values():
        answers = [normalize(a) for a in q.answers if len(normalize(a)) >= 3 and normalize(a) not in ("yes", "no")]
        if not answers:
            continue
        q.gold_ids = [i for i, t in norm_corpus if any(f" {a} " in f" {t} " for a in answers)]
        if q.gold_ids:
            kept.append(q)
    return kept


def load_mini_wikipedia():
    from datasets import load_dataset

    corp = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    cs, qs = corp["passages"], qa["test"]
    corpus = [Doc(id=str(r["id"]), text=r["passage"]) for r in cs]
    questions = [Question(id=str(r["id"]), text=r["question"], answers=[str(r["answer"])]) for r in qs]
    return corpus, attach_gold(corpus, questions)


def load_folder(path, chunk_chars=800):
    """Your own document set: <path>/*.txt|*.md plus <path>/questions.jsonl
    with lines like {"id": "1", "question": "...", "answers": ["..."]}."""
    corpus = []
    for fn in sorted(os.listdir(path)):
        if fn.endswith((".txt", ".md")):
            text = open(os.path.join(path, fn), encoding="utf-8").read()
            paras, buf = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()], ""
            chunks = []
            for p in paras:
                if len(buf) + len(p) > chunk_chars and buf:
                    chunks.append(buf)
                    buf = ""
                buf += (" " if buf else "") + p
            if buf:
                chunks.append(buf)
            corpus += [Doc(id=f"{fn}#{i}", text=c) for i, c in enumerate(chunks)]
    questions = []
    with open(os.path.join(path, "questions.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            questions.append(Question(id=str(r["id"]), text=r["question"], answers=r["answers"]))
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Each question in questions.jsonl must have a unique ID.")
    return corpus, attach_gold(corpus, questions)


def toy_dataset(n=90, seed=0):
    """Synthetic facts for smoke-testing the pipeline offline."""
    rng = random.Random(seed)
    syl = "zor van qui mel dra tek lor bax nim sul ker phi".split()
    used = set()

    def uname():
        while True:
            w = (rng.choice(syl) + rng.choice(syl) + rng.choice(syl)).capitalize()
            if w not in used:
                used.add(w)
                return w

    corpus, questions = [], []
    for i in range(n):
        c, cap, cur = uname(), uname(), uname()
        region, thing = rng.choice(["north", "south", "east", "west"]), rng.choice(["honey", "glass", "silk", "tea"])
        corpus.append(Doc(id=f"p{i}", text=f"{c} is a country in the {region} of the continent. The capital of {c} is {cap}. "
                                           f"The official currency of {c} is the {cur}. {c} is famous for its {thing}."))
        if i % 2 == 0:
            questions.append(Question(id=f"q{i}", text=f"What is the capital of {c}?", answers=[cap]))
        else:
            questions.append(Question(id=f"q{i}", text=f"What is the official currency of {c}?", answers=[cur]))
    return corpus, attach_gold(corpus, questions)


def load_dataset_by_name(name):
    if name == "toy":
        return toy_dataset()
    if name == "mini_wiki":
        return load_mini_wikipedia()
    return load_folder(name)  # any other value is treated as a folder path
