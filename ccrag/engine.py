import re
import time
from dataclasses import dataclass, field

from . import prompts
from .common import Doc, is_correct
from .noise import corrupt

ALL_STAGES = ("direct", "rewrite", "web")


@dataclass
class Stage:
    name: str
    score: float      # sufficiency score in [0, 1]
    answer: str
    correct: bool
    calls: int        # LLM calls spent in this stage
    web_calls: int
    latency: float
    docs: list = field(default_factory=list)
    generation_calls: int = 0
    generation_latency: float = 0.0
    retrieval_latency: float = 0.0


@dataclass
class Outcome:
    answered: bool
    answer: str
    route: str
    correct: bool
    calls: int
    web_calls: int
    latency: float


def decide(stages, tau):
    """Pure routing rule: answer at the first stage whose sufficiency score >= tau, else abstain.
    Used both online (demo) and offline (calibration / simulation), so they can never disagree."""
    if tau == float("inf"):
        return Outcome(False, "", "abstain", False, 0, 0, 0.0)
    calls = web = lat = 0
    for st in stages:
        calls, web, lat = calls + st.calls, web + st.web_calls, lat + st.latency
        if st.score >= tau:
            return Outcome(True, st.answer, st.name, st.correct, calls + st.generation_calls,
                           web, lat + st.generation_latency)
    return Outcome(False, "", "abstain", False, calls, web, lat)


def standard_rag(stage):
    """Replay only retrieval and generation from the first stage."""
    return Outcome(True, stage.answer, stage.name, stage.correct, stage.generation_calls,
                   0, stage.retrieval_latency + stage.generation_latency)


class Engine:
    def __init__(self, llm, retriever, web, corpus, stale_pool, k=4, seed=0):
        self.llm, self.retriever, self.web = llm, retriever, web
        self.pool, self.stale_pool, self.k, self.seed = corpus, stale_pool, k, seed
        self.noise = ("none", 0.0)

    # ---- LLM building blocks -------------------------------------------------
    @staticmethod
    def _ctx(docs):
        return "\n".join(f"[{i + 1}] {d.text[:1200]}" for i, d in enumerate(docs)) or "[no documents]"

    def generate(self, q, docs):
        return self.llm.complete(prompts.ANSWER.format(context=self._ctx(docs), question=q.text)).split("\n")[0].strip()

    @staticmethod
    def _parse_score(text):
        m = re.fullmatch(r"(\d{1,3})\s*(?:%|/\s*100)?", text.strip())
        return int(m[1]) / 100 if m and int(m[1]) <= 100 else -1.0

    def sufficiency(self, q, docs):
        if not docs:
            return -1.0  # No evidence can pass a sufficiency threshold in [0, 1].
        return self._parse_score(self.llm.complete(prompts.SUFFICIENCY.format(context=self._ctx(docs), question=q.text)))

    def chunk_score(self, q, doc):
        text = self.llm.complete(prompts.CHUNK.format(context=self._ctx([doc]), question=q.text))
        return self._parse_score(text.partition("\n")[0])

    def rewrite(self, qtext):
        return self.llm.complete(prompts.REWRITE.format(question=qtext)).split("\n")[0].strip().strip('"')

    def _local(self, q, query, stage):
        docs = self.retriever.retrieve(query, self.k)
        return corrupt(docs, q, self.noise, self.pool, self.stale_pool, stage, self.seed)

    def _timed_local(self, q, query, stage):
        start = time.perf_counter()
        docs = self._local(q, query, stage)
        return docs, time.perf_counter() - start

    def _snap(self):
        lc, ll = self.llm.snapshot()
        wc, wl = self.web.snapshot() if self.web else (0, 0.0)
        return lc, wc, ll + wl

    # ---- Proposed pipeline: staged evidence gathering ---------------------------
    def stages(self, q, tau=None, names=ALL_STAGES, open_mode=True):
        """Compute stage records. tau=None -> compute every stage (for calibration/simulation);
        tau given -> generate only at the first sufficient stage (online behaviour)."""
        if tau == float("inf"):
            return []  # An uncertified calibrated rule abstains without calling providers.
        out, rq = [], None
        for name in names:
            if name == "web" and not (open_mode and self.web):
                continue
            c0, w0, l0 = self._snap()
            retrieval_s = 0.0
            if name == "direct":
                docs, retrieval_s = self._timed_local(q, q.text, "direct")
            elif name == "rewrite":
                rq = self.rewrite(q.text)
                docs, retrieval_s = self._timed_local(q, rq or q.text, "rewrite")
            else:
                docs = self.web.search(rq or q.text, self.k)
            s = self.sufficiency(q, docs)
            c1, w1, l1 = self._snap()
            passes = tau is not None and s >= tau
            a = self.generate(q, docs) if passes or tau is None else ""
            c2, _, l2 = self._snap()
            out.append(Stage(name, s, a, is_correct(a, q.answers), c1 - c0, w1 - w0,
                             l1 - l0 + retrieval_s, docs, c2 - c1, l2 - l1, retrieval_s))
            if passes:
                break
        return out

    # ---- CRAG baseline (Yan et al., 2024) with hand-set thresholds ---------------
    def crag(self, q, open_mode=True, up=0.7, low=0.3):
        c0, w0, l0 = self._snap()
        docs, retrieval_s = self._timed_local(q, q.text, "direct")
        scores = [self.chunk_score(q, d) for d in docs]
        mx = max(scores) if scores else 0.0
        good = [d for d, s in zip(docs, scores) if s >= low]
        if mx >= up:
            route, ctx = "crag:correct", good
        else:
            rq = self.rewrite(q.text) or q.text
            if open_mode and self.web:
                extra = self.web.search(rq, self.k)
            else:
                extra, extra_s = self._timed_local(q, rq, "rewrite")
                retrieval_s += extra_s
            route, ctx = ("crag:incorrect", extra) if mx < low else ("crag:ambiguous", good + extra)
        ans = self.generate(q, ctx)
        c1, w1, l1 = self._snap()
        return Outcome(True, ans, route, is_correct(ans, q.answers), c1 - c0, w1 - w0, l1 - l0 + retrieval_s)
