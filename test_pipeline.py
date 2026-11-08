import math
import unittest

from ccrag.common import Doc, Question, is_correct
from ccrag.conformal import calibrate_tau
from ccrag.engine import Engine, Stage, decide, standard_rag
from ccrag.noise import corrupt


class RecordingLLM:
    def __init__(self):
        self.tasks = []

    def snapshot(self):
        return len(self.tasks), 0.0

    def complete(self, prompt):
        task = prompt.splitlines()[0].split()[-1]
        self.tasks.append(task)
        if task == "REWRITE":
            return "repaired query"
        if task == "SUFFICIENCY":
            return "95" if "Paris is the capital" in prompt else "15"
        return "Paris"


class StaticRetriever:
    def retrieve(self, query, k):
        if query == "repaired query":
            return [Doc("p1", "Paris is the capital of France.")]
        return [Doc("p2", "Europe is a continent.")]


class UnavailableWeb:
    def snapshot(self):
        return 0, 0.0

    def search(self, query, k):
        raise AssertionError("Closed mode attempted a web search")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM()
        self.engine = Engine(self.llm, StaticRetriever(), UnavailableWeb(), [], [])
        self.question = Question("q1", "What is the capital of France?", ["Paris"], ["p1"])

    def test_live_and_replay_agree_without_generating_on_rejected_stages(self):
        live = decide(self.engine.stages(self.question, tau=0.8, open_mode=False), 0.8)
        self.assertEqual(self.llm.tasks, ["SUFFICIENCY", "REWRITE", "SUFFICIENCY", "ANSWER"])
        replay = decide(self.engine.stages(self.question, open_mode=False), 0.8)
        self.assertEqual((live.route, live.answer, live.calls), ("rewrite", "Paris", 4))
        self.assertEqual((live.route, live.answer, live.calls), (replay.route, replay.answer, replay.calls))

    def test_abstention_and_standard_rag_costs(self):
        stages = self.engine.stages(self.question, open_mode=False)
        self.assertEqual(standard_rag(stages[0]).calls, 1)
        before = len(self.llm.tasks)
        self.assertEqual(self.engine.stages(self.question, tau=math.inf), [])
        self.assertEqual(decide(stages, math.inf).calls, 0)
        self.assertEqual(len(self.llm.tasks), before)

    def test_calibration_can_certify_after_an_empty_high_threshold(self):
        stages = [[Stage("direct", 0.85, "Paris", True, 0, 0, 0.0)]] * 36
        tau, table = calibrate_tau(stages)
        self.assertEqual(tau, 0.8)
        self.assertFalse(table[0]["certified"])
        self.assertIsNone(calibrate_tau(stages[:8])[0])

    def test_empty_context_and_invalid_grade_never_pass(self):
        self.assertEqual(self.engine.sufficiency(self.question, []), -1)
        self.assertEqual(self.engine._parse_score("1/10"), -1)

    def test_noise_is_defined_for_small_corpora(self):
        only = [Doc("p1", "Paris is the capital of France.")]
        for mode in ("irrelevant", "missing", "partial"):
            self.assertEqual(corrupt(only, self.question, (mode, 1.0), only, [], "direct"), [])
        stale = corrupt(only, self.question, ("outdated", 1.0), only, ["Paris"], "direct")
        self.assertNotIn("Paris", stale[0].text)

    def test_answer_match_requires_complete_span(self):
        self.assertTrue(is_correct("The answer is New York City.", ["New York City"]))
        self.assertFalse(is_correct("York", ["New York City"]))
        self.assertFalse(is_correct("comparison", ["Paris"]))


if __name__ == "__main__":
    unittest.main()
