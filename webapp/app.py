"""A small fullstack demo: a real Flask backend that runs the actual
conformal_crag pipeline (the same retriever, noise injector, grader, reader
and router classes the test suite and experiments use -- nothing here is a
JavaScript reimplementation) against a question the viewer types in, with a
noise level and method they choose, and returns the step-by-step routing
trace for the frontend to render.

Run with:
    pip install -r ../requirements.txt flask
    pip install -e ..
    python app.py
Then open http://localhost:5000
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template_string, request  # noqa: E402

from conformal_crag.baselines import ALL_CONFIGS  # noqa: E402
from conformal_crag.conformal import calibrate_threshold  # noqa: E402
from conformal_crag.data import load_dataset, split_calibration_test  # noqa: E402
from conformal_crag.grader import SufficiencyGrader  # noqa: E402
from conformal_crag.llm import ExtractiveReader, STOPWORDS, is_correct, tokenize  # noqa: E402
from conformal_crag.noise import NoiseInjector  # noqa: E402
from conformal_crag.pipeline import stage_trace_for_calibration  # noqa: E402
from conformal_crag.retriever import Retriever  # noqa: E402
from conformal_crag.router import PipelineConfig, _merge_docs  # noqa: E402
from conformal_crag.web_fallback import WebFallback  # noqa: E402

app = Flask(__name__)

ALPHA = 0.25
DELTA = 0.1
CALIBRATION_NOISE = 0.3
SEED = 0

# --- built once at startup: the same objects pipeline.py uses ------------
DATASET = load_dataset(seed=SEED)
RETRIEVER = Retriever(DATASET.documents)
CLEAN_RETRIEVER = Retriever(DATASET.documents)
WEB_FALLBACK = WebFallback(CLEAN_RETRIEVER)
GRADER = SufficiencyGrader(variant="autorater_sim", seed=SEED)
READER = ExtractiveReader()

CONFIGS = {c.name: c for c in ALL_CONFIGS}
QUESTIONS_BY_ID = {q.qid: q for q in DATASET.questions}

CALIB_QS, TEST_QS = split_calibration_test(DATASET, calib_frac=0.5, seed=SEED)


def _calibrate() -> float:
    conformal_cfg = CONFIGS["conformal_crag"]
    router_for_calib = _RouterLike()
    traces = [
        stage_trace_for_calibration(router_for_calib, q, conformal_cfg, CALIBRATION_NOISE, "mixed")
        for q in CALIB_QS
    ]
    return calibrate_threshold(traces, alpha=ALPHA, delta=DELTA)


class _RouterLike:
    """Just enough surface for stage_trace_for_calibration/pipeline helpers,
    reusing the exact same global objects the live /api/run endpoint uses."""
    def __init__(self):
        self.retriever = RETRIEVER
        self.grader = GRADER
        self.reader = READER
        self.noise_injector = NoiseInjector(DATASET.documents, seed=SEED)
        self.web_fallback = WEB_FALLBACK

    def _retrieve(self, query, k, noise_level, noise_mode):
        retrieved = self.retriever.retrieve(query, k)
        if noise_level > 0:
            retrieved = self.noise_injector.corrupt(retrieved, noise_level, noise_mode)
        return retrieved


CALIBRATED_THRESHOLD = _calibrate()


def doc_view(doc):
    return {"doc_id": doc.doc_id, "text": doc.text}


def run_pipeline(question_text: str, config: PipelineConfig, noise_level: float, noise_mode: str,
                  threshold: float, gold_aliases: list[str] | None, seed: int) -> dict:
    """Mirrors router.run()'s logic stage by stage, but returns the full
    intermediate state (retrieved docs, scores, decisions) for the UI,
    using the exact same package objects (retriever/grader/reader/noise
    injector/web fallback) the CLI pipeline and tests use."""
    noise_injector = NoiseInjector(DATASET.documents, seed=seed)
    stages = []

    def retrieve(q, k=3):
        docs = RETRIEVER.retrieve(q, k)
        if noise_level > 0:
            docs = noise_injector.corrupt(docs, noise_level, noise_mode)
        return docs

    query = question_text
    retrieved = retrieve(query)
    score = GRADER.score(query, retrieved)
    answer, _ = READER.answer(query, retrieved)
    stages.append({
        "stage": "generate", "query": query, "docs": [doc_view(d) for d in retrieved],
        "score": score, "answer": answer, "passed": score >= threshold,
    })

    final_answer, final_action, final_score = None, None, score
    if score >= threshold:
        final_answer, final_action, final_score = answer, "generate", score
    elif config.allow_rewrite:
        exclude = set(tokenize(query)) - STOPWORDS
        entities = READER.extract_rewrite_entities(retrieved, exclude)
        if entities:
            new_query = query + " " + entities[0]
            retrieved2 = retrieve(new_query)
            merged = _merge_docs(retrieved, retrieved2)
            score2 = GRADER.score(new_query, merged)
            answer2, _ = READER.answer(new_query, merged)
            stages.append({
                "stage": "rewrite", "query": new_query, "docs": [doc_view(d) for d in merged],
                "score": score2, "answer": answer2, "passed": score2 >= threshold,
                "new_entity": entities[0],
            })
            retrieved, score, answer = merged, score2, answer2
            if score2 >= threshold:
                final_answer, final_action, final_score = answer2, "generate", score2
        else:
            stages.append({"stage": "rewrite", "query": query, "docs": [], "score": score,
                            "answer": None, "passed": False, "new_entity": None,
                            "note": "no new entity found in context to rewrite with"})

    if final_action is None and config.allow_web:
        web_docs = WEB_FALLBACK.search(query, k=2)
        merged = _merge_docs(retrieved, web_docs)
        score3 = GRADER.score(query, merged)
        answer3, _ = READER.answer(query, merged)
        stages.append({
            "stage": "web_fallback", "query": query, "docs": [doc_view(d) for d in merged],
            "score": score3, "answer": answer3, "passed": score3 >= threshold,
        })
        if score3 >= threshold:
            final_answer, final_action, final_score = answer3, "generate", score3
        else:
            retrieved, score, answer = merged, score3, answer3

    if final_action is None:
        if config.allow_abstain:
            final_action = "abstain"
            final_answer = None
            final_score = score
        else:
            final_action = "generate_forced"
            final_answer = answer
            final_score = score

    correct = None
    if gold_aliases is not None and final_answer is not None:
        correct = is_correct(final_answer, gold_aliases)

    return {
        "stages": stages,
        "final_action": final_action,
        "final_answer": final_answer,
        "final_score": final_score,
        "threshold": threshold,
        "correct": correct,
        "abstained": final_answer is None,
    }


INDEX_HTML = r"""
<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conformal-CRAG -- Live Demo</title>
<style>
  :root { color-scheme: light; --bg:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ed; --ink:#0b0b0b; --ink-2:#52514e; --ink-muted:#898781; --border:rgba(11,11,11,0.10); --accent:#2a78d6; --good:#0ca30c; --critical:#d03b3b; --warn:#eda100; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; padding:24px 16px 80px; }
  .wrap { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 28px; margin: 0 0 6px; }
  .sub { color: var(--ink-2); font-size: 14.5px; margin: 0 0 28px; max-width: 640px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; }
  label { display:block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-muted); margin-bottom: 6px; }
  textarea, select, input[type=range] { width: 100%; font: inherit; }
  textarea { background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; color: var(--ink); resize: vertical; min-height: 54px; }
  select { background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; color: var(--ink); }
  .row { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 14px; }
  .row > div { flex: 1 1 180px; min-width: 0; }
  .noise-val { font-weight: 600; }
  button.run { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 11px 20px; font: inherit; font-weight: 600; cursor: pointer; margin-top: 16px; }
  button.run:disabled { opacity: 0.5; cursor: default; }
  .examples { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
  .ex-btn { font-size: 12px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px; padding: 5px 10px; cursor: pointer; color: var(--ink-2); }
  .ex-btn:hover { background: var(--border); }

  .stage { border-left: 3px solid var(--border); padding: 4px 0 4px 14px; margin-bottom: 14px; }
  .stage.passed { border-left-color: var(--good); }
  .stage-head { display:flex; align-items:center; gap:8px; font-weight:600; font-size:13.5px; margin-bottom:6px; }
  .badge { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; padding: 2px 8px; border-radius: 999px; background: var(--surface-2); color: var(--ink-muted); }
  .badge.pass { background: color-mix(in srgb, var(--good) 18%, var(--surface)); color: var(--good); }
  .score { font-variant-numeric: tabular-nums; color: var(--ink-2); font-size: 13px; }
  .doc { background: var(--surface-2); border-radius: 8px; padding: 8px 10px; font-size: 12.5px; color: var(--ink-2); margin-bottom: 6px; }
  .doc .id { font-weight: 600; color: var(--ink); margin-right: 6px; }
  .answer-line { font-size: 13.5px; margin-top: 4px; }
  .final { border-radius: 10px; padding: 16px 18px; margin-top: 6px; }
  .final.answered { background: color-mix(in srgb, var(--good) 10%, var(--surface)); border: 1px solid color-mix(in srgb, var(--good) 35%, var(--border)); }
  .final.abstained { background: color-mix(in srgb, var(--warn) 10%, var(--surface)); border: 1px solid color-mix(in srgb, var(--warn) 35%, var(--border)); }
  .final .tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 6px; }
  .final.answered .tag { color: var(--good); }
  .final.abstained .tag { color: var(--warn); }
  .correct-pill { font-size: 11px; padding: 2px 8px; border-radius: 999px; margin-left: 8px; }
  .correct-pill.yes { background: color-mix(in srgb, var(--good) 20%, var(--surface)); color: var(--good); }
  .correct-pill.no { background: color-mix(in srgb, var(--critical) 20%, var(--surface)); color: var(--critical); }
  .meta-line { font-size: 12.5px; color: var(--ink-muted); margin-top: 10px; }
  .loading { color: var(--ink-muted); font-size: 13px; }
  footer { margin-top: 40px; color: var(--ink-muted); font-size: 12px; }
  code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
</style></head>
<body>
<div class="wrap">
  <h1>Conformal-CRAG &mdash; Live Demo</h1>
  <p class="sub">A real Flask backend running the actual <code>conformal_crag</code> package &mdash; the same retriever, sufficiency grader, noise injector and router the test suite uses &mdash; against whatever question and noise level you set below.</p>

  <div class="card">
    <label>Question</label>
    <textarea id="question" placeholder="Type a question, or pick an example below">What is the capital of Mongolia?</textarea>
    <div class="examples" id="examples"></div>

    <div class="row">
      <div>
        <label>Method</label>
        <select id="method"></select>
      </div>
      <div>
        <label>Noise mode</label>
        <select id="noise_mode">
          <option value="mixed">mixed</option>
          <option value="irrelevant">irrelevant</option>
          <option value="outdated">outdated</option>
          <option value="partial">partial</option>
        </select>
      </div>
      <div>
        <label>Noise level: <span class="noise-val" id="noise-val">0.30</span></label>
        <input type="range" id="noise" min="0" max="1" step="0.05" value="0.3">
      </div>
    </div>
    <button class="run" id="run">Run pipeline</button>
    <div class="meta-line" id="threshold-line"></div>
  </div>

  <div id="result"></div>

  <footer>Calibrated once at startup on noise=<span id="calib-noise"></span>, &alpha;=<span id="calib-alpha"></span>, &delta;=<span id="calib-delta"></span> &mdash; see <code>webapp/app.py</code>.</footer>
</div>
<script>
  var METHODS = {{ methods | tojson }};
  var EXAMPLES = {{ examples | tojson }};
  var THRESHOLD = {{ threshold }};
  document.getElementById("threshold-line").textContent = "Calibrated threshold τ = " + THRESHOLD.toFixed(3) + " (Conformal-CRAG only; baselines use their own fixed/none threshold)";
  document.getElementById("calib-noise").textContent = "{{ calib_noise }}";
  document.getElementById("calib-alpha").textContent = "{{ alpha }}";
  document.getElementById("calib-delta").textContent = "{{ delta }}";

  var methodSel = document.getElementById("method");
  METHODS.forEach(function (m) {
    var o = document.createElement("option"); o.value = m.name; o.textContent = m.label;
    methodSel.appendChild(o);
  });

  var exWrap = document.getElementById("examples");
  EXAMPLES.forEach(function (ex) {
    var b = document.createElement("button");
    b.className = "ex-btn"; b.textContent = "[" + ex.source + "] " + ex.question;
    b.addEventListener("click", function () { document.getElementById("question").value = ex.question; });
    exWrap.appendChild(b);
  });

  var noiseInput = document.getElementById("noise");
  var noiseVal = document.getElementById("noise-val");
  noiseInput.addEventListener("input", function () { noiseVal.textContent = parseFloat(noiseInput.value).toFixed(2); });

  function escapeHtml(s) {
    return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderResult(data) {
    var html = "";
    data.stages.forEach(function (st) {
      html += '<div class="card stage ' + (st.passed ? "passed" : "") + '">';
      html += '<div class="stage-head"><span>' + st.stage + '</span>';
      html += '<span class="badge ' + (st.passed ? "pass" : "") + '">' + (st.passed ? "cleared τ" : "below τ") + '</span>';
      html += '<span class="score">score = ' + st.score.toFixed(3) + '</span></div>';
      if (st.new_entity) html += '<div class="answer-line">rewrote query with entity: <code>' + escapeHtml(st.new_entity) + '</code></div>';
      if (st.note) html += '<div class="answer-line" style="color:var(--ink-muted)">' + escapeHtml(st.note) + '</div>';
      (st.docs || []).forEach(function (d) {
        html += '<div class="doc"><span class="id">' + escapeHtml(d.doc_id) + '</span>' + escapeHtml(d.text) + '</div>';
      });
      if (st.answer) html += '<div class="answer-line">extracted: &ldquo;' + escapeHtml(st.answer) + '&rdquo;</div>';
      html += '</div>';
    });

    var finalClass = data.abstained ? "abstained" : "answered";
    html += '<div class="card final ' + finalClass + '">';
    html += '<div class="tag">' + data.final_action + '</div>';
    if (data.abstained) {
      html += '<div>No stage cleared τ = ' + data.threshold.toFixed(3) + ' (score stayed at ' + data.final_score.toFixed(3) + '). The system abstains rather than guess.</div>';
    } else {
      html += '<div>&ldquo;' + escapeHtml(data.final_answer) + '&rdquo;';
      if (data.correct !== null) {
        html += '<span class="correct-pill ' + (data.correct ? "yes" : "no") + '">' + (data.correct ? "correct" : "incorrect") + '</span>';
      }
      html += '</div>';
    }
    html += '</div>';
    document.getElementById("result").innerHTML = html;
  }

  document.getElementById("run").addEventListener("click", function () {
    var btn = this;
    btn.disabled = true;
    document.getElementById("result").innerHTML = '<p class="loading">Running the real pipeline&hellip;</p>';
    fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: document.getElementById("question").value,
        method: methodSel.value,
        noise_level: parseFloat(noiseInput.value),
        noise_mode: document.getElementById("noise_mode").value
      })
    }).then(function (r) { return r.json(); }).then(function (data) {
      renderResult(data);
      btn.disabled = false;
    }).catch(function (err) {
      document.getElementById("result").innerHTML = '<p style="color:var(--critical)">Error: ' + escapeHtml(String(err)) + '</p>';
      btn.disabled = false;
    });
  });
</script>
</body></html>
"""


@app.route("/")
def index():
    methods = [
        {"name": "conformal_crag", "label": "Conformal-CRAG (calibrated)"},
        {"name": "standard_rag", "label": "Standard RAG"},
        {"name": "crag_fixed_threshold", "label": "CRAG (fixed threshold)"},
        {"name": "sufficiency_only_abstention", "label": "Sufficiency-only abstention"},
        {"name": "targ_style_gate", "label": "TARG-style gate"},
    ]
    examples = [
        {"question": q.question, "source": q.source}
        for q in sorted(DATASET.questions, key=lambda q: q.qid)[:3]
    ] + [
        q2 for src in ("hotpotqa", "musique", "own")
        for q2 in [{"question": next(q.question for q in DATASET.questions if q.source == src), "source": src}]
    ]
    return render_template_string(
        INDEX_HTML, methods=methods, examples=examples, threshold=CALIBRATED_THRESHOLD,
        calib_noise=CALIBRATION_NOISE, alpha=ALPHA, delta=DELTA,
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    question_text = (body.get("question") or "").strip()
    method = body.get("method", "conformal_crag")
    noise_level = float(body.get("noise_level", 0.3))
    noise_mode = body.get("noise_mode", "mixed")

    if not question_text:
        return jsonify({"error": "question is required"}), 400
    config = CONFIGS.get(method)
    if config is None:
        return jsonify({"error": f"unknown method {method}"}), 400

    gold_q = next((q for q in DATASET.questions if q.question.strip().lower() == question_text.lower()), None)
    threshold = CALIBRATED_THRESHOLD if config.use_calibrated_threshold else config.fixed_threshold

    result = run_pipeline(
        question_text, config, noise_level, noise_mode, threshold,
        gold_q.answer_aliases if gold_q else None, seed=SEED,
    )
    return jsonify(result)


if __name__ == "__main__":
    print(f"Calibrated threshold (alpha={ALPHA}, noise={CALIBRATION_NOISE}): {CALIBRATED_THRESHOLD:.3f}")
    app.run(host="0.0.0.0", port=5000, debug=False)
