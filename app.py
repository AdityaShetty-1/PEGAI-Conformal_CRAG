"""Interactive Conformal CRAG demo. Start with python -m streamlit run app.py."""
import json
import math
import os
from pathlib import Path
from time import perf_counter

import streamlit as st

from ccrag.common import Question
from ccrag.engine import Engine, decide
from ccrag.factory import build
from ccrag.noise import spec_label

st.set_page_config(page_title="Conformal CRAG", layout="wide")
st.title("Conformal Corrective RAG")
st.caption("Inspect the retrieved evidence, sufficiency score, and routing decision at each stage.")

mock = os.getenv("USE_MOCK") == "1"
dataset = os.getenv("DATASET", "toy" if mock else "mini_wiki")
with st.spinner("Loading documents and retrieval index…"):
    if ("build", dataset, mock) not in st.session_state:
        try:
            st.session_state[("build", dataset, mock)] = build(dataset, mock)
        except Exception as exc:
            st.error(f"Could not load the demo ({type(exc).__name__}). Check the dataset and model setup in the README.")
            st.stop()
env = st.session_state[("build", dataset, mock)]
if mock:
    st.info("Mock mode uses a deterministic stand-in for the language model and web search.")
else:
    st.caption(f"Dataset · {dataset}    |    Model · {env.profile['generator']}")

path = Path(os.getenv("CC_RESULTS", "results")) / "taus.json"
try:
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
except (OSError, ValueError):
    saved = {}
compatible = saved.get("calibration_version") == 2 and saved.get("profile") == env.profile
k = saved["k"] if compatible else 4

with st.sidebar:
    modes = ["closed", "open"] if env.web else ["closed"]
    mode = st.radio("Retrieval mode", modes, help="Closed mode never calls web search.")
    noise_labels = {
        "Clean retrieval": ("none", 0.0),
        "Irrelevant passages (30%)": ("irrelevant", 0.3),
        "Irrelevant passages (100%)": ("irrelevant", 1.0),
        "Missing evidence (30%)": ("missing", 0.3),
        "Partial evidence (30%)": ("partial", 0.3),
        "Outdated evidence (30%)": ("outdated", 0.3),
    }
    noise = noise_labels[st.selectbox("Retrieval condition", list(noise_labels))]
    if noise[0] == "none":
        thresholds, key = saved.get("taus", {}), f"{mode}_full"
    else:
        thresholds, key = saved.get("noise_taus", {}).get(spec_label(noise), {}), mode
    calibrated = compatible and key in thresholds
    manual = not calibrated or st.checkbox("Use a manual threshold")
    if manual:
        tau = st.slider("Manual threshold", 0.0, 1.0, 0.7, 0.01)
        st.caption("Manual thresholds have no risk certification.")
    else:
        selected = thresholds[key]
        tau = math.inf if selected is None else selected
        if selected is None:
            st.warning("Calibration certified no threshold. This rule always abstains.")
        else:
            st.write(f"Calibrated threshold **{tau:.2f}**")
            st.caption(f"{saved['risk_def'].title()} risk target {saved['alpha']:.0%} at "
                       f"{1 - saved['delta']:.0%} confidence on matching calibration data.")

test_ids = set(saved.get("test_ids", [])) if compatible else set()
examples = sorted(env.questions, key=lambda q: q.id not in test_ids)[:15]
selected = st.selectbox("Sample question", examples, format_func=lambda q: q.text)
custom = st.text_input("Or enter your own question").strip()

if st.button("Run", type="primary"):
    known = next((q for q in env.questions if q.text.casefold() == custom.casefold()), None) if custom else selected
    if not custom and known is None:
        st.warning("Enter a question to run the demo.")
        st.stop()
    if known is None and noise[0] in {"missing", "partial", "outdated"}:
        st.warning("Choose a sample question to simulate this type of corruption. It requires labeled evidence.")
        st.stop()
    q = known or Question(custom.casefold(), custom, [])
    engine = Engine(env.llm, env.retriever, env.web, env.corpus, env.stale_pool,
                    k=k, seed=saved["seed"] if compatible else 0)
    engine.noise = noise
    start = perf_counter()
    try:
        with st.spinner("Running the router…"):
            stages = engine.stages(q, tau=tau, open_mode=mode == "open")
            result = decide(stages, tau)
    except Exception as exc:
        st.error(f"The router could not complete this request ({type(exc).__name__}). "
                 "Check the model endpoint or web-search connection and retry.")
        st.stop()
    if result.answered:
        st.success(f"{result.answer}  ·  {result.route.title()}")
    else:
        st.warning("I don't know. The router abstained.")
    st.caption(f"{result.calls} model calls · {result.web_calls} search calls · {perf_counter() - start:.2f} s elapsed")
    if stages:
        st.subheader("Route trace")
        st.dataframe([{"Stage": s.name, "Score": round(s.score, 2) if s.score >= 0 else None,
                       "Passed": s.score >= tau, "Evidence passages": len(s.docs)} for s in stages],
                     width="stretch", hide_index=True)
        for stage in stages:
            with st.expander(f"Evidence · {stage.name}", expanded=result.route == stage.name):
                if not stage.docs:
                    st.caption("No passages returned.")
                for doc in stage.docs:
                    st.caption(doc.id)
                    st.text(doc.text[:1200])
