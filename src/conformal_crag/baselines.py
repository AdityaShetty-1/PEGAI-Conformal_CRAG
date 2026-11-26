"""Named pipeline configurations: Conformal-CRAG plus the four baselines the
proposal names in its evaluation setup. All five run through the same
`ConformalCRAGRouter` engine (router.py) -- only which actions are enabled
and whether the threshold is conformally calibrated differ.
"""
from __future__ import annotations

from .router import PipelineConfig

CONFORMAL_CRAG = PipelineConfig(
    name="conformal_crag",
    allow_rewrite=True,
    allow_web=True,
    allow_abstain=True,
    use_calibrated_threshold=True,
)

STANDARD_RAG = PipelineConfig(
    name="standard_rag",
    allow_rewrite=False,
    allow_web=False,
    allow_abstain=False,
    use_calibrated_threshold=False,
    fixed_threshold=0.0,  # no grading gate at all: always answer from top-k
)

CRAG_FIXED = PipelineConfig(
    name="crag_fixed_threshold",
    allow_rewrite=True,
    allow_web=True,
    allow_abstain=False,  # CRAG never abstains -- forced answer if repairs fail
    use_calibrated_threshold=False,
    fixed_threshold=0.5,  # hand-set, uncalibrated cutoff
)

SUFFICIENCY_ONLY = PipelineConfig(
    name="sufficiency_only_abstention",
    allow_rewrite=False,  # abstains only -- no correction path
    allow_web=False,
    allow_abstain=True,
    use_calibrated_threshold=False,
    fixed_threshold=0.5,
)

TARG_STYLE_GATE = PipelineConfig(
    name="targ_style_gate",
    allow_rewrite=True,   # decides whether to retrieve again...
    allow_web=False,      # ...but has no repair/fallback path
    allow_abstain=False,  # ...and never abstains
    use_calibrated_threshold=False,
    fixed_threshold=0.5,
)

ALL_BASELINES = [STANDARD_RAG, CRAG_FIXED, SUFFICIENCY_ONLY, TARG_STYLE_GATE]
ALL_CONFIGS = [CONFORMAL_CRAG, *ALL_BASELINES]
