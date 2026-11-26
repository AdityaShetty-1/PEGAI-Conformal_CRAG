# Conformal-CRAG -- shetty's branch (method & pipeline core)

This repo is shared by four people, one branch each (`sam`, `shetty`,
`vinju`, `apeksha`) off a common `main` that holds the full project. This
branch is trimmed to shetty's slice to work on; `git log --all --graph
--oneline` shows how it relates to the others, and `git checkout main`
recovers the complete project at any time. Push this branch to the shared
remote and open a PR into `main` to merge it back in.

## What's on this branch

- `src/conformal_crag/llm.py` -- the extractive answer generator
- `src/conformal_crag/grader.py` -- the sufficiency grader (3 variants)
- `src/conformal_crag/conformal.py` -- conformal risk control threshold
  calibration (the core statistical contribution)
- `src/conformal_crag/router.py` -- the 4-action router engine
- `src/conformal_crag/baselines.py` -- named configs for the 4 literature
  baselines
- `src/conformal_crag/metrics.py` -- evaluation metrics with bootstrap CIs
- `src/conformal_crag/pipeline.py` -- experiment orchestration

The data/retrieval layer, the tests, the experiments, and the docs/slides
live on the other three branches -- see `main` for how it all fits
together.
