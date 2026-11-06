# PEGAI-Conformal_CRAG

Conformal Corrective RAG is a retrieval-augmented question-answering system that decides how much to trust its own retrieved evidence. Standard RAG answers confidently even when retrieval fails, and Corrective RAG (CRAG) improves on this by rewriting queries or searching the web when documents look poor, but it relies on hand-set thresholds, never abstains, and gives no guarantee on its error rate. This project adds a calibrated router that grades the sufficiency of the evidence at each stage (direct retrieval, query rewrite and re-retrieval, optional web fallback) and answers only when the score passes a threshold chosen on a held-out set with a binomial-test procedure, so the error rate of answered queries is bounded by a target alpha with high probability. If no stage produces sufficient evidence, the system abstains with “I don’t know”. We evaluate it against standard RAG, CRAG and abstention-only baselines under controlled retrieval noise (irrelevant, missing, partial and outdated documents), measuring accuracy, hallucination rate, abstention rate, coverage, latency and web calls.

Aditya Shetty - PES1UG24AM018

Aryan Vinju - PES1UG24AM056

Apeksha Ashok - PES1UG24AM048

Samarth Maladkar - PES1UG24AM396
