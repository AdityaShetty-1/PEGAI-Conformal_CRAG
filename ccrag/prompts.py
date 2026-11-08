# The "### TASK" tag is only used by the mock LLM to route prompts; real LLMs ignore it.

ANSWER = """### TASK: ANSWER
Answer the question using the context below. Reply with the shortest possible answer (a few words) and no explanation. If the context does not contain the answer, still give your best guess.

Context:
{context}

Question: {question}
Answer:"""

SUFFICIENCY = """### TASK: SUFFICIENCY
You are a strict evaluator. Decide whether the context contains enough information to answer the question correctly and completely. Reply with one integer from 0 to 100 (0 = context is irrelevant or does not contain the answer, 100 = context clearly contains the answer). Reply with the number only.

Context:
{context}

Question: {question}
Score:"""

CHUNK = """### TASK: CHUNK
Classify how much evidence this passage supplies for the question. Treat the passage as the source of truth, including unfamiliar names.
Return exactly one number:
100 = the passage explicitly states the requested fact about the question's subject.
50 = the passage supplies some relevant evidence but does not state the requested fact.
0 = the passage supplies no relevant evidence.

Context:
{context}

Question: {question}
Score:"""

REWRITE = """### TASK: REWRITE
Rewrite the question as a short, self-contained web search query that would find a page containing the answer. Reply with the query only.

Question: {question}
Query:"""
