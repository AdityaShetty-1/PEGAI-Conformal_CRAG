# Conformal Corrective RAG

An experimental question-answering system for UE24AM343AB5, Prompt Engineering and Optimization for Generative AI. It grades retrieved evidence, tries a query rewrite when the evidence is weak, optionally searches the web, and abstains when no stage clears the threshold.

## Run the demo on Windows

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2
ollama pull all-minilm
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Use Python 3.10 or newer and start Ollama before launching the app. Skip setup steps for software and models you already have. The sample settings use the built-in synthetic corpus, the local Llama 3.2 model, and local all-MiniLM embeddings. Open the address Streamlit prints, select a sample question, and click **Run**. The trace shows each attempted stage, its grade, and the evidence presented to the model.

The app reads the thresholds in the folder specified by **CC_RESULTS** in **.env**. It uses saved calibration only when the corpus, embedding configuration, generator, prompts, and selected retrieval mode and noise condition match. If calibration found no acceptable threshold, the calibrated rule abstains immediately. For an arbitrary question, the displayed calibration comes from the benchmark population. The **manual threshold** control lets you explore routing on individual examples.

For a key-free smoke test without Ollama, set **USE_MOCK=1** and **CC_RESULTS=results_mock** in **.env**, or use these PowerShell commands:

```powershell
$env:USE_MOCK = "1"
$env:DATASET = "toy"
$env:CC_RESULTS = "results_mock"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

## Run the benchmarks

In the following commands, **python** means the interpreter in your project environment. Use **--out** to keep runs separate.

```powershell
# Offline plumbing check, including mock web fallback
python run_experiment.py --dataset toy --mock --n-questions 90 --noise none irrelevant:0.3 missing:0.3 partial:0.3 outdated:0.3 --modes closed open --recal --out results_mock

# Real local models on synthetic facts, with a small held-out split
python run_experiment.py --dataset toy --n-questions 60 --cal-frac 0.6 --k 2 --noise none irrelevant:0.3 --recal --out results_local

# Real local models on public QA data. The dataset downloads on first use.
python run_experiment.py --dataset mini_wiki --n-questions 90 --cal-frac 0.5 --k 2 --noise none irrelevant:0.3 missing:0.3 partial:0.3 outdated:0.3 --recal --out results_mini_wiki
python make_plots.py results_mini_wiki
```

To run live web fallback, set **TAVILY_API_KEY** in your local **.env** and add **--modes closed open**. Only the open mode uses Tavily. Closed mode still calls the configured generation and embedding providers, so use Ollama for both to keep those requests local. The mock web provider uses a clean copy of the synthetic corpus.

The model settings also support an OpenAI-compatible endpoint and a Hugging Face embedding model such as **BAAI/bge-small-en-v1.5**. Use separate results folders when changing model, dataset, or seed. A smaller **--n-questions** shortens runtime but makes calibration less likely to certify a threshold. Run several **--seed** values for evidence about the stability of the comparison.

To use your own data, point **--dataset** at a directory containing **.txt** or **.md** source files and **questions.jsonl**. Each line needs a unique ID, question, and list of accepted answers.

```json
{"id":"1","question":"What is the capital of France?","answers":["Paris"]}
```

## Routing and calibration

The router grades direct evidence, then rewritten-query evidence, then web evidence in open mode. It answers at the first stage that reaches the selected threshold. A live request generates an answer only at the selected stage. The benchmark generates candidates at every eligible stage once so that all thresholds can be replayed on the same examples. Its call counts and latency estimates describe the path each method would use, including retrieval and original provider time.

The calibration split tests eleven pre-specified thresholds from 0.0 to 1.0. For each threshold it evaluates the complete routing rule with an exact one-sided binomial test, then applies Holm correction across the candidates for that router. It chooses the certified candidate with the greatest calibration coverage, breaking ties by fewer calibration errors and then a higher threshold. With no certified candidate the calibrated policy always abstains.

- **Selective risk** is the error rate among answered queries and is undefined at zero coverage. It is the default.
- **Joint risk** is the rate of wrong answers across all queries. Select it with **--risk-def joint**.
- The statistical statement applies to the population risk of one selected router at the requested **alpha** and **delta**, assuming i.i.d. calibration draws from the deployment distribution and a fixed pipeline. It does not bound the realized error fraction in every finite test split.
- Clean calibration under noisy retrieval is a distribution-shift test. **--recal** fits a separate rule on calibration questions under the same noise condition. Calibration over multiple routers and noise conditions is reported separately.
- Correctness is measured by normalized answer-span matching. It can miss paraphrases and cannot establish that an answer is grounded. Mini Wikipedia passage labels use answer matching rather than human annotations, so their corruption is approximate. Outdated noise substitutes a different answer string as a proxy for stale facts.

## Methods and results

| Method | Decision |
|---|---|
| Standard RAG | Retrieve and generate without grading |
| CRAG-style baseline | Grade each chunk as direct evidence, partial evidence, or irrelevant. Route using fixed cutoffs and attempt repair without abstention. |
| Sufficiency-only | Grade the initial evidence and apply a fixed 0.5 threshold |
| Conformal direct | Calibrate the initial evidence threshold with no repair |
| Naive full | Select a full-router threshold using empirical calibration risk |
| Conformal full | Certify a full-router threshold on clean calibration examples |
| Conformal full recalibrated | Certify separately under each noisy calibration condition |

Irrelevant noise replaces retrieved passages with non-gold passages and can affect direct retrieval and rewritten retrieval independently. Missing, partial, and outdated noise use a persistent draw per question. They remove the labeled passage, remove answer-bearing sentences, or substitute a stale answer in the retrieved local context. Web results are excluded from synthetic corruption.

Each run writes **summary.csv**, **sweep.csv**, **per_query.jsonl**, the **calibration_*.csv** tables, and **taus.json**. The summary includes accuracy, coverage, error rates, abstention, route usage, call counts, and estimated latency. Bootstrap intervals are descriptive estimates from the held-out split. The bundled project tests one prompted grader. Comparing learned graders, running the full multi-dataset study, and measuring live Tavily performance require separate experiments.

## References

- Yan et al., *Corrective Retrieval Augmented Generation*. [Paper](https://arxiv.org/abs/2401.15884)
- Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*. [Paper](https://arxiv.org/abs/2110.01052)
- [Mini Wikipedia dataset](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
