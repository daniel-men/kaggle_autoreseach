# kaggle-autoresearch

An agentic pipeline that takes a Kaggle competition URL and automatically:

1. Downloads the competition data and scrapes/cleans the competition docs
   (description, evaluation, data) — [src/challenge_info.py](src/challenge_info.py),
   [src/kaggle_docs_scrape.py](src/kaggle_docs_scrape.py)
2. Produces an initial ML research plan with a Deep Agent —
   [src/research.py](src/research.py), [src/main.py](src/main.py)
3. Builds a preprocessing pipeline and infers evaluation metrics —
   [src/preprocessing_graph.py](src/preprocessing_graph.py), [src/metrics_graph.py](src/metrics_graph.py)
4. Implements and iterates on experiments from the plan, using an LLM coding
   agent that repairs its own code on failure —
   [src/graph.py](src/graph.py), [src/dcode.py](src/dcode.py), [src/runner.py](src/runner.py)
5. Writes a Markdown research report — [src/report.py](src/report.py)

All outputs for a given competition are written under `runs/<competition-slug>/`
(data, generated solution code, reports, and graph diagrams).

## Prerequisites

- Python 3.11+
- A Kaggle account with an API token (used to download competition data)
- [Ollama](https://ollama.com/) running locally, with the models this project
  uses pulled:
  ```bash
  ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M   # research & metrics planning
  ollama pull qwen3-coder:30b                        # code generation/repair (dcode)
  ollama pull qwen2.5:7b                              # challenge-doc cleanup
  ```
- Google Chrome or Chromium on `PATH` (or in the default macOS Applications
  folder) — used headlessly to render Kaggle competition pages when scraping docs.


## Setup

1. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project (editable install, including test dependencies):
   ```bash
   pip install -e ".[dev]"
   ```
   The `dcode` extra pulls in the optional CLI coding-agent backend
   (`deepagents-code`, `langchain-experimental`):
   ```bash
   pip install -e ".[dcode]"
   ```
3. Configure credentials:
   ```bash
   cp .env.example .env
   ```
   Fill in `KAGGLE_USERNAME` and `KAGGLE_KEY` from a token generated at
   https://www.kaggle.com/settings. Alternatively, use the standard Kaggle CLI
   location (`~/.kaggle/kaggle.json`) instead of the `.env` file.

## Running

The entry point runs the full pipeline against a competition URL. By default
`src/main.py` targets the Titanic competition:

```bash
python -m src.main
```

To target a different competition, either edit the `main(...)` call at the
bottom of [src/main.py](src/main.py), or run it directly:

```bash
python -c "from src.main import main; main('https://www.kaggle.com/competitions/<slug>')"
```

Results appear under `runs/<slug>/`:
- `data/` — downloaded and preprocessed competition data
- `solution/` — generated preprocessing, metrics, and experiment code
- `reports/` — scraped challenge docs and the final `research_report.md`
- `*_graph.png` — LangGraph diagrams of each pipeline stage

## Testing

```bash
pytest
```

The test suite only exercises the parts of the codebase with no heavy
external dependencies (file/code generation utilities, the experiment
runner, and competition data handling), so it does not require Ollama, a
Kaggle token, or Chrome to be installed.

## Project layout

```
src/
  main.py                 # pipeline entry point
  challenge_info.py       # Kaggle data download
  kaggle_docs_scrape.py   # competition doc scraping + LLM cleanup
  research.py             # initial research plan (Deep Agent)
  graph.py                # experiment iteration loop
  preprocessing_graph.py  # preprocessing pipeline generation
  metrics_graph.py        # metric inference/implementation
  dcode.py                # LLM coding agent (implements/repairs solution code)
  runner.py               # sandboxed execution of generated code
  report.py               # Markdown report generation
  llms.py                 # LLM provider wiring (Ollama/OpenAI-compatible/Anthropic)
  schemas/                # Pydantic schemas for plans
tests/                    # unit tests (no external services required)
runs/                     # per-competition outputs (git-ignored)
```
