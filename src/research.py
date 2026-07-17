"""
agent.py

A Deep Agent that reads:
- a Markdown ML task description
- a CSV training dataset

and proposes an initial machine learning research plan.

It does NOT train a model.
It only inspects the task/data and produces a structured plan.

Run:

    python agent.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_ollama import ChatOllama
import pandas as pd
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool

# ---------------------------------------------------------------------
# 1. Choose your model
# ---------------------------------------------------------------------
# Hosted OpenAI example:
#
#   pip install langchain-openai
#   export OPENAI_API_KEY="..."
#
# Local Ollama example:
#
#   pip install langchain-ollama
#   ollama pull qwen2.5-coder:7b
#
# Use ONE of the two blocks below.


from src.llms import get_llm
from src.prompts import RESEARCH_SYSTEM_PROMPT
from src.schemas.MLResearchPlan import MLResearchPlan



#


# ---------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------
# Keep this pointed at a small working directory that contains task.md
# and your CSV file.


# ---------------------------------------------------------------------
# 3. Structured output schema
# ---------------------------------------------------------------------
# Deep Agents themselves do not have to use structured output,
# but defining the desired schema in the prompt makes the result easier
# to consume.
#
# If you want strict validation, parse the final JSON with this model.


# ---------------------------------------------------------------------
# 4. Tools
# ---------------------------------------------------------------------
# These tools let the agent inspect the markdown and CSV without loading
# the entire dataset into the LLM context.
#
# This is important. You generally do NOT want to paste a full CSV into
# the model prompt.





@tool
def read_task_markdown(markdown_filename: str) -> dict:
    """
    Read the Markdown task description.

    Use this before inspecting the CSV so the target, metric, and task
    objective come from the competition description.
    """

    path = Path(markdown_filename)
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "n_chars": len(text),
        "content": text,
    }


@tool
def inspect_csv_schema(csv_filename: str) -> dict:
    """
    Inspect the CSV schema, row count, column names, dtypes, missing values,
    uniqueness, and a few sample values.

    Use this before proposing the ML research plan.
    """

    df = pd.read_csv(csv_filename)

    # Limit sample values so we do not dump too much data into the context.
    columns = []

    for col in df.columns:
        series = df[col]

        example_values = series.dropna().astype(str).head(3).tolist()

        columns.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "missing_pct": round(float(series.isna().mean()), 3),
                "n_unique": int(series.nunique(dropna=True)),
                "sample_values": example_values,
            }
        )

    # Cap at 60 columns — the model doesn't need the full schema of wide datasets.
    truncated = len(df.columns) > 60
    return {
        "path": str(csv_filename),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": columns[:60],
        **({"truncated_columns": True} if truncated else {}),
    }


@tool
def inspect_target_distribution(csv_filename: str, target_column: str) -> dict:
    """
    Inspect the distribution of a possible target column.

    Use this after you infer or suspect a likely target column.
    """

    df = pd.read_csv(csv_filename)

    if target_column not in df.columns:
        return {
            "error": f"Column not found: {target_column}",
            "available_columns": list(df.columns),
        }

    series = df[target_column]

    value_counts = series.value_counts(dropna=False).head(20).to_dict()

    return {
        "target_column": target_column,
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_percent": float(series.isna().mean()),
        "n_unique": int(series.nunique(dropna=True)),
        "top_values": {str(k): int(v) for k, v in value_counts.items()},
    }


@tool
def inspect_numeric_summary(csv_filename) -> dict:
    """
    Return summary statistics for numeric columns.

    Use this to identify scales, outliers, sparse numeric fields,
    constant columns, and possible regression targets.
    """

    df = pd.read_csv(csv_filename)
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return {"message": "No numeric columns found."}

    # Keep only the most diagnostic stats; drop 25%/75% to halve the output size.
    keep_stats = {"mean", "std", "min", "50%", "max"}
    summary = numeric.describe().loc[list(keep_stats & set(numeric.describe().index))].transpose()

    result = {
        col: {
            stat: None if pd.isna(value) else round(float(value), 4)
            for stat, value in summary.loc[col].to_dict().items()
        }
        for col in summary.index[:50]  # cap at 50 numeric columns
    }
    if len(numeric.columns) > 50:
        result["_truncated"] = f"{len(numeric.columns) - 50} additional numeric columns omitted"
    return result


@tool
def inspect_sample_rows(csv_filename: str, n_rows: int = 5) -> list[dict]:
    """
    Return a small number of sample rows from the CSV.

    Use this only when column names and schema are not enough.
    Do not request a large number of rows.
    """

    n_rows = max(1, min(int(n_rows), 10))

    df = pd.read_csv(csv_filename)
    return df.head(n_rows).where(pd.notna(df), None).to_dict(orient="records")


# ---------------------------------------------------------------------
# 5. System prompt
# ---------------------------------------------------------------------
# The prompt makes the agent behave like an ML research lead.
# The key instruction: produce a research plan, not code or training.




# ---------------------------------------------------------------------
# 6. Create the Deep Agent
# ---------------------------------------------------------------------
# Deep Agents are useful here because they can plan, inspect files/data
# through tools, and produce a multi-step research plan.
#
# The default deep agent harness includes planning and context management.
# We provide custom tools for safe CSV inspection.
def _get_research_model() -> ChatOllama:
    return get_llm(
        provider="ollama",
        model="hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        temperature=0.05,
    )


def create_research_agent():
    model = _get_research_model()
    agent = create_deep_agent(
        model=model,
        tools=[
            read_task_markdown,
            inspect_csv_schema,
            inspect_target_distribution,
            inspect_numeric_summary,
            inspect_sample_rows,
        ],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        response_format=MLResearchPlan,
        backend=FilesystemBackend(),
    )
    return agent


# ---------------------------------------------------------------------
# 7. Invoke the agent
# ---------------------------------------------------------------------


def _research_input_paths(slug: str) -> tuple[Path, Path]:
    workspace = Path("runs") / slug
    markdown_path = workspace / "reports" / "challenge_docs" / "combined.md"
    train_csv_path = workspace / "data" / "train.csv"

    if not train_csv_path.exists():
        data_dir = workspace / "data"
        explicit_candidates = (
            data_dir / "train.csv.gz",
            data_dir / "training.csv",
            data_dir / "training.csv.gz",
        )
        for candidate in explicit_candidates:
            if candidate.exists():
                train_csv_path = candidate
                break
        else:
            csv_candidates = [
                path
                for path in data_dir.rglob("*")
                if path.is_file()
                and (
                    path.name.endswith(".csv")
                    or path.name.endswith(".csv.gz")
                )
            ]

            def score(path: Path) -> tuple[int, int, str]:
                parts = {part.lower() for part in path.parts}
                name = path.name.lower()
                stem = path.name.lower().removesuffix(".gz").removesuffix(".csv")
                bad_name = (
                    "sample_submission" in name
                    or "submission" in name
                    or stem.startswith("test")
                )
                bad_dir = bool(parts & {"test", "valid", "validation"})
                train_hint = int("train" in parts or stem.startswith("train") or "training" in stem)
                return (
                    train_hint * 100 - int(bad_name) * 50 - int(bad_dir) * 25,
                    path.stat().st_size,
                    str(path),
                )

            if csv_candidates:
                train_csv_path = max(csv_candidates, key=score)

    missing_paths = [
        path
        for path in (markdown_path, train_csv_path)
        if not path.exists()
    ]
    if missing_paths:
        missing = "\n".join(f"- {path.resolve()}" for path in missing_paths)
        raise FileNotFoundError(
            "Research inputs are missing at the expected paths:\n"
            f"{missing}"
        )

    return markdown_path.resolve(), train_csv_path.resolve()


def get_initial_research_plan(slug) -> MLResearchPlan:
    markdown_path, train_csv_path = _research_input_paths(slug)
    agent = create_research_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Create an initial research plan for the ML task in competition slug {slug!r}. "
                        f"The workspace directory is {(Path('runs') / slug).resolve()}. "
                        f"The Markdown description file exists at {markdown_path}. "
                        f"The training CSV file exists at {train_csv_path}. "
                        "Focus on what would be the best machine learning method to solve this task."
                    ),
                }
            ]
        }
    )

    if "structured_response" in result:
        return result["structured_response"]

    # The agent ended its turn without calling the structured-output tool
    # (small local models often just stop instead of calling it). Fall back
    # to Ollama's native JSON-schema constrained decoding over the
    # conversation so far, which is more reliable than tool-calling here.
    structured_model = _get_research_model().with_structured_output(
        MLResearchPlan, method="json_schema"
    )
    return structured_model.invoke(
        result["messages"]
        + [
            {
                "role": "user",
                "content": (
                    "Based on the investigation above, produce the final "
                    "research plan now."
                ),
            }
        ]
    )


def improve_experiment(experiments):
    prompt = f"""
    You are a senior machine learning research lead.

    Your job is to think of a new experiment based on the already conducted experiments, their code and their result.
    {experiments}

    """
