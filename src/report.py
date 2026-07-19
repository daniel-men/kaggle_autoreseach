from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from src.schemas.DataPreprocessingPlan import DataPreprocessingPlan
from src.utils import get_file_content

if TYPE_CHECKING:
    from src.graph import ResearchIterationState


def write_markdown_report(
    state: "ResearchIterationState",
    preprocessing_plan: DataPreprocessingPlan,
    output_path: str | Path | None = None,
) -> Path:
    """Write a Markdown report for a completed research iteration state."""
    slug = state["slug"]
    report_path = Path(output_path) if output_path else _default_report_path(slug)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessing_code = get_file_content(
        Path.cwd() / "runs" / slug / "solution" / "preprocessing.py"
    )

    parts = [
        f"# Research Report: {slug}",
        "## Preprocessing Plan",
        _render_fields(preprocessing_plan),
        _details("Preprocessing code", _code_block(preprocessing_code, "python")),
        "## Research Plan",
        _render_fields(state["research_plan"], exclude={"experiments"}),
        "## Experiments",
        _render_experiment_plans(
            state["research_plan"].experiments, state.get("experiment_results", [])
        ),
        "",
    ]

    report_path.write_text(
        "\n\n".join(part for part in parts if part), encoding="utf-8"
    )
    return report_path


def _default_report_path(slug: str) -> Path:
    return Path.cwd() / "runs" / slug / "reports" / "research_report.md"


def _to_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _render_fields(value: Any, exclude: set[str] | None = None) -> str:
    data = _to_data(value)
    if not isinstance(data, Mapping):
        return _render_value(data)

    exclude = exclude or set()
    lines: list[str] = []
    for key, item in data.items():
        if key in exclude:
            continue
        title = key.replace("_", " ").title()
        rendered = _render_value(item)
        if "\n" in rendered:
            lines.append(f"### {title}\n\n{rendered}")
        else:
            lines.append(f"- **{title}:** {rendered}")
    return "\n\n".join(lines)


def _render_value(value: Any, indent: int = 0) -> str:
    value = _to_data(value)
    prefix = "  " * indent

    if isinstance(value, Mapping):
        return "\n".join(
            f"{prefix}- **{key.replace('_', ' ').title()}:** "
            f"{_render_value(item, indent + 1).lstrip()}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        if not value:
            return "None"
        return "\n".join(
            f"{prefix}- {_render_value(item, indent + 1).lstrip()}" for item in value
        )
    if value is None:
        return "None"
    return str(value)


def _render_experiment_plans(
    experiments: list[Any], results: list[dict[str, Any]]
) -> str:
    if not experiments:
        return "No planned experiments."

    sections = []
    for index, experiment in enumerate(experiments, start=1):
        data = _to_data(experiment)
        name = (
            data.get("name", f"Experiment {index}")
            if isinstance(data, Mapping)
            else f"Experiment {index}"
        )

        lines = [
            f"### {index}. {name}",
            _render_fields(experiment),
            "#### Result",
            _render_experiment_result(
                results[index - 1] if index <= len(results) else None
            ),
        ]
        sections.append("\n\n".join(lines))
    return "\n\n".join(sections)


def _render_experiment_result(result: dict[str, Any] | None) -> str:
    if not result:
        return "No result recorded."

    predict_result = result.get("predict_result") or {}
    metrics = predict_result.get("metrics") or {}
    status = "success" if predict_result.get("success") else "failed"
    lines = [
        f"- **Status:** {status}",
        f"- **Attempts:** {result.get('attempts', 'unknown')}",
    ]
    if metrics:
        lines.append(f"- **Metrics:** {_render_inline_mapping(metrics)}")

    solution_content = predict_result.get("solution_content")
    if solution_content:
        lines.append(_details("Final code", _code_block(solution_content, "python")))

    return "\n\n".join(lines)


def _render_inline_mapping(mapping: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in mapping.items())


def _details(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def _code_block(content: str, language: str = "") -> str:
    longest_backticks = max(
        (len(match.group(0)) for match in re.finditer(r"`+", content)), default=0
    )
    fence = "`" * max(3, longest_backticks + 1)
    return f"{fence}{language}\n{content.rstrip()}\n{fence}"
