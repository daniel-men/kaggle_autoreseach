from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.llm_calls import repair_code as _repair_code
from src.llms import get_llm
from src.schemas.CodeResultModel import CodeResultModel
from src.schemas.MLResearchPlan import MLResearchPlan
from src.utils import write_python_code_to_file
from src.llm_calls import implement_metric as _implement_metric

MAX_PREDICT_ATTEMPTS = 10


class MetricImplementationState(TypedDict):
    research_plan: MLResearchPlan
    slug: str
    metric_names: List[str] = []
    normalized_metric_names: List[str] = []
    attempt: int
    max_attempts: int
    metric_index: int
    predict_result: Optional[dict]
    code_result: Optional[dict]
    feedback: Optional[str]


def normalize_metric_name_with_llm(metric_name: str):
    normalized_name = None
    while normalized_name is None:
        llm = get_llm(
            provider="ollama",
            model="hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
            temperature=0.0,
        )
        prompt = (
            f"Normalize the following metric name: {metric_name} to a function name. "
            "It should only contain lowercase letters and underscores. "
            "Return only the normalized name."
        )

        model_output = llm.invoke(prompt)
        normalized_name = model_output.content

    return normalized_name


def select_metric(state: MetricImplementationState) -> dict:

    print(f"Implementing {state['metric_names'][state['metric_index']]}")
    return {"attempt": 0}


def normalize_metric_name(state: MetricImplementationState) -> dict:
    normalized_name = normalize_metric_name_with_llm(
        state["metric_names"][state["metric_index"]]
    )
    return {"normalized_metric_names": state["normalized_metric_names"] + [normalized_name]}


def implement_metric(state: MetricImplementationState) -> dict:
    code_result = _implement_metric(
        slug=state["slug"],
        metric=state["normalized_metric_names"][state["metric_index"]],
    )

    if isinstance(code_result, CodeResultModel):
        code_result = code_result.python_code

    content = getattr(code_result, "content", code_result)
    if isinstance(content, dict):
        content = content.get("content") or content.get("text") or str(content)
    else:
        content = str(content)

    write_python_code_to_file(
        content=content, filename="inferred_metrics.py", slug=state["slug"], append=True
    )
    return {"attempt": 0}


def load_metric_module(slug: str):
    module_path = Path.cwd() / "runs" / slug / "solution" / "inferred_metrics.py"
    spec = importlib.util.spec_from_file_location("inferred_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import(state: MetricImplementationState) -> dict:
    normalized_name = state["normalized_metric_names"][state["metric_index"]]
    predict_result = {}
    try:
        module = load_metric_module(slug=state["slug"])
        metric = getattr(module, normalized_name)
        metric([0, 1, 1, 0], [0, 1, 0, 0])
    except Exception as e:
        predict_result["success"] = False
        feedback = str(e)
        return {"predict_result": predict_result, "feedback": feedback}
    predict_result["success"] = True
    predict_result["feedback"] = None
    return {"predict_result": predict_result}


def repair_code(state: MetricImplementationState) -> dict:
    attempt = state["attempt"] + 1
    if state["feedback"]:
        print(
            f"--- Generating code (attempt {attempt}/{state['max_attempts']}, retry after failure) ---"
        )
    else:
        print(f"--- Generating code (attempt {attempt}/{state['max_attempts']}) ---")

    path = f"{os.getcwd()}/runs/{state['slug']}/solution/inferred_metrics.py"
    metric_name = state["metric_names"][state["metric_index"]]
    normalized_name = state["normalized_metric_names"][state["metric_index"]]
    context = (
        f"Metric to implement: {metric_name} "
        f"(function name: {normalized_name}, arguments: y_pred, y_test)"
    )

    code_result = _repair_code(
        slug=state["slug"], file_path=path, traceback=state["feedback"], context=context
    )
    content = getattr(code_result, "content", code_result)
    if isinstance(content, dict):
        content = content.get("content") or content.get("text") or str(content)
    else:
        content = str(content)

    write_python_code_to_file(
        content=content,
        filename="inferred_metrics.py",
        slug=state["slug"],
        append=False,
    )

    return {"code_result": code_result, "attempt": attempt}


def increase_index(state: MetricImplementationState):
    return {"metric_index": state["metric_index"] + 1}


def route_after_import(state: MetricImplementationState) -> str:
    if state["predict_result"]["success"] or state["attempt"] >= state["max_attempts"]:
        if state["metric_index"] == len(state["metric_names"]) - 1:
            return END
        return "increase_index"

    return "repair_code"


def build_graph():
    graph = StateGraph(MetricImplementationState)
    graph.add_node("select_metric", select_metric)
    graph.add_node("normalize_metric_name", normalize_metric_name)
    graph.add_node("implement_metric", implement_metric)
    graph.add_node("test_import", test_import)
    graph.add_node("repair_code", repair_code)
    graph.add_node("increase_index", increase_index)

    graph.set_entry_point("select_metric")
    graph.add_edge("select_metric", "normalize_metric_name")
    graph.add_edge("normalize_metric_name", "implement_metric")
    graph.add_edge("implement_metric", "test_import")
    graph.add_conditional_edges(
        "test_import",
        route_after_import,
        {"repair_code": "repair_code", "increase_index": "increase_index", END: END},
    )
    graph.add_edge("increase_index", "select_metric")
    graph.add_edge("repair_code", "test_import")

    return graph.compile()


def implement_metrics_from_research_plan(slug: str, plan: MLResearchPlan):
    metric_names = [plan.primary_metric] + plan.secondary_metrics

    graph = build_graph()
    graph.get_graph().draw_mermaid_png(
        output_file_path=f"{os.getcwd()}/runs/{slug}/metrics_graph.png"
    )

    final_state = graph.invoke(
        {
            "slug": slug,
            "metric_index": 0,
            "metric_names": metric_names,
            "normalized_metric_names": [],
            "research_plan": plan,
            "max_attempts": MAX_PREDICT_ATTEMPTS,
        }
    )
    normalized_metric_names = final_state["normalized_metric_names"]
    return normalized_metric_names[0], normalized_metric_names[1:]
