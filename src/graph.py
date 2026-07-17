from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.dcode import ask_for_code
from src.dcode import repair_code as _repair_code
from src.metrics_graph import load_metric_module
from src.runner import run_function
from src.schemas.ExperimentPlan import ExperimentPlan
from src.schemas.MLResearchPlan import MLResearchPlan
from src.utils import write_python_code_to_file
from src.preprocessing_graph import implement_preprocessing as _implement_preprocessing

MAX_PREDICT_ATTEMPTS = 10


class ResearchIterationState(TypedDict):
    slug: str
    research_plan: MLResearchPlan
    max_attempts: int

    experiment_index: int
    experiment_plan: Optional[ExperimentPlan]
    attempt: int
    feedback: Optional[str]
    code_result: Optional[dict]
    predict_result: Optional[dict]

    experiment_results: list[dict]


def select_experiment(state: ResearchIterationState) -> dict:
    experiment_plan = ExperimentPlan.from_research_plan(
        state["experiment_index"], state["research_plan"]
    )
    n_experiments = len(state["research_plan"].experiments)
    print(
        f"\n=== Experiment {state['experiment_index'] + 1}/{n_experiments}: "
        f"{experiment_plan.experiment.name} ==="
    )
    return {
        "experiment_plan": experiment_plan,
        "attempt": 0,
        "feedback": None,
        "code_result": None,
        "predict_result": None,
    }

#def implement_preprocessing(state: ResearchIterationState) -> dict:
#    state["experiment_plan"]
#    _implement_preprocessing(slug=state["slug"], context=state["experiment_plan"].model_dump_json())

def repair_code(state: ResearchIterationState) -> dict:
    attempt = state["attempt"] + 1
    if state["feedback"]:
        print(f"--- Generating code (attempt {attempt}/{state['max_attempts']}, retry after failure) ---")
    else:
        print(f"--- Generating code (attempt {attempt}/{state['max_attempts']}) ---")

    solution_path = f"{os.getcwd()}/runs/{state['slug']}/solution/solution.py"
    context = state["experiment_plan"].model_dump_json()

    code_result = _repair_code(slug=state["slug"], file_path=solution_path, traceback=state["feedback"], context=context)
    print("Code repair done.")
    return {"code_result": code_result, "attempt": attempt} 


def generate_code(state: ResearchIterationState) -> dict:
    context = state["experiment_plan"].model_dump_json()
    code_result = ask_for_code(slug=state["slug"], context=context, stream=False)
    return {"code_result": code_result, "attempt": 0}
    

  


def get_metric(state: ResearchIterationState) -> Callable:
    metric_name = state["experiment_plan"].primary_metric
    normalized_name = metric_name.lower().replace(" ", "_").replace("-", "_")
    metric_module = load_metric_module(state["slug"])
    metric = getattr(metric_module, normalized_name, None)
    
    if metric is None:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metric


def get_metrics(state: ResearchIterationState) -> dict[str, Callable]:
    metric_names = [state["experiment_plan"].primary_metric]
    metric_names.extend(state["experiment_plan"].secondary_metrics or [])
    metric_names = [name for name in metric_names if name]

    metrics = {}
    for metric_name in metric_names:
        normalized_name = metric_name.lower().replace(" ", "_").replace("-", "_")
        metric_module = load_metric_module(state["slug"])
        metric = getattr(metric_module, normalized_name, None)
      
        if metric is None:
            raise ValueError(f"Unknown metric: {metric_name}")
        metrics[metric_name] = metric
    return metrics


def run_predict_node(state: ResearchIterationState) -> dict:
    print("--- Running predict ---")
    predict_result = run_function(slug=state["slug"], filename="solution.py", fn_name="predict", args=None)
    feedback = None if predict_result["success"] else predict_result["traceback"]
    if predict_result["success"]:
        
        try:
            y_true, y_pred = predict_result["result"]
        except Exception as e:
            predict_result["success"] = False
            feedback = str(e)
            return {"predict_result": predict_result, "feedback": feedback}
        
        print("Predict succeeded.")
        metrics = get_metrics(state)
        predict_result["metrics"] = {}
        for metric_name, metric in metrics.items():
            
            try:
                result = metric(y_true, y_pred)
            except Exception as e:
                print(e)
                result = None
            
            predict_result["metrics"][metric_name] = result
        
            
        predict_result["experiment"] = state["experiment_plan"].experiment.model_dump()
        solution_path = f"{os.getcwd()}/runs/{state['slug']}/solution/solution.py"
        with open(solution_path, "r", encoding="utf-8") as solution_file:
            predict_result["solution_content"] = solution_file.read()

    else:
        print(f"Predict failed:\n{feedback}")
    return {"predict_result": predict_result, "feedback": feedback}


def record_result(state: ResearchIterationState) -> dict:
    # TODO check if correct
    entry = {
        "experiment": state["experiment_plan"].experiment.name,
        "attempts": state["attempt"],
        "predict_result": state["predict_result"],
    }
    status = "success" if state["predict_result"]["success"] else "failed"
    print(
        f"=== Experiment {state['experiment_index'] + 1} result: {status} "
        f"after {state['attempt']} attempt(s) ==="
    )
    return {
        "experiment_results": state["experiment_results"] + [entry],
        "experiment_index": state["experiment_index"] + 1,
    }


def route_after_predict(state: ResearchIterationState) -> str:
    if state["predict_result"]["success"]:
        return "record_result"
    if state["attempt"] >= state["max_attempts"]:
        return "record_result"
    return "repair_code"


def route_after_record(state: ResearchIterationState) -> str:
    if state["experiment_index"] < len(state["research_plan"].experiments):
        return "select_experiment"
    return END

def write_code(state: ResearchIterationState) -> dict:
    content = getattr(state["code_result"], "content", state["code_result"])
    if isinstance(content, dict):
        content = content.get("content") or content.get("text") or str(content)
    else:
        content = str(content)

    write_python_code_to_file(content=content, filename="solution.py", slug=state["slug"])
    return {}


def build_research_iteration_graph():
    graph = StateGraph(ResearchIterationState)

    graph.add_node("select_experiment", select_experiment)
    graph.add_node("generate_code", generate_code)
    graph.add_node("write_code", write_code)
    graph.add_node("run_predict", run_predict_node)
    graph.add_node("record_result", record_result)
    graph.add_node("repair_code", repair_code)

    graph.set_entry_point("select_experiment")
    graph.add_edge("select_experiment", "generate_code")
    graph.add_edge("write_code", "run_predict")
    graph.add_edge("generate_code", "write_code")
    graph.add_edge("repair_code", "write_code")
    graph.add_conditional_edges(
        "run_predict",
        route_after_predict,
        {"repair_code": "repair_code", "record_result": "record_result"},
    )
    graph.add_conditional_edges(
        "record_result",
        route_after_record,
        {"select_experiment": "select_experiment", END: END},
    )

    return graph.compile()


def iterate_research_plan(
    slug: str, research_plan: MLResearchPlan, max_attempts: int = MAX_PREDICT_ATTEMPTS
) -> list[dict]:
    graph = build_research_iteration_graph()
    graph.get_graph().draw_mermaid_png(output_file_path=f"{os.getcwd()}/runs/{slug}/graph.png")


    final_state = graph.invoke(
        {
            "slug": slug,
            "research_plan": research_plan,
            "max_attempts": max_attempts,
            "experiment_index": 0,
            "experiment_plan": None,
            "attempt": 0,
            "feedback": None,
            "code_result": None,
            "predict_result": None,
            "experiment_results": [],
        },
        #{"recursion_limit": recursion_limit},
    )

    return final_state
