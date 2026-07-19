import os
import re
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.schemas.CodeResultModel import CodeResultModel
from src.runner import run_function
from src.schemas.DataPreprocessingPlan import DataPreprocessingPlan
from src.llm_calls import implement_preprocessing
from src.llm_calls import repair_code as _repair_code
from src.utils import write_python_code_to_file

class PreprocessingState(TypedDict):
    slug: str
    preprocessing_plan: DataPreprocessingPlan
    max_attempts: int
    attempt: int
    feedback: Optional[str]
    code_result: Optional[dict]


def write_preprocessing_code(state: PreprocessingState) -> dict:
    content = getattr(state["code_result"], "content", state["code_result"])
    if isinstance(content, dict):
        content = content.get("content") or content.get("text") or str(content)
    else:
        content = str(content)

    write_python_code_to_file(content=content, filename="preprocessing.py", slug=state["slug"])
    return {}


def implement_preprocessing_node(state: PreprocessingState) -> dict:
    model_output = implement_preprocessing(slug=state["slug"], context=state["preprocessing_plan"].model_dump_json())
    if isinstance(model_output, CodeResultModel):
        code_result = model_output.python_code
    else:
        code_result = model_output.content
    return {"code_result": code_result, "attempt": 0}

def run_preprocessing_node(state: PreprocessingState) -> dict:
    print("--- Running preprocessing ---")
    slug = state["slug"]
    data_path = Path(os.getcwd()) / "runs" / slug / "data" / "train.csv"
    # args should be an iterable (tuple/list) of positional arguments for the target function
    result = run_function(slug=slug, filename="preprocessing.py", fn_name="preprocess", args=(str(data_path),))
    feedback = None if result["success"] else result["traceback"]
    if result["success"]:
        try:
            
            output_path = Path(os.getcwd()) / "runs" / slug / "data" / "preprocessed_data.csv"
            result["result"].to_csv(output_path, index=False)
        except Exception as e:
            result["success"] = False
            return {"success": result["success"], "feedback": str(e)}
        
        print("Preprocessing succeeded.")
    else:
        print(f"Preprocessing failed:\n{feedback}")
    return {"success": result["success"], "feedback": feedback}


def repair_code(state: PreprocessingState) -> dict:
    attempt = state["attempt"] + 1
    if state["feedback"]:
        print(f"--- Generating code (attempt {attempt}/{state['max_attempts']}, retry after failure) ---")
    else:
        print(f"--- Generating code (attempt {attempt}/{state['max_attempts']}) ---")

    solution_path = f"{os.getcwd()}/runs/{state['slug']}/solution/preprocessing.py"
    context = state["preprocessing_plan"].model_dump_json()
    code_result = _repair_code(slug=state["slug"], file_path=solution_path, traceback=state["feedback"], context=context)
    if isinstance(code_result, CodeResultModel):
        code_result = code_result.python_code
    
    print("Code repair done.")
    return {"code_result": code_result, "attempt": attempt}

def route_after_run(state: PreprocessingState) -> str:
    if state["feedback"] is None:
        return END
    if state["attempt"] >= state["max_attempts"]:
        return END
    return "repair_code"


def build_preprocessing_graph():
    graph = StateGraph(PreprocessingState)

    graph.add_node("implement_preprocessing", implement_preprocessing_node)
    graph.add_node("write_preprocessing_code", write_preprocessing_code)
    graph.add_node("run_preprocessing", run_preprocessing_node)
    graph.add_node("repair_code", repair_code)

    graph.set_entry_point("implement_preprocessing")
    graph.add_edge("implement_preprocessing", "write_preprocessing_code")
    graph.add_edge("write_preprocessing_code", "run_preprocessing")
    graph.add_edge("repair_code", "write_preprocessing_code")
    graph.add_conditional_edges(
        "run_preprocessing",
        route_after_run,
        {"repair_code": "repair_code", END: END},
    )

    return graph.compile()



def preprocess_data(slug: str, plan: DataPreprocessingPlan):
    graph = build_preprocessing_graph()
    graph.get_graph().draw_mermaid_png(output_file_path=f"{os.getcwd()}/runs/{slug}/preprocessing_graph.png")

    final_state = graph.invoke(
        {
            "slug": slug,
            "preprocessing_plan": plan,
            "max_attempts": 10,
            "attempt": 0,
            "feedback": None,
            "code_result": None,
        },
        #{"recursion_limit": recursion_limit},
    )

    return final_state
