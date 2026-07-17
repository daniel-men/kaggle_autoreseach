import shutil
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.challenge_info import download_competition

from src.kaggle_docs_scrape import write_challenge_docs
from src.metrics_graph import implement_metrics_from_research_plan
from src.preprocessing_graph import preprocess_data
from src.report import write_markdown_report
from src.schemas.DataPreprocessingPlan import DataPreprocessingPlan
from src.schemas.MLResearchPlan import MLResearchPlan
from src.utils import _workspace_for_challenge, ensure_dirs, slug_from_url
from src.research import get_initial_research_plan
from src.graph import iterate_research_plan


class InitialResearchPlanState(TypedDict):
    slug: str
    plan: Optional[MLResearchPlan]


def generate_initial_research_plan(state: InitialResearchPlanState) -> dict:
    return {"plan": get_initial_research_plan(slug=state["slug"])}


def route_after_research_plan(state: InitialResearchPlanState) -> str:
    if state["plan"] is None:
        return "generate_initial_research_plan"
    return END


def build_initial_research_plan_graph():
    graph = StateGraph(InitialResearchPlanState)
    graph.add_node("generate_initial_research_plan", generate_initial_research_plan)
    graph.set_entry_point("generate_initial_research_plan")
    graph.add_conditional_edges(
        "generate_initial_research_plan",
        route_after_research_plan,
        {"generate_initial_research_plan": "generate_initial_research_plan", END: END},
    )
    return graph.compile()


def initialize_research_plan(slug: str) -> MLResearchPlan:
    graph = build_initial_research_plan_graph()
    final_state = graph.invoke(
        {"slug": slug, "plan": None},
        {"recursion_limit": 1000},
    )
    return final_state["plan"]

def remove_existing_files(slug: str):
    workspace = _workspace_for_challenge(slug=slug)

    solution_dir = workspace / "solution"
    if solution_dir.exists():
        for path in solution_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    (workspace / "data" / "preprocessed_data.csv").unlink(missing_ok=True)
    (workspace / "reports" / "research_report.md").unlink(missing_ok=True)

def initialize_workspace(challenge_url: str):
    slug = slug_from_url(challenge_url)
    workspace = _workspace_for_challenge(slug=slug)
    workspace.mkdir(parents=True, exist_ok=True)
    ensure_dirs(workspace=workspace)
    remove_existing_files(slug=slug)
    download_competition(slug=slug, data_dir=workspace / "data")
    write_challenge_docs(workspace=workspace, challenge=challenge_url)



    


def main(challenge_url: str):
    initialize_workspace(challenge_url=challenge_url)
    slug = slug_from_url(challenge_url)
    plan = initialize_research_plan(slug=slug)

    # Run preprocessing
    preprocessing_plan = DataPreprocessingPlan.from_research_plan(plan)
    preprocessing_result = preprocess_data(slug=slug, plan=preprocessing_plan)
    
    # Infer metrics
    primary_metric, secondary_metrics = implement_metrics_from_research_plan(slug=slug, plan=plan)
    plan.primary_metric = primary_metric
    plan.secondary_metrics = secondary_metrics

    # Run experiments
    experiment_results = iterate_research_plan(slug=slug, research_plan=plan)
    write_markdown_report(state=experiment_results, preprocessing_plan=preprocessing_plan)

if __name__ == "__main__":
    main("https://www.kaggle.com/competitions/titanic")
    #main("https://www.kaggle.com/competitions/home-data-for-ml-course")
    #main("https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction")
