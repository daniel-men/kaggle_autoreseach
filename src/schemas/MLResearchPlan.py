
from typing import Literal

from pydantic import BaseModel, Field

class ExperimentIdea(BaseModel):
    name: str = Field(description="Short experiment name.")
    goal: str = Field(description="What this experiment is meant to learn.")
    method: str = Field(description="Modeling or analysis method.")
    success_metric: str = Field(description="Metric used to judge success.")
    expected_insight: str = Field(description="What we expect to learn.")


class DataRisk(BaseModel):
    risk: str = Field(description="Data, modeling, evaluation, or deployment risk.")
    why_it_matters: str = Field(description="Why this risk matters.")
    mitigation: str = Field(description="How to reduce or test the risk.")


class MLResearchPlan(BaseModel):
    task_summary: str
    inferred_task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "forecasting",
        "ranking",
        "clustering",
        "nlp",
        "computer_vision",
        "unknown",
    ]
    likely_target_column: str | None
    primary_metric: str
    secondary_metrics: list[str]
    validation_strategy: str
    baseline_models: list[str]
    feature_engineering_ideas: list[str]
    experiments: list[ExperimentIdea]
    data_risks: list[DataRisk]
    recommended_first_steps: list[str]
    open_questions_for_user: list[str]
