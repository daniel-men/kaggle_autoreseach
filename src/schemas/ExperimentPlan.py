from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.MLResearchPlan import ExperimentIdea, MLResearchPlan

class ExperimentPlan(BaseModel):
    experiment: ExperimentIdea
    likely_target_column: str | None
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
    primary_metric: str
    secondary_metrics: list[str]
    validation_strategy: str

    def from_research_plan(experiment_index: int, plan: MLResearchPlan):
        return ExperimentPlan(
            experiment=plan.experiments[experiment_index],
            inferred_task_type=plan.inferred_task_type,
            likely_target_column=plan.likely_target_column,
            primary_metric=plan.primary_metric,
            secondary_metrics=plan.secondary_metrics,
            validation_strategy=plan.validation_strategy
        )