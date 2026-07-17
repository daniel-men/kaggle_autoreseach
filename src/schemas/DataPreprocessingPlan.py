from typing import Literal

from pydantic import BaseModel

from src.schemas.MLResearchPlan import DataRisk, MLResearchPlan

class DataPreprocessingPlan(BaseModel):
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
    feature_engineering_ideas: list[str]
    data_risks: list[DataRisk]
    preprocessed_data_path: str | None


    def from_research_plan(plan: MLResearchPlan):
        return DataPreprocessingPlan(
            inferred_task_type=plan.inferred_task_type,
            likely_target_column=plan.likely_target_column,
            feature_engineering_ideas=plan.feature_engineering_ideas,
            data_risks=plan.data_risks,
            preprocessed_data_path=None
        )