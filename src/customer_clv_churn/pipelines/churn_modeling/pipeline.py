"""Pipeline for churn model training, calibration and survival analysis."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    analyze_survival,
    build_churn_model_input,
    split_churn_data,
    train_churn_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``churn_modeling`` pipeline."""
    return pipeline(
        [
            node(
                func=build_churn_model_input,
                inputs=["transactions_curated", "params:temporal", "params:churn_modeling"],
                outputs="churn_model_input",
                name="build_churn_model_input_node",
            ),
            node(
                func=split_churn_data,
                inputs=["churn_model_input", "params:churn_modeling"],
                outputs=["train_data", "validation_data", "test_data"],
                name="split_churn_data_node",
            ),
            node(
                func=train_churn_models,
                inputs=["train_data", "validation_data", "test_data", "params:churn_modeling"],
                outputs=[
                    "churn_model_metrics",
                    "churn_test_predictions",
                    "churn_feature_importance",
                ],
                name="train_churn_models_node",
            ),
            node(
                func=analyze_survival,
                inputs=[
                    "customer_features",
                    "churn_labels",
                    "customer_segments",
                    "params:churn_modeling",
                ],
                outputs=["survival_curves", "survival_report"],
                name="analyze_survival_node",
            ),
        ]
    )
