"""Pipeline for CLV modelling and revenue-at-risk."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_value_and_risk, train_clv_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``clv`` pipeline."""
    return pipeline(
        [
            node(
                func=train_clv_model,
                inputs=["train_data", "validation_data", "test_data", "params:clv"],
                outputs=["clv_model_metrics", "clv_test_predictions", "clv_feature_importance"],
                name="train_clv_model_node",
            ),
            node(
                func=compute_value_and_risk,
                inputs=[
                    "test_data",
                    "clv_test_predictions",
                    "churn_test_predictions",
                    "params:clv",
                ],
                outputs=["customer_value_risk", "revenue_at_risk_summary"],
                name="compute_value_and_risk_node",
            ),
        ]
    )
