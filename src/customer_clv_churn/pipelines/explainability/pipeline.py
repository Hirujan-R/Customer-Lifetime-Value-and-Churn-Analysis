"""Pipeline for SHAP explainability and risk factors."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import explain_churn_predictions


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``explainability`` pipeline."""
    return pipeline(
        [
            node(
                func=explain_churn_predictions,
                inputs=["churn_model", "test_data", "customer_segments", "params:explainability"],
                outputs=[
                    "shap_global_importance",
                    "customer_risk_factors",
                    "explainability_report",
                ],
                name="explain_churn_predictions_node",
            ),
        ]
    )
