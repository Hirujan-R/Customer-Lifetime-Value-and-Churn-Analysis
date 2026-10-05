"""Pipeline wiring tests for prioritisation, explainability and recommendations."""

from __future__ import annotations

from customer_clv_churn.pipelines.explainability.pipeline import create_pipeline as explain_pipeline
from customer_clv_churn.pipelines.prioritisation.pipeline import (
    create_pipeline as prioritisation_pipeline,
)
from customer_clv_churn.pipelines.recommendations.pipeline import (
    create_pipeline as recommendations_pipeline,
)


def test_prioritisation_pipeline_outputs() -> None:
    outputs = prioritisation_pipeline().all_outputs()
    assert {"retention_priorities", "prioritisation_report"} <= outputs


def test_explainability_pipeline_outputs() -> None:
    outputs = explain_pipeline().all_outputs()
    assert {"shap_global_importance", "customer_risk_factors", "explainability_report"} <= outputs


def test_recommendations_pipeline_outputs() -> None:
    outputs = recommendations_pipeline().all_outputs()
    assert {"customer_recommendations", "recommendations_report"} <= outputs
