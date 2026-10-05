"""Tests for project-level pipeline registration."""

from __future__ import annotations

from kedro.framework.project import configure_project

from customer_clv_churn.pipeline_registry import register_pipelines

configure_project("customer_clv_churn")


def test_data_ingestion_pipeline_is_registered() -> None:
    pipelines = register_pipelines()
    assert {"data_ingestion", "data_quality", "feature_engineering", "baseline"} <= set(pipelines)
    assert "__default__" in pipelines
    assert "transactions" in pipelines["data_ingestion"].all_outputs()
    assert "transactions_curated" in pipelines["data_quality"].all_outputs()
    assert "customer_features" in pipelines["feature_engineering"].all_outputs()
    assert "baseline_report" in pipelines["baseline"].all_outputs()
