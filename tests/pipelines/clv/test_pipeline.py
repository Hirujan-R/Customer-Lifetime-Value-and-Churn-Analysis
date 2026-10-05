"""Tests that the CLV pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.clv.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {"train_clv_model_node", "compute_value_and_risk_node"}
    for output in (
        "clv_model_metrics",
        "clv_test_predictions",
        "clv_feature_importance",
        "customer_value_risk",
        "revenue_at_risk_summary",
    ):
        assert output in pipe.all_outputs()
