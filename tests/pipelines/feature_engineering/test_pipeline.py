"""Tests that the feature_engineering pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.feature_engineering.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {"build_customer_features_node", "build_churn_labels_node"}
    assert "customer_features" in pipe.all_outputs()
    assert "churn_labels" in pipe.all_outputs()
