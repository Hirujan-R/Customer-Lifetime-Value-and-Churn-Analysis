"""Tests that the data_quality pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.data_quality.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {
        "curate_transactions_node",
        "validate_curated_transactions_node",
    }
    assert "transactions_curated" in pipe.all_outputs()
    assert "feature_quality_report" in pipe.all_outputs()
