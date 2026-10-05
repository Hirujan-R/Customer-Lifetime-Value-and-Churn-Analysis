"""Tests that the segmentation pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.segmentation.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {"segment_customers_node", "profile_segments_node"}
    for output in (
        "customer_segments",
        "segmentation_model_report",
        "segment_profiles",
        "rule_segment_profiles",
        "segmentation_report",
    ):
        assert output in pipe.all_outputs()
