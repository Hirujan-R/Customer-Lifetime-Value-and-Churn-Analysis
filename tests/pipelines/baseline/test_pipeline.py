"""Tests that the baseline pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.baseline.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {"score_baseline_recency_node", "evaluate_baseline_node"}
    assert "baseline_scores" in pipe.all_outputs()
    assert "baseline_report" in pipe.all_outputs()
