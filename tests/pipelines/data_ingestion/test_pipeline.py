"""Tests that the data_ingestion pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.data_ingestion.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {
        "download_online_retail_ii_node",
        "load_raw_transactions_node",
        "clean_transactions_node",
        "validate_transactions_node",
        "summarise_transactions_node",
    }
    assert "transactions" in pipe.all_outputs()
    assert "transactions_raw" in pipe.all_outputs()
    assert "data_quality_report" in pipe.all_outputs()
    assert "transactions_summary" in pipe.all_outputs()
