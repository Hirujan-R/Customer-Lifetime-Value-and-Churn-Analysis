"""Tests that the churn_modeling pipeline is wired correctly."""

from __future__ import annotations

from customer_clv_churn.pipelines.churn_modeling.pipeline import create_pipeline


def test_pipeline_has_expected_nodes_and_outputs() -> None:
    pipe = create_pipeline()
    node_names = {node.name for node in pipe.nodes}
    assert node_names == {
        "build_churn_model_input_node",
        "split_churn_data_node",
        "train_churn_models_node",
        "analyze_survival_node",
    }
    for output in (
        "churn_model_input",
        "train_data",
        "validation_data",
        "test_data",
        "churn_model_metrics",
        "churn_test_predictions",
        "churn_feature_importance",
        "churn_model",
        "survival_curves",
        "survival_report",
    ):
        assert output in pipe.all_outputs()
