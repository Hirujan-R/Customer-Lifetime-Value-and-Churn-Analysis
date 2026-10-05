"""Pipeline building the customer feature store and churn labels."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_churn_labels, build_customer_features


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``feature_engineering`` pipeline."""
    return pipeline(
        [
            node(
                func=build_customer_features,
                inputs=["transactions_curated", "params:temporal"],
                outputs="customer_features",
                name="build_customer_features_node",
            ),
            node(
                func=build_churn_labels,
                inputs=["transactions_curated", "params:temporal"],
                outputs="churn_labels",
                name="build_churn_labels_node",
            ),
        ]
    )
