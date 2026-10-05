"""Pipeline for feature-quality curation of the cleaned transactions."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import curate_transactions, validate_curated_transactions


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``data_quality`` pipeline."""
    return pipeline(
        [
            node(
                func=curate_transactions,
                inputs=[
                    "transactions",
                    "params:data_quality.residual_admin_stock_codes",
                    "params:data_quality.winsorise",
                    "params:data_quality.country_grouping",
                ],
                outputs="transactions_curated",
                name="curate_transactions_node",
            ),
            node(
                func=validate_curated_transactions,
                inputs=["transactions_curated", "params:data_quality.expected"],
                outputs="feature_quality_report",
                name="validate_curated_transactions_node",
            ),
        ]
    )
