"""Pipeline for retention prioritisation."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_prioritisation


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``prioritisation`` pipeline."""
    return pipeline(
        [
            node(
                func=build_prioritisation,
                inputs=[
                    "customer_value_risk",
                    "test_data",
                    "customer_segments",
                    "params:prioritisation",
                ],
                outputs=["retention_priorities", "prioritisation_report"],
                name="build_prioritisation_node",
            ),
        ]
    )
