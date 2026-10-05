"""Pipeline for retention recommendations."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_recommendations


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``recommendations`` pipeline."""
    return pipeline(
        [
            node(
                func=build_recommendations,
                inputs=[
                    "retention_priorities",
                    "customer_risk_factors",
                    "params:recommendations",
                ],
                outputs=["customer_recommendations", "recommendations_report"],
                name="build_recommendations_node",
            ),
        ]
    )
