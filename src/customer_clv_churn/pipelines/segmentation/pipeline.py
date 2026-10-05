"""Pipeline for customer segmentation and profiling."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import profile_segments, segment_customers


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``segmentation`` pipeline."""
    return pipeline(
        [
            node(
                func=segment_customers,
                inputs=["customer_features", "params:segmentation"],
                outputs=["customer_segments", "segmentation_model_report"],
                name="segment_customers_node",
            ),
            node(
                func=profile_segments,
                inputs=[
                    "customer_segments",
                    "customer_features",
                    "churn_labels",
                    "baseline_scores",
                    "segmentation_model_report",
                    "params:segmentation",
                ],
                outputs=["segment_profiles", "rule_segment_profiles", "segmentation_report"],
                name="profile_segments_node",
            ),
        ]
    )
