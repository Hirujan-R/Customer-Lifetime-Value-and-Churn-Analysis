"""Pipeline implementing and evaluating the naive recency baseline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_baseline, score_baseline_recency


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``baseline`` pipeline."""
    return pipeline(
        [
            node(
                func=score_baseline_recency,
                inputs=["customer_features", "params:baseline"],
                outputs="baseline_scores",
                name="score_baseline_recency_node",
            ),
            node(
                func=evaluate_baseline,
                inputs=["baseline_scores", "churn_labels", "params:baseline"],
                outputs="baseline_report",
                name="evaluate_baseline_node",
            ),
        ]
    )
