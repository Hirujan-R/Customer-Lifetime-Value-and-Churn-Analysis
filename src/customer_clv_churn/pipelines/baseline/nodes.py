"""The current / naive company strategy: a 90-day recency rule.

This is the baseline the ML strategy must beat. It is deliberately simple:

    "Any customer who has not purchased in more than N days is treated as at
     risk and receives a generic retention intervention."

The module produces per-customer targeting scores and an evaluation report so
the ML ranking can be compared against it later.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def score_baseline_recency(
    customer_features: pd.DataFrame,
    baseline: dict[str, Any],
) -> pd.DataFrame:
    """Apply the recency rule and attach a historical-value proxy.

    Args:
        customer_features: Output of ``build_customer_features``.
        baseline: Config with ``inactivity_days``.

    Returns:
        One row per customer with a ``baseline_target`` flag and value proxy.

    Note:
        ``historical_value_proxy`` annualises the observed monthly net revenue.
        It is an explicit **proxy** for future CLV until the dedicated CLV model
        is built, and is labelled as such everywhere.
    """
    inactivity = int(baseline["inactivity_days"])

    columns = [
        "customer_id",
        "recency_days",
        "n_orders",
        "gross_revenue",
        "net_revenue",
        "return_value",
        "revenue_per_month",
        "country_grouped",
    ]
    scores = customer_features[columns].copy()
    scores["baseline_target"] = scores["recency_days"] > inactivity
    scores["historical_value_proxy"] = scores["revenue_per_month"] * 12

    logger.info(
        "Baseline recency rule (inactive > %s days): targets %s of %s customers (%.1f%%)",
        inactivity,
        f"{int(scores['baseline_target'].sum()):,}",
        f"{len(scores):,}",
        scores["baseline_target"].mean() * 100,
    )
    return scores


def evaluate_baseline(
    baseline_scores: pd.DataFrame,
    churn_labels: pd.DataFrame,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the baseline rule against the temporally-held-out churn labels.

    Args:
        baseline_scores: Output of :func:`score_baseline_recency`.
        churn_labels: Output of ``build_churn_labels``.
        baseline: Config with ``inactivity_days``.

    Returns:
        JSON-serialisable evaluation report.
    """
    inactivity = int(baseline["inactivity_days"])
    df = baseline_scores.merge(
        churn_labels[["customer_id", "churn", "future_revenue", "future_orders"]],
        on="customer_id",
        how="inner",
    )

    targeted = df["baseline_target"]
    churned = df["churn"].eq(1)
    value = df["historical_value_proxy"]

    true_positives = int((targeted & churned).sum())
    false_positives = int((targeted & ~churned).sum())
    false_negatives = int((~targeted & churned).sum())
    true_negatives = int((~targeted & ~churned).sum())

    n_customers = len(df)
    n_targeted = int(targeted.sum())
    n_churners = int(churned.sum())
    precision = true_positives / n_targeted if n_targeted else 0.0
    recall = true_positives / n_churners if n_churners else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    report: dict[str, Any] = {
        "strategy": "baseline_recency_rule",
        "rule": f"recency_days > {inactivity}",
        "value_proxy_definition": "12 x observed monthly net revenue (annualised run-rate)",
        "cutoff_date": str(churn_labels["cutoff_date"].iloc[0].date()),
        "performance_end_date": str(churn_labels["performance_end_date"].iloc[0].date()),
        "n_customers": n_customers,
        "n_targeted": n_targeted,
        "targeted_share": round(n_targeted / n_customers, 4) if n_customers else 0.0,
        "n_actual_churners": n_churners,
        "overall_churn_rate": round(n_churners / n_customers, 4) if n_customers else 0.0,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
        },
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "historical_revenue_targeted": round(float(df.loc[targeted, "gross_revenue"].sum()), 2),
        "future_value_proxy_targeted": round(float(value.loc[targeted].sum()), 2),
        "future_value_proxy_all_churners": round(float(value.loc[churned].sum()), 2),
        "potential_revenue_protected_proxy": round(float(value.loc[targeted & churned].sum()), 2),
        "observed_future_revenue_targeted": round(
            float(df.loc[targeted, "future_revenue"].sum()), 2
        ),
    }

    logger.info(
        "Baseline evaluation: precision=%.3f recall=%.3f f1=%.3f | potential protected (proxy)=£%s",
        precision,
        recall,
        f1,
        f"{report['potential_revenue_protected_proxy']:,.0f}",
    )
    return report
