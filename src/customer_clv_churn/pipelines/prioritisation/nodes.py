"""Retention prioritisation engine.

The central business insight: the customer with the highest probability of
churn is **not** necessarily the one to prioritise. A £50 customer at 90% risk
(£45 at risk) matters far less than a £5,000 customer at 55% risk (£2,750 at
risk). The engine therefore ranks by **expected revenue at risk** and compares
that ranking against naive alternatives (recency rule, probability-only,
value-only) at realistic targeting budgets.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STRATEGY_SCORES = {
    "recency_rule": "recency_days",  # larger = more likely to have lapsed
    "churn_probability": "churn_probability",
    "expected_clv": "expected_future_clv",
    "revenue_at_risk": "revenue_at_risk",
}


def build_prioritisation(
    customer_value_risk: pd.DataFrame,
    test_data: pd.DataFrame,
    customer_segments: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank customers by expected revenue at risk and compare strategies.

    Args:
        customer_value_risk: Output of the CLV pipeline.
        test_data: Held-out cohort (recency, observed churn, future revenue).
        customer_segments: Segment assignments.
        params: Prioritisation config (``budgets``, ``high_value_quantile``).

    Returns:
        ``(retention_priorities, prioritisation_report)``.
    """
    df = customer_value_risk.merge(
        test_data[["customer_id", "cohort_date", "recency_days", "churn"]],
        on=["customer_id", "cohort_date"],
        how="left",
    ).merge(
        customer_segments[["customer_id", "segment_name"]],
        on="customer_id",
        how="left",
    )
    df["segment_name"] = df["segment_name"].fillna("Unknown")

    df["priority_rank"] = df["revenue_at_risk"].rank(ascending=False, method="first").astype(int)
    df["risk_level"] = pd.cut(
        df["churn_probability"],
        bins=[-0.01, 0.34, 0.67, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    budgets = [int(b) for b in params.get("budgets", [100, 250, 500, 1000])]
    for budget in budgets:
        df[f"target_top_{budget}"] = (df["priority_rank"] <= budget).astype(int)

    df = df.sort_values("priority_rank").reset_index(drop=True)
    report = _compare_strategies(df, budgets, params)

    logger.info(
        "Prioritisation: %s customers ranked by revenue at risk | total £%s",
        f"{len(df):,}",
        f"{df['revenue_at_risk'].sum():,.0f}",
    )
    return df, report


def _metrics_at_k(
    df: pd.DataFrame,
    scores: pd.Series,
    k: int,
    high_value: pd.Series,
) -> dict[str, Any]:
    order = np.argsort(-scores.to_numpy())[: min(k, len(df))]
    selected = df.iloc[order]
    high_value_count = int(high_value.iloc[order].sum())
    return {
        "n_targeted": int(len(selected)),
        "revenue_at_risk_captured": round(float(selected["revenue_at_risk"].sum()), 2),
        "expected_clv_captured": round(float(selected["expected_future_clv"].sum()), 2),
        "observed_future_revenue_captured": round(
            float(selected["actual_future_revenue"].sum()), 2
        ),
        "actual_churners_captured": int(selected["churn"].sum()),
        "actual_churn_precision": round(float(selected["churn"].mean()), 4),
        "high_value_customers_captured": high_value_count,
    }


def _compare_strategies(
    df: pd.DataFrame,
    budgets: list[int],
    params: dict[str, Any],
) -> dict[str, Any]:
    high_value_quantile = float(params.get("high_value_quantile", 0.75))
    threshold = df["expected_future_clv"].quantile(high_value_quantile)
    high_value = df["expected_future_clv"] >= threshold

    totals = {
        "n_customers": int(len(df)),
        "total_revenue_at_risk": round(float(df["revenue_at_risk"].sum()), 2),
        "total_expected_clv": round(float(df["expected_future_clv"].sum()), 2),
        "total_actual_churners": int(df["churn"].sum()),
        "total_observed_future_revenue": round(float(df["actual_future_revenue"].sum()), 2),
        "high_value_threshold_clv": round(float(threshold), 2),
        "total_high_value_customers": int(high_value.sum()),
    }

    results: dict[str, Any] = {"budgets": budgets, "totals": totals, "strategies": {}}
    fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    for strategy, column in STRATEGY_SCORES.items():
        scores = df[column]
        budget_metrics = {str(b): _metrics_at_k(df, scores, b, high_value) for b in budgets}
        curve = {
            str(f): _metrics_at_k(df, scores, max(1, int(len(df) * f)), high_value)[
                "revenue_at_risk_captured"
            ]
            for f in fractions
        }
        results["strategies"][strategy] = {
            "description": {
                "recency_rule": "rank by days since last purchase (naive)",
                "churn_probability": "rank by churn probability only",
                "expected_clv": "rank by expected CLV only",
                "revenue_at_risk": "rank by P(churn) x expected CLV (recommended)",
            }[strategy],
            "by_budget": budget_metrics,
            "capture_curve": curve,
        }

    # The full recency rule (target everyone inactive > 90 days) for reference.
    recency_targets = df["recency_days"] > int(params.get("recency_threshold_days", 90))
    results["recency_rule_full"] = {
        "n_targeted": int(recency_targets.sum()),
        "coverage": round(float(recency_targets.mean()), 4),
        "revenue_at_risk_captured": round(
            float(df.loc[recency_targets, "revenue_at_risk"].sum()), 2
        ),
        "actual_churners_captured": int(df.loc[recency_targets, "churn"].sum()),
        "actual_churn_precision": round(float(df.loc[recency_targets, "churn"].mean()), 4),
        "high_value_customers_captured": int(high_value[recency_targets].sum()),
    }

    # Headline uplift: revenue at risk captured by the recommended ranking vs
    # the naive recency ranking at the smallest budget.
    sample_budget = str(min(budgets))
    recommended = results["strategies"]["revenue_at_risk"]["by_budget"][sample_budget]
    naive = results["strategies"]["recency_rule"]["by_budget"][sample_budget]
    uplift = recommended["revenue_at_risk_captured"] - naive["revenue_at_risk_captured"]
    results["recommended_strategy"] = "revenue_at_risk"
    results["headline"] = {
        "at_budget": int(min(budgets)),
        "recommended_revenue_at_risk_captured": recommended["revenue_at_risk_captured"],
        "naive_revenue_at_risk_captured": naive["revenue_at_risk_captured"],
        "absolute_uplift": round(float(uplift), 2),
        "relative_uplift": round(float(uplift / naive["revenue_at_risk_captured"]), 4)
        if naive["revenue_at_risk_captured"]
        else None,
    }
    return results
