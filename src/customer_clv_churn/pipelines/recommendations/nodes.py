"""Retention recommendations.

Maps each customer's segment and dominant SHAP risk factor to a recommended
retention action, then aggregates actions into an executive list ranked by
expected revenue at risk.

Observational caveat: these are hypotheses for experimentation. The dataset
cannot estimate treatment effects — A/B tests are required before claiming any
recovered revenue.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_recommendations(
    retention_priorities: pd.DataFrame,
    customer_risk_factors: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign recommended retention actions and aggregate executive actions.

    Args:
        retention_priorities: Output of the prioritisation engine.
        customer_risk_factors: Per-customer SHAP risk factors.
        params: Config with ``segment_actions``, ``risk_factor_actions`` and
            ``default_action``.

    Returns:
        ``(customer_recommendations, recommendations_report)``.
    """
    segment_actions: dict[str, str] = params.get("segment_actions", {})
    risk_factor_actions: dict[str, str] = params.get("risk_factor_actions", {})
    default_action = str(
        params.get("default_action", "Include in the next retention test; monitor.")
    )

    dominant = (
        customer_risk_factors.loc[
            customer_risk_factors["rank"] == 1, ["customer_id", "feature", "direction"]
        ]
        .rename(columns={"feature": "dominant_risk_factor", "direction": "dominant_direction"})
        .drop_duplicates("customer_id")
    )

    data = retention_priorities.merge(dominant, on="customer_id", how="left")
    data["dominant_risk_factor"] = data["dominant_risk_factor"].fillna("unknown")
    data["dominant_direction"] = data["dominant_direction"].fillna("unknown")

    def _action(row: pd.Series) -> tuple[str, str]:
        factor_action = risk_factor_actions.get(row["dominant_risk_factor"])
        segment_action = segment_actions.get(row["segment_name"])
        if factor_action:
            return factor_action, f"dominant risk factor: {row['dominant_risk_factor']}"
        if segment_action:
            return segment_action, f"segment: {row['segment_name']}"
        return default_action, "no specific signal"

    actions = data.apply(_action, axis=1, result_type="expand")
    data["primary_action"] = actions[0]
    data["rationale"] = actions[1]

    columns = [
        "customer_id",
        "segment_name",
        "risk_level",
        "priority_rank",
        "churn_probability",
        "expected_future_clv",
        "revenue_at_risk",
        "dominant_risk_factor",
        "dominant_direction",
        "primary_action",
        "rationale",
    ]
    customer_recommendations = (
        data[columns].sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)
    )

    # --- Executive aggregation by action ----------------------------------
    grouped = customer_recommendations.groupby("primary_action")
    action_rows = []
    for action, group in grouped:
        top_segments = (
            group.groupby("segment_name")["customer_id"]
            .count()
            .sort_values(ascending=False)
            .head(3)
        )
        action_rows.append(
            {
                "action": action,
                "n_customers": int(len(group)),
                "total_expected_clv": round(float(group["expected_future_clv"].sum()), 2),
                "total_revenue_at_risk": round(float(group["revenue_at_risk"].sum()), 2),
                "mean_churn_probability": round(float(group["churn_probability"].mean()), 4),
                "top_segments": [
                    {"segment": s, "customers": int(c)} for s, c in top_segments.items()
                ],
            }
        )
    action_rows.sort(key=lambda r: r["total_revenue_at_risk"], reverse=True)

    report: dict[str, Any] = {
        "n_customers": int(len(customer_recommendations)),
        "actions": action_rows,
        "causal_caveat": (
            "The dataset is observational: these are prioritised hypotheses. "
            "A/B testing or causal experimentation is required to estimate the "
            "actual treatment effect of each intervention."
        ),
    }

    logger.info(
        "Recommendations: %s customers mapped to %s distinct actions",
        f"{len(customer_recommendations):,}",
        len(action_rows),
    )
    return customer_recommendations, report
