"""Tests for retention recommendations."""

from __future__ import annotations

import pandas as pd

from customer_clv_churn.pipelines.recommendations.nodes import build_recommendations

PARAMS = {
    "segment_actions": {"Champions": "Reward loyalty.", "New customers": "Onboarding."},
    "risk_factor_actions": {"return_value_ratio": "Investigate dissatisfaction."},
    "default_action": "Monitor.",
}


def test_build_recommendations() -> None:
    priorities = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "segment_name": ["Champions", "New customers", "Unknown"],
            "risk_level": ["Low", "High", "Medium"],
            "priority_rank": [3, 1, 2],
            "churn_probability": [0.2, 0.8, 0.5],
            "expected_future_clv": [100.0, 500.0, 50.0],
            "revenue_at_risk": [20.0, 400.0, 25.0],
        }
    )
    risk_factors = pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "feature": [
                "recency_days",
                "n_orders",
                "aov",
                "return_value_ratio",
                "n_orders",
                "aov",
                "n_orders",
                "recency_days",
                "aov",
            ],
            "shap_value": [0.1] * 9,
            "abs_shap": [0.3, 0.2, 0.1] * 3,
            "direction": ["increases_risk"] * 9,
        }
    )

    recommendations, report = build_recommendations(priorities, risk_factors, PARAMS)

    assert len(recommendations) == 3
    # Customer 2's dominant factor (return_value_ratio) overrides its segment action.
    customer2 = recommendations.set_index("customer_id").loc[2]
    assert customer2["primary_action"] == "Investigate dissatisfaction."
    # Customer 1 has no mapped risk factor -> falls back to segment action.
    customer1 = recommendations.set_index("customer_id").loc[1]
    assert customer1["primary_action"] == "Reward loyalty."
    assert "causal" in report["causal_caveat"].lower()
    assert (
        report["actions"][0]["total_revenue_at_risk"]
        >= report["actions"][-1]["total_revenue_at_risk"]
    )
