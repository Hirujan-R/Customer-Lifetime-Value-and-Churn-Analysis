"""Tests for the prioritisation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_clv_churn.pipelines.prioritisation.nodes import build_prioritisation

PARAMS = {"budgets": [10, 25, 50], "high_value_quantile": 0.75, "recency_threshold_days": 90}


def _make_inputs():
    rng = np.random.default_rng(5)
    n = 100
    customers = np.arange(n)
    cohort = pd.Timestamp("2011-09-01")
    churn_probability = rng.uniform(0.05, 0.95, n)
    expected_clv = rng.lognormal(6, 1.0, n)
    # Deliberately make recency weakly related to value.
    recency = rng.uniform(0, 400, n)
    value_risk = pd.DataFrame(
        {
            "customer_id": customers,
            "cohort_date": cohort,
            "historical_value": rng.uniform(0, 5000, n),
            "churn_probability": churn_probability,
            "expected_future_clv": expected_clv,
            "revenue_at_risk": churn_probability * expected_clv,
            "actual_future_revenue": np.where(rng.uniform(0, 1, n) < churn_probability, 0.0, 100.0),
        }
    )
    test_data = pd.DataFrame(
        {
            "customer_id": customers,
            "cohort_date": cohort,
            "recency_days": recency,
            "churn": rng.integers(0, 2, n),
        }
    )
    segments = pd.DataFrame(
        {"customer_id": customers, "segment_name": rng.choice(["A", "B", "C"], n)}
    )
    return value_risk, test_data, segments


def test_prioritisation_ranks_by_revenue_at_risk() -> None:
    value_risk, test_data, segments = _make_inputs()
    priorities, report = build_prioritisation(value_risk, test_data, segments, PARAMS)

    assert len(priorities) == 100
    assert priorities["priority_rank"].iloc[0] == 1
    assert priorities["revenue_at_risk"].iloc[0] == priorities["revenue_at_risk"].max()
    assert priorities["target_top_10"].sum() == 10
    assert priorities["target_top_25"].sum() == 25
    assert {"Low", "Medium", "High"} >= set(priorities["risk_level"])


def test_prioritisation_report_structure() -> None:
    value_risk, test_data, segments = _make_inputs()
    _, report = build_prioritisation(value_risk, test_data, segments, PARAMS)

    assert set(report["strategies"]) == {
        "recency_rule",
        "churn_probability",
        "expected_clv",
        "revenue_at_risk",
    }
    top10 = report["strategies"]["revenue_at_risk"]["by_budget"]["10"]
    expected = float(value_risk["revenue_at_risk"].nlargest(10).sum())
    assert top10["revenue_at_risk_captured"] == round(expected, 2)
    assert report["headline"]["at_budget"] == 10
    assert "recency_rule_full" in report
    assert report["recommended_strategy"] == "revenue_at_risk"
