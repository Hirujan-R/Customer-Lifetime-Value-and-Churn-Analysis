"""Tests for the baseline (recency rule) pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_clv_churn.pipelines.baseline.nodes import (
    evaluate_baseline,
    score_baseline_recency,
)

BASELINE = {"inactivity_days": 90}


@pytest.fixture()
def customer_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "recency_days": [39, 190, 8, 525, 220, 20],
            "n_orders": [2, 1, 1, 1, 1, 1],
            "gross_revenue": [150.0, 200.0, 300.0, 400.0, 250.0, 80.0],
            "net_revenue": [130.0, 200.0, 300.0, 400.0, 250.0, 80.0],
            "return_value": [20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "revenue_per_month": [10.0, 20.0, 30.0, 40.0, 25.0, 8.0],
            "country_grouped": ["United Kingdom"] * 6,
        }
    )


@pytest.fixture()
def churn_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "churn": [0, 1, 0, 1, 0, 1],
            "future_revenue": [30.0, 0.0, 70.0, 0.0, 90.0, 0.0],
            "future_orders": [1, 0, 1, 0, 1, 0],
            "cutoff_date": [pd.Timestamp("2011-06-09")] * 6,
            "performance_end_date": [pd.Timestamp("2011-09-07")] * 6,
        }
    )


def test_score_baseline_targets_inactive_customers(customer_features: pd.DataFrame) -> None:
    scores = score_baseline_recency(customer_features, BASELINE).set_index("customer_id")
    assert set(scores.loc[scores["baseline_target"]].index) == {2, 4, 5}
    assert scores.loc[1, "historical_value_proxy"] == pytest.approx(120.0)


def test_evaluate_baseline_metrics(
    customer_features: pd.DataFrame, churn_labels: pd.DataFrame
) -> None:
    scores = score_baseline_recency(customer_features, BASELINE)
    report = evaluate_baseline(scores, churn_labels, BASELINE)

    assert report["n_customers"] == 6
    assert report["n_targeted"] == 3
    assert report["n_actual_churners"] == 3
    assert report["confusion_matrix"] == {
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 2,
    }
    assert report["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["recall"] == pytest.approx(2 / 3, abs=1e-4)
    # Value proxy of the correctly-captured churners (customers 2 and 4): 20*12 + 40*12.
    assert report["potential_revenue_protected_proxy"] == pytest.approx(720.0)
