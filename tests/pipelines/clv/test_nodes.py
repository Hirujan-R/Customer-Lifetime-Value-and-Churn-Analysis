"""Tests for the CLV pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_clv_churn.pipelines.clv.nodes import (
    compute_value_and_risk,
    train_clv_model,
)

COHORTS = ["2011-06-01", "2011-07-01", "2011-08-01", "2011-09-01"]

CLV_PARAMS = {
    "target": "future_revenue",
    "id_columns": ["customer_id", "cohort_date"],
    "categorical_features": ["country_grouped"],
    "exclude_features": ["churn", "future_orders"],
    "models": ["ridge", "random_forest", "lightgbm"],
    "model_params": {
        "ridge": {"alpha": 1.0},
        "random_forest": {"n_estimators": 20, "min_samples_leaf": 5},
        "lightgbm": {"n_estimators": 20, "num_leaves": 15, "objective": "tweedie"},
    },
    "selection_metric": "mae",
    "random_state": 42,
    "horizon_days": 365,
    "discount_rate_annual": 0.0,
    "min_retention_probability": 0.1,
    "mlflow": {"enabled": False},
}


def _make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frames = []
    offset = 0
    for cohort in COHORTS:
        n = 200
        net_revenue = rng.lognormal(5, 1.0, n)
        n_orders = rng.integers(1, 15, n)
        churn = rng.binomial(1, 0.6, n)
        future_revenue = np.where(churn == 1, 0.0, net_revenue * rng.uniform(0.05, 0.5, n))
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": np.arange(offset, offset + n),
                    "cohort_date": pd.Timestamp(cohort),
                    "recency_days": rng.uniform(0, 300, n),
                    "n_orders": n_orders,
                    "net_revenue": net_revenue,
                    "aov": net_revenue / n_orders,
                    "avg_interval": rng.uniform(5, 120, n),
                    "return_value_ratio": rng.uniform(0, 0.5, n),
                    "country_grouped": rng.choice(["United Kingdom", "Germany", "Other"], n),
                    "churn": churn,
                    "future_orders": np.where(churn == 1, 0, 1),
                    "future_revenue": future_revenue,
                }
            )
        )
        offset += n
    return pd.concat(frames, ignore_index=True)


def test_train_clv_model_outputs() -> None:
    panel = _make_panel()
    train = panel[panel["cohort_date"].isin(pd.to_datetime(COHORTS[:2]))]
    validation = panel[panel["cohort_date"] == pd.Timestamp(COHORTS[2])]
    test = panel[panel["cohort_date"] == pd.Timestamp(COHORTS[3])]

    metrics, predictions, importance = train_clv_model(train, validation, test, CLV_PARAMS)

    assert metrics["chosen_model"] in CLV_PARAMS["models"]
    assert {"mae", "rmse", "r2", "spearman"} <= set(
        metrics["test_metrics"][metrics["chosen_model"]]
    )
    assert len(predictions) == len(test)
    assert (predictions["predicted_90d_value"] >= 0).all()
    assert not importance.empty


def test_compute_value_and_risk_formula() -> None:
    n = 5
    customers = np.arange(n)
    cohort = pd.Timestamp("2011-09-01")
    test_data = pd.DataFrame(
        {
            "customer_id": customers,
            "cohort_date": cohort,
            "net_revenue": [100.0, 200.0, 50.0, 400.0, 80.0],
            "n_orders": [2, 3, 1, 5, 1],
        }
    )
    clv_predictions = pd.DataFrame(
        {
            "customer_id": customers,
            "cohort_date": cohort,
            "actual_future_revenue": [0.0, 10.0, 0.0, 40.0, 5.0],
            "predicted_90d_value": [10.0, 40.0, 5.0, 80.0, 20.0],
        }
    )
    churn_predictions = pd.DataFrame(
        {
            "customer_id": customers,
            "cohort_date": cohort,
            "churn_probability": [0.9, 0.4, 0.8, 0.2, 0.6],
        }
    )

    value_risk, summary = compute_value_and_risk(
        test_data, clv_predictions, churn_predictions, CLV_PARAMS
    )

    assert len(value_risk) == n
    assert (value_risk["revenue_at_risk"] >= 0).all()
    assert (value_risk["expected_future_clv"] >= 0).all()
    # Revenue at risk is defined as P(churn) x expected CLV.
    np.testing.assert_allclose(
        value_risk["revenue_at_risk"],
        value_risk["churn_probability"] * value_risk["expected_future_clv"],
        rtol=1e-9,
    )
    # Higher churn probability (all else equal) should not reduce revenue at risk
    # for customers with the same predicted value.
    assert summary["n_customers"] == n
    assert summary["horizon_days"] == 365
    assert summary["value_provenance"]["historical_value"] == "observed"
    assert summary["value_provenance"]["expected_future_clv"] == "modelled"
    assert summary["total_revenue_at_risk"] > 0
