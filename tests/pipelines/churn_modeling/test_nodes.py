"""Tests for the churn modelling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_clv_churn.pipelines.churn_modeling.nodes import (
    analyze_survival,
    build_churn_model_input,
    split_churn_data,
    train_churn_models,
)

COHORTS = [
    "2011-03-01",
    "2011-04-01",
    "2011-05-01",
    "2011-06-01",
    "2011-07-01",
    "2011-08-01",
    "2011-09-01",
]

PARAMS = {
    "cohort_dates": COHORTS,
    "train_cohorts": COHORTS[:5],
    "validation_cohorts": [COHORTS[5]],
    "test_cohorts": [COHORTS[6]],
    "target": "churn",
    "id_columns": ["customer_id", "cohort_date"],
    "categorical_features": ["country_grouped"],
    "exclude_features": ["future_revenue", "future_orders"],
    "models": ["logistic_regression", "random_forest", "lightgbm"],
    "model_params": {
        "logistic_regression": {"max_iter": 500},
        "random_forest": {"n_estimators": 30, "min_samples_leaf": 5},
        "lightgbm": {"n_estimators": 30, "num_leaves": 15},
    },
    "calibration_method": "isotonic",
    "random_state": 42,
    "survival_features": ["recency_days", "n_orders", "net_revenue", "return_value_ratio"],
    "mlflow": {"enabled": False},
}


def _make_model_input() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    customer_offset = 0
    for cohort in COHORTS:
        n = 200
        recency = rng.uniform(0, 300, n)
        n_orders = rng.integers(1, 15, n)
        net_revenue = rng.lognormal(5, 1.0, n)
        feature_frame = pd.DataFrame(
            {
                "customer_id": np.arange(customer_offset, customer_offset + n),
                "cohort_date": pd.Timestamp(cohort),
                "recency_days": recency,
                "n_orders": n_orders,
                "net_revenue": net_revenue,
                "aov": net_revenue / n_orders,
                "avg_interval": rng.uniform(5, 120, n),
                "n_unique_products": rng.integers(1, 40, n),
                "return_value_ratio": rng.uniform(0, 0.5, n),
                "recent_revenue_share": rng.uniform(0, 1, n),
                "tenure_days": rng.uniform(30, 700, n),
                "product_diversity": rng.uniform(0.1, 0.8, n),
                "country_grouped": rng.choice(["United Kingdom", "Germany", "France", "Other"], n),
            }
        )
        customer_offset += n
        logit = -0.01 * recency + 0.12 * n_orders - 0.0002 * net_revenue
        probability = 1 / (1 + np.exp(-logit))
        feature_frame["churn"] = rng.binomial(1, probability)
        feature_frame["future_revenue"] = np.where(feature_frame["churn"] == 1, 0.0, 100.0)
        feature_frame["future_orders"] = np.where(feature_frame["churn"] == 1, 0, 1)
        frames.append(feature_frame)
    return pd.concat(frames, ignore_index=True)


def test_split_churn_data_uses_time_cohorts() -> None:
    model_input = _make_model_input()
    train, validation, test = split_churn_data(model_input, PARAMS)
    assert set(train["cohort_date"]) == set(pd.to_datetime(PARAMS["train_cohorts"]))
    assert set(validation["cohort_date"]) == set(pd.to_datetime(PARAMS["validation_cohorts"]))
    assert set(test["cohort_date"]) == set(pd.to_datetime(PARAMS["test_cohorts"]))
    assert len(train) + len(validation) + len(test) == len(model_input)


def test_train_churn_models_outputs() -> None:
    model_input = _make_model_input()
    train, validation, test = split_churn_data(model_input, PARAMS)
    metrics, predictions, importance = train_churn_models(train, validation, test, PARAMS)

    assert metrics["chosen_model"] in PARAMS["models"]
    assert 0.0 <= metrics["final_test_metrics"]["roc_auc"] <= 1.0
    assert 0.0 <= metrics["final_test_metrics"]["brier"] <= 1.0
    assert isinstance(metrics["calibration_applied"], bool)
    assert len(predictions) == len(test)
    assert predictions["churn_probability"].between(0, 1).all()
    assert set(predictions["churn_prediction"]).issubset({0, 1})
    assert not importance.empty
    assert {"top_10pct", "top_20pct"} <= set(metrics["business_metrics"])
    assert {"test_brier_raw", "test_brier_calibrated"} <= set(metrics["calibration_effect"])


def _make_transactions() -> pd.DataFrame:
    def line(customer, invoice, date, revenue, cancel=False):
        return {
            "customer_id": customer,
            "invoice_no": invoice,
            "invoice_date": pd.Timestamp(date),
            "stock_code": "85123A",
            "quantity": -1 if cancel else 1,
            "unit_price": revenue,
            "line_revenue": -revenue if cancel else revenue,
            "is_cancellation": cancel,
            "line_revenue_gross": 0.0 if cancel else revenue,
            "return_value": revenue if cancel else 0.0,
            "country_grouped": "United Kingdom",
            "invoice_month": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
        }

    rows = []
    for customer in range(1, 6):
        rows.append(line(customer, f"{customer}A", "2011-01-01", 100.0))
        rows.append(line(customer, f"{customer}B", "2011-05-01", 50.0))
    # performance purchases for some customers around the first cohort
    rows.append(line(1, "1C", "2011-07-01", 40.0))
    return pd.DataFrame(rows)


def test_build_churn_model_input_multi_cohort() -> None:
    transactions = _make_transactions()
    temporal = {"performance_window_days": 90, "recent_windows": [30, 90]}
    params = {"cohort_dates": ["2011-06-01", "2011-07-01"]}
    model_input = build_churn_model_input(transactions, temporal, params)
    assert set(model_input["cohort_date"]) == {
        pd.Timestamp("2011-06-01"),
        pd.Timestamp("2011-07-01"),
    }
    assert {"churn", "future_revenue", "country_grouped"} <= set(model_input.columns)


def test_analyze_survival() -> None:
    rng = np.random.default_rng(1)
    n = 200
    features = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_days": rng.uniform(30, 700, n),
            "recency_days": rng.uniform(0, 300, n),
            "n_orders": rng.integers(1, 15, n),
            "net_revenue": rng.lognormal(5, 1, n),
            "aov": rng.uniform(50, 500, n),
            "return_value_ratio": rng.uniform(0, 0.5, n),
            "avg_interval": rng.uniform(5, 120, n),
        }
    )
    labels = pd.DataFrame({"customer_id": np.arange(n), "churn": rng.integers(0, 2, n)})
    segments = pd.DataFrame(
        {"customer_id": np.arange(n), "segment_name": rng.choice(["A", "B"], n)}
    )
    curves, report = analyze_survival(features, labels, segments, PARAMS)
    assert not curves.empty
    assert {"time_days", "survival", "segment"} <= set(curves.columns)
    assert "concordance_index" in report
    assert report["n_customers"] == n
