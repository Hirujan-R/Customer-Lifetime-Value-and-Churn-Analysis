"""Tests for SHAP explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_clv_churn.pipelines.churn_modeling.nodes import _build_estimator
from customer_clv_churn.pipelines.explainability.nodes import explain_churn_predictions

EXPLAIN_PARAMS = {
    "id_columns": ["customer_id", "cohort_date"],
    "target": "churn",
    "exclude_features": ["future_revenue", "future_orders"],
    "top_factors": 3,
}


def _make_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recency = rng.uniform(0, 300, n)
    n_orders = rng.integers(1, 15, n)
    net_revenue = rng.lognormal(5, 1, n)
    churn = (recency > 120).astype(int)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "cohort_date": pd.Timestamp("2011-09-01"),
            "recency_days": recency,
            "n_orders": n_orders,
            "net_revenue": net_revenue,
            "aov": net_revenue / n_orders,
            "return_value_ratio": rng.uniform(0, 0.5, n),
            "country_grouped": rng.choice(["United Kingdom", "Germany", "Other"], n),
            "churn": churn,
            "future_revenue": np.where(churn == 1, 0.0, 100.0),
            "future_orders": np.where(churn == 1, 0, 1),
        }
    )


def _fit_model(train: pd.DataFrame):
    model_params = {"random_state": 42, "model_params": {"random_forest": {"n_estimators": 25}}}
    numeric = ["recency_days", "n_orders", "net_revenue", "aov", "return_value_ratio"]
    model = _build_estimator("random_forest", numeric, ["country_grouped"], model_params)
    feature_columns = [
        c
        for c in train.columns
        if c not in {"customer_id", "cohort_date", "churn", "future_revenue", "future_orders"}
    ]
    model.fit(train[feature_columns], train["churn"])
    return model


def test_explain_churn_predictions() -> None:
    train = _make_data(150, seed=1)
    test = _make_data(80, seed=2)
    segments = pd.DataFrame(
        {"customer_id": test["customer_id"], "segment_name": np.where(test["churn"] == 1, "A", "B")}
    )
    model = _fit_model(train)

    importance, risk_factors, report = explain_churn_predictions(
        model, test, segments, EXPLAIN_PARAMS
    )

    assert not importance.empty
    assert importance["mean_abs_shap"].iloc[0] == importance["mean_abs_shap"].max()
    # One top-3 factor per customer.
    assert len(risk_factors) == len(test) * 3
    assert set(risk_factors["direction"]).issubset({"increases_risk", "decreases_risk"})
    assert report["n_customers_explained"] == len(test)
    assert "segment_risk_factors" in report
    assert "causal" in report["interpretation_note"].lower()
