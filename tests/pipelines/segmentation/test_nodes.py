"""Tests for the customer segmentation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_clv_churn.pipelines.segmentation.nodes import (
    RULE_SEGMENTS,
    profile_segments,
    segment_customers,
)

SEG_PARAMS = {
    "cluster_features": [
        "recency_days",
        "n_orders",
        "net_revenue",
        "aov",
        "avg_interval",
        "n_unique_products",
        "return_value_ratio",
        "recent_revenue_share",
        "tenure_days",
        "product_diversity",
    ],
    "log_features": ["n_orders", "net_revenue", "aov", "n_unique_products", "tenure_days"],
    "k_range": [2, 3, 4],
    "models": ["kmeans", "gmm", "hierarchical"],
    "random_state": 42,
    "high_return_threshold": 0.5,
    "new_customer_days": 180,
    "strategy_map": {"Champions": "reward", "Lost customers": "win-back"},
}


def _make_features() -> pd.DataFrame:
    rng = np.random.default_rng(7)

    def block(n, start, rec, orders, rev, aov, interval, tenure, ret, recent):
        return pd.DataFrame(
            {
                "customer_id": np.arange(start, start + n),
                "recency_days": rng.normal(rec, 5, n).clip(0),
                "n_orders": rng.normal(orders, 1, n).clip(1),
                "net_revenue": rng.normal(rev, 50, n).clip(1),
                "aov": rng.normal(aov, 5, n).clip(1),
                "avg_interval": rng.normal(interval, 3, n).clip(1),
                "n_unique_products": rng.normal(20, 3, n).clip(1),
                "return_value_ratio": rng.normal(ret, 0.05, n).clip(0, 1),
                "recent_revenue_share": rng.normal(recent, 0.05, n).clip(0, 1),
                "tenure_days": rng.normal(tenure, 10, n).clip(1),
                "product_diversity": rng.normal(0.4, 0.05, n).clip(0, 1),
            }
        )

    return pd.concat(
        [
            block(
                40,
                1,
                rec=10,
                orders=12,
                rev=3000,
                aov=300,
                interval=25,
                tenure=700,
                ret=0.02,
                recent=0.6,
            ),
            block(
                40,
                41,
                rec=250,
                orders=1,
                rev=100,
                aov=100,
                interval=120,
                tenure=300,
                ret=0.05,
                recent=0.0,
            ),
            block(
                40,
                81,
                rec=60,
                orders=4,
                rev=600,
                aov=150,
                interval=60,
                tenure=500,
                ret=0.2,
                recent=0.2,
            ),
        ],
        ignore_index=True,
    )


@pytest.fixture()
def customer_features() -> pd.DataFrame:
    return _make_features()


@pytest.fixture()
def churn_labels(customer_features: pd.DataFrame) -> pd.DataFrame:
    # First block churns less, second block churns most.
    churn = np.where(customer_features["recency_days"] > 100, 1, 0)
    return pd.DataFrame(
        {
            "customer_id": customer_features["customer_id"],
            "churn": churn,
            "future_revenue": np.where(churn == 1, 0.0, 50.0),
        }
    )


@pytest.fixture()
def baseline_scores(customer_features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": customer_features["customer_id"],
            "historical_value_proxy": customer_features["net_revenue"] / 12 * 12,
        }
    )


def test_segment_customers_assigns_all_customers(customer_features: pd.DataFrame) -> None:
    segments, report = segment_customers(customer_features, SEG_PARAMS)
    assert len(segments) == len(customer_features)
    assert set(segments["customer_id"]) == set(customer_features["customer_id"])
    assert segments["segment_name"].notna().all()
    assert set(segments["rule_segment"]).issubset(set(RULE_SEGMENTS))
    assert report["chosen_model"] in SEG_PARAMS["models"]
    assert report["chosen_k"] in SEG_PARAMS["k_range"]
    assert len(report["candidates"]) == 9  # 3 k x 3 models


def test_profile_segments(customer_features, churn_labels, baseline_scores) -> None:
    segments, model_report = segment_customers(customer_features, SEG_PARAMS)
    profiles, rule_profiles, report = profile_segments(
        segments, customer_features, churn_labels, baseline_scores, model_report, SEG_PARAMS
    )

    assert profiles["n_customers"].sum() == len(customer_features)
    assert rule_profiles["n_customers"].sum() == len(customer_features)
    assert {
        "observed_churn_rate",
        "avg_value_proxy",
        "revenue_at_risk_proxy",
        "recommended_strategy",
    } <= set(profiles.columns)
    assert set(rule_profiles["rule_segment"]).issubset(set(RULE_SEGMENTS))
    assert (profiles["observed_churn_rate"].between(0, 1)).all()
    assert (profiles["revenue_at_risk_proxy"] >= 0).all()
    assert len(report["segments"]) == len(profiles)
    assert len(report["rule_based_segments"]) == len(rule_profiles)
    assert report["key_behavioural_characteristics"]
    assert report["chosen_model"] == model_report["chosen_model"]
