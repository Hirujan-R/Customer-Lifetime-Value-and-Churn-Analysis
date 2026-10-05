"""Tests for customer feature engineering and churn labelling."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_clv_churn.pipelines.feature_engineering.nodes import (
    build_churn_labels,
    build_customer_features,
)

TEMPORAL = {
    "cutoff_date": "2011-06-09",
    "performance_window_days": 90,
    "recent_windows": [30, 90, 180, 365],
}


def _line(
    customer: int,
    invoice: str,
    date: str,
    revenue: float,
    *,
    quantity: int = 1,
    cancel: bool = False,
    stock_code: str = "85123A",
) -> dict:
    return {
        "customer_id": customer,
        "invoice_no": invoice,
        "invoice_date": pd.Timestamp(date),
        "stock_code": stock_code,
        "quantity": -abs(quantity) if cancel else quantity,
        "unit_price": abs(revenue) / abs(quantity),
        "line_revenue": -abs(revenue) if cancel else revenue,
        "is_cancellation": cancel,
        "line_revenue_gross": 0.0 if cancel else revenue,
        "return_value": abs(revenue) if cancel else 0.0,
        "country_grouped": "United Kingdom",
        "invoice_month": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
    }


@pytest.fixture()
def transactions() -> pd.DataFrame:
    rows = [
        _line(1, "A1", "2011-01-01", 100.0),
        _line(1, "A2", "2011-05-01", 50.0),
        _line(1, "CA1", "2011-02-01", 20.0, cancel=True),
        _line(2, "B1", "2010-12-01", 200.0),
        _line(3, "C1", "2011-06-01", 300.0),
        _line(4, "D1", "2010-01-01", 400.0),
        _line(5, "E1", "2010-11-01", 250.0),
        _line(6, "F1", "2011-05-20", 80.0),
        # performance-window purchases (must NOT leak into features)
        _line(1, "P1", "2011-07-01", 30.0),
        _line(3, "P3", "2011-08-01", 70.0),
        _line(5, "P5", "2011-07-15", 90.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def features(transactions: pd.DataFrame) -> pd.DataFrame:
    return build_customer_features(transactions, TEMPORAL).set_index("customer_id")


def test_feature_columns_present(features: pd.DataFrame) -> None:
    expected = {
        "recency_days",
        "n_orders",
        "gross_revenue",
        "net_revenue",
        "aov",
        "avg_interval",
        "n_orders_90",
        "revenue_90",
        "revenue_growth_90",
        "order_value_cv",
        "product_diversity",
        "return_value_ratio",
        "cancellation_rate",
        "tenure_days",
        "active_months",
        "revenue_trend_slope",
        "interval_change",
        "country_grouped",
    }
    assert expected.issubset(features.columns)


def test_features_do_not_leak_performance_window(features: pd.DataFrame) -> None:
    # Customer 1 made a 30 GBP purchase in the performance window; it must be absent.
    assert features.loc[1, "gross_revenue"] == pytest.approx(150.0)
    # Customer 3's 70 GBP performance purchase must be absent.
    assert features.loc[3, "gross_revenue"] == pytest.approx(300.0)


def test_recency_and_returns(features: pd.DataFrame) -> None:
    assert (
        features.loc[1, "recency_days"]
        == (pd.Timestamp("2011-06-09") - pd.Timestamp("2011-05-01")).days
    )
    assert features.loc[1, "return_value"] == pytest.approx(20.0)
    assert features.loc[1, "net_revenue"] == pytest.approx(130.0)
    assert features.loc[1, "cancellation_rate"] == pytest.approx(1 / 3, abs=1e-6)


def test_recent_windows(features: pd.DataFrame) -> None:
    # Only the 2011-05-01 order falls in the last 90 days for customer 1.
    assert features.loc[1, "n_orders_90"] == 1.0
    assert features.loc[1, "revenue_90"] == pytest.approx(50.0)
    # The 2011-06-01 order falls in the last 30 days for customer 3.
    assert features.loc[3, "n_orders_30"] == 1.0


def test_churn_labels(features: pd.DataFrame, transactions: pd.DataFrame) -> None:
    labels = build_churn_labels(transactions, TEMPORAL).set_index("customer_id")
    assert labels.loc[1, "churn"] == 0  # purchased 2011-07-01
    assert labels.loc[2, "churn"] == 1  # last purchase 2010-12-01
    assert labels.loc[3, "churn"] == 0
    assert labels.loc[4, "churn"] == 1
    assert labels.loc[5, "churn"] == 0
    assert labels.loc[6, "churn"] == 1
    assert set(labels.index) == set(features.index)


def test_labels_have_future_revenue(transactions: pd.DataFrame) -> None:
    labels = build_churn_labels(transactions, TEMPORAL).set_index("customer_id")
    assert labels.loc[1, "future_revenue"] == pytest.approx(30.0)
    assert labels.loc[2, "future_revenue"] == pytest.approx(0.0)
