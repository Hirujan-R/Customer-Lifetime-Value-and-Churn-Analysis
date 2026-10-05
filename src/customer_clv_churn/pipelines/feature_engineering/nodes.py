"""Customer feature engineering and churn labelling.

Everything in this module is **temporally safe**:

* features are computed from transactions **at or before** the cutoff ``T``;
* the churn label is computed from the performance window ``(T, T+horizon]``;
* the two never overlap, so the model cannot leak future information.

The result is one row per *eligible* customer (a customer with at least one
purchase order in the observation window).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order_value_trend(group: pd.DataFrame) -> float:
    """Slope of order value over time (GBP per day) for one customer."""
    if len(group) < 2:
        return 0.0
    x = group["order_date"].astype("int64").to_numpy() / 86_400_000_000_000
    y = group["order_revenue"].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _interval_change(group: pd.DataFrame) -> float:
    """Change in mean inter-purchase interval: recent half minus older half."""
    dates = group.sort_values("order_date")["order_date"]
    intervals = dates.diff().dt.days.dropna()
    if len(intervals) < 2:
        return 0.0
    half = len(intervals) // 2
    older = intervals.iloc[:half].mean()
    recent = intervals.iloc[half:].mean()
    if pd.isna(older) or pd.isna(recent):
        return 0.0
    return float(recent - older)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator + EPS)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_customer_features(
    transactions: pd.DataFrame,
    temporal: dict[str, Any],
) -> pd.DataFrame:
    """Build the customer-level feature store from the observation window.

    Args:
        transactions: Curated transactions (``transactions_curated``).
        temporal: Temporal config with ``cutoff_date`` and ``recent_windows``.

    Returns:
        One row per eligible customer with RFM + behavioural features.
    """
    cutoff = pd.Timestamp(temporal["cutoff_date"])
    windows = list(temporal.get("recent_windows", [30, 90, 180, 365]))

    obs = transactions.loc[transactions["invoice_date"] <= cutoff].copy()
    purchases = obs.loc[~obs["is_cancellation"]].copy()
    cancellations = obs.loc[obs["is_cancellation"]].copy()

    # --- Order-level table (one row per purchase invoice) -----------------
    orders = (
        purchases.groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(
            order_date=("invoice_date", "min"),
            order_revenue=("line_revenue", "sum"),
            order_lines=("invoice_no", "size"),
            order_quantity=("quantity", "sum"),
        )
        .sort_values(["customer_id", "order_date"])
        .reset_index(drop=True)
    )

    # --- Core RFM aggregates ----------------------------------------------
    features = orders.groupby("customer_id").agg(
        n_orders=("invoice_no", "nunique"),
        gross_revenue=("order_revenue", "sum"),
        aov=("order_revenue", "mean"),
        median_order_value=("order_revenue", "median"),
        max_order_value=("order_revenue", "max"),
        std_order_value=("order_revenue", "std"),
        total_lines=("order_lines", "sum"),
        avg_basket_lines=("order_lines", "mean"),
        total_quantity=("order_quantity", "sum"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
    )

    # --- Inter-purchase intervals -----------------------------------------
    orders["prev_order_date"] = orders.groupby("customer_id")["order_date"].shift()
    orders["interval_days"] = (orders["order_date"] - orders["prev_order_date"]).dt.days
    intervals = (
        orders.dropna(subset=["interval_days"])
        .groupby("customer_id")["interval_days"]
        .agg(
            avg_interval="mean",
            std_interval="std",
            median_interval="median",
            max_interval="max",
        )
    )
    last_interval = (
        orders.dropna(subset=["interval_days"])
        .sort_values("order_date")
        .groupby("customer_id")["interval_days"]
        .last()
        .rename("days_since_previous_purchase")
    )

    # --- Rolling windows --------------------------------------------------
    window_frames = []
    for window in windows:
        mask = orders["order_date"] > (cutoff - pd.Timedelta(window, "D"))
        window_frame = (
            orders.loc[mask]
            .groupby("customer_id")
            .agg(
                **{
                    f"n_orders_{window}": ("invoice_no", "nunique"),
                    f"revenue_{window}": ("order_revenue", "sum"),
                }
            )
        )
        window_frames.append(window_frame)

    previous_mask = (orders["order_date"] > cutoff - pd.Timedelta(180, "D")) & (
        orders["order_date"] <= cutoff - pd.Timedelta(90, "D")
    )
    previous_90 = (
        orders.loc[previous_mask]
        .groupby("customer_id")
        .agg(
            revenue_prev_90=("order_revenue", "sum"),
            orders_prev_90=("invoice_no", "nunique"),
        )
    )

    # --- Product, return and geography signals ----------------------------
    products = purchases.groupby("customer_id").agg(
        n_unique_products=("stock_code", "nunique"),
        n_lines=("invoice_no", "size"),
    )
    returns = cancellations.groupby("customer_id").agg(
        n_cancel_orders=("invoice_no", "nunique"),
        return_value=("return_value", "sum"),
    )
    active_months = (
        purchases.groupby("customer_id")["invoice_month"].nunique().rename("active_months")
    )
    countries = (
        purchases.groupby("customer_id")["country_grouped"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Other")
        .rename("country_grouped")
    )

    # --- Time trends (per-customer regression) ----------------------------
    trend = orders.groupby("customer_id")[["order_date", "order_revenue"]].apply(_order_value_trend)
    trend = trend.rename("revenue_trend_slope")
    interval_change = orders.groupby("customer_id")[["order_date"]].apply(_interval_change)
    interval_change = interval_change.rename("interval_change")

    # --- Assemble ----------------------------------------------------------
    features = features.join(
        [
            intervals,
            last_interval,
            *window_frames,
            previous_90,
            products,
            returns,
            active_months,
            countries,
            trend,
            interval_change,
        ],
        how="left",
    )

    count_columns = [
        c
        for c in features.columns
        if c.startswith(("n_orders_", "revenue_", "orders_prev", "n_cancel", "return_value"))
    ]
    features[count_columns] = features[count_columns].fillna(0.0)
    features["std_order_value"] = features["std_order_value"].fillna(0.0)
    features["std_interval"] = features["std_interval"].fillna(0.0)

    # --- Derived behavioural features -------------------------------------
    features["tenure_days"] = (cutoff - features["first_order_date"]).dt.days
    features["recency_days"] = (cutoff - features["last_order_date"]).dt.days
    features["tenure_months"] = (features["tenure_days"] / 30.44).clip(lower=1.0)
    features["purchase_frequency"] = features["n_orders"] / features["tenure_months"]
    features["overall_order_rate"] = features["n_orders"] / features["tenure_days"].clip(lower=1)
    features["net_revenue"] = features["gross_revenue"] - features["return_value"]
    features["revenue_per_month"] = features["net_revenue"] / features["tenure_months"]
    features["return_value_ratio"] = _safe_ratio(
        features["return_value"], features["gross_revenue"]
    )
    features["cancellation_rate"] = _safe_ratio(
        features["n_cancel_orders"], features["n_orders"] + features["n_cancel_orders"]
    )
    features["avg_unit_price"] = _safe_ratio(
        features["gross_revenue"], features["total_quantity"].clip(lower=1)
    )
    features["product_diversity"] = _safe_ratio(
        features["n_unique_products"], features["n_lines"].clip(lower=1)
    )
    features["order_value_cv"] = _safe_ratio(features["std_order_value"], features["aov"])
    features["interval_cv"] = _safe_ratio(features["std_interval"], features["avg_interval"])
    features["is_single_purchase"] = (features["n_orders"] == 1).astype(int)
    features["active_month_ratio"] = _safe_ratio(
        features["active_months"], features["tenure_months"]
    )

    # Recent vs historical activity
    features["recent_order_share"] = _safe_ratio(features["n_orders_90"], features["n_orders"])
    features["recent_revenue_share"] = _safe_ratio(
        features["revenue_90"], features["gross_revenue"]
    )
    features["recent_order_rate"] = features["n_orders_90"] / 90.0
    features["recent_revenue_rate"] = features["revenue_90"] / 90.0
    features["order_rate_ratio"] = _safe_ratio(
        features["recent_order_rate"], features["overall_order_rate"]
    )
    features["revenue_rate_ratio"] = _safe_ratio(
        features["recent_revenue_rate"],
        features["gross_revenue"] / features["tenure_days"].clip(lower=1),
    )

    # Growth / momentum
    features["revenue_growth_90"] = _safe_ratio(
        features["revenue_90"] - features["revenue_prev_90"], features["revenue_prev_90"] + 1
    )
    features["order_growth_90"] = _safe_ratio(
        features["n_orders_90"] - features["orders_prev_90"], features["orders_prev_90"] + 1
    )
    features["aov_last_90"] = _safe_ratio(features["revenue_90"], features["n_orders_90"])
    features["aov_prev_90"] = _safe_ratio(features["revenue_prev_90"], features["orders_prev_90"])
    features["aov_change_90"] = _safe_ratio(
        features["aov_last_90"] - features["aov_prev_90"], features["aov_prev_90"] + 1
    )
    features["recency_to_interval_ratio"] = _safe_ratio(
        features["recency_days"], features["avg_interval"].fillna(features["tenure_days"])
    )

    # --- Housekeeping ------------------------------------------------------
    # Drop redundant / intermediate-only columns to avoid feature bloat.
    intermediate_columns = [
        "n_lines",  # identical to total_lines
        "aov_last_90",  # inputs to aov_change_90
        "aov_prev_90",
        "recent_order_rate",  # inputs to the rate ratios
        "recent_revenue_rate",
        "tenure_months",  # derived from tenure_days
        "orders_prev_90",  # input to order_growth_90
        "median_interval",  # summarised by avg_interval / interval_cv
        "max_interval",
    ]
    features = features.drop(columns=[c for c in intermediate_columns if c in features.columns])

    features = features.reset_index()
    features = features.replace([np.inf, -np.inf], np.nan)

    date_columns = ["first_order_date", "last_order_date"]
    features = features.drop(columns=date_columns)
    features["country_grouped"] = features["country_grouped"].fillna("Other").astype(str)

    logger.info(
        "Built customer features: %s customers x %s columns (cutoff %s)",
        f"{len(features):,}",
        features.shape[1],
        cutoff.date(),
    )
    return features


# ---------------------------------------------------------------------------
# Churn labels
# ---------------------------------------------------------------------------


def build_churn_labels(
    transactions: pd.DataFrame,
    temporal: dict[str, Any],
) -> pd.DataFrame:
    """Create the churn label and observed future value per eligible customer.

    Churn is defined as **no purchase in the performance window**
    ``(T, T + performance_window_days]``.

    Args:
        transactions: Curated transactions.
        temporal: Temporal config with ``cutoff_date`` and
            ``performance_window_days``.

    Returns:
        One row per eligible customer with ``churn`` and future value columns.
    """
    cutoff = pd.Timestamp(temporal["cutoff_date"])
    horizon = int(temporal["performance_window_days"])
    performance_end = cutoff + pd.Timedelta(horizon, "D")

    obs = transactions.loc[transactions["invoice_date"] <= cutoff]
    performance = transactions.loc[
        (transactions["invoice_date"] > cutoff) & (transactions["invoice_date"] <= performance_end)
    ]

    eligible = obs.loc[~obs["is_cancellation"], "customer_id"].dropna().unique()
    future = (
        performance.loc[~performance["is_cancellation"]]
        .groupby("customer_id")
        .agg(
            future_orders=("invoice_no", "nunique"),
            future_revenue=("line_revenue", "sum"),
        )
    )

    labels = pd.DataFrame({"customer_id": pd.Index(eligible, name="customer_id")})
    labels = labels.merge(future, on="customer_id", how="left")
    labels["future_orders"] = labels["future_orders"].fillna(0).astype(int)
    labels["future_revenue"] = labels["future_revenue"].fillna(0.0)
    labels["churn"] = (labels["future_orders"] == 0).astype(int)
    labels["cutoff_date"] = cutoff
    labels["performance_end_date"] = performance_end

    logger.info(
        "Built churn labels: %s customers | churn rate %.1f%% (window %s -> %s)",
        f"{len(labels):,}",
        labels["churn"].mean() * 100,
        cutoff.date(),
        performance_end.date(),
    )
    return labels
