"""Tests for the feature-quality (data_quality) nodes."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_clv_churn.pipelines.data_quality.nodes import (
    curate_transactions,
    validate_curated_transactions,
)

WINSORISE = {"lower_quantile": 0.0, "upper_quantile": 0.75}
COUNTRY_GROUPING = {"min_transactions_per_country": 2, "other_label": "Other"}


@pytest.fixture()
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invoice_no": ["536365", "536366", "C536367", "536368", "536369", "536370"],
            "stock_code": ["85123A", "71053", "85123A", "85099B", "12345", "ADJUST"],
            "description": ["a", "b", "a", "c", "d", "adj"],
            "quantity": [2, 1, -1, 1, 99999, 1],
            "invoice_date": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-01-02",
                    "2010-01-03",
                    "2010-01-04",
                    "2010-01-05",
                    "2010-01-06",
                ]
            ),
            "unit_price": [2.0, 3.0, 2.0, 0.0, 100.0, 10.0],
            "customer_id": pd.array([1, 1, 1, 2, 3, 3], dtype="Int64"),
            "country": [
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "Rareland",
                "United Kingdom",
            ],
            "is_cancellation": [False, False, True, False, False, False],
            "is_return": [False, False, True, False, False, False],
            "line_revenue": [4.0, 3.0, -2.0, 0.0, 9_999_900.0, 10.0],
            "is_non_product": [False, False, False, False, False, False],
            "invoice_month": ["2010-01"] * 6,
        }
    )


def _curate(df: pd.DataFrame) -> pd.DataFrame:
    return curate_transactions(df, ["ADJUST", "ADJUST2"], WINSORISE, COUNTRY_GROUPING)


def test_curate_removes_residual_admin_codes(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    assert "ADJUST" not in set(curated["stock_code"])
    assert len(curated) == len(transactions) - 1


def test_curate_flags_zero_price_and_monetary_eligibility(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    zero = curated.loc[curated["stock_code"] == "85099B"].iloc[0]
    assert bool(zero["is_zero_price"]) is True
    assert bool(zero["is_monetary_eligible"]) is False


def test_curate_splits_net_and_gross_revenue(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    cancellation = curated.loc[curated["is_cancellation"]].iloc[0]
    assert cancellation["line_revenue_gross"] == 0.0
    assert cancellation["return_value"] == pytest.approx(2.0)
    purchase = curated.loc[~curated["is_cancellation"]].iloc[0]
    assert purchase["line_revenue_gross"] == purchase["line_revenue"]


def test_curate_winsorises_outliers(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    assert "quantity_winsorised" in curated.columns
    extreme = curated.loc[curated["stock_code"] == "12345"].iloc[0]
    assert bool(extreme["is_outlier"]) is True
    assert extreme["quantity_winsorised"] < extreme["quantity"]


def test_curate_groups_rare_countries(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    rare = curated.loc[curated["country"] == "Rareland"].iloc[0]
    assert rare["country_grouped"] == "Other"
    assert (
        curated.loc[curated["country"] == "United Kingdom", "country_grouped"]
        .eq("United Kingdom")
        .all()
    )


def test_validate_curated_passes(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    report = validate_curated_transactions(curated, {"stock_code_pattern": r"^[0-9]{5}[A-Z]{0,3}$"})
    assert report["passed"] is True
    assert report["n_rows"] == len(curated)
    assert report["n_outliers_capped"] >= 1
    assert any(i["level"] == "info" for i in report["issues"])  # zero-price info


def test_validate_flags_bad_stock_code(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    curated.loc[0, "stock_code"] = "BAD!!"
    report = validate_curated_transactions(curated)
    assert report["passed"] is False
    assert any("stock codes fail format" in i["message"] for i in report["issues"])


def test_validate_flags_negative_price(transactions: pd.DataFrame) -> None:
    curated = _curate(transactions)
    curated.loc[0, "unit_price"] = -1.0
    report = validate_curated_transactions(curated)
    assert report["passed"] is False
    assert any("negative unit_price" in i["message"] for i in report["issues"])


def test_validate_flags_empty_frame() -> None:
    report = validate_curated_transactions(pd.DataFrame())
    assert report["passed"] is False
    assert any(i["level"] == "error" for i in report["issues"])
