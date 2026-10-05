"""Tests for the data ingestion nodes."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_clv_churn.pipelines.data_ingestion.nodes import (
    clean_transactions,
    load_raw_transactions,
    summarise_transactions,
    validate_transactions,
)


@pytest.fixture()
def raw_transactions() -> pd.DataFrame:
    """A small canonical transaction frame with realistic edge cases."""
    return pd.DataFrame(
        {
            "invoice_no": ["536365", "536365", "C536366", "536367", "536368", "536369"],
            "stock_code": ["85123A", "71053", "85123A", "85123A", "POST", "TEST001"],
            "description": ["a", "b", "c", "d", "postage", "test"],
            "quantity": [2, 1, -1, 1, 1, 1],
            "invoice_date": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-01-01",
                    "2010-01-05",
                    "2010-01-06",
                    "2010-01-07",
                    "2010-01-08",
                ]
            ),
            "unit_price": [2.0, 3.0, 2.0, 5.0, 18.0, 1.0],
            "customer_id": pd.array([1, 1, 1, 2, 2, None], dtype="Int64"),
            "country": ["United Kingdom"] * 6,
        }
    )


def test_clean_transactions_flags_and_filters(raw_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(raw_transactions)

    # Missing customers and non-product codes are removed.
    assert cleaned["customer_id"].notna().all()
    assert "POST" not in set(cleaned["stock_code"])
    assert "TEST001" not in set(cleaned["stock_code"])
    assert set(cleaned["customer_id"].astype(int)) == {1, 2}

    # Derived columns exist and are correct.
    assert cleaned.loc[cleaned["invoice_no"] == "C536366", "is_cancellation"].all()
    assert cleaned.loc[cleaned["invoice_no"] == "C536366", "line_revenue"].iloc[0] == -2.0
    assert "invoice_month" in cleaned.columns


def test_clean_transactions_can_keep_non_products(raw_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(raw_transactions, drop_non_product_stock_codes=False)
    assert "POST" in set(cleaned["stock_code"])


def test_clean_transactions_raises_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        clean_transactions(pd.DataFrame({"invoice_no": ["1"]}))


def test_clean_transactions_drops_missing_customer(raw_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(raw_transactions, drop_missing_customer=True)
    assert cleaned["customer_id"].isna().sum() == 0


def test_load_raw_transactions_reads_all_sheets(tmp_path) -> None:
    path = tmp_path / "online_retail_II.xlsx"
    sheet_one = pd.DataFrame(
        {
            "Invoice": ["1"],
            "StockCode": ["A"],
            "Description": ["x"],
            "Quantity": [1],
            "InvoiceDate": pd.to_datetime(["2010-01-01"]),
            "Price": [1.0],
            "Customer ID": [1.0],
            "Country": ["UK"],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet_one.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        sheet_one.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    raw = load_raw_transactions(str(path))
    assert len(raw) == 2
    assert set(raw["source_sheet"]) == {"Year 2009-2010", "Year 2010-2011"}
    assert "invoice_no" in raw.columns
    assert "customer_id" in raw.columns


def test_validate_transactions_passes_on_clean_data(raw_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(raw_transactions)
    report = validate_transactions(cleaned, {"min_invoice_date": "2009-12-01"})
    assert report["passed"] is True
    assert report["n_customers"] == 2
    assert report["n_cancellations"] == 1
    assert report["issues"] == []


def test_validate_transactions_flags_empty_frame() -> None:
    empty = pd.DataFrame(columns=["invoice_no", "customer_id", "invoice_date", "line_revenue"])
    report = validate_transactions(empty)
    assert report["passed"] is False
    assert any(issue["level"] == "error" for issue in report["issues"])


def test_summarise_transactions_aggregates_by_month(raw_transactions: pd.DataFrame) -> None:
    cleaned = clean_transactions(raw_transactions)
    summary = summarise_transactions(cleaned)
    assert "invoice_month" in summary.columns
    assert summary["net_revenue"].sum() == pytest.approx(cleaned["line_revenue"].sum())
    assert summary["n_orders"].iloc[0] == cleaned["invoice_no"].nunique()
