"""Nodes for the ``data_ingestion`` pipeline.

This module downloads the UCI *Online Retail II* dataset, parses the two
yearly sheets into a single transaction-level frame, performs defensive
cleaning and produces a data-quality report plus a monthly business summary.

The ingestion layer is intentionally the *only* place where raw files are
touched. Downstream pipelines consume the cleaned ``transactions`` dataset.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical column names expected by the rest of the platform.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
)

#: Maps loosely formatted source headers to the canonical schema.
_STANDARD_COLUMNS: dict[str, str] = {
    "invoice": "invoice_no",
    "invoiceno": "invoice_no",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "country": "country",
}

#: Stock codes that are administrative rather than real products.
DEFAULT_NON_PRODUCT_CODES: tuple[str, ...] = (
    "POST",
    "D",
    "M",
    "BANK CHARGES",
    "PADS",
    "DOT",
    "C2",
    "CRUK",
    "S",
    "AMAZONFEE",
    "B",
    "TEST",
)


def _normalise_column_name(name: Any) -> str:
    """Lowercase a column name and keep only alphanumeric characters."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_online_retail_ii(
    source_urls: list[str],
    raw_dir: str,
    filename: str,
    force_download: bool = False,
    timeout_seconds: int = 300,
) -> str:
    """Download the Online Retail II workbook, returning the local file path.

    The UCI archive is sometimes served as a ``.zip`` archive and sometimes as
    a bare ``.xlsx`` workbook, so both are handled transparently.

    Args:
        source_urls: Candidate URLs, tried in order.
        raw_dir: Directory in which to cache the raw workbook.
        filename: File name used for the cached workbook.
        force_download: Re-download even if the cached file already exists.
        timeout_seconds: Per-request HTTP timeout.

    Returns:
        Path to the local Excel workbook.

    Raises:
        RuntimeError: If every candidate URL fails.
    """
    raw_path = Path(raw_dir) / filename

    if raw_path.exists() and not force_download:
        logger.info("Reusing cached Online Retail II workbook at %s", raw_path)
        return str(raw_path)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for url in source_urls:
        try:
            logger.info("Downloading Online Retail II from %s", url)
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.content
            buffer = io.BytesIO(payload)

            if zipfile.is_zipfile(buffer):
                with zipfile.ZipFile(buffer) as archive:
                    members = [
                        name
                        for name in archive.namelist()
                        if name.lower().endswith((".xlsx", ".xls"))
                    ]
                    if not members:
                        raise ValueError("Zip archive does not contain an Excel file")
                    raw_path.write_bytes(archive.read(members[0]))
            else:
                raw_path.write_bytes(payload)

            size_mb = raw_path.stat().st_size / 1_000_000
            logger.info("Saved raw workbook to %s (%.1f MB)", raw_path, size_mb)
            return str(raw_path)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url} -> {exc!r}")
            logger.warning("Download failed from %s: %s", url, exc)

    raise RuntimeError(
        "Unable to download the Online Retail II dataset from any source.\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_raw_transactions(
    raw_path: str,
    sheet_names: list[str] | None = None,
) -> pd.DataFrame:
    """Read the raw workbook and concat all sheets into one transaction frame.

    Args:
        raw_path: Path to the Excel workbook.
        sheet_names: Sheets to read. ``None`` reads every sheet.

    Returns:
        Transaction-level frame with canonical column names and a
        ``source_sheet`` provenance column.
    """
    path = Path(raw_path)
    logger.info("Loading raw workbook %s", path)
    sheets = pd.read_excel(path, sheet_name=sheet_names, engine="openpyxl")

    if isinstance(sheets, pd.DataFrame):
        sheets = {path.stem: sheets}

    frames = []
    for sheet, sheet_frame in sheets.items():
        renamed = sheet_frame.rename(
            columns=lambda c: _STANDARD_COLUMNS.get(
                _normalise_column_name(c), _normalise_column_name(c)
            )
        )
        renamed["source_sheet"] = sheet
        frames.append(renamed)

    raw = pd.concat(frames, ignore_index=True)

    # Excel infers mixed types for identifier columns (e.g. numeric invoice
    # numbers alongside cancellations such as ``C489449``). Cast them to a
    # consistent string type so they can be persisted safely.
    for column in ("invoice_no", "stock_code", "description", "country", "source_sheet"):
        if column in raw.columns:
            raw[column] = raw[column].astype("string")

    logger.info("Loaded %s raw transaction rows from %s sheet(s)", f"{len(raw):,}", len(sheets))
    return raw


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def clean_transactions(
    raw_transactions: pd.DataFrame,
    drop_missing_customer: bool = True,
    drop_non_product_stock_codes: bool = True,
    non_product_stock_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Clean and standardise the raw transaction records.

    The function is deliberately defensive: it validates the schema, coerces
    dtypes, derives cancellation/revenue flags and removes rows that cannot be
    used for customer-level modelling.

    Args:
        raw_transactions: Output of :func:`load_raw_transactions`.
        drop_missing_customer: Remove rows without a Customer ID. These cannot
            be attributed to a customer and would corrupt customer features.
        drop_non_product_stock_codes: Remove administrative stock codes such as
            postage and bank charges.
        non_product_stock_codes: Override for the default administrative codes.

    Returns:
        Cleaned, transaction-level frame.
    """
    df = raw_transactions.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    keep = list(REQUIRED_COLUMNS)
    if "source_sheet" in df.columns:
        keep.append("source_sheet")
    df = df[keep]

    rows_in = len(df)

    # --- Coerce dtypes -----------------------------------------------------
    df["invoice_no"] = df["invoice_no"].astype("string").str.strip()
    df["stock_code"] = df["stock_code"].astype("string").str.strip().str.upper()
    df["description"] = df["description"].astype("string").str.strip()
    df["country"] = df["country"].astype("string").str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")

    # Rows without a date/quantity/price are unusable.
    df = df.dropna(subset=["invoice_no", "stock_code", "invoice_date", "quantity", "unit_price"])

    if drop_missing_customer:
        df = df.dropna(subset=["customer_id"])

    # --- Derived business flags -------------------------------------------
    df["is_cancellation"] = (
        df["invoice_no"].str.upper().str.startswith("C").fillna(False).astype(bool)
    )
    df["is_return"] = df["is_cancellation"] & (df["quantity"] < 0)
    df["line_revenue"] = (df["quantity"] * df["unit_price"]).astype("float64")

    codes = {str(c).upper() for c in (non_product_stock_codes or DEFAULT_NON_PRODUCT_CODES)}
    df["is_non_product"] = df["stock_code"].isin(codes).astype(bool)
    if drop_non_product_stock_codes:
        df = df.loc[~df["is_non_product"]].copy()

    # --- De-duplicate and enrich ------------------------------------------
    df = df.drop_duplicates()
    df["invoice_month"] = df["invoice_date"].dt.to_period("M").astype(str)

    df = df.reset_index(drop=True)

    logger.info(
        "Cleaned transactions: %s -> %s rows (%s dropped)",
        f"{rows_in:,}",
        f"{len(df):,}",
        f"{rows_in - len(df):,}",
    )
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate_transactions(
    transactions: pd.DataFrame,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run data-quality checks and return a JSON-serialisable report.

    Args:
        transactions: Cleaned transaction frame.
        expected: Optional expectations, e.g. ``min_invoice_date``.

    Returns:
        Report dictionary with a ``passed`` flag and a list of ``issues``.
    """
    expected = expected or {}
    issues: list[dict[str, str]] = []
    df = transactions

    def _issue(level: str, message: str) -> None:
        issues.append({"level": level, "message": message})

    if df.empty:
        _issue("error", "Cleaned transactions dataset is empty")

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            _issue("error", f"Missing required column: {column}")

    if not df.empty:
        if df["customer_id"].isna().any():
            _issue("error", "Null customer_id values present after cleaning")
        if df["invoice_date"].isna().any():
            _issue("error", "Null invoice_date values present after cleaning")
        if df["customer_id"].nunique() < 2:
            _issue("warning", "Fewer than two unique customers in the dataset")

    invoice_date_min = pd.Timestamp(df["invoice_date"].min()) if not df.empty else None
    invoice_date_max = pd.Timestamp(df["invoice_date"].max()) if not df.empty else None

    if invoice_date_min is not None:
        min_expected = (
            pd.Timestamp(expected.get("min_invoice_date"))
            if expected.get("min_invoice_date")
            else None
        )
        max_expected = (
            pd.Timestamp(expected.get("max_invoice_date"))
            if expected.get("max_invoice_date")
            else None
        )
        if min_expected is not None and invoice_date_min < min_expected:
            _issue(
                "warning",
                f"Earliest invoice {invoice_date_min.date()} precedes {min_expected.date()}",
            )
        if max_expected is not None and invoice_date_max > max_expected:
            _issue(
                "warning",
                f"Latest invoice {invoice_date_max.date()} is after {max_expected.date()}",
            )

    non_cancelled = df.loc[~df["is_cancellation"]] if not df.empty else df
    report: dict[str, Any] = {
        "passed": not any(i["level"] == "error" for i in issues),
        "n_rows": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()) if not df.empty else 0,
        "n_invoices": int(df["invoice_no"].nunique()) if not df.empty else 0,
        "n_products": int(df["stock_code"].nunique()) if not df.empty else 0,
        "n_countries": int(df["country"].nunique()) if not df.empty else 0,
        "n_cancellations": int(df["is_cancellation"].sum()) if not df.empty else 0,
        "total_net_revenue": round(float(df["line_revenue"].sum()), 2) if not df.empty else 0.0,
        "gross_revenue": round(float(non_cancelled["line_revenue"].sum()), 2)
        if not df.empty
        else 0.0,
        "invoice_date_min": invoice_date_min.isoformat() if invoice_date_min is not None else None,
        "invoice_date_max": invoice_date_max.isoformat() if invoice_date_max is not None else None,
        "issues": issues,
    }

    logger.info(
        "Data-quality report: passed=%s, rows=%s, customers=%s, issues=%s",
        report["passed"],
        f"{report['n_rows']:,}",
        f"{report['n_customers']:,}",
        len(issues),
    )
    return report


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


def summarise_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build a monthly business summary of activity for reporting.

    Args:
        transactions: Cleaned transaction frame.

    Returns:
        Monthly summary with transactions, orders, customers, cancellations
        and net revenue per calendar month.
    """
    summary = (
        transactions.groupby("invoice_month", as_index=False)
        .agg(
            n_transaction_lines=("invoice_no", "size"),
            n_orders=("invoice_no", "nunique"),
            n_customers=("customer_id", "nunique"),
            n_cancellations=("is_cancellation", "sum"),
            net_revenue=("line_revenue", "sum"),
        )
        .sort_values("invoice_month")
        .reset_index(drop=True)
    )
    logger.info("Built monthly summary with %s months", len(summary))
    return summary
