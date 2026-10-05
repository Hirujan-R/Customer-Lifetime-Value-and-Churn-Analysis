"""Feature-quality nodes.

The ingestion pipeline produces a schema-clean transaction table. This layer
applies the *analysis*-driven decisions from the EDA:

* remove the last residual administrative stock codes;
* flag zero-price lines and define monetary eligibility;
* separate net vs gross line revenue and quantify returns;
* winsorise heavy-tailed quantity / revenue without deleting genuine customers;
* group long-tail countries so categorical models stay stable.

Crucially, it does **not** impute ``customer_id`` (missing-not-at-random
identity) — those rows were already excluded at ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns referenced downstream; protected against accidental schema drift.
DEFAULT_STOCK_CODE_PATTERN = r"^[0-9]{5}[A-Z]{0,3}$"


def curate_transactions(
    transactions: pd.DataFrame,
    residual_admin_stock_codes: list[str],
    winsorise: dict[str, float],
    country_grouping: dict[str, Any],
) -> pd.DataFrame:
    """Apply feature-quality curation to the cleaned transaction table.

    Args:
        transactions: Cleaned transactions from the ingestion pipeline.
        residual_admin_stock_codes: Administrative codes the ingestion filter
            missed (e.g. ``ADJUST``, ``TEST001``).
        winsorise: Dict with ``lower_quantile`` and ``upper_quantile`` used to
            cap ``quantity`` and ``line_revenue``.
        country_grouping: Dict with ``min_transactions_per_country`` and
            ``other_label`` for collapsing long-tail countries.

    Returns:
        Curated transaction frame with quality flags and robust columns.
    """
    df = transactions.copy()

    # --- Remove residual administrative stock codes -----------------------
    admin = {str(code).upper() for code in residual_admin_stock_codes}
    n_admin = int(df["stock_code"].isin(admin).sum())
    df = df.loc[~df["stock_code"].isin(admin)].copy()

    # --- Price quality and monetary eligibility ---------------------------
    df["is_zero_price"] = (df["unit_price"] <= 0).astype(bool)
    df["is_monetary_eligible"] = (~df["is_cancellation"]) & (~df["is_zero_price"])

    # --- Explicit net vs gross revenue and returns ------------------------
    df["line_revenue_gross"] = df["line_revenue"].where(~df["is_cancellation"], 0.0)
    df["return_value"] = (-df["line_revenue"]).where(df["is_cancellation"], 0.0)

    # --- Robust (winsorised) tails ----------------------------------------
    lower_q = float(winsorise.get("lower_quantile", 0.001))
    upper_q = float(winsorise.get("upper_quantile", 0.999))
    outlier_mask = pd.Series(False, index=df.index)
    for column in ("quantity", "line_revenue"):
        lower = df[column].quantile(lower_q)
        upper = df[column].quantile(upper_q)
        capped = df[column].clip(lower, upper)
        df[f"{column}_winsorised"] = capped
        is_outlier = df[column] != capped
        df[f"{column}_is_outlier"] = is_outlier
        outlier_mask = outlier_mask | is_outlier
    df["is_outlier"] = outlier_mask

    # --- Country grouping --------------------------------------------------
    min_count = int(country_grouping.get("min_transactions_per_country", 100))
    other_label = str(country_grouping.get("other_label", "Other"))
    counts = df["country"].value_counts()
    keep = set(counts[counts >= min_count].index)
    df["country_grouped"] = df["country"].where(df["country"].isin(keep), other_label)

    df = df.reset_index(drop=True)

    logger.info(
        "Curated transactions: %s rows | removed %s admin rows | %s zero-price | "
        "%s outliers capped | %s countries grouped into '%s'",
        f"{len(df):,}",
        f"{n_admin:,}",
        f"{int(df['is_zero_price'].sum()):,}",
        f"{int(df['is_outlier'].sum()):,}",
        f"{df['country'].nunique() - df['country_grouped'].nunique()}",
        other_label,
    )
    return df


def validate_curated_transactions(
    transactions_curated: pd.DataFrame,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the curated dataset and return a JSON-serialisable report.

    Args:
        transactions_curated: Output of :func:`curate_transactions`.
        expected: Optional expectations (``stock_code_pattern``,
            ``max_invoice_date``).

    Returns:
        Report with a ``passed`` flag and a list of ``issues``.
    """
    expected = expected or {}
    df = transactions_curated
    issues: list[dict[str, str]] = []

    def _issue(level: str, message: str) -> None:
        issues.append({"level": level, "message": message})

    if df.empty:
        _issue("error", "Curated transactions dataset is empty")

    required = {
        "is_zero_price",
        "is_monetary_eligible",
        "line_revenue_gross",
        "return_value",
        "quantity_winsorised",
        "line_revenue_winsorised",
        "is_outlier",
        "country_grouped",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _issue("error", f"Missing curated columns: {missing}")

    if not df.empty:
        pattern = expected.get("stock_code_pattern", DEFAULT_STOCK_CODE_PATTERN)
        bad_codes = ~df["stock_code"].astype(str).str.match(pattern)
        n_bad = int(bad_codes.sum())
        if n_bad:
            sample = sorted(set(df.loc[bad_codes, "stock_code"].head(5)))
            _issue("error", f"{n_bad} stock codes fail format {pattern}: {sample}")

        n_negative_price = int((df["unit_price"] < 0).sum())
        if n_negative_price:
            _issue("error", f"{n_negative_price} rows have negative unit_price")

        n_null = int(df[["customer_id", "invoice_date", "line_revenue"]].isna().any(axis=1).sum())
        if n_null:
            _issue("error", f"{n_null} rows have nulls in critical columns")

    n_zero_price = int(df["is_zero_price"].sum()) if "is_zero_price" in df else 0
    if n_zero_price:
        _issue(
            "info", f"{n_zero_price} zero-price lines flagged and excluded from monetary features"
        )

    report: dict[str, Any] = {
        "passed": not any(i["level"] == "error" for i in issues),
        "n_rows": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()) if not df.empty else 0,
        "n_zero_price_lines": n_zero_price,
        "n_outliers_capped": int(df["is_outlier"].sum()) if not df.empty else 0,
        "n_returns": int(df["is_cancellation"].sum()) if not df.empty else 0,
        "n_countries": int(df["country"].nunique()) if not df.empty else 0,
        "n_countries_grouped": int(df["country_grouped"].nunique()) if not df.empty else 0,
        "gross_revenue": round(float(df["line_revenue_gross"].sum()), 2) if not df.empty else 0.0,
        "net_revenue": round(float(df["line_revenue"].sum()), 2) if not df.empty else 0.0,
        "issues": issues,
    }

    logger.info(
        "Curated data-quality report: passed=%s, rows=%s, outliers=%s, issues=%s",
        report["passed"],
        f"{report['n_rows']:,}",
        report["n_outliers_capped"],
        len(issues),
    )
    return report
