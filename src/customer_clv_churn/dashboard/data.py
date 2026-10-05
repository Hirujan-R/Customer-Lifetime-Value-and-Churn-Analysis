"""Data access layer for the C-suite dashboard.

Reads the Kedro pipeline outputs from the ``data/`` directory. All loaders are
cached and fail soft (return empty structures) so the dashboard can render a
helpful "run the pipeline" message instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking up to ``pyproject.toml``."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


DATA_DIR = find_project_root() / "data"


@st.cache_data(show_spinner=False)
def _read_parquet(relative_path: str) -> pd.DataFrame:
    path = DATA_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_csv(relative_path: str) -> pd.DataFrame:
    path = DATA_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(relative_path: str) -> dict:
    path = DATA_DIR / relative_path
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


# --- Convenience accessors -------------------------------------------------


def customer_recommendations() -> pd.DataFrame:
    return _read_parquet("07_model_output/customer_recommendations.parquet")


def retention_priorities() -> pd.DataFrame:
    return _read_parquet("07_model_output/retention_priorities.parquet")


def customer_risk_factors() -> pd.DataFrame:
    return _read_parquet("07_model_output/customer_risk_factors.parquet")


def customer_value_risk() -> pd.DataFrame:
    return _read_parquet("07_model_output/customer_value_risk.parquet")


def customer_features() -> pd.DataFrame:
    return _read_parquet("04_feature/customer_features.parquet")


def transactions() -> pd.DataFrame:
    return _read_parquet("03_primary/transactions_curated.parquet")


def segment_profiles() -> pd.DataFrame:
    return _read_csv("08_reporting/segment_profiles.csv")


def rule_segment_profiles() -> pd.DataFrame:
    return _read_csv("08_reporting/rule_segment_profiles.csv")


def prioritisation_report() -> dict:
    return _read_json("08_reporting/prioritisation_report.json")


def recommendations_report() -> dict:
    return _read_json("08_reporting/recommendations_report.json")


def explainability_report() -> dict:
    return _read_json("08_reporting/explainability_report.json")


def segmentation_report() -> dict:
    return _read_json("08_reporting/segmentation_report.json")


def revenue_at_risk_summary() -> dict:
    return _read_json("08_reporting/revenue_at_risk_summary.json")


def baseline_report() -> dict:
    return _read_json("08_reporting/baseline_report.json")


def churn_model_metrics() -> dict:
    return _read_json("08_reporting/churn_model_metrics.json")


def data_available() -> bool:
    """True if the key pipeline outputs exist."""
    return not customer_recommendations().empty
