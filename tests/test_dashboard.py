"""Smoke tests for the dashboard package (imports + helpers, no server)."""

from __future__ import annotations

from pathlib import Path

from customer_clv_churn.dashboard import app, data


def test_dashboard_modules_import() -> None:
    for name in (
        "render_executive_overview",
        "render_segmentation",
        "render_churn_risk",
        "render_revenue_impact",
        "render_recommended_actions",
        "main",
    ):
        assert callable(getattr(app, name))


def test_find_project_root() -> None:
    root = data.find_project_root(Path(__file__).resolve().parent)
    assert (root / "pyproject.toml").exists()


def test_money_formatting() -> None:
    assert app.money(960359.42) == "£960,359"
