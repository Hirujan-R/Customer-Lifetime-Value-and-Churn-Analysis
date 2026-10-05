"""Customer Lifetime Value (CLV) modelling and revenue-at-risk.

What this module produces
-------------------------
For every customer in the scored (test) cohort:

* ``historical_value`` — **observed** net revenue in the observation window.
* ``predicted_90d_value`` — **modelled** expected net revenue in the next
  90 days, from a supervised regression on the temporal panel.
* ``expected_future_clv`` — **modelled** expected value over a configurable
  horizon (default 365 days), using the calibrated churn probability as a
  per-period retention assumption.
* ``revenue_at_risk`` — ``P(churn) × expected_future_clv`` (the spec's
  definition of expected revenue at risk).

Modelled estimates are clearly separated from observed revenue — nothing here
claims realised revenue.

CLV assumptions (all explicit)
------------------------------
1. The 90-day-ahead revenue is predicted by a model trained only on
   observation-window features (no leakage).
2. Conditional on being active, expected value per 90-day period is assumed
   stationary: ``v = predicted_90d_value / retention_probability``.
3. Per-period retention is constant and equal to
   ``1 − calibrated_90d_churn_probability``.
4. The horizon is a configurable number of days; an optional discount rate may
   be applied per 90-day period.
5. No inflation, seasonality or margin assumptions beyond the above.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-y_pred)
    top_n = max(1, int(len(y_true) * 0.10))
    total = float(np.sum(y_true))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "spearman": round(float(spearmanr(y_true, y_pred).statistic), 4),
        "mean_actual": round(float(np.mean(y_true)), 4),
        "mean_predicted": round(float(np.mean(y_pred)), 4),
        "prediction_to_actual_ratio": round(float(np.mean(y_pred) / (np.mean(y_true) + 1e-6)), 4),
        "top_decile_value_capture": round(float(np.sum(y_true[order[:top_n]]) / (total + 1e-6)), 4),
    }


def _build_regressor(
    name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
) -> SkPipeline:
    model_params = dict(params.get("model_params", {}).get(name, {}))
    random_state = int(params.get("random_state", 42))

    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if name == "ridge":
        numeric_steps.append(("scale", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", SkPipeline(numeric_steps), numeric_columns),
    ]
    if categorical_columns:
        transformers.append(
            (
                "cat",
                SkPipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers)

    if name == "ridge":
        model = Ridge(alpha=float(model_params.get("alpha", 1.0)), random_state=random_state)
    elif name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(model_params.get("n_estimators", 300)),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 5)),
            n_jobs=-1,
            random_state=random_state,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor  # noqa: PLC0415

        model = LGBMRegressor(
            n_estimators=int(model_params.get("n_estimators", 400)),
            learning_rate=float(model_params.get("learning_rate", 0.05)),
            num_leaves=int(model_params.get("num_leaves", 31)),
            subsample=float(model_params.get("subsample", 0.8)),
            colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
            objective=str(model_params.get("objective", "tweedie")),
            tweedie_variance_power=float(model_params.get("tweedie_variance_power", 1.5)),
            random_state=random_state,
            verbose=-1,
        )
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"Unknown CLV model: {name}")

    return SkPipeline([("preprocessor", preprocessor), ("model", model)])


def _regressor_importance(pipeline: SkPipeline, model_name: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    else:  # pragma: no cover
        return pd.DataFrame(columns=["feature", "importance", "model"])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": importance, "model": model_name})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def train_clv_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train, select and evaluate supervised CLV regression models.

    Args:
        train_data: Training cohort rows (from the churn panel).
        validation_data: Validation cohort rows.
        test_data: Held-out latest cohort rows.
        params: CLV config.

    Returns:
        ``(metrics, test_predictions, feature_importance)``.
    """
    target = params["target"]
    id_columns = set(params.get("id_columns", ["customer_id", "cohort_date"]))
    exclude = set(params.get("exclude_features", []))
    categorical = [c for c in params.get("categorical_features", []) if c in train_data.columns]
    feature_columns = [
        c for c in train_data.columns if c not in id_columns and c not in exclude and c != target
    ]
    numeric = [c for c in feature_columns if c not in categorical]

    x_train, y_train = train_data[feature_columns], train_data[target].to_numpy(dtype=float)
    x_val, y_val = validation_data[feature_columns], validation_data[target].to_numpy(dtype=float)
    x_test, y_test = test_data[feature_columns], test_data[target].to_numpy(dtype=float)

    models = list(params.get("models", ["ridge", "random_forest", "lightgbm"]))
    selection_metric = params.get("selection_metric", "mae")
    fitted: dict[str, SkPipeline] = {}
    validation_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}

    for name in models:
        pipeline = _build_regressor(name, numeric, categorical, params)
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        validation_metrics[name] = _regression_metrics(y_val, pipeline.predict(x_val))
        test_metrics[name] = _regression_metrics(y_test, pipeline.predict(x_test))
        logger.info(
            "CLV model %-14s | val MAE %.2f | test MAE %.2f | test R2 %.3f",
            name,
            validation_metrics[name]["mae"],
            test_metrics[name]["mae"],
            test_metrics[name]["r2"],
        )

    best_name = min(validation_metrics, key=lambda n: validation_metrics[n][selection_metric])
    best_pipeline = fitted[best_name]
    logger.info("Selected best CLV model by validation %s: %s", selection_metric, best_name)

    predictions = pd.DataFrame(
        {
            "customer_id": test_data["customer_id"].to_numpy(),
            "cohort_date": test_data["cohort_date"].to_numpy(),
            "actual_future_revenue": y_test,
            "predicted_90d_value": best_pipeline.predict(x_test).clip(min=0.0),
        }
    )
    importance = _regressor_importance(best_pipeline, best_name)

    metrics: dict[str, Any] = {
        "chosen_model": best_name,
        "selection_metric": selection_metric,
        "target": target,
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)),
        "test_rows": int(len(test_data)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "top_features": importance.head(20).to_dict(orient="records"),
        "note": (
            "predicted_90d_value is a modelled estimate of net revenue in the "
            "next 90 days; it is not observed revenue."
        ),
    }

    _log_to_mlflow(params, metrics, best_pipeline, importance)
    return metrics, predictions, importance


def compute_value_and_risk(
    test_data: pd.DataFrame,
    clv_predictions: pd.DataFrame,
    churn_predictions: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine CLV and churn probability into CLV and revenue-at-risk.

    Args:
        test_data: Held-out cohort rows (provides historical value).
        clv_predictions: Output of :func:`train_clv_model`.
        churn_predictions: ``churn_test_predictions`` with calibrated
            ``churn_probability``.
        params: CLV config (``horizon_days``, ``min_retention_probability``,
            ``discount_rate_annual``).

    Returns:
        ``(customer_value_risk, revenue_at_risk_summary)``.
    """
    horizon_days = float(params.get("horizon_days", 365))
    min_retention = float(params.get("min_retention_probability", 0.1))
    annual_discount = float(params.get("discount_rate_annual", 0.0))
    period_days = 90.0
    periods = horizon_days / period_days
    period_discount = (1.0 + annual_discount) ** (-period_days / 365.0)

    data = (
        test_data[["customer_id", "cohort_date", "net_revenue", "n_orders"]]
        .rename(columns={"net_revenue": "historical_value", "n_orders": "historical_orders"})
        .merge(clv_predictions, on=["customer_id", "cohort_date"], how="inner")
        .merge(
            churn_predictions[["customer_id", "cohort_date", "churn_probability"]],
            on=["customer_id", "cohort_date"],
            how="inner",
        )
    )

    data["retention_probability"] = (1.0 - data["churn_probability"]).clip(
        lower=min_retention, upper=0.999
    )
    # Value per active period, then geometric sum over the horizon.
    data["value_per_active_period"] = data["predicted_90d_value"] / data["retention_probability"]
    ratio = (data["retention_probability"] * period_discount).clip(upper=0.999999)
    geometric = (1.0 - ratio.pow(periods)) / (1.0 - ratio)
    data["expected_future_clv"] = data["value_per_active_period"] * geometric
    data["revenue_at_risk"] = data["churn_probability"] * data["expected_future_clv"]

    columns = [
        "customer_id",
        "cohort_date",
        "historical_value",
        "historical_orders",
        "churn_probability",
        "retention_probability",
        "predicted_90d_value",
        "value_per_active_period",
        "expected_future_clv",
        "revenue_at_risk",
        "actual_future_revenue",
    ]
    customer_value_risk = (
        data[columns].sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)
    )

    total_risk = float(customer_value_risk["revenue_at_risk"].sum())
    risk_order = customer_value_risk["revenue_at_risk"].to_numpy()
    top_decile = max(1, int(len(risk_order) * 0.10))
    top_decile_share = (
        float(np.sort(risk_order)[::-1][:top_decile].sum() / total_risk) if total_risk else 0.0
    )

    summary: dict[str, Any] = {
        "n_customers": int(len(customer_value_risk)),
        "horizon_days": horizon_days,
        "discount_rate_annual": annual_discount,
        "min_retention_probability": min_retention,
        "assumptions": [
            "90-day-ahead revenue is modelled from observation-window features only",
            "value per active 90-day period is assumed stationary",
            "per-period retention = 1 - calibrated 90-day churn probability",
            f"horizon = {horizon_days:.0f} days ({periods:.2f} periods)",
            "no margin/inflation assumptions",
        ],
        "mean_churn_probability": round(float(customer_value_risk["churn_probability"].mean()), 4),
        "total_historical_value": round(float(customer_value_risk["historical_value"].sum()), 2),
        "total_predicted_90d_value": round(
            float(customer_value_risk["predicted_90d_value"].sum()), 2
        ),
        "total_expected_future_clv": round(
            float(customer_value_risk["expected_future_clv"].sum()), 2
        ),
        "total_revenue_at_risk": round(total_risk, 2),
        "revenue_at_risk_top_decile_share": round(top_decile_share, 4),
        "value_provenance": {
            "historical_value": "observed",
            "predicted_90d_value": "modelled",
            "expected_future_clv": "modelled",
            "revenue_at_risk": "modelled = P(churn) x expected_future_clv",
            "actual_future_revenue": "observed (held-out; not available at scoring time)",
        },
    }

    logger.info(
        "CLV/risk: %s customers | total expected CLV £%s | total revenue at risk £%s (top 10%% = %.1f%%)",
        f"{len(customer_value_risk):,}",
        f"{summary['total_expected_future_clv']:,.0f}",
        f"{total_risk:,.0f}",
        top_decile_share * 100,
    )
    return customer_value_risk, summary


def _log_to_mlflow(
    params: dict[str, Any],
    metrics: dict[str, Any],
    model: Any,
    importance: pd.DataFrame,
) -> None:
    """Log the CLV run to MLflow, tolerating a missing backend."""
    mlflow_config = params.get("mlflow", {})
    if not mlflow_config.get("enabled", False):
        return
    try:
        import os  # noqa: PLC0415

        import mlflow  # noqa: PLC0415
        import mlflow.sklearn  # noqa: PLC0415

        tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_config.get("experiment_name", "customer-clv-churn"))
        with mlflow.start_run(run_name=f"clv-{metrics['chosen_model']}"):
            mlflow.log_params(
                {
                    "chosen_model": metrics["chosen_model"],
                    "selection_metric": metrics["selection_metric"],
                    "target": metrics["target"],
                    "train_rows": metrics["train_rows"],
                    "validation_rows": metrics["validation_rows"],
                    "test_rows": metrics["test_rows"],
                }
            )
            for metric, value in metrics["test_metrics"][metrics["chosen_model"]].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_{metric}", float(value))
            mlflow.log_dict(metrics, "clv_metrics.json")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "clv_feature_importance.csv"
                importance.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
            mlflow.sklearn.log_model(model, name="clv_model")
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        logger.warning("MLflow logging skipped: %s", exc)
