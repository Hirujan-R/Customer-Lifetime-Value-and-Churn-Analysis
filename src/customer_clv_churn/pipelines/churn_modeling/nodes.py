"""Churn modelling: multi-cohort training, calibration and evaluation.

Modelling design
----------------
A single cutoff gives only one prediction date, so a random split would let the
model memorise the future. Instead we build **multiple temporal cohorts**
(monthly cutoffs), each with its own observation features and 90-day label, and
split them by *time*:

    train cohorts (earlier)  ->  validation cohort  ->  test cohort (latest)

This mirrors deployment: the model is always trained on the past and evaluated
on a period strictly after it.

Models: logistic regression, random forest and LightGBM. The best model (by
validation PR-AUC) is **calibrated** on the validation cohort, because churn
probabilities are later multiplied by CLV — a badly calibrated 0.8 is worse than
a flat ranking. Experiments are tracked with MLflow.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_clv_churn.pipelines.feature_engineering.nodes import (
    build_churn_labels,
    build_customer_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_churn_model_input(
    transactions: pd.DataFrame,
    temporal: dict[str, Any],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Build a multi-cohort (panel) modelling table.

    Args:
        transactions: Curated transactions.
        temporal: Base temporal config (provides ``performance_window_days`` and
            ``recent_windows``).
        params: Churn-modelling config including ``cohort_dates``.

    Returns:
        Long frame: one row per customer per cohort, with features, ``churn``
        and observed ``future_revenue``.
    """
    cohorts = list(params["cohort_dates"])
    frames = []
    for cohort in cohorts:
        cohort_temporal = {**temporal, "cutoff_date": cohort}
        features = build_customer_features(transactions, cohort_temporal)
        labels = build_churn_labels(transactions, cohort_temporal)
        merged = features.merge(
            labels[["customer_id", "churn", "future_revenue", "future_orders"]],
            on="customer_id",
            how="inner",
        )
        merged["cohort_date"] = pd.Timestamp(cohort)
        frames.append(merged)

    model_input = pd.concat(frames, ignore_index=True)
    logger.info(
        "Built churn model input: %s rows across %s cohorts (%s -> %s)",
        f"{len(model_input):,}",
        len(cohorts),
        cohorts[0],
        cohorts[-1],
    )
    return model_input


def split_churn_data(
    model_input: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the panel into train / validation / test by cohort date.

    Args:
        model_input: Output of :func:`build_churn_model_input`.
        params: Config with ``train_cohorts``, ``validation_cohorts``,
            ``test_cohorts``.

    Returns:
        ``(train, validation, test)``.
    """

    def _select(cohorts: list[str]) -> pd.DataFrame:
        wanted = pd.to_datetime(cohorts)
        return model_input.loc[model_input["cohort_date"].isin(wanted)].reset_index(drop=True)

    train = _select(params["train_cohorts"])
    validation = _select(params["validation_cohorts"])
    test = _select(params["test_cohorts"])
    logger.info(
        "Split by time -> train %s rows, validation %s rows, test %s rows",
        f"{len(train):,}",
        f"{len(validation):,}",
        f"{len(test):,}",
    )
    return train, validation, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted confidence and observed frequency."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, bins[1:-1])
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        confidence = prob[mask].mean()
        accuracy = y_true[mask].mean()
        error += mask.mean() * abs(accuracy - confidence)
    return float(error)


def _classification_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, prob)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6))), 4),
        "expected_calibration_error": round(_expected_calibration_error(y_true, prob), 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {
            "true_positives": int(((pred == 1) & (y_true == 1)).sum()),
            "false_positives": int(((pred == 1) & (y_true == 0)).sum()),
            "false_negatives": int(((pred == 0) & (y_true == 1)).sum()),
            "true_negatives": int(((pred == 0) & (y_true == 0)).sum()),
        },
    }


def _best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _top_fraction_capture(
    y_true: np.ndarray, prob: np.ndarray, fraction: float
) -> dict[str, float]:
    n = len(y_true)
    k = max(1, int(n * fraction))
    top = np.argsort(-prob)[:k]
    captured = int(y_true[top].sum())
    total = int(y_true.sum())
    base_rate = total / n if n else 0.0
    return {
        "capture_rate": round(captured / total, 4) if total else 0.0,
        "precision": round(captured / k, 4) if k else 0.0,
        "lift": round((captured / k) / base_rate, 4) if base_rate else 0.0,
    }


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _build_estimator(
    name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
) -> SkPipeline:
    model_params = dict(params.get("model_params", {}).get(name, {}))
    random_state = int(params.get("random_state", 42))

    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if name == "logistic_regression":
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

    if name == "logistic_regression":
        model = LogisticRegression(
            max_iter=int(model_params.get("max_iter", 2000)),
            C=float(model_params.get("C", 1.0)),
            class_weight="balanced",
            random_state=random_state,
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(model_params.get("n_estimators", 400)),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 5)),
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        model = LGBMClassifier(
            n_estimators=int(model_params.get("n_estimators", 400)),
            learning_rate=float(model_params.get("learning_rate", 0.05)),
            num_leaves=int(model_params.get("num_leaves", 31)),
            subsample=float(model_params.get("subsample", 0.8)),
            colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"Unknown model: {name}")

    return SkPipeline([("preprocessor", preprocessor), ("model", model)])


def _feature_importance(pipeline: SkPipeline, model_name: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    else:  # pragma: no cover - all supported models expose one of the two
        return pd.DataFrame(columns=["feature", "importance", "model"])

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": importance, "model": model_name})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_churn_models(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train, select, calibrate and evaluate churn models.

    Args:
        train_data: Training cohort rows.
        validation_data: Validation cohort rows (model selection + calibration).
        test_data: Held-out latest cohort rows.
        params: Churn-modelling config.

    Returns:
        ``(metrics, test_predictions, feature_importance, base_model)`` where
        ``base_model`` is the fitted uncalibrated pipeline (used downstream for
        SHAP explanations).
    """
    target = params["target"]
    id_columns = set(params.get("id_columns", ["customer_id", "cohort_date"]))
    exclude = set(params.get("exclude_features", []))
    categorical = [c for c in params.get("categorical_features", []) if c in train_data.columns]
    feature_columns = [
        c for c in train_data.columns if c not in id_columns and c not in exclude and c != target
    ]
    numeric = [c for c in feature_columns if c not in categorical]

    x_train, y_train = train_data[feature_columns], train_data[target].to_numpy()
    x_val, y_val = validation_data[feature_columns], validation_data[target].to_numpy()
    x_test, y_test = test_data[feature_columns], test_data[target].to_numpy()

    models = list(params.get("models", ["logistic_regression", "random_forest", "lightgbm"]))
    fitted: dict[str, SkPipeline] = {}
    validation_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}

    for name in models:
        pipeline = _build_estimator(name, numeric, categorical, params)
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        validation_metrics[name] = _classification_metrics(
            y_val, pipeline.predict_proba(x_val)[:, 1]
        )
        test_metrics[name] = _classification_metrics(y_test, pipeline.predict_proba(x_test)[:, 1])
        logger.info(
            "Model %-20s | val PR-AUC %.4f | test PR-AUC %.4f",
            name,
            validation_metrics[name]["pr_auc"],
            test_metrics[name]["pr_auc"],
        )

    best_name = max(validation_metrics, key=lambda n: validation_metrics[n]["pr_auc"])
    best_pipeline = fitted[best_name]
    logger.info("Selected best model by validation PR-AUC: %s", best_name)

    # Calibrate on the validation cohort using the frozen base estimator.
    # Calibration is only *applied* if it genuinely improves validation
    # calibration (random forests are often already well calibrated).
    calibration_method = params.get("calibration_method", "isotonic")
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(best_pipeline), method=calibration_method
    ).fit(x_val, y_val)

    raw_val_prob = best_pipeline.predict_proba(x_val)[:, 1]
    calibrated_val_prob = calibrated.predict_proba(x_val)[:, 1]
    raw_val_brier = float(brier_score_loss(y_val, raw_val_prob))
    calibrated_val_brier = float(brier_score_loss(y_val, calibrated_val_prob))
    use_calibrated = calibrated_val_brier < raw_val_brier

    raw_test_prob = best_pipeline.predict_proba(x_test)[:, 1]
    calibrated_test_prob = calibrated.predict_proba(x_test)[:, 1]
    if use_calibrated:
        final_val_prob, final_test_prob = calibrated_val_prob, calibrated_test_prob
    else:
        final_val_prob, final_test_prob = raw_val_prob, raw_test_prob

    threshold = _best_f1_threshold(y_val, final_val_prob)
    final_metrics = _classification_metrics(y_test, final_test_prob, threshold=threshold)

    importance = _feature_importance(best_pipeline, best_name)

    metrics: dict[str, Any] = {
        "chosen_model": best_name,
        "selection_metric": "pr_auc (validation)",
        "calibration_method": calibration_method,
        "calibration_applied": use_calibrated,
        "probability_source": "calibrated" if use_calibrated else "raw",
        "decision_threshold": threshold,
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)),
        "test_rows": int(len(test_data)),
        "churn_rate": {
            "train": round(float(y_train.mean()), 4),
            "validation": round(float(y_val.mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "final_test_metrics": final_metrics,
        "calibration_effect": {
            "validation_brier_raw": round(raw_val_brier, 4),
            "validation_brier_calibrated": round(calibrated_val_brier, 4),
            "test_brier_raw": round(float(brier_score_loss(y_test, raw_test_prob)), 4),
            "test_brier_calibrated": round(
                float(brier_score_loss(y_test, calibrated_test_prob)), 4
            ),
            "test_ece_raw": round(_expected_calibration_error(y_test, raw_test_prob), 4),
            "test_ece_calibrated": round(
                _expected_calibration_error(y_test, calibrated_test_prob), 4
            ),
        },
        "business_metrics": {
            "top_10pct": _top_fraction_capture(y_test, final_test_prob, 0.10),
            "top_20pct": _top_fraction_capture(y_test, final_test_prob, 0.20),
        },
        "top_features": importance.head(20).to_dict(orient="records"),
    }

    predictions = pd.DataFrame(
        {
            "customer_id": test_data["customer_id"].to_numpy(),
            "cohort_date": test_data["cohort_date"].to_numpy(),
            "y_true": y_test,
            "churn_probability_raw": raw_test_prob,
            "churn_probability_calibrated": calibrated_test_prob,
            "churn_probability": final_test_prob,
            "churn_prediction": (final_test_prob >= threshold).astype(int),
        }
    )

    final_model = calibrated if use_calibrated else best_pipeline
    _log_to_mlflow(params, metrics, final_model, importance)

    logger.info(
        "Chosen %s | calibration_applied=%s | test PR-AUC %.4f, ROC-AUC %.4f, Brier %.4f, ECE %.4f",
        best_name,
        use_calibrated,
        final_metrics["pr_auc"],
        final_metrics["roc_auc"],
        final_metrics["brier"],
        final_metrics["expected_calibration_error"],
    )
    return metrics, predictions, importance, best_pipeline


def _log_to_mlflow(
    params: dict[str, Any],
    metrics: dict[str, Any],
    model: Any,
    importance: pd.DataFrame,
) -> None:
    """Log the run to MLflow, tolerating a missing/unreachable backend."""
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
        with mlflow.start_run(run_name=f"churn-{metrics['chosen_model']}"):
            mlflow.log_params(
                {
                    "chosen_model": metrics["chosen_model"],
                    "calibration_method": metrics["calibration_method"],
                    "calibration_applied": metrics["calibration_applied"],
                    "probability_source": metrics["probability_source"],
                    "decision_threshold": metrics["decision_threshold"],
                    "train_rows": metrics["train_rows"],
                    "validation_rows": metrics["validation_rows"],
                    "test_rows": metrics["test_rows"],
                }
            )
            for model_name, model_metrics in metrics["test_metrics"].items():
                for metric, value in model_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"test_{model_name}_{metric}", float(value))
            for metric, value in metrics["final_test_metrics"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"final_test_{metric}", float(value))
            mlflow.log_dict(metrics, "churn_metrics.json")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "feature_importance.csv"
                importance.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
            mlflow.sklearn.log_model(model, name="churn_model")
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        logger.warning("MLflow logging skipped: %s", exc)


# ---------------------------------------------------------------------------
# Survival analysis (exploratory)
# ---------------------------------------------------------------------------


def analyze_survival(
    customer_features: pd.DataFrame,
    churn_labels: pd.DataFrame,
    customer_segments: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Kaplan-Meier curves and a Cox proportional-hazards model.

    Duration is days from first purchase to last purchase for churned customers,
    and to the cutoff for those still active (censored).

    Args:
        customer_features: Observation-window feature store.
        churn_labels: Churn labels.
        customer_segments: Segment assignments for stratified curves.
        params: Config with ``survival_features``.

    Returns:
        ``(survival_curves, survival_report)``.
    """
    from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: PLC0415

    data = (
        customer_features.merge(
            churn_labels[["customer_id", "churn"]], on="customer_id", how="inner"
        )
        .merge(customer_segments[["customer_id", "segment_name"]], on="customer_id", how="left")
        .reset_index(drop=True)
    )
    data["event"] = data["churn"].astype(int)
    duration_if_event = (data["tenure_days"] - data["recency_days"]).clip(lower=1)
    data["duration"] = np.where(
        data["event"] == 1, duration_if_event, data["tenure_days"].clip(lower=1)
    )

    curves = []
    km_overall = KaplanMeierFitter()
    km_overall.fit(data["duration"], data["event"])
    for time, survival in km_overall.survival_function_.itertuples():
        curves.append({"segment": "Overall", "time_days": float(time), "survival": float(survival)})
    for segment_name, group in data.groupby("segment_name"):
        km = KaplanMeierFitter()
        km.fit(group["duration"], group["event"])
        for time, survival in km.survival_function_.itertuples():
            curves.append(
                {"segment": segment_name, "time_days": float(time), "survival": float(survival)}
            )
    curves_df = pd.DataFrame(curves)

    survival_features = [
        c
        for c in params.get("survival_features", [])
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
    ]
    cox_data = data[survival_features + ["duration", "event"]].copy()
    cox_data[survival_features] = cox_data[survival_features].fillna(
        cox_data[survival_features].median()
    )
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(cox_data, duration_col="duration", event_col="event")
    cox_summary = cox.summary[
        ["coef", "exp(coef)", "z", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"]
    ].reset_index()

    report: dict[str, Any] = {
        "n_customers": int(len(data)),
        "n_events": int(data["event"].sum()),
        "median_survival_days_overall": float(km_overall.median_survival_time_),
        "concordance_index": round(float(cox.concordance_index_), 4),
        "cox_hazard_ratios": cox_summary.round(4).to_dict(orient="records"),
        "segments": sorted(data["segment_name"].dropna().unique().tolist()),
        "note": (
            "Exploratory survival analysis. Duration is time-to-last-purchase; "
            "association, not causation."
        ),
    }
    logger.info(
        "Survival analysis: concordance=%.3f, events=%s",
        cox.concordance_index_,
        f"{int(data['event'].sum()):,}",
    )
    return curves_df, report
