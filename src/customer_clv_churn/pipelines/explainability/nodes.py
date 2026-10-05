"""Explainability and risk factors via SHAP.

Explains the *base* churn model (the uncalibrated pipeline persisted by the
churn pipeline) for the scored cohort. Produces:

* global feature importance (mean absolute SHAP);
* per-customer top risk factors (feature, direction, SHAP value);
* per-segment aggregated risk factors.

Correlational/predictive factors only — these are **not** proven causal
drivers; causal claims would require experimentation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _positive_class_shap(shap_values: Any) -> np.ndarray:
    """Extract the positive-class SHAP matrix across shap/library versions."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def explain_churn_predictions(
    churn_model: Any,
    test_data: pd.DataFrame,
    customer_segments: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute SHAP values and derive global, customer and segment risk factors.

    Args:
        churn_model: Fitted uncalibrated sklearn pipeline.
        test_data: Scored cohort rows.
        customer_segments: Segment assignments.
        params: Explainability config (``feature_columns``/``top_factors``).

    Returns:
        ``(shap_global_importance, customer_risk_factors, explainability_report)``.
    """
    import shap  # noqa: PLC0415

    id_columns = set(params.get("id_columns", ["customer_id", "cohort_date"]))
    target = params.get("target", "churn")
    exclude = set(params.get("exclude_features", []))
    feature_columns = [
        c for c in test_data.columns if c not in id_columns and c not in exclude and c != target
    ]

    x_raw = test_data[feature_columns]
    preprocessor = churn_model.named_steps["preprocessor"]
    estimator = churn_model.named_steps["model"]
    x_transformed = preprocessor.transform(x_raw)
    feature_names = [_clean_feature_name(n) for n in preprocessor.get_feature_names_out()]

    explainer = shap.TreeExplainer(estimator)
    shap_values = _positive_class_shap(explainer.shap_values(x_transformed, check_additivity=False))
    if shap_values.shape[1] != len(feature_names):  # pragma: no cover - defensive
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=x_raw.index)
    abs_shap = shap_df.abs()

    # --- Global importance -------------------------------------------------
    global_importance = (
        abs_shap.mean()
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    global_importance["rank"] = global_importance.index + 1

    # --- Per-customer top risk factors ------------------------------------
    top_k = int(params.get("top_factors", 5))
    customer_ids = test_data["customer_id"].to_numpy()
    names = shap_df.columns.to_numpy()
    values = shap_df.to_numpy()
    abs_values = abs_shap.to_numpy()

    records = []
    top_idx_matrix = np.argsort(-abs_values, axis=1)[:, :top_k]
    for row_position, customer_id in enumerate(customer_ids):
        for rank, feature_position in enumerate(top_idx_matrix[row_position], start=1):
            value = float(values[row_position, feature_position])
            records.append(
                {
                    "customer_id": int(customer_id),
                    "rank": rank,
                    "feature": names[feature_position],
                    "shap_value": value,
                    "abs_shap": abs(value),
                    "direction": "increases_risk" if value > 0 else "decreases_risk",
                }
            )
    customer_risk_factors = pd.DataFrame(records)

    # --- Per-segment aggregation ------------------------------------------
    shap_with_segment = shap_df.copy()
    shap_with_segment["customer_id"] = customer_ids
    shap_with_segment = shap_with_segment.merge(
        customer_segments[["customer_id", "segment_name"]], on="customer_id", how="left"
    )
    segment_abs = shap_with_segment.groupby("segment_name")[feature_names].mean().abs()
    segment_factors: dict[str, list[dict[str, Any]]] = {}
    for segment_name, row in segment_abs.iterrows():
        top = row.sort_values(ascending=False).head(top_k)
        segment_factors[segment_name] = [
            {"feature": feature, "mean_abs_shap": round(float(row[feature]), 4)}
            for feature in top.index
        ]

    report: dict[str, Any] = {
        "explained_model": type(estimator).__name__,
        "n_customers_explained": int(len(test_data)),
        "n_features": int(len(feature_names)),
        "top_global_features": global_importance.head(20).to_dict(orient="records"),
        "segment_risk_factors": segment_factors,
        "interpretation_note": (
            "SHAP values are predictive/correlational attributions, not causal "
            "effects. Direction is relative to churn probability."
        ),
    }

    logger.info(
        "Explainability: explained %s customers x %s features | top: %s",
        f"{len(test_data):,}",
        len(feature_names),
        global_importance.iloc[0]["feature"],
    )
    return global_importance, customer_risk_factors, report
