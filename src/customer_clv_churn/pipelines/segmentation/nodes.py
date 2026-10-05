"""Customer segmentation.

Segments are discovered with unsupervised clustering (K-Means, Gaussian
Mixture Models, Agglomerative hierarchical) on **observation-window features
only** — the churn label is never used to form the segments. The best model is
selected by silhouette score.

Clusters are then *interpreted*: compared against an interpretable RFM
rule-based segmentation, named by their dominant rule segment, profiled on
value / churn / behaviour, and mapped to a recommended retention strategy.

Churn rate, value and "revenue at risk" here use the **observed** performance
window and the annualised value **proxy**. Once the churn and CLV models land,
``observed_churn_rate`` is replaced by the calibrated model probability.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RULE_SEGMENTS = (
    "Champions",
    "Loyal customers",
    "High-value at risk",
    "New customers",
    "Occasional buyers",
    "High-return customers",
    "Lost customers",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_matrix(features: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """Select, log-transform, impute and scale the clustering features."""
    columns = list(params["cluster_features"])
    matrix = features[columns].copy()

    for column in params.get("log_features", []):
        if column in matrix.columns:
            matrix[column] = np.log1p(matrix[column].clip(lower=0))

    matrix = matrix.fillna(matrix.median(numeric_only=True))
    return StandardScaler().fit_transform(matrix)


def _fit_labels(model: str, k: int, matrix: np.ndarray, random_state: int) -> np.ndarray | None:
    """Fit a single candidate clustering and return labels (or ``None``)."""
    if model == "kmeans":
        estimator = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = estimator.fit_predict(matrix)
    elif model == "gmm":
        estimator = GaussianMixture(
            n_components=k, covariance_type="full", random_state=random_state
        )
        labels = estimator.fit_predict(matrix)
    elif model == "hierarchical":
        estimator = AgglomerativeClustering(n_clusters=k)
        labels = estimator.fit_predict(matrix)
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"Unknown segmentation model: {model}")

    if len(set(labels)) < 2:
        return None
    return labels


def _rule_based_segments(features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Assign interpretable RFM-style business-rule segments."""
    recency = features["recency_days"]
    frequency = features["n_orders"]
    monetary = features["net_revenue"]

    r_score = 6 - pd.qcut(recency.rank(method="first"), 5, labels=False)
    f_score = pd.qcut(frequency.rank(method="first"), 5, labels=False) + 1
    m_score = pd.qcut(monetary.rank(method="first"), 5, labels=False) + 1

    conditions = [
        features["return_value_ratio"] > float(params.get("high_return_threshold", 0.5)),
        (features["tenure_days"] < int(params.get("new_customer_days", 180))) & (f_score <= 2),
        (r_score >= 4) & (f_score >= 4) & (m_score >= 4),
        (m_score >= 4) & (r_score <= 2),
        r_score <= 1,
        (f_score <= 2) & (m_score <= 2),
        (r_score >= 3) & (f_score >= 3),
    ]
    choices = [
        "High-return customers",
        "New customers",
        "Champions",
        "High-value at risk",
        "Lost customers",
        "Occasional buyers",
        "Loyal customers",
    ]
    return pd.Series(
        np.select(conditions, choices, default="Occasional buyers"),
        index=features.index,
        name="rule_segment",
    )


def _name_clusters(segments: pd.DataFrame) -> pd.DataFrame:
    """Name each cluster after its dominant rule-based segment."""
    majority = (
        segments.groupby("cluster")["rule_segment"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    counts = segments["cluster"].value_counts().to_dict()
    duplicate_names = {
        name for name in majority.values() if list(majority.values()).count(name) > 1
    }

    names = {}
    for cluster, name in majority.items():
        names[cluster] = f"{name} #{cluster}" if name in duplicate_names else name
    segments["segment_name"] = segments["cluster"].map(names)
    logger.info("Cluster sizes: %s", {names[k]: v for k, v in counts.items()})
    return segments


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def segment_customers(
    customer_features: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster customers and interpret the clusters.

    Args:
        customer_features: Observation-window feature store.
        params: Segmentation config (features, k range, models, strategies).

    Returns:
        ``(customer_segments, model_report)`` where ``customer_segments`` has
        one row per customer with ``cluster``, ``segment_name`` and
        ``rule_segment``.
    """
    features = customer_features.reset_index(drop=True)
    matrix = _prepare_matrix(features, params)
    random_state = int(params.get("random_state", 42))
    k_range = list(params.get("k_range", [2, 3, 4, 5, 6, 7, 8]))
    models = list(params.get("models", ["kmeans", "gmm", "hierarchical"]))

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for k in k_range:
        for model in models:
            labels = _fit_labels(model, k, matrix, random_state)
            if labels is None:
                continue
            score = float(silhouette_score(matrix, labels))
            candidates.append({"model": model, "k": k, "silhouette": round(score, 4)})
            if best is None or score > best["silhouette"]:
                best = {"model": model, "k": k, "silhouette": score, "labels": labels}

    if best is None:  # pragma: no cover - guarded by config
        raise RuntimeError("No valid segmentation model could be fitted")

    segments = pd.DataFrame(
        {
            "customer_id": features["customer_id"].to_numpy(),
            "cluster": best["labels"],
        }
    )
    segments["rule_segment"] = _rule_based_segments(features, params).to_numpy()
    segments = _name_clusters(segments)

    cross_tab = pd.crosstab(segments["segment_name"], segments["rule_segment"]).to_dict(
        orient="index"
    )

    model_report: dict[str, Any] = {
        "chosen_model": best["model"],
        "chosen_k": best["k"],
        "chosen_silhouette": round(best["silhouette"], 4),
        "candidates": sorted(candidates, key=lambda c: c["silhouette"], reverse=True),
        "cluster_sizes": segments["segment_name"].value_counts().to_dict(),
        "cluster_vs_rule_crosstab": cross_tab,
        "note": (
            "Segments are formed on observation-window features only; the churn "
            "label is not used for clustering."
        ),
    }

    logger.info(
        "Chosen segmentation: %s with k=%s (silhouette=%.3f)",
        best["model"],
        best["k"],
        best["silhouette"],
    )
    return segments, model_report


def _profile_group(
    data: pd.DataFrame,
    group_column: str,
    cluster_column: str | None,
    strategies: dict[str, str],
) -> pd.DataFrame:
    """Aggregate business metrics for a given segmentation column."""
    grouped = data.groupby(group_column)
    profiles = grouped.agg(
        n_customers=("customer_id", "nunique"),
        total_net_revenue=("net_revenue", "sum"),
        avg_net_revenue=("net_revenue", "mean"),
        avg_value_proxy=("historical_value_proxy", "mean"),
        observed_churn_rate=("churn", "mean"),
        avg_recency_days=("recency_days", "mean"),
        avg_n_orders=("n_orders", "mean"),
        avg_aov=("aov", "mean"),
        avg_return_ratio=("return_value_ratio", "mean"),
        avg_recent_revenue_share=("recent_revenue_share", "mean"),
        avg_tenure_days=("tenure_days", "mean"),
    ).reset_index()
    if cluster_column is not None:
        profiles[cluster_column] = grouped[cluster_column].first().to_numpy()

    churners = data.loc[data["churn"] == 1]
    revenue_at_risk = (
        churners.groupby(group_column)["historical_value_proxy"]
        .sum()
        .rename("revenue_at_risk_proxy")
    )
    churn_counts = churners.groupby(group_column)["customer_id"].nunique().rename("n_churners")
    profiles = profiles.merge(revenue_at_risk, on=group_column, how="left").merge(
        churn_counts, on=group_column, how="left"
    )
    profiles["n_churners"] = profiles["n_churners"].fillna(0).astype(int)
    profiles["revenue_at_risk_proxy"] = profiles["revenue_at_risk_proxy"].fillna(0.0)
    profiles["share_of_customers"] = (
        profiles["n_customers"] / profiles["n_customers"].sum()
    ).round(4)
    profiles["recommended_strategy"] = profiles[group_column].map(
        lambda name: strategies.get(
            str(name).split(" #")[0], "Monitor and re-evaluate at the next scoring cycle."
        )
    )
    return profiles.sort_values("revenue_at_risk_proxy", ascending=False).reset_index(drop=True)


def _segment_characteristics(
    data: pd.DataFrame,
    params: dict[str, Any],
    group_column: str,
) -> dict[str, list[dict[str, Any]]]:
    """Top standardised feature deviations per group (correlational, not causal)."""
    numeric_columns = [
        c
        for c in params["cluster_features"]
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
    ]
    standardised = (data[numeric_columns] - data[numeric_columns].mean()) / data[
        numeric_columns
    ].std(ddof=0)
    standardised[group_column] = data[group_column].to_numpy()
    deviations = standardised.groupby(group_column)[numeric_columns].mean()

    characteristics: dict[str, list[dict[str, Any]]] = {}
    for name, row in deviations.iterrows():
        top = row.abs().sort_values(ascending=False).head(5)
        characteristics[name] = [
            {
                "feature": feature,
                "direction": "higher" if row[feature] > 0 else "lower",
                "z_score": round(float(row[feature]), 2),
            }
            for feature in top.index
        ]
    return characteristics


def profile_segments(
    customer_segments: pd.DataFrame,
    customer_features: pd.DataFrame,
    churn_labels: pd.DataFrame,
    baseline_scores: pd.DataFrame,
    segmentation_model_report: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Profile clusters **and** rule-based segments, with recommendations.

    Args:
        customer_segments: Output of :func:`segment_customers`.
        customer_features: Observation-window feature store.
        churn_labels: Held-out churn labels and observed future revenue.
        baseline_scores: Provides the annualised value proxy.
        segmentation_model_report: Model-selection report to fold in.
        params: Segmentation config including ``strategy_map``.

    Returns:
        ``(segment_profiles, rule_segment_profiles, segmentation_report)``.
    """
    data = (
        customer_segments.merge(customer_features, on="customer_id", how="left")
        .merge(
            churn_labels[["customer_id", "churn", "future_revenue"]],
            on="customer_id",
            how="left",
        )
        .merge(
            baseline_scores[["customer_id", "historical_value_proxy"]],
            on="customer_id",
            how="left",
        )
    )
    data["churn"] = data["churn"].fillna(0).astype(int)
    data["historical_value_proxy"] = data["historical_value_proxy"].fillna(0.0)

    strategies = params.get("strategy_map", {})
    segment_profiles = _profile_group(data, "segment_name", "cluster", strategies)
    rule_segment_profiles = _profile_group(data, "rule_segment", None, strategies)

    report: dict[str, Any] = {
        "chosen_model": segmentation_model_report["chosen_model"],
        "chosen_k": segmentation_model_report["chosen_k"],
        "chosen_silhouette": segmentation_model_report["chosen_silhouette"],
        "n_segments": int(len(segment_profiles)),
        "value_proxy_definition": "12 x observed monthly net revenue (annualised run-rate)",
        "churn_metric_note": (
            "observed_churn_rate is the held-out 90-day churn rate; it will be "
            "replaced by calibrated model churn probability in the modelling stage."
        ),
        "segments": segment_profiles.round(4).to_dict(orient="records"),
        "rule_based_segments": rule_segment_profiles.round(4).to_dict(orient="records"),
        "key_behavioural_characteristics": _segment_characteristics(data, params, "segment_name"),
        "cluster_vs_rule_crosstab": segmentation_model_report["cluster_vs_rule_crosstab"],
        "candidate_models": segmentation_model_report["candidates"],
    }

    logger.info(
        "Profiled %s clusters and %s rule-based segments",
        len(segment_profiles),
        len(rule_segment_profiles),
    )
    return segment_profiles, rule_segment_profiles, report
