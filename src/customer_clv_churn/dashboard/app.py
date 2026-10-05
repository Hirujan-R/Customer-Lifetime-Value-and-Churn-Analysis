"""C-suite dashboard for the Customer CLV & Churn Engine.

Run with::

    streamlit run src/customer_clv_churn/dashboard/app.py

Five pages: Executive Overview, Customer Segmentation, Churn Risk, Revenue
Impact and Recommended Actions. All figures are *estimated* / *potential*;
the dataset is observational and no recovered revenue is claimed.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from customer_clv_churn.dashboard import data as ds

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def money(value: float) -> str:
    return f"£{value:,.0f}"


def _require_data() -> bool:
    if not ds.data_available():
        st.warning(
            "No pipeline outputs found. Run `kedro run` first to generate the "
            "customer intelligence datasets, then refresh this page."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Page 1 — Executive overview
# ---------------------------------------------------------------------------


def render_executive_overview() -> None:
    st.title("Executive Overview")
    st.caption("How serious is the churn problem, and where should management focus?")

    recs = ds.customer_recommendations()
    summary = ds.revenue_at_risk_summary()

    total_customers = len(recs)
    at_risk = recs[recs["churn_probability"] >= 0.5]
    active = total_customers - len(at_risk)
    high_value_threshold = recs["expected_future_clv"].quantile(0.75)
    high_value_at_risk = int((at_risk["expected_future_clv"] >= high_value_threshold).sum())

    row1 = st.columns(3)
    row1[0].metric("Total customers", f"{total_customers:,}")
    row1[1].metric("Active customers", f"{active:,}")
    row1[2].metric("Customers at risk (P≥0.5)", f"{len(at_risk):,}")

    row2 = st.columns(3)
    row2[0].metric("Revenue at risk (potential)", money(recs["revenue_at_risk"].sum()))
    row2[1].metric("Future CLV at risk (potential)", money(at_risk["expected_future_clv"].sum()))
    row2[2].metric("High-value customers at risk", f"{high_value_at_risk:,}")

    if summary:
        st.caption(
            f"Estimated over a {summary.get('horizon_days', 365):.0f}-day horizon · "
            f"mean churn probability {summary.get('mean_churn_probability', 0):.0%} · "
            "all values are modelled estimates, not realised revenue."
        )

    left, right = st.columns(2)
    with left:
        risk_counts = recs["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
        risk_counts.columns = ["Risk level", "Customers"]
        fig = px.bar(
            risk_counts,
            x="Risk level",
            y="Customers",
            color="Risk level",
            color_discrete_map=RISK_COLORS,
            title="Customer risk distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_segment = (
            recs.groupby("segment_name")["revenue_at_risk"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            by_segment,
            x="revenue_at_risk",
            y="segment_name",
            orientation="h",
            title="Revenue at risk by segment",
            labels={"revenue_at_risk": "Revenue at risk (£)", "segment_name": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top customers by expected revenue at risk")
    st.dataframe(
        recs.head(10)[
            [
                "customer_id",
                "segment_name",
                "risk_level",
                "churn_probability",
                "expected_future_clv",
                "revenue_at_risk",
                "primary_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    report = ds.recommendations_report()
    if report.get("actions"):
        st.subheader("Management focus — highest-value actions")
        for action in report["actions"][:3]:
            st.markdown(
                f"**{action['action']}** — {action['n_customers']:,} customers, "
                f"potential CLV at risk **{money(action['total_revenue_at_risk'])}**"
            )


# ---------------------------------------------------------------------------
# Page 2 — Customer segmentation
# ---------------------------------------------------------------------------


def render_segmentation() -> None:
    st.title("Customer Segmentation")
    st.caption("Behavioural clusters compared with interpretable business rules.")

    profiles = ds.segment_profiles()
    if profiles.empty:
        st.info("Segment profiles not found — run the segmentation pipeline.")
        return

    st.dataframe(
        profiles[
            [
                "segment_name",
                "n_customers",
                "total_net_revenue",
                "avg_value_proxy",
                "observed_churn_rate",
                "revenue_at_risk_proxy",
                "recommended_strategy",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            profiles, names="segment_name", values="n_customers", title="Customers by segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            profiles.sort_values("revenue_at_risk_proxy", ascending=True),
            x="revenue_at_risk_proxy",
            y="segment_name",
            orientation="h",
            title="Revenue at risk by segment (proxy)",
            labels={"revenue_at_risk_proxy": "Revenue at risk (£)", "segment_name": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    selected = st.selectbox("Inspect a segment", profiles["segment_name"].tolist())
    seg_report = ds.segmentation_report()
    characteristics = seg_report.get("key_behavioural_characteristics", {}).get(selected, [])
    risk_factors = ds.explainability_report().get("segment_risk_factors", {}).get(selected, [])
    strategy = profiles.set_index("segment_name").loc[selected, "recommended_strategy"]

    st.subheader(f"Segment detail — {selected}")
    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.markdown("**Key behavioural characteristics**")
        for item in characteristics:
            st.markdown(f"- {item['feature']} ({item['direction']})")
    with detail_cols[1]:
        st.markdown("**Main churn risk factors (SHAP)**")
        for item in risk_factors:
            st.markdown(f"- {item['feature']}")
    with detail_cols[2]:
        st.markdown("**Recommended intervention**")
        st.success(strategy)


# ---------------------------------------------------------------------------
# Page 3 — Churn risk
# ---------------------------------------------------------------------------


def render_churn_risk() -> None:
    st.title("Churn Risk")
    st.caption("Search, filter and explain individual customers.")

    recs = ds.customer_recommendations()
    if recs.empty:
        st.info("Customer risk table not found — run the pipeline.")
        return

    filters = st.columns(3)
    risk_filter = filters[0].multiselect("Risk level", RISK_ORDER, default=RISK_ORDER)
    segment_filter = filters[1].multiselect(
        "Segment", sorted(recs["segment_name"].dropna().unique()), default=None
    )
    search = filters[2].text_input("Search customer ID")

    view = recs[recs["risk_level"].isin(risk_filter)]
    if segment_filter:
        view = view[view["segment_name"].isin(segment_filter)]
    if search:
        view = view[view["customer_id"].astype(str).str.contains(search)]

    st.dataframe(
        view[
            [
                "customer_id",
                "segment_name",
                "churn_probability",
                "expected_future_clv",
                "revenue_at_risk",
                "risk_level",
            ]
        ].head(500),
        use_container_width=True,
        hide_index=True,
    )

    if view.empty:
        return
    selected_id = st.selectbox(
        "Select a customer to explain", view["customer_id"].head(200).tolist()
    )
    customer = recs[recs["customer_id"] == selected_id].iloc[0]

    metrics = st.columns(4)
    metrics[0].metric("Churn probability", f"{customer['churn_probability']:.1%}")
    metrics[1].metric("Expected CLV", money(customer["expected_future_clv"]))
    metrics[2].metric("Revenue at risk", money(customer["revenue_at_risk"]))
    metrics[3].metric("Segment", str(customer["segment_name"]))

    factors = ds.customer_risk_factors()
    customer_factors = factors[factors["customer_id"] == selected_id].sort_values("rank")
    if not customer_factors.empty:
        st.subheader("Key risk factors")
        left, right = st.columns([1, 1])
        with left:
            for _, row in customer_factors.iterrows():
                arrow = "↑" if row["direction"] == "increases_risk" else "↓"
                st.markdown(f"{arrow} **{row['feature']}** (SHAP {row['shap_value']:+.3f})")
        with right:
            fig = px.bar(
                customer_factors,
                x="shap_value",
                y="feature",
                orientation="h",
                title="SHAP contribution to churn probability",
                color="direction",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer profile (observation window)")
    features = ds.customer_features()
    profile = features[features["customer_id"] == selected_id]
    if not profile.empty:
        profile = profile.iloc[0]
        st.dataframe(
            profile[
                [
                    "recency_days",
                    "n_orders",
                    "net_revenue",
                    "aov",
                    "avg_interval",
                    "return_value_ratio",
                    "tenure_days",
                    "active_months",
                    "country_grouped",
                ]
            ].to_frame("value"),
            use_container_width=True,
        )

    st.subheader("Recommended action")
    st.success(str(customer["primary_action"]))
    if customer.get("dominant_risk_factor"):
        st.caption(f"Dominant risk factor: {customer['dominant_risk_factor']}")


# ---------------------------------------------------------------------------
# Page 4 — Revenue impact
# ---------------------------------------------------------------------------


def render_revenue_impact() -> None:
    st.title("Revenue Impact")
    st.caption("Naive recency rule vs ML-driven, value-weighted prioritisation.")

    report = ds.prioritisation_report()
    priorities = ds.retention_priorities()
    if not report or priorities.empty:
        st.info("Prioritisation report not found — run the pipeline.")
        return

    budgets = sorted(int(b) for b in report.get("budgets", [100, 250, 500, 1000]))
    budget = st.select_slider("Customers we can target", options=budgets, value=budgets[0])

    strategies = report["strategies"]
    comparison = []
    for name in ["recency_rule", "churn_probability", "expected_clv", "revenue_at_risk"]:
        metrics = strategies[name]["by_budget"][str(budget)]
        comparison.append(
            {
                "Strategy": name.replace("_", " ").title(),
                "Customers targeted": metrics["n_targeted"],
                "Revenue at risk captured": metrics["revenue_at_risk_captured"],
                "Expected CLV captured": metrics["expected_clv_captured"],
                "Observed future revenue captured": metrics["observed_future_revenue_captured"],
                "High-value captured": metrics["high_value_customers_captured"],
            }
        )
    comparison_df = pd.DataFrame(comparison)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        comparison_df,
        x="Strategy",
        y="Revenue at risk captured",
        title=f"Revenue at risk captured at budget = {budget}",
    )
    st.plotly_chart(fig, use_container_width=True)

    headline = report.get("headline", {})
    if headline:
        st.info(
            f"At a budget of {headline['at_budget']} customers, value-weighted "
            f"prioritisation captures **{money(headline['recommended_revenue_at_risk_captured'])}** "
            f"of revenue at risk versus **{money(headline['naive_revenue_at_risk_captured'])}** "
            f"for the recency rule "
            f"({headline['relative_uplift']:.1f}× more)."
        )

    st.subheader(f"Who to prioritise (top {budget})")
    top = priorities.head(budget)
    st.dataframe(
        top[
            [
                "customer_id",
                "segment_name",
                "churn_probability",
                "expected_future_clv",
                "revenue_at_risk",
                "priority_rank",
            ]
        ].head(200),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("**Segment mix of the prioritised customers**")
    mix = top.groupby("segment_name")["revenue_at_risk"].sum().sort_values(ascending=False)
    st.bar_chart(mix)


# ---------------------------------------------------------------------------
# Page 5 — Recommended actions
# ---------------------------------------------------------------------------


def render_recommended_actions() -> None:
    st.title("Recommended Actions")
    st.caption("Executive action list generated from the analysis.")

    report = ds.recommendations_report()
    if not report.get("actions"):
        st.info("Recommendations report not found — run the pipeline.")
        return

    for action in report["actions"][:8]:
        segments = ", ".join(
            f"{s['segment']} ({s['customers']})" for s in action.get("top_segments", [])
        )
        st.markdown(
            f"### {action['action']}\n"
            f"- **{action['n_customers']:,} customers** "
            f"({action['mean_churn_probability']:.0%} mean churn probability)\n"
            f"- **{money(action['total_revenue_at_risk'])}** potential CLV at risk"
            + (f"\n- Top segments: {segments}" if segments else "")
        )

    st.warning(report.get("causal_caveat", ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Customer CLV & Churn Engine",
        page_icon="📊",
        layout="wide",
    )
    st.sidebar.title("Customer Intelligence")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Overview",
            "Customer Segmentation",
            "Churn Risk",
            "Revenue Impact",
            "Recommended Actions",
        ],
    )
    st.sidebar.caption(
        "Estimates only. Observational data — A/B tests required before claiming recovered revenue."
    )

    if page == "Executive Overview":
        render_executive_overview()
    elif page == "Customer Segmentation":
        render_segmentation()
    elif page == "Churn Risk":
        render_churn_risk()
    elif page == "Revenue Impact":
        render_revenue_impact()
    else:
        render_recommended_actions()


if __name__ == "__main__":
    main()
