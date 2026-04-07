"""Streamlit dashboard for HealthBeauty360."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models.model_utils import DATA_DIR, MODELS_DIR, SYNTHETIC_DIR, standardize_transactions

st.set_page_config(page_title="HealthBeauty360", page_icon="HB", layout="wide")

PALETTE = {
    "bg": "#11161D",
    "panel": "#171F28",
    "panel_soft": "#202B36",
    "grid": "rgba(223, 228, 235, 0.10)",
    "text": "#E9EEF2",
    "muted": "#9FB0BF",
    "teal": "#4FA89A",
    "amber": "#D49A45",
    "coral": "#C86457",
    "steel": "#6F8AA6",
    "lavender": "#8D7BAF",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(79,168,154,0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(212,154,69,0.08), transparent 24%),
            linear-gradient(180deg, #0f1419 0%, #131920 100%);
        color: {PALETTE['text']};
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3 {{
        font-family: 'Fraunces', Georgia, serif;
        color: {PALETTE['text']};
        letter-spacing: -0.02em;
    }}
    .stMarkdown, .stMetric, .stDataFrame, label, p, span, div {{
        font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        color: {PALETTE['text']};
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #121820 0%, #171f28 100%);
        border-right: 1px solid rgba(233,238,242,0.08);
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(23,31,40,0.95), rgba(17,22,29,0.92));
        border: 1px solid rgba(233,238,242,0.08);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
    }}
    .hero {{
        padding: 1.3rem 1.5rem;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(23,31,40,0.98), rgba(32,43,54,0.96));
        border: 1px solid rgba(233,238,242,0.08);
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }}
    .hero p {{
        color: {PALETTE['muted']};
        margin-bottom: 0;
    }}
    .insight {{
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(23,31,40,0.92), rgba(17,22,29,0.95));
        border-left: 4px solid {PALETTE['teal']};
        border-top: 1px solid rgba(233,238,242,0.06);
        border-right: 1px solid rgba(233,238,242,0.06);
        border-bottom: 1px solid rgba(233,238,242,0.06);
        margin: 0.35rem 0 1rem 0;
        color: {PALETTE['muted']};
    }}
    .section-note {{
        color: {PALETTE['muted']};
        margin-bottom: 0.8rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_transactions() -> pd.DataFrame:
    path = SYNTHETIC_DIR / "transactions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return standardize_transactions(pd.read_parquet(path))


def fmt_gbp(value: float) -> str:
    return f"GBP {value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def executive_kpis(transactions: pd.DataFrame) -> dict[str, float]:
    if transactions.empty:
        return {
            "total_revenue_gbp": 0.0,
            "units_sold": 0,
            "avg_order_value_gbp": 0.0,
            "gross_margin_pct": 0.0,
        }
    positive = transactions[transactions["quantity"] > 0].copy()
    revenue = float(positive["net_revenue_gbp"].sum())
    units = int(positive["quantity"].sum())
    orders = max(int(positive["invoice_id"].nunique()), 1)
    margin = float(positive["gross_margin_gbp"].sum() / revenue) if revenue else 0.0
    return {
        "total_revenue_gbp": round(revenue, 2),
        "units_sold": units,
        "avg_order_value_gbp": round(revenue / orders, 2),
        "gross_margin_pct": round(margin * 100, 2),
    }


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Space Grotesk"),
        title_font=dict(color=PALETTE["text"], family="Fraunces", size=22),
        legend_font=dict(color=PALETTE["muted"]),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=PALETTE["muted"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False, color=PALETTE["muted"])
    return fig


def insight_box(title: str, text: str) -> None:
    st.markdown(
        f"<div class='insight'><strong>{title}</strong><br>{text}</div>",
        unsafe_allow_html=True,
    )


def explain_weekly_revenue(weekly: pd.DataFrame) -> str:
    latest = float(weekly.iloc[-1]["net_revenue_gbp"])
    start = float(weekly.iloc[0]["net_revenue_gbp"])
    change_pct = ((latest - start) / start * 100) if start else 0.0
    peak = weekly.loc[weekly["net_revenue_gbp"].idxmax()]
    trough = weekly.loc[weekly["net_revenue_gbp"].idxmin()]
    direction = "up" if change_pct >= 0 else "down"
    return (
        f"Weekly revenue is trending {direction} by {abs(change_pct):.1f}% from the start of the visible period. "
        f"The strongest trading week was {peak['week_start']:%d %b %Y} at {fmt_gbp(float(peak['net_revenue_gbp']))}, "
        f"while the softest week was {trough['week_start']:%d %b %Y}."
    )


def explain_category_mix(category_mix: pd.DataFrame) -> str:
    leader = category_mix.iloc[0]
    share = float(leader["net_revenue_gbp"] / category_mix["net_revenue_gbp"].sum() * 100)
    runner_up = category_mix.iloc[1] if len(category_mix) > 1 else leader
    gap = float(leader["net_revenue_gbp"] - runner_up["net_revenue_gbp"])
    return (
        f"{leader['category_l1']} is the top revenue driver, contributing {share:.1f}% of category sales. "
        f"Its lead over the next category is {fmt_gbp(gap)}, which indicates where the assortment is currently concentrated."
    )


def explain_channel_mix(channel_mix: pd.DataFrame) -> str:
    leader = channel_mix.iloc[0]
    return (
        f"{leader['channel']} is the dominant channel at {leader['revenue_share_pct']:.1f}% of revenue. "
        f"If that share is materially larger than the rest, trading risk is concentrated in a single route to market."
    )


def explain_forecast(sku_forecast: pd.DataFrame) -> str:
    first_row = sku_forecast.iloc[0]
    peak_row = sku_forecast.loc[sku_forecast["forecast_units"].idxmax()]
    avg_forecast = float(sku_forecast["forecast_units"].mean())
    uncertainty = float((sku_forecast["forecast_upper"] - sku_forecast["forecast_lower"]).mean())
    return (
        f"This SKU is expected to average {avg_forecast:.1f} units per forecast period. "
        f"The peak forecast arrives on {peak_row['forecast_date']:%d %b %Y} at {peak_row['forecast_units']:.1f} units, "
        f"and the average forecast band width is {uncertainty:.1f} units, which gives a sense of planning uncertainty."
    )


def explain_segments(segment_mix: pd.DataFrame) -> str:
    leader = segment_mix.iloc[0]
    return (
        f"{leader['segment_name']} is the largest segment with {leader['customers']:,} customers. "
        f"That tells you where most of the customer base currently sits before prioritising activation or retention campaigns."
    )


def explain_churn(risk_counts: pd.DataFrame) -> str:
    high = int(risk_counts.loc[risk_counts["risk_band"] == "High", "customers"].sum())
    total = int(risk_counts["customers"].sum())
    high_share = (high / total * 100) if total else 0.0
    return (
        f"High-risk customers make up {high_share:.1f}% of the scored base. "
        f"That is the immediate retention queue, while medium-risk customers are the next-best audience for preventative CRM treatment."
    )


def explain_inventory(inventory_snapshot: pd.DataFrame) -> str:
    reorder_count = int(inventory_snapshot["reorder_recommended"].sum())
    avg_days_cover = float(inventory_snapshot["days_cover"].mean())
    top_risk = inventory_snapshot.iloc[0]
    return (
        f"{reorder_count:,} SKUs are flagged for reorder, with average days cover at {avg_days_cover:.1f}. "
        f"The most exposed SKU is {top_risk['stock_code']} with a stockout risk score of {top_risk['stockout_risk_score']:.1f}."
    )


def explain_trends(trend_counts: pd.DataFrame, trend_snapshot: pd.DataFrame) -> str:
    leader = trend_counts.iloc[0]
    hottest = trend_snapshot.sort_values("relative_trend_slope", ascending=False).iloc[0]
    return (
        f"{leader['trend_label']} is the most common demand state across the assortment. "
        f"The strongest positive momentum currently belongs to {hottest['stock_code']} in {hottest['category_l1']}, "
        f"with a relative trend slope of {hottest['relative_trend_slope']:.3f}."
    )


transactions = load_transactions()
forecast_df = read_parquet(MODELS_DIR / "demand_forecast_predictions.parquet")
segment_df = read_parquet(MODELS_DIR / "customer_segments.parquet")
churn_df = read_parquet(MODELS_DIR / "customer_churn_scores.parquet")
inventory_df = read_parquet(MODELS_DIR / "inventory_scores.parquet")
trend_df = read_parquet(MODELS_DIR / "trend_detection_sku.parquet")
trend_category_df = read_parquet(MODELS_DIR / "trend_detection_category.parquet")

st.markdown(
    """
    <div class="hero">
        <h1>HealthBeauty360 Control Room</h1>
        <p>Muted, operational view of demand, customers, inventory, and trend risk across the demo retail stack.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline_summary = read_json(DATA_DIR / "reports" / "weekly_pipeline_summary.json") or read_json(DATA_DIR / "reports" / "daily_pipeline_summary.json")

with st.sidebar:
    st.markdown("### System State")
    if pipeline_summary:
        st.metric("Pipeline Status", str(pipeline_summary.get("status", "unknown")).upper())
        st.metric("DQ Status", str(pipeline_summary.get("data_quality_status", "unknown")).upper())
    st.markdown("<p class='section-note'>Charts below explain themselves using the currently loaded data and model outputs.</p>", unsafe_allow_html=True)

kpis = executive_kpis(transactions)
columns = st.columns(4)
columns[0].metric("Revenue", fmt_gbp(kpis["total_revenue_gbp"]))
columns[1].metric("Units Sold", f"{kpis['units_sold']:,}")
columns[2].metric("Avg Order Value", f"GBP {kpis['avg_order_value_gbp']:,.2f}")
columns[3].metric("Gross Margin", fmt_pct(kpis["gross_margin_pct"]))

tab_overview, tab_forecast, tab_customer, tab_inventory, tab_trends = st.tabs(
    ["Overview", "Forecast", "Customers", "Inventory", "Trends"]
)

with tab_overview:
    if transactions.empty:
        st.info("Synthetic input files are not available yet. Run a daily or weekly pipeline first.")
    else:
        positive = transactions[transactions["quantity"] > 0].copy()
        weekly = (
            positive.assign(week_start=lambda data: data["invoice_date"].dt.to_period("W").dt.start_time)
            .groupby("week_start", as_index=False)["net_revenue_gbp"]
            .sum()
        )
        channel_mix = (
            positive.groupby("channel", as_index=False)["net_revenue_gbp"]
            .sum()
            .rename(columns={"net_revenue_gbp": "revenue_gbp"})
            .sort_values("revenue_gbp", ascending=False)
        )
        channel_mix["revenue_share_pct"] = channel_mix["revenue_gbp"] / channel_mix["revenue_gbp"].sum() * 100
        category_mix = (
            positive.groupby("category_l1", as_index=False)["net_revenue_gbp"]
            .sum()
            .sort_values("net_revenue_gbp", ascending=False)
        )
        monthly_category = (
            positive.assign(month_name=lambda data: data["invoice_date"].dt.strftime("%b"))
            .groupby(["month_name", "category_l1"], as_index=False)["net_revenue_gbp"]
            .sum()
        )
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_pivot = monthly_category.pivot(index="month_name", columns="category_l1", values="net_revenue_gbp").reindex(month_order).fillna(0)

        left, right = st.columns([1.7, 1])
        with left:
            revenue_fig = px.area(
                weekly,
                x="week_start",
                y="net_revenue_gbp",
                title="Weekly Revenue Rhythm",
                color_discrete_sequence=[PALETTE["teal"]],
            )
            revenue_fig.update_traces(line=dict(width=2), fillcolor="rgba(79,168,154,0.25)")
            st.plotly_chart(style_figure(revenue_fig), width="stretch")
            insight_box("Revenue insight", explain_weekly_revenue(weekly))
        with right:
            channel_fig = px.pie(
                channel_mix,
                names="channel",
                values="revenue_gbp",
                hole=0.58,
                title="Channel Revenue Mix",
                color_discrete_sequence=[PALETTE["amber"], PALETTE["teal"], PALETTE["steel"], PALETTE["coral"]],
            )
            st.plotly_chart(style_figure(channel_fig), width="stretch")
            insight_box("Channel insight", explain_channel_mix(channel_mix))

        left, right = st.columns([1.1, 1.2])
        with left:
            category_fig = px.bar(
                category_mix,
                x="net_revenue_gbp",
                y="category_l1",
                orientation="h",
                title="Category Revenue Ladder",
                color="net_revenue_gbp",
                color_continuous_scale=[PALETTE["steel"], PALETTE["amber"]],
            )
            category_fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(style_figure(category_fig), width="stretch")
            insight_box("Category insight", explain_category_mix(category_mix))
        with right:
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=monthly_pivot.values,
                    x=list(monthly_pivot.columns),
                    y=list(monthly_pivot.index),
                    colorscale=[[0, "#1B2530"], [0.4, PALETTE["steel"]], [0.75, PALETTE["teal"]], [1, PALETTE["amber"]]],
                )
            )
            heatmap.update_layout(title="Seasonality by Month and Category")
            st.plotly_chart(style_figure(heatmap), width="stretch")
            strongest_month = monthly_pivot.sum(axis=1).idxmax()
            st.markdown(
                f"<div class='insight'><strong>Seasonality insight</strong><br>{strongest_month} is currently the strongest month across all categories in the loaded demo history, which is the key seasonal planning anchor in this view.</div>",
                unsafe_allow_html=True,
            )

with tab_forecast:
    if forecast_df.empty:
        st.info("No forecast artifact found. Run the weekly pipeline to populate this view.")
    else:
        sku_options = sorted(forecast_df["stock_code"].unique().tolist())
        selected_sku = st.selectbox("Select SKU", sku_options)
        sku_forecast = forecast_df[forecast_df["stock_code"] == selected_sku].copy().sort_values("forecast_date")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Mean Forecast", f"{sku_forecast['forecast_units'].mean():.1f} units")
        metric_cols[1].metric("Peak Forecast", f"{sku_forecast['forecast_units'].max():.1f} units")
        metric_cols[2].metric("Avg Band Width", f"{(sku_forecast['forecast_upper'] - sku_forecast['forecast_lower']).mean():.1f} units")

        left, right = st.columns([1.65, 1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sku_forecast["forecast_date"], y=sku_forecast["forecast_units"], mode="lines+markers", name="Forecast", line=dict(color=PALETTE["amber"], width=3)))
            fig.add_trace(go.Scatter(x=sku_forecast["forecast_date"], y=sku_forecast["forecast_upper"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=sku_forecast["forecast_date"], y=sku_forecast["forecast_lower"], mode="lines", fill="tonexty", fillcolor="rgba(111,138,166,0.24)", line=dict(width=0), name="Uncertainty"))
            fig.update_layout(title=f"Demand Forecast for {selected_sku}")
            st.plotly_chart(style_figure(fig), width="stretch")
            insight_box("Forecast insight", explain_forecast(sku_forecast))
        with right:
            sku_forecast["week_over_week_change"] = sku_forecast["forecast_units"].diff().fillna(0.0)
            delta_fig = px.bar(
                sku_forecast,
                x="forecast_date",
                y="week_over_week_change",
                title="Week-on-Week Forecast Change",
                color="week_over_week_change",
                color_continuous_scale=[PALETTE["coral"], PALETTE["steel"], PALETTE["teal"]],
            )
            delta_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_figure(delta_fig), width="stretch")
            strongest_move = sku_forecast.loc[sku_forecast["week_over_week_change"].abs().idxmax()]
            insight_box(
                "Change insight",
                f"The biggest forecast step-change lands on {strongest_move['forecast_date']:%d %b %Y}, shifting by {strongest_move['week_over_week_change']:.1f} units versus the prior period.",
            )

        st.dataframe(sku_forecast, width="stretch")

with tab_customer:
    left, right = st.columns(2)
    with left:
        if segment_df.empty:
            st.info("Customer segmentation artifact not found.")
        else:
            segment_mix = segment_df["segment_name"].value_counts().rename_axis("segment_name").reset_index(name="customers")
            segment_fig = px.pie(
                segment_mix,
                names="segment_name",
                values="customers",
                title="Customer Segment Mix",
                hole=0.5,
                color_discrete_sequence=[PALETTE["amber"], PALETTE["teal"], PALETTE["steel"], PALETTE["coral"], PALETTE["lavender"]],
            )
            st.plotly_chart(style_figure(segment_fig), width="stretch")
            insight_box("Segmentation insight", explain_segments(segment_mix))
    with right:
        if churn_df.empty:
            st.info("Churn scores artifact not found.")
        else:
            risk_order = ["Low", "Medium", "High"]
            risk_counts = churn_df["risk_band"].value_counts().reindex(risk_order, fill_value=0).rename_axis("risk_band").reset_index(name="customers")
            risk_fig = px.bar(
                risk_counts,
                x="risk_band",
                y="customers",
                color="risk_band",
                title="Churn Risk Distribution",
                color_discrete_map={"High": PALETTE["coral"], "Medium": PALETTE["amber"], "Low": PALETTE["teal"]},
            )
            st.plotly_chart(style_figure(risk_fig), width="stretch")
            insight_box("Churn insight", explain_churn(risk_counts))

    if not segment_df.empty and {"frequency", "monetary_value", "segment_name"}.issubset(segment_df.columns):
        scatter = px.scatter(
            segment_df,
            x="frequency",
            y="monetary_value",
            color="segment_name",
            title="Customer Value Landscape",
            hover_data=["customer_id"],
            color_discrete_sequence=[PALETTE["amber"], PALETTE["teal"], PALETTE["steel"], PALETTE["coral"], PALETTE["lavender"]],
            opacity=0.7,
        )
        st.plotly_chart(style_figure(scatter), width="stretch")
        champion_candidate = segment_df.sort_values("monetary_value", ascending=False).iloc[0]
        insight_box(
            "Value insight",
            f"The most valuable observed customer in this sample sits in {champion_candidate['segment_name']} with monetary value of GBP {champion_candidate['monetary_value']:,.0f}. The upper-right quadrant shows where retention and premium upsell effort should focus.",
        )

with tab_inventory:
    if inventory_df.empty:
        st.info("Inventory scoring artifact not found.")
    else:
        inventory_snapshot = inventory_df.sort_values("stockout_risk_score", ascending=False).copy()
        top_risk = inventory_snapshot.head(20)
        left, right = st.columns([1.5, 1])
        with left:
            risk_fig = px.bar(
                top_risk,
                x="stock_code",
                y="stockout_risk_score",
                color="dead_stock_score",
                title="Top Stockout Risk SKUs",
                color_continuous_scale=[[0, PALETTE["steel"]], [1, PALETTE["coral"]]],
            )
            st.plotly_chart(style_figure(risk_fig), width="stretch")
            insight_box("Inventory insight", explain_inventory(inventory_snapshot))
        with right:
            scatter = px.scatter(
                inventory_snapshot,
                x="days_cover",
                y="stockout_risk_score",
                color="reorder_recommended",
                title="Coverage vs Stockout Risk",
                color_discrete_map={True: PALETTE["coral"], False: PALETTE["teal"]},
                hover_data=["stock_code", "abc_class", "xyz_class"],
            )
            st.plotly_chart(style_figure(scatter), width="stretch")
            low_cover = int((inventory_snapshot["days_cover"] < 14).sum())
            insight_box("Coverage insight", f"{low_cover:,} SKUs have less than 14 days of cover, which is the near-term replenishment pressure zone in this view.")

        heatmap_source = (
            inventory_snapshot.groupby(["abc_class", "xyz_class"], as_index=False)["stockout_risk_score"]
            .mean()
            .pivot(index="abc_class", columns="xyz_class", values="stockout_risk_score")
            .fillna(0)
        )
        matrix = go.Figure(
            data=go.Heatmap(
                z=heatmap_source.values,
                x=list(heatmap_source.columns),
                y=list(heatmap_source.index),
                colorscale=[[0, "#1B2530"], [0.5, PALETTE["amber"]], [1, PALETTE["coral"]]],
            )
        )
        matrix.update_layout(title="Mean Risk by ABC/XYZ Class")
        st.plotly_chart(style_figure(matrix), width="stretch")
        hottest_cell_value = float(heatmap_source.max().max())
        insight_box("ABC/XYZ insight", f"The riskiest ABC/XYZ pocket averages {hottest_cell_value:.1f} stockout-risk points. That matrix helps identify whether the problem sits in high-value staples or volatile tail items.")

        st.dataframe(
            inventory_snapshot[["stock_code", "abc_class", "xyz_class", "days_cover", "stockout_risk_score", "dead_stock_score", "reorder_recommended", "reorder_quantity_suggestion"]].head(50),
            width="stretch",
        )

with tab_trends:
    if trend_df.empty:
        st.info("Trend detection artifact not found.")
    else:
        trend_counts = trend_df["trend_label"].value_counts().rename_axis("trend_label").reset_index(name="sku_count")
        left, right = st.columns([1.2, 1.3])
        with left:
            trend_fig = px.bar(
                trend_counts,
                x="trend_label",
                y="sku_count",
                color="trend_label",
                title="Trend-State Distribution",
                color_discrete_map={"accelerating": PALETTE["teal"], "growing": PALETTE["amber"], "declining": PALETTE["coral"], "stable": PALETTE["steel"]},
            )
            st.plotly_chart(style_figure(trend_fig), width="stretch")
            insight_box("Trend insight", explain_trends(trend_counts, trend_df))
        with right:
            slope_scatter = px.scatter(
                trend_df,
                x="relative_trend_slope",
                y="z_score_recent_sales",
                color="trend_label",
                title="Momentum vs Recent Spike",
                hover_data=["stock_code", "category_l1"],
                color_discrete_map={"accelerating": PALETTE["teal"], "growing": PALETTE["amber"], "declining": PALETTE["coral"], "stable": PALETTE["steel"]},
                opacity=0.7,
            )
            st.plotly_chart(style_figure(slope_scatter), width="stretch")
            spike_count = int((trend_df["z_score_recent_sales"] >= 1.5).sum())
            insight_box("Spike insight", f"{spike_count:,} SKUs are currently showing a strong recent demand spike versus their own baseline, which is the fast-watchlist for trend-led replenishment or promotion response.")

        if not trend_category_df.empty:
            category_fig = px.bar(
                trend_category_df.sort_values("avg_relative_trend_slope", ascending=False),
                x="category_l1",
                y="avg_relative_trend_slope",
                color="category_label",
                title="Category Momentum",
                color_discrete_map={"hot": PALETTE["teal"], "steady": PALETTE["steel"], "cooling": PALETTE["coral"]},
            )
            st.plotly_chart(style_figure(category_fig), width="stretch")
            top_category = trend_category_df.sort_values("avg_relative_trend_slope", ascending=False).iloc[0]
            insight_box("Category momentum insight", f"{top_category['category_l1']} is the strongest category-level mover right now, with average relative slope of {top_category['avg_relative_trend_slope']:.3f}.")

        st.dataframe(
            trend_df[["stock_code", "category_l1", "relative_trend_slope", "z_score_recent_sales", "trend_label"]].sort_values(["relative_trend_slope", "z_score_recent_sales"], ascending=[False, False]).head(50),
            width="stretch",
        )

if pipeline_summary:
    st.caption(f"Last pipeline status: {pipeline_summary.get('status', 'unknown')} | DQ: {pipeline_summary.get('data_quality_status', 'unknown')}")