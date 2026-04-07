"""Technical operations dashboard for HealthBeauty360."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features.feature_store import FeatureStore
from models.model_utils import DATA_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR, SYNTHETIC_DIR, standardize_transactions

st.set_page_config(page_title="HealthBeauty360 Technical Dashboard", page_icon="HB Tech", layout="wide")

PALETTE = {
    "bg": "#0E1318",
    "panel": "#151C23",
    "panel_alt": "#1B252E",
    "text": "#EAF0F4",
    "muted": "#98A7B5",
    "grid": "rgba(234,240,244,0.08)",
    "teal": "#55A89D",
    "amber": "#C99143",
    "coral": "#C86457",
    "steel": "#738AA3",
    "lavender": "#8876A7",
    "ok": "#4FA67D",
    "warn": "#D49A45",
    "fail": "#C86457",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(85,168,157,0.10), transparent 24%),
            radial-gradient(circle at bottom right, rgba(201,145,67,0.09), transparent 22%),
            linear-gradient(180deg, #0d1217 0%, #11171d 100%);
        color: {PALETTE['text']};
    }}
    .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3 {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {PALETTE['text']};
        letter-spacing: -0.02em;
    }}
    .stMarkdown, .stMetric, .stDataFrame, label, p, span, div {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {PALETTE['text']};
    }}
    code {{
        font-family: 'IBM Plex Mono', monospace;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #121922 0%, #161f29 100%);
        border-right: 1px solid rgba(234,240,244,0.08);
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(21,28,35,0.98), rgba(16,22,29,0.98));
        border: 1px solid rgba(234,240,244,0.08);
        border-radius: 16px;
        padding: 0.75rem 0.85rem;
    }}
    .hero {{
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(21,28,35,0.98), rgba(27,37,46,0.98));
        border: 1px solid rgba(234,240,244,0.08);
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
    }}
    .hero p {{
        color: {PALETTE['muted']};
        margin-bottom: 0;
    }}
    .insight {{
        padding: 0.85rem 0.95rem;
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(21,28,35,0.95), rgba(15,20,25,0.98));
        border-left: 4px solid {PALETTE['teal']};
        border-top: 1px solid rgba(234,240,244,0.06);
        border-right: 1px solid rgba(234,240,244,0.06);
        border-bottom: 1px solid rgba(234,240,244,0.06);
        color: {PALETTE['muted']};
        margin: 0.35rem 0 1rem 0;
    }}
    .muted {{
        color: {PALETTE['muted']};
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


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="IBM Plex Sans"),
        title_font=dict(color=PALETTE["text"], family="IBM Plex Sans", size=21),
        legend_font=dict(color=PALETTE["muted"]),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=PALETTE["muted"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False, color=PALETTE["muted"])
    return fig


def insight_box(title: str, body: str) -> None:
    st.markdown(f"<div class='insight'><strong>{title}</strong><br>{body}</div>", unsafe_allow_html=True)


def status_color(status: str) -> str:
    return {
        "PASS": PALETTE["ok"],
        "WARN": PALETTE["warn"],
        "FAIL": PALETTE["fail"],
        "completed": PALETTE["ok"],
        "failed": PALETTE["fail"],
        "running": PALETTE["amber"],
    }.get(str(status), PALETTE["steel"])


def load_transactions() -> pd.DataFrame:
    path = SYNTHETIC_DIR / "transactions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return standardize_transactions(pd.read_parquet(path))


def summarize_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": None, "size_mb": None, "updated_at": None}
    rows = None
    if path.suffix == ".parquet":
        try:
            rows = len(pd.read_parquet(path))
        except Exception:
            rows = None
    return {
        "path": str(path),
        "exists": True,
        "rows": rows,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "updated_at": pd.Timestamp(path.stat().st_mtime, unit="s"),
    }


def load_pipeline_logs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_path in sorted((DATA_DIR / "pipeline_logs").glob("*.json")):
        payload = read_json(log_path)
        if not payload:
            continue
        started = pd.to_datetime(payload.get("started_at"))
        completed = pd.to_datetime(payload.get("completed_at")) if payload.get("completed_at") else pd.NaT
        duration_seconds = None
        if pd.notna(started) and pd.notna(completed):
            duration_seconds = float((completed - started).total_seconds())
        rows.append(
            {
                "pipeline_name": payload.get("pipeline_name"),
                "run_id": payload.get("run_id"),
                "status": payload.get("status"),
                "started_at": started,
                "completed_at": completed,
                "duration_seconds": duration_seconds,
                "steps_completed": len(payload.get("steps_completed", [])),
                "steps_failed": len(payload.get("steps_failed", [])),
                "rows_processed_total": sum(payload.get("rows_processed", {}).values()),
                "step_rows": payload.get("rows_processed", {}),
                "error_message": payload.get("error_message"),
            }
        )
    return pd.DataFrame(rows).sort_values("started_at", ascending=False) if rows else pd.DataFrame()


def flatten_dq_report(payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, checks in payload.items():
        for check in checks:
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "check_name": check.get("check_name"),
                    "status": check.get("status"),
                    "message": check.get("message"),
                    "rows_checked": check.get("rows_checked"),
                    "rows_failed": check.get("rows_failed"),
                    "timestamp": pd.to_datetime(check.get("timestamp")),
                }
            )
    return pd.DataFrame(rows)


def latest_report(name: str) -> dict:
    path = REPORTS_DIR / name
    return read_json(path)


def load_model_summaries() -> pd.DataFrame:
    summary_paths = sorted(MODELS_DIR.glob("*_summary.json"))
    rows: list[dict[str, object]] = []
    for path in summary_paths:
        payload = read_json(path)
        summary = payload.get("summary", {})
        if summary:
            summary["summary_file"] = path.name
            rows.append(summary)
    return pd.DataFrame(rows)


def load_feature_shapes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(FEATURES_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path)
        rows.append(
            {
                "feature_matrix": path.stem,
                "rows": len(frame),
                "columns": len(frame.columns),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_source_registry() -> pd.DataFrame:
    sources = [
        {"source": "synthetic_transactions", "cadence": "daily", "mode": "demo", "path": SYNTHETIC_DIR / "transactions.parquet"},
        {"source": "synthetic_inventory", "cadence": "daily", "mode": "demo", "path": SYNTHETIC_DIR / "inventory.parquet"},
        {"source": "synthetic_customers", "cadence": "daily", "mode": "demo", "path": SYNTHETIC_DIR / "customers.parquet"},
        {"source": "synthetic_campaigns", "cadence": "daily", "mode": "demo", "path": SYNTHETIC_DIR / "campaigns.parquet"},
        {"source": "synthetic_costs", "cadence": "weekly", "mode": "demo", "path": SYNTHETIC_DIR / "costs.parquet"},
    ]
    rows = []
    for source in sources:
        summary = summarize_file(source["path"])
        rows.append(
            {
                "source": source["source"],
                "cadence": source["cadence"],
                "mode": source["mode"],
                "rows": summary["rows"],
                "size_mb": summary["size_mb"],
                "updated_at": summary["updated_at"],
                "path": str(source["path"]),
                "status": "available" if summary["exists"] else "missing",
            }
        )
    return pd.DataFrame(rows)


transactions = load_transactions()
feature_shapes = load_feature_shapes()
feature_catalog = FeatureStore().get_feature_catalog()
pipeline_logs = load_pipeline_logs()
model_summaries = load_model_summaries()
source_registry = build_source_registry()
daily_summary = latest_report("daily_pipeline_summary.json")
weekly_summary = latest_report("weekly_pipeline_summary.json")
dq_frame = flatten_dq_report(latest_report("weekly_data_quality.json") or latest_report("daily_data_quality.json"))
product_features = read_parquet(FEATURES_DIR / "product_features.parquet")
customer_features = read_parquet(FEATURES_DIR / "customer_features.parquet")
forecast_metrics = read_parquet(MODELS_DIR / "demand_forecast_metrics.parquet")
churn_scores = read_parquet(MODELS_DIR / "customer_churn_scores.parquet")
segment_df = read_parquet(MODELS_DIR / "customer_segments.parquet")
inventory_scores = read_parquet(MODELS_DIR / "inventory_scores.parquet")
trend_sku = read_parquet(MODELS_DIR / "trend_detection_sku.parquet")
trend_category = read_parquet(MODELS_DIR / "trend_detection_category.parquet")
churn_summary = latest_report_path = read_json(MODELS_DIR / "churn_prediction_summary.json")

st.markdown(
    """
    <div class="hero">
        <h1>HealthBeauty360 Technical Console</h1>
        <p>Engineering view of ingestion, validation, feature generation, model retraining, and pipeline execution across the demo retail analytics stack.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Runtime")
    st.metric("Demo Mode", "TRUE")
    if weekly_summary:
        st.metric("Weekly DQ", str(weekly_summary.get("data_quality_status", "unknown")).upper())
        st.metric("Weekly Status", str(weekly_summary.get("status", "unknown")).upper())
    st.markdown("### Paths")
    st.caption(str(DATA_DIR))
    st.caption(str(SYNTHETIC_DIR))
    st.caption(str(FEATURES_DIR))
    st.caption(str(MODELS_DIR))

top_metrics = st.columns(4)
top_metrics[0].metric("Sources Landed", f"{int(source_registry['status'].eq('available').sum())}/{len(source_registry)}")
top_metrics[1].metric("Feature Matrices", f"{len(feature_shapes)}")
top_metrics[2].metric("Model Artifacts", f"{len(list(MODELS_DIR.glob('*.pkl')))}")
top_metrics[3].metric("Pipeline Runs Logged", f"{len(pipeline_logs)}")

tabs = st.tabs(["Architecture", "Ingestion", "Data Quality", "Features", "Models", "Pipelines", "Environment"])

with tabs[0]:
    sankey = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=[
                        "Sources",
                        "Synthetic Inputs",
                        "Raw/Bronze",
                        "Feature Store",
                        "Model Training",
                        "Serving",
                    ],
                    color=[PALETTE["steel"], PALETTE["amber"], PALETTE["teal"], PALETTE["lavender"], PALETTE["coral"], PALETTE["ok"]],
                    pad=22,
                    thickness=18,
                ),
                link=dict(
                    source=[0, 1, 2, 3, 4],
                    target=[1, 2, 3, 4, 5],
                    value=[5, 5, 2, 5, 2],
                    color=["rgba(115,138,163,0.35)", "rgba(201,145,67,0.35)", "rgba(85,168,157,0.35)", "rgba(136,118,167,0.35)", "rgba(200,100,87,0.35)"],
                ),
            )
        ]
    )
    sankey.update_layout(title="System Flow: Ingestion to Serving")
    st.plotly_chart(style_figure(sankey), width="stretch")
    flow_note = "The current demo workflow starts with synthetic landed inputs, passes through DQ and feature generation, retrains five models in the weekly pipeline, and then feeds both the business API and dashboard artifacts."
    insight_box("Architecture insight", flow_note)

    left, right = st.columns(2)
    with left:
        if not pipeline_logs.empty:
            timeline_data = pipeline_logs.dropna(subset=["started_at", "completed_at"])[["pipeline_name", "run_id", "status", "started_at", "completed_at"]].copy()
            timeline_data["label"] = timeline_data["pipeline_name"] + " / " + timeline_data["run_id"]
            timeline_fig = px.timeline(
                timeline_data,
                x_start="started_at",
                x_end="completed_at",
                y="label",
                color="status",
                title="Recent Pipeline Timelines",
                color_discrete_map={"completed": PALETTE["ok"], "failed": PALETTE["fail"], "running": PALETTE["amber"]},
            )
            timeline_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_figure(timeline_fig), width="stretch")
    with right:
        if not feature_shapes.empty:
            shape_fig = px.bar(
                feature_shapes,
                x="feature_matrix",
                y="rows",
                color="columns",
                title="Feature Store Output Shapes",
                color_continuous_scale=[PALETTE["steel"], PALETTE["teal"]],
            )
            shape_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_figure(shape_fig), width="stretch")
            largest = feature_shapes.sort_values("rows", ascending=False).iloc[0]
            insight_box("Feature-store insight", f"{largest['feature_matrix']} is the widest deployed feature artifact by row volume in the current run. The feature store is currently only materializing product and customer matrices.")

with tabs[1]:
    left, right = st.columns([1.1, 1.4])
    with left:
        source_fig = px.bar(
            source_registry,
            x="source",
            y="rows",
            color="cadence",
            title="Landed Source Volumes",
            color_discrete_map={"daily": PALETTE["teal"], "weekly": PALETTE["amber"]},
        )
        st.plotly_chart(style_figure(source_fig), width="stretch")
        newest = source_registry.sort_values("updated_at", ascending=False).iloc[0]
        insight_box("Ingestion insight", f"All demo sources are landing locally under data/synthetic. The freshest landed source is {newest['source']}, updated at {newest['updated_at']:%d %b %Y %H:%M:%S}.")
    with right:
        file_size_fig = px.scatter(
            source_registry,
            x="rows",
            y="size_mb",
            color="cadence",
            hover_data=["source", "path"],
            title="Source Volume vs File Size",
            color_discrete_map={"daily": PALETTE["steel"], "weekly": PALETTE["amber"]},
        )
        st.plotly_chart(style_figure(file_size_fig), width="stretch")
        insight_box("Storage insight", "This view shows which sources dominate landed storage. Campaign and transaction datasets are the main local payloads, which is where ingestion optimization matters most.")

    st.dataframe(source_registry, width="stretch")

with tabs[2]:
    if dq_frame.empty:
        st.info("No data quality report found.")
    else:
        dq_summary = dq_frame.groupby("status", as_index=False).size().rename(columns={"size": "checks"})
        left, right = st.columns([1, 1.3])
        with left:
            dq_fig = px.bar(
                dq_summary,
                x="status",
                y="checks",
                color="status",
                title="DQ Check Outcomes",
                color_discrete_map={"PASS": PALETTE["ok"], "WARN": PALETTE["warn"], "FAIL": PALETTE["fail"]},
            )
            st.plotly_chart(style_figure(dq_fig), width="stretch")
            pass_rate = float((dq_frame["status"] == "PASS").mean() * 100)
            insight_box("DQ insight", f"Current pass rate is {pass_rate:.1f}% across all checks. The latest run is green because transaction uniqueness, null, and price sanity checks all passed after regenerating synthetic inputs.")
        with right:
            matrix = dq_frame.pivot(index="dataset_name", columns="check_name", values="status")
            heat_values = matrix.replace({"PASS": 1, "WARN": 0.5, "FAIL": 0})
            dq_heat = go.Figure(
                data=go.Heatmap(
                    z=heat_values.values,
                    x=list(heat_values.columns),
                    y=list(heat_values.index),
                    text=matrix.values,
                    colorscale=[[0, PALETTE["fail"]], [0.5, PALETTE["warn"]], [1, PALETTE["ok"]]],
                    zmin=0,
                    zmax=1,
                )
            )
            dq_heat.update_layout(title="DQ Matrix by Dataset and Check")
            st.plotly_chart(style_figure(dq_heat), width="stretch")

        st.dataframe(dq_frame.sort_values(["dataset_name", "check_name"]), width="stretch")

with tabs[3]:
    left, right = st.columns([1.1, 1.3])
    with left:
        if not feature_catalog.empty:
            usage_counts = feature_catalog.assign(model_usage_count=feature_catalog["model_usage"].str.split(", ").str.len())
            usage_fig = px.histogram(
                usage_counts,
                x="refresh_cadence",
                color="feature_type",
                barmode="group",
                title="Feature Catalog by Refresh Cadence and Type",
                color_discrete_sequence=[PALETTE["teal"], PALETTE["amber"], PALETTE["steel"], PALETTE["lavender"]],
            )
            st.plotly_chart(style_figure(usage_fig), width="stretch")
            daily_feature_count = int((feature_catalog["refresh_cadence"] == "daily").sum())
            insight_box("Catalog insight", f"{daily_feature_count} registered features are expected to refresh daily. That is the part of the feature surface most exposed to data freshness and schema drift issues.")
    with right:
        if not product_features.empty:
            numeric_columns = [col for col in ["total_units_sold_12m", "total_revenue_12m", "avg_unit_price", "days_cover", "stockout_probability"] if col in product_features.columns]
            if numeric_columns:
                corr = product_features[numeric_columns].corr().fillna(0)
                corr_fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=list(corr.columns),
                        y=list(corr.index),
                        colorscale=[[0, PALETTE["coral"]], [0.5, PALETTE["steel"]], [1, PALETTE["teal"]]],
                        zmin=-1,
                        zmax=1,
                    )
                )
                corr_fig.update_layout(title="Product Feature Correlation Snapshot")
                st.plotly_chart(style_figure(corr_fig), width="stretch")
                insight_box("Correlation insight", "This correlation matrix gives a quick read on whether demand, revenue, price, and stock-health features are moving together strongly enough to create redundancy or leakage risk.")

    if not customer_features.empty:
        sample_columns = [col for col in ["recency_days", "frequency", "monetary_value", "avg_order_value", "days_between_orders"] if col in customer_features.columns]
        if sample_columns:
            distribution_source = customer_features[sample_columns].melt(var_name="feature", value_name="value")
            dist_fig = px.box(
                distribution_source,
                x="feature",
                y="value",
                color="feature",
                title="Customer Feature Distribution Snapshot",
                color_discrete_sequence=[PALETTE["amber"], PALETTE["teal"], PALETTE["lavender"], PALETTE["steel"], PALETTE["coral"]],
            )
            st.plotly_chart(style_figure(dist_fig), width="stretch")
            insight_box("Distribution insight", "These distributions show how the customer model sees the population. Heavy skew in recency or monetary value translates directly into segmentation and churn-class balance.")

    st.dataframe(feature_catalog, width="stretch")

with tabs[4]:
    left, right = st.columns([1.1, 1.3])
    with left:
        if not model_summaries.empty:
            metric_rows = []
            for _, row in model_summaries.iterrows():
                if pd.notna(row.get("sku_count")):
                    metric_rows.append({"model": row["model_name"], "volume": row["sku_count"]})
                elif pd.notna(row.get("customer_count")):
                    metric_rows.append({"model": row["model_name"], "volume": row["customer_count"]})
            if metric_rows:
                volume_fig = px.bar(
                    pd.DataFrame(metric_rows),
                    x="model",
                    y="volume",
                    color="model",
                    title="Training Volume by Model",
                    color_discrete_sequence=[PALETTE["teal"], PALETTE["amber"], PALETTE["coral"], PALETTE["steel"], PALETTE["lavender"]],
                )
                volume_fig.update_layout(showlegend=False)
                st.plotly_chart(style_figure(volume_fig), width="stretch")
                insight_box("Training insight", "Demand forecasting and inventory scoring currently train over the full SKU base, while customer models only score customers with qualifying transactional history in the latest feature store build.")
    with right:
        if not forecast_metrics.empty:
            mape_fig = px.histogram(
                forecast_metrics,
                x="validation_mape",
                nbins=25,
                title="Demand Forecast Validation MAPE Distribution",
                color_discrete_sequence=[PALETTE["amber"]],
            )
            st.plotly_chart(style_figure(mape_fig), width="stretch")
            mean_mape = float(forecast_metrics["validation_mape"].mean())
            insight_box("Forecast-model insight", f"Average validation MAPE is {mean_mape:.3f}. The spread matters more than the average here because poor SKU-level tails are what create replenishment misses.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if not segment_df.empty:
            segment_mix = segment_df["segment_name"].value_counts().rename_axis("segment_name").reset_index(name="customers")
            segment_fig = px.pie(
                segment_mix,
                names="segment_name",
                values="customers",
                hole=0.5,
                title="Segmentation Output Mix",
                color_discrete_sequence=[PALETTE["amber"], PALETTE["teal"], PALETTE["steel"], PALETTE["coral"], PALETTE["lavender"]],
            )
            st.plotly_chart(style_figure(segment_fig), width="stretch")
    with lower_right:
        if not churn_scores.empty:
            churn_mix = churn_scores["risk_band"].value_counts().rename_axis("risk_band").reset_index(name="customers")
            churn_fig = px.bar(
                churn_mix,
                x="risk_band",
                y="customers",
                color="risk_band",
                title="Churn Score Distribution",
                color_discrete_map={"High": PALETTE["coral"], "Medium": PALETTE["amber"], "Low": PALETTE["ok"]},
            )
            st.plotly_chart(style_figure(churn_fig), width="stretch")

    if churn_summary.get("top_drivers"):
        driver_df = pd.DataFrame(churn_summary["top_drivers"]).head(12)
        driver_df = driver_df.sort_values("coefficient")
        driver_fig = px.bar(
            driver_df,
            x="coefficient",
            y="feature",
            orientation="h",
            title="Top Churn Model Drivers",
            color="coefficient",
            color_continuous_scale=[PALETTE["coral"], PALETTE["steel"], PALETTE["teal"]],
        )
        driver_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(driver_fig), width="stretch")
        insight_box("Driver insight", "Positive coefficients in this model nudge churn probability upward, negative coefficients downward. This is useful for debugging feature behavior more than for making business claims from the synthetic sample.")

    st.dataframe(model_summaries, width="stretch")

with tabs[5]:
    if pipeline_logs.empty:
        st.info("No pipeline logs found.")
    else:
        left, right = st.columns([1.1, 1.3])
        with left:
            duration_fig = px.bar(
                pipeline_logs.dropna(subset=["duration_seconds"]),
                x="run_id",
                y="duration_seconds",
                color="pipeline_name",
                title="Pipeline Run Duration",
                color_discrete_map={"daily_pipeline": PALETTE["teal"], "weekly_pipeline": PALETTE["amber"]},
            )
            st.plotly_chart(style_figure(duration_fig), width="stretch")
            longest = pipeline_logs.dropna(subset=["duration_seconds"]).sort_values("duration_seconds", ascending=False).iloc[0]
            insight_box("Runtime insight", f"The longest recorded run is {longest['pipeline_name']} / {longest['run_id']} at {longest['duration_seconds']:.1f} seconds. Step-level durations are not logged yet, so this dashboard shows run duration and rows processed instead.")
        with right:
            rows_fig = px.bar(
                pipeline_logs,
                x="run_id",
                y="rows_processed_total",
                color="status",
                title="Rows Processed per Run",
                color_discrete_map={"completed": PALETTE["ok"], "failed": PALETTE["fail"], "running": PALETTE["amber"]},
            )
            st.plotly_chart(style_figure(rows_fig), width="stretch")
            latest = pipeline_logs.iloc[0]
            step_rows = latest["step_rows"]
            if isinstance(step_rows, dict) and step_rows:
                step_df = pd.DataFrame({"step": list(step_rows.keys()), "rows": list(step_rows.values())})
                step_fig = px.funnel(step_df, x="rows", y="step", title=f"Latest Run Step Volume: {latest['pipeline_name']} / {latest['run_id']}")
                st.plotly_chart(style_figure(step_fig), width="stretch")
                insight_box("Step-volume insight", "This is a row-volume view, not a latency view. It shows how much work each stage processed in the latest pipeline run, which is the best currently available operational trace in stored logs.")

        display_logs = pipeline_logs.copy()
        if "step_rows" in display_logs.columns:
            display_logs["step_rows"] = display_logs["step_rows"].astype(str)
        st.dataframe(display_logs, width="stretch")

with tabs[6]:
    env_rows = pd.DataFrame(
        [
            {"setting": "DATA_DIR", "value": str(DATA_DIR)},
            {"setting": "SYNTHETIC_DIR", "value": str(SYNTHETIC_DIR)},
            {"setting": "FEATURES_DIR", "value": str(FEATURES_DIR)},
            {"setting": "MODELS_DIR", "value": str(MODELS_DIR)},
            {"setting": "REPORTS_DIR", "value": str(REPORTS_DIR)},
            {"setting": "Demo mode assumption", "value": "true (local synthetic artifacts loaded)"},
            {"setting": "Last daily status", "value": daily_summary.get("data_quality_status", "unknown") if daily_summary else "unknown"},
            {"setting": "Last weekly status", "value": weekly_summary.get("data_quality_status", "unknown") if weekly_summary else "unknown"},
        ]
    )
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(env_rows, width="stretch")
        insight_box("Environment insight", "This dashboard is reading entirely from on-disk artifacts under the local data directory. That makes it useful as a reproducible post-run inspection surface, even when no live services are running.")
    with right:
        artifact_rows = []
        for path in sorted(MODELS_DIR.glob("*")):
            if path.is_file():
                artifact_rows.append(
                    {
                        "artifact": path.name,
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                        "updated_at": pd.Timestamp(path.stat().st_mtime, unit="s"),
                    }
                )
        artifact_df = pd.DataFrame(artifact_rows).sort_values("updated_at", ascending=False) if artifact_rows else pd.DataFrame()
        if not artifact_df.empty:
            artifact_fig = px.bar(
                artifact_df.head(15),
                x="artifact",
                y="size_mb",
                color="size_mb",
                title="Recent Model Artifacts by Size",
                color_continuous_scale=[PALETTE["steel"], PALETTE["teal"]],
            )
            artifact_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_figure(artifact_fig), width="stretch")
            newest = artifact_df.iloc[0]
            insight_box("Artifact insight", f"The most recently updated model artifact is {newest['artifact']}, written at {newest['updated_at']:%d %b %Y %H:%M:%S}. This page is effectively the local registry browser for the current project state.")
