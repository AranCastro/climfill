"""
ClimFill — Streamlit Web Application.
Launch: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime

from climfill.engine import ClimFillEngine
from climfill.reanalysis import get_open_meteo_data

# ──────────────────────────────────────
# Page Config
# ──────────────────────────────────────
st.set_page_config(
    page_title="ClimFill — Climate Data Gap-Filler",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1B4332; }
    .sub-header { font-size: 1.1rem; color: #52796F; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# Session State
# ──────────────────────────────────────
for key, default in [
    ("engine", None),
    ("step", 1),
    ("gap_report", None),
    ("filled", False),
    ("reanalysis_merged", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────
# Header
# ──────────────────────────────────────
st.markdown('<p class="main-header">🌍 ClimFill</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Automated Climate Station Data Gap-Filler</p>',
    unsafe_allow_html=True,
)

# Progress bar
cols = st.columns(5)
steps = ["📤 Upload", "🔍 Detect", "🌐 Reanalysis", "🔧 Fill", "📥 Export"]
for i, (col, name) in enumerate(zip(cols, steps)):
    with col:
        if i + 1 < st.session_state.step:
            st.success(name)
        elif i + 1 == st.session_state.step:
            st.info(name)
        else:
            st.markdown(f"⬜ {name}")

st.divider()

# ──────────────────────────────────────
# Sidebar
# ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("Station Location")
    lat = st.number_input("Latitude", value=10.0, min_value=-90.0, max_value=90.0, step=0.01)
    lon = st.number_input("Longitude", value=79.0, min_value=-180.0, max_value=180.0, step=0.01)

    st.subheader("Gap Filling")
    fill_method = st.selectbox(
        "Method",
        ["auto", "linear", "spline", "seasonal", "random_forest", "xgboost"],
    )
    use_reanalysis = st.checkbox("Use Reanalysis (Open-Meteo)", value=True)
    compute_uncertainty = st.checkbox("Compute Uncertainty", value=True)
    n_bootstrap = st.slider("Bootstrap Samples", 20, 200, 50, step=10)

    st.subheader("Outlier Detection")
    detect_outliers = st.checkbox("Flag Outliers", value=True)
    outlier_threshold = st.slider("Threshold (IQR)", 1.5, 5.0, 3.0, step=0.5)

    st.divider()
    st.markdown("**ClimFill v1.0** | Free & Open Source")

# ══════════════════════════════════════
# STEP 1: Upload
# ══════════════════════════════════════
st.header("📤 Step 1: Upload Station Data")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
use_sample = st.checkbox("Use sample data for demo")

if use_sample and not uploaded_file:
    # Generate sample inline
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear

    temp = 25 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 2, n)
    precip = np.maximum(0, np.random.exponential(3, n))
    humidity = 60 + 15 * np.sin(2 * np.pi * (doy - 200) / 365) + np.random.normal(0, 5, n)
    wind = np.maximum(0, 5 + np.random.normal(0, 1.5, n))

    sample_df = pd.DataFrame({
        "Date": dates,
        "Temperature": np.round(temp, 1),
        "Precipitation": np.round(precip, 1),
        "Humidity": np.round(humidity, 1),
        "WindSpeed": np.round(wind, 1),
    })

    # Random gaps
    for col in ["Temperature", "Precipitation", "Humidity", "WindSpeed"]:
        mask = np.random.random(n) < 0.03
        sample_df.loc[mask, col] = np.nan
    # Sensor failures
    for _ in range(6):
        s = np.random.randint(0, n - 20)
        l = np.random.randint(5, 15)
        c = np.random.choice(["Temperature", "Precipitation", "Humidity", "WindSpeed"])
        sample_df.loc[s : s + l, c] = np.nan
    # Long gaps
    sample_df.loc[400:440, "Temperature"] = np.nan
    sample_df.loc[400:440, "Humidity"] = np.nan

    buf = io.StringIO()
    sample_df.to_csv(buf, index=False)
    buf.seek(0)
    uploaded_file = buf

if uploaded_file:
    engine = ClimFillEngine()
    load_info = engine.load_csv(uploaded_file)
    st.session_state.engine = engine
    st.session_state.step = max(st.session_state.step, 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", load_info["shape"][0])
    c2.metric("Variables", load_info["shape"][1] - 1)
    c3.metric("Date Column", load_info["date_column"])
    c4.metric("Sentinels Fixed", sum(load_info["sentinels_replaced"].values()) if load_info["sentinels_replaced"] else 0)

    st.caption(f"📅 {load_info['date_range']}")
    with st.expander("Preview Raw Data"):
        st.dataframe(engine.df.head(50), use_container_width=True)


# ══════════════════════════════════════
# STEP 2: Detect Gaps
# ══════════════════════════════════════
if st.session_state.engine is not None and st.session_state.step >= 2:
    st.header("🔍 Step 2: Gap Detection")

    if st.button("Run Gap Detection", type="primary"):
        with st.spinner("Analyzing..."):
            engine = st.session_state.engine
            result = engine.detect_gaps(check_outliers=detect_outliers)
            st.session_state.gap_report = result
            st.session_state.step = max(st.session_state.step, 3)

    if st.session_state.gap_report:
        result = st.session_state.gap_report
        engine = st.session_state.engine
        gap_report = result["gap_report"]
        numeric_cols = list(gap_report.keys())

        # Temporal check
        if result["temporal_regularity"]:
            reg = result["temporal_regularity"]
            if reg["regularity_pct"] >= 95:
                st.success(f"✅ Regular time series ({reg['expected_interval']})")
            else:
                st.warning(f"⚠️ Irregular intervals ({reg['regularity_pct']}% regular)")

        # Summary
        total_missing = sum(gap_report[c]["missing_count"] for c in numeric_cols)
        total_records = gap_report[numeric_cols[0]]["total_records"] if numeric_cols else 0
        max_gap = max((gap_report[c]["max_gap_length"] for c in numeric_cols), default=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Missing", f"{total_missing:,}")
        c2.metric("Missing %", f"{total_missing / (total_records * len(numeric_cols)) * 100:.1f}%" if total_records > 0 else "0%")
        c3.metric("Longest Gap", f"{max_gap} records")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Heatmap", "📈 Time Series", "📋 Details"])

        with tab1:
            gap_matrix = pd.DataFrame()
            for col in numeric_cols:
                gap_matrix[col] = engine.df[col].isna().astype(int)
            if engine.date_col in engine.df.columns:
                gap_matrix.index = engine.df[engine.date_col]

            fig = go.Figure(data=go.Heatmap(
                z=gap_matrix.T.values, x=gap_matrix.index, y=gap_matrix.columns,
                colorscale=[[0, "#2D6A4F"], [1, "#E63946"]],
                colorbar=dict(title="", tickvals=[0, 1], ticktext=["Present", "Missing"]),
            ))
            fig.update_layout(title="Data Availability", xaxis_title="Date", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            sel_var = st.selectbox("Variable", numeric_cols)
            if sel_var:
                dates = engine.df[engine.date_col]
                series = engine.df[sel_var]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=series, mode="lines", name="Data", line=dict(color="#2D6A4F", width=1)))
                gap_mask = series.isna()
                if gap_mask.any():
                    fig.add_trace(go.Scatter(x=dates[gap_mask], y=[series.mean()] * gap_mask.sum(), mode="markers", name="Missing", marker=dict(color="#E63946", size=3, symbol="x")))
                fig.update_layout(title=f"{sel_var} with Gaps", height=400)
                st.plotly_chart(fig, use_container_width=True)

                if gap_report[sel_var]["n_gaps"] > 0:
                    dist = gap_report[sel_var]["gap_length_distribution"]
                    fig2 = go.Figure(data=[go.Bar(x=list(dist.keys()), y=list(dist.values()), marker_color="#2D6A4F")])
                    fig2.update_layout(title="Gap Length Distribution", height=300)
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            for col in numeric_cols:
                gr = gap_report[col]
                with st.expander(f"**{col}** — {gr['missing_count']} missing ({gr['missing_pct']}%)"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Missing", gr["missing_count"])
                    c2.metric("%", f"{gr['missing_pct']}%")
                    c3.metric("Gaps", gr["n_gaps"])
                    c4.metric("Max Gap", gr["max_gap_length"])

        if detect_outliers and result.get("outliers"):
            st.subheader("⚠️ Outliers")
            for col, info in result["outliers"].items():
                st.warning(f"**{col}**: {info['count']} potential outliers")


# ══════════════════════════════════════
# STEP 3: Reanalysis
# ══════════════════════════════════════
if st.session_state.engine is not None and st.session_state.step >= 3:
    st.header("🌐 Step 3: Reanalysis Integration")

    if not use_reanalysis:
        st.info("Reanalysis disabled. Enable in sidebar.")
        st.session_state.step = max(st.session_state.step, 4)
    else:
        if st.button("Download Reanalysis Data", type="primary"):
            engine = st.session_state.engine
            dates = engine.df[engine.date_col]

            with st.spinner(f"Downloading for ({lat}, {lon})..."):
                try:
                    rean_df = get_open_meteo_data(
                        lat=lat, lon=lon,
                        start_date=dates.min().strftime("%Y-%m-%d"),
                        end_date=dates.max().strftime("%Y-%m-%d"),
                    )

                    engine.df["_merge_date"] = engine.df[engine.date_col].dt.date
                    rean_df["_merge_date"] = rean_df["date"].dt.date
                    merged = engine.df.merge(rean_df, on="_merge_date", how="left")
                    merged = merged.drop(columns=["_merge_date", "date"], errors="ignore")
                    engine.df = merged
                    st.session_state.engine = engine
                    st.session_state.reanalysis_merged = True

                    st.success(f"✅ Downloaded {len(rean_df)} days of reanalysis data!")

                    with st.expander("Preview merged data"):
                        st.dataframe(merged.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed: {str(e)}")

            st.session_state.step = max(st.session_state.step, 4)

        elif st.session_state.reanalysis_merged:
            st.success("✅ Reanalysis already merged.")
            st.session_state.step = max(st.session_state.step, 4)
        else:
            if st.button("Skip Reanalysis"):
                st.session_state.step = max(st.session_state.step, 4)
                st.rerun()


# ══════════════════════════════════════
# STEP 4: Fill Gaps
# ══════════════════════════════════════
if st.session_state.engine is not None and st.session_state.step >= 4:
    st.header("🔧 Step 4: Gap Filling")

    engine = st.session_state.engine
    st.markdown(f"**Method:** `{fill_method}` | **Uncertainty:** {'Yes' if compute_uncertainty else 'No'}")

    if st.button("🚀 Fill All Gaps", type="primary"):
        with st.spinner("Filling gaps..."):
            try:
                engine.fill_gaps(method=fill_method)
                if compute_uncertainty:
                    with st.spinner("Computing uncertainty..."):
                        engine.compute_uncertainty(n_bootstrap=n_bootstrap)
                st.session_state.filled = True
                st.session_state.step = max(st.session_state.step, 5)
                st.success("✅ Gap filling complete!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.filled:
        original_df = engine.df
        filled_df = engine.filled_df
        numeric_cols = [
            c for c in original_df.select_dtypes(include=[np.number]).columns
            if c != engine.date_col and not c.startswith("reanalysis_")
        ]

        # Summary
        orig_miss = sum(original_df[c].isna().sum() for c in numeric_cols)
        remain_miss = sum(filled_df[c].isna().sum() for c in numeric_cols)
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Missing", f"{orig_miss:,}")
        c2.metric("Remaining", f"{remain_miss:,}")
        fill_rate = (orig_miss - remain_miss) / max(orig_miss, 1) * 100
        c3.metric("Fill Rate", f"{fill_rate:.1f}%")

        # Before/after plot
        compare_var = st.selectbox("Compare Variable", numeric_cols, key="cmp")
        if compare_var:
            dates = filled_df[engine.date_col]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Original", "Filled"], vertical_spacing=0.08)

            fig.add_trace(go.Scatter(x=dates, y=original_df[compare_var], mode="lines", name="Original", line=dict(color="#2D6A4F", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=dates, y=filled_df[compare_var], mode="lines", name="Filled", line=dict(color="#2D6A4F", width=1)), row=2, col=1)

            fill_mask = original_df[compare_var].isna() & filled_df[compare_var].notna()
            if fill_mask.any():
                fig.add_trace(go.Scatter(x=dates[fill_mask], y=filled_df.loc[fill_mask, compare_var], mode="markers", name="Filled Values", marker=dict(color="#E63946", size=4)), row=2, col=1)

            if compare_var in engine.uncertainty_results:
                unc = engine.uncertainty_results[compare_var]
                if not unc.empty:
                    unc_dates = filled_df.loc[unc.index, engine.date_col]
                    fig.add_trace(go.Scatter(
                        x=pd.concat([unc_dates, unc_dates[::-1]]),
                        y=pd.concat([unc["ci_upper_95"], unc["ci_lower_95"][::-1]]),
                        fill="toself", fillcolor="rgba(230,57,70,0.15)", line=dict(width=0), name="95% CI",
                    ), row=2, col=1)

            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        # Method summary
        if engine.filler and engine.filler.fill_metadata:
            report = engine.generate_report()
            st.subheader("📋 Method Summary")
            for col in numeric_cols:
                if col in report.get("fill_summary", {}):
                    fs = report["fill_summary"][col]
                    gs = report["gap_summary"].get(col, {})
                    with st.expander(f"**{col}** — {gs.get('filled_count', 0)}/{gs.get('original_missing', 0)} filled"):
                        methods = {k: v for k, v in fs.get("methods_used", {}).items()}
                        if methods:
                            fig = go.Figure(data=[go.Pie(labels=list(methods.keys()), values=list(methods.values()), hole=0.4)])
                            fig.update_layout(height=250, title=f"{col} Methods")
                            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════
# STEP 5: Export
# ══════════════════════════════════════
if st.session_state.filled and st.session_state.step >= 5:
    st.header("📥 Step 5: Export")

    engine = st.session_state.engine
    c1, c2 = st.columns(2)
    with c1:
        include_flags = st.checkbox("Include quality flags", value=True)
        include_unc = st.checkbox("Include uncertainty", value=compute_uncertainty)
    with c2:
        include_rean = st.checkbox("Include reanalysis columns", value=False)

    export_df = engine.export(include_uncertainty=include_unc, include_metadata=include_flags)
    if not include_rean:
        rean_cols = [c for c in export_df.columns if c.startswith("reanalysis_")]
        export_df = export_df.drop(columns=rean_cols, errors="ignore")

    st.subheader("Preview")
    st.dataframe(export_df.head(20), use_container_width=True)
    st.caption(f"Shape: {export_df.shape[0]} × {export_df.shape[1]}")

    c1, c2 = st.columns(2)
    with c1:
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download CSV", csv_buf.getvalue(),
            f"climfill_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", type="primary",
        )
    with c2:
        report = engine.generate_report()
        st.download_button(
            "📄 Download Report",
            json.dumps(report, indent=2, default=str),
            f"climfill_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
        )

    st.divider()
    st.success("🎉 Done! Your gap-filled dataset is ready for publication.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;'>"
    "ClimFill v1.0 — Free, Open-Source | Built for Earth Sciences</div>",
    unsafe_allow_html=True,
)
