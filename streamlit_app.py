"""
ClimFill — Streamlit web application.

Launch:   streamlit run streamlit_app.py
Requires: streamlit>=1.51, plotly>=5.10  (pip install "climfill[app]")
"""

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from climfill import __version__
from climfill.engine import HAS_XGBOOST, ClimFillEngine
from climfill.reanalysis import get_open_meteo_data, suggest_reanalysis_mapping

SAMPLE_PATH = Path(__file__).with_name("sample_station_data.csv")
DEFAULT_LAT, DEFAULT_LON = 10.78, 79.13
NO_MATCH = "— none —"

# Mid-tone colours that stay legible on both light and dark backgrounds
COLORS = {
    "observed": "#2E86AB",
    "filled": "#F18F01",
    "reanalysis": "#8D99AE",
    "missing": "#D1495B",
    "suspicious": "#9B5DE5",
}
METHOD_COLORS = {
    "linear_interpolation": "#F18F01",
    "spline_interpolation": "#E9C46A",
    "era5_bias_corrected": "#577590",
    "seasonal_decomposition": "#2A9D8F",
    "random_forest": "#C1666B",
    "xgboost": "#9B5DE5",
}
METHOD_NAMES = {
    "linear_interpolation": "Linear",
    "spline_interpolation": "Spline",
    "era5_bias_corrected": "Reanalysis (bias-corrected)",
    "seasonal_decomposition": "Seasonal",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
METHODS = {
    "auto": "Auto cascade (recommended)",
    "linear": "Linear interpolation (gaps ≤ 3 records)",
    "spline": "Cubic spline (gaps ≤ 7 records)",
    "seasonal": "Monthly climatology + interpolated anomaly",
    "random_forest": "Random Forest regression",
    "xgboost": "XGBoost regression",
}
METHOD_HELP = (
    "**Auto** fills each variable in stages: linear (≤ 3 records), spline "
    "(≤ 7), bias-corrected reanalysis (if enabled), seasonal climatology, "
    "then XGBoost/Random Forest for anything left. ML methods use the other "
    "variables, reanalysis series, lags and calendar terms as predictors."
)
FLAG_LEGEND = (
    "Flags: **O** original · **F** filled · **S** suspicious (statistical "
    "outlier in the original record) · blank = still missing."
)

STATE_DEFAULTS = {
    "data_key": None,
    "data_name": None,
    "engine": None,
    "load_info": None,
    "load_error": None,
    "detection": None,
    "detect_params": None,
    "inserted_rows": 0,
    "rean_df": None,
    "rean_info": None,
    "rean_suggested": None,
    "rean_map": None,
    "fill_params": None,
    "validation": None,
    "unc_params": None,
}

st.set_page_config(
    page_title="ClimFill — Climate Data Gap-Filler",
    page_icon=":material/water_drop:",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/AranCastro/climfill",
        "Report a bug": "https://github.com/AranCastro/climfill/issues",
        "About": f"ClimFill v{__version__} — open-source gap-filling for "
                 "climate station data (MIT licence).",
    },
)


# ──────────────────────────────────────────────────────────────
# State helpers
# ──────────────────────────────────────────────────────────────
def reset_state():
    for key, value in STATE_DEFAULTS.items():
        st.session_state[key] = value


def clear_fill():
    """Drop fill results; called whenever their inputs change."""
    s = st.session_state
    s.fill_params = s.validation = s.unc_params = None
    if s.engine is not None:
        s.engine.filler = None
        s.engine.filled_df = None
        s.engine.uncertainty_results = {}


def load_dataset(raw, name, key):
    """Build the engine once per distinct file; reruns reuse it."""
    s = st.session_state
    if s.data_key == key:
        return
    reset_state()
    s.data_key, s.data_name = key, name
    engine = ClimFillEngine()
    try:
        info = engine.load_csv(io.BytesIO(raw))
    except Exception as exc:  # malformed CSV, no date column, bad encoding
        s.load_error = f"Could not read **{name}**: {exc}"
        return
    if not engine.station_columns():
        s.load_error = f"**{name}** has no numeric variables to gap-fill."
        return
    s.engine, s.load_info = engine, info


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_reanalysis(lat, lon, start, end):
    return get_open_meteo_data(lat=lat, lon=lon, start_date=start, end_date=end)


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def clean_nan(records):
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        for r in records
    ]


def fmt_date(ts):
    return pd.Timestamp(ts).strftime("%d %b %Y") if pd.notna(ts) else "—"


def fmt_interval(text):
    td = pd.Timedelta(text)
    if td % pd.Timedelta(days=1) == pd.Timedelta(0):
        days = td.days
        return f"{days} day" + ("s" if days != 1 else "")
    return str(td)


def render_progress():
    steps = [
        ("Data loaded", s.engine is not None),
        ("Gaps detected", s.detection is not None),
        ("Reanalysis merged (optional)", s.rean_info is not None),
        ("Gaps filled", s.fill_params is not None),
        ("Uncertainty estimated (optional)", s.unc_params is not None),
    ]
    with progress_box:
        for label, done in steps:
            if done:
                st.markdown(f":green[:material/check_circle:] {label}")
            else:
                st.markdown(f":gray[:material/radio_button_unchecked: {label}]")


def date_column_config(df, col):
    """Show dates without a redundant 00:00:00 for daily data."""
    dates = df[col].dropna()
    daily = bool((dates == dates.dt.normalize()).all())
    fmt = "YYYY-MM-DD" if daily else "YYYY-MM-DD HH:mm"
    return {col: st.column_config.DatetimeColumn(format=fmt)}


def chart_layout(fig, height, **kwargs):
    layout = dict(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_layout(**{**layout, **kwargs})
    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
for _key, _value in STATE_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
s = st.session_state

with st.sidebar:
    st.markdown("### :material/water_drop: ClimFill")
    st.caption("Gap-filling for climate station time series")

    source = st.segmented_control(
        "Data source", ["Upload", "Sample"], default="Upload", key="source"
    ) or "Upload"
    if source == "Upload":
        uploaded = st.file_uploader(
            "Station CSV",
            type=["csv"],
            help="One date column plus numeric variables. Sentinels such as "
                 "-999 and -9999 are treated as missing.",
        )
        if uploaded is not None:
            raw = uploaded.getvalue()
            load_dataset(raw, uploaded.name, hashlib.sha1(raw).hexdigest())
        elif s.data_key is not None:
            reset_state()
    else:
        st.caption(
            "Synthetic daily record, 2019–2023, eight variables, with random "
            "single-day gaps, sensor failures and maintenance outages."
        )
        load_dataset(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name, "sample")

    st.divider()
    # Filled at the end of the script, after this run's button actions
    progress_box = st.container()
    st.divider()
    st.caption(
        f"ClimFill v{__version__} · MIT licence · "
        "[Source code](https://github.com/AranCastro/climfill)"
    )


# ──────────────────────────────────────────────────────────────
# Header / landing
# ──────────────────────────────────────────────────────────────
st.title("ClimFill")
st.caption("Detect, fill and quality-flag gaps in climate station records")

if s.load_error:
    st.error(s.load_error, icon=":material/error:")

if s.engine is None:
    st.info(
        "Upload a station CSV in the sidebar, or choose **Sample** to try the "
        "workflow.",
        icon=":material/upload_file:",
    )
    cols = st.columns(4)
    intro = [
        (":material/search:", "Detect",
         "Locate gaps, irregular timestamps and statistical outliers."),
        (":material/public:", "Reanalysis",
         "Optionally merge free Open-Meteo (ERA5) daily series as predictors."),
        (":material/auto_fix_high:", "Fill",
         "Interpolation, seasonal climatology, quantile-mapped reanalysis "
         "and ML, with hold-out validation."),
        (":material/download:", "Export",
         "CSV with per-value flags, methods, confidence and 95 % intervals."),
    ]
    for col, (icon, title, text) in zip(cols, intro):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)

    st.subheader("Expected input")
    st.dataframe(
        pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Temperature_Mean": [24.1, None, 23.8],
            "Precipitation": [0.0, 2.4, -999],
            "Relative_Humidity": [71.2, 74.0, None],
        }),
        hide_index=True,
        width="content",
    )
    st.caption(
        "The date column is detected automatically; every other numeric "
        "column is treated as a variable. Daily data are recommended "
        "(the reanalysis source is daily)."
    )
    render_progress()
    st.stop()

engine = s.engine
date_col = engine.date_col
station_cols = engine.station_columns()

tab_data, tab_gaps, tab_rean, tab_fill, tab_export = st.tabs([
    ":material/table_chart: Data",
    ":material/search: Gaps",
    ":material/public: Reanalysis",
    ":material/auto_fix_high: Fill",
    ":material/download: Export",
])


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
with tab_data:
    info = s.load_info
    dates = engine.df[date_col]
    sentinels = info["sentinels_replaced"] or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(engine.df):,}", border=True)
    c2.metric("Variables", len(station_cols), border=True)
    c3.metric(
        "Period", f"{fmt_date(dates.min())} – {fmt_date(dates.max())}",
        border=True,
    )
    c4.metric(
        "Sentinels replaced", f"{sum(sentinels.values()):,}", border=True,
        help="Values such as -999 or -9999 converted to missing.",
    )

    st.markdown(f"**{s.data_name}** · date column `{date_col}`")
    summary = pd.DataFrame([
        {
            "Variable": col,
            "Observed": int(engine.df[col].notna().sum()),
            "Missing": int(engine.df[col].isna().sum()),
            "Missing %": float(engine.df[col].isna().mean() * 100),
            "Sentinels": sentinels.get(col, 0),
            "Min": engine.df[col].min(),
            "Mean": engine.df[col].mean(),
            "Max": engine.df[col].max(),
        }
        for col in station_cols
    ])
    st.dataframe(
        summary,
        hide_index=True,
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                format="%.1f %%", min_value=0, max_value=100
            ),
            "Min": st.column_config.NumberColumn(format="%.2f"),
            "Mean": st.column_config.NumberColumn(format="%.2f"),
            "Max": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    with st.expander("Raw data preview (first 200 rows)"):
        st.dataframe(engine.df.head(200), height=320, hide_index=True,
                     column_config=date_column_config(engine.df, date_col))


# ──────────────────────────────────────────────────────────────
# Gaps
# ──────────────────────────────────────────────────────────────
with tab_gaps:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
        check_outliers = c1.toggle("Flag outliers", value=True)
        outlier_method = c2.selectbox(
            "Outlier test",
            ["iqr", "zscore"],
            format_func={"iqr": "IQR fence", "zscore": "Z-score"}.get,
            disabled=not check_outliers,
        )
        outlier_threshold = c3.number_input(
            "Threshold", min_value=1.0, max_value=6.0, value=3.0, step=0.5,
            disabled=not check_outliers,
            help="IQR multiplier (Q1 − k·IQR, Q3 + k·IQR) or |z| limit.",
        )
        run_detect = c4.button(
            "Detect gaps", type="primary", icon=":material/search:",
            width="stretch",
        )
        outlier_columns = st.multiselect(
            "Variables to screen for outliers", station_cols,
            default=station_cols, disabled=not check_outliers,
            help="Zero-inflated variables such as precipitation are screened "
                 "on their non-zero values only.",
        )

    detect_params = {
        "check_outliers": check_outliers,
        "outlier_method": outlier_method,
        "outlier_threshold": outlier_threshold,
        "outlier_columns": outlier_columns,
    }
    if run_detect:
        s.detection = engine.detect_gaps(**detect_params)
        s.detect_params = detect_params
        clear_fill()

    if s.detection is None:
        st.info("Run gap detection to analyse missing values.",
                icon=":material/info:")
    else:
        if s.detect_params != detect_params:
            st.caption(":orange[:material/warning: Settings changed since the "
                       "last run — detect again to apply them.]")
        detection = s.detection
        gap_report = detection["gap_report"]

        reg = detection["temporal_regularity"]
        if reg:
            interval = fmt_interval(reg["expected_interval"])
            if reg["irregular_intervals"] == 0:
                st.success(f"Regular time series: one record every {interval}.",
                           icon=":material/check_circle:")
            else:
                c1, c2 = st.columns([3, 1], vertical_alignment="center")
                c1.warning(
                    f"{reg['irregular_intervals']:,} of {reg['total_intervals']:,} "
                    f"intervals differ from the expected {interval} "
                    f"({reg['regularity_pct']} % regular). Records absent from "
                    "the file are not counted as gaps until they are inserted.",
                    icon=":material/warning:",
                )
                if c2.button("Insert missing records", icon=":material/add_row_below:",
                             width="stretch"):
                    added = engine.insert_missing_timestamps()
                    if s.rean_df is not None:
                        engine.merge_reanalysis(s.rean_df)
                    s.inserted_rows += added
                    s.detection = engine.detect_gaps(**s.detect_params)
                    clear_fill()
                    st.toast(
                        f"Inserted {added:,} empty records." if added else
                        "No absent timestamps found; the irregular intervals "
                        "come from duplicate or off-grid timestamps.",
                        icon=":material/add_row_below:",
                    )
                    st.rerun()
        if s.inserted_rows:
            st.caption(f"{s.inserted_rows:,} empty records were inserted for "
                       "absent timestamps.")

        total_missing = sum(gap_report[c]["missing_count"] for c in station_cols)
        n_cells = len(engine.df) * len(station_cols)
        n_gaps = sum(gap_report[c]["n_gaps"] for c in station_cols)
        longest_col = max(station_cols, key=lambda c: gap_report[c]["max_gap_length"])
        longest = gap_report[longest_col]["max_gap_length"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing values", f"{total_missing:,}", border=True)
        c2.metric("Missing share", f"{total_missing / max(n_cells, 1) * 100:.1f} %",
                  border=True)
        c3.metric("Gap segments", f"{n_gaps:,}", border=True)
        c4.metric("Longest gap", f"{longest:,} records", border=True,
                  help=f"Variable: {longest_col}" if longest else None)

        if total_missing == 0:
            st.success("No missing values found — nothing to fill.",
                       icon=":material/celebration:")

        # Availability heatmap
        avail = engine.df[station_cols].notna().astype(int).T
        fig = go.Figure(go.Heatmap(
            z=avail.values,
            x=engine.df[date_col],
            y=station_cols,
            colorscale=[[0, COLORS["missing"]], [1, COLORS["observed"]]],
            zmin=0, zmax=1, showscale=False,
            hovertemplate="%{y}<br>%{x|%d %b %Y}<extra></extra>",
        ))
        chart_layout(fig, 70 + 28 * len(station_cols),
                     title="Data availability (blue = observed, red = missing)",
                     hovermode="closest")
        st.plotly_chart(fig)

        outliers = detection.get("outliers") or {}
        rows = []
        for col in station_cols:
            gr = gap_report[col]
            longest_gap = max(gr["gaps"], key=lambda g: g["length"], default=None)
            rows.append({
                "Variable": col,
                "Missing": gr["missing_count"],
                "Missing %": gr["missing_pct"],
                "Gaps": gr["n_gaps"],
                "Mean length": gr["mean_gap_length"],
                "Longest": gr["max_gap_length"],
                "Longest gap period": (
                    f"{fmt_date(longest_gap['start_date'])} – "
                    f"{fmt_date(longest_gap['end_date'])}"
                    if longest_gap and "start_date" in longest_gap else "—"
                ),
                "Outliers": outliers.get(col, {}).get("count", 0),
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    format="%.1f %%", min_value=0, max_value=100
                ),
            },
        )

        st.subheader("Explore a variable")
        var = st.selectbox("Variable", station_cols, key="gap_var",
                           label_visibility="collapsed")
        series = engine.df[var]
        c1, c2 = st.columns([3, 1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=engine.df[date_col], y=series, mode="lines", name="Observed",
            line=dict(color=COLORS["observed"], width=1),
        ))
        miss = series.isna()
        if miss.any():
            fig.add_trace(go.Scatter(
                x=engine.df.loc[miss, date_col],
                y=np.full(int(miss.sum()), series.min()),
                mode="markers", name="Missing",
                marker=dict(color=COLORS["missing"], symbol="line-ns-open",
                            size=10),
                hovertemplate="Missing<extra></extra>",
            ))
        if var in outliers:
            idx = outliers[var]["indices"]
            fig.add_trace(go.Scatter(
                x=engine.df.loc[idx, date_col], y=engine.df.loc[idx, var],
                mode="markers", name="Outlier",
                marker=dict(color=COLORS["suspicious"], size=7,
                            symbol="circle-open", line=dict(width=2)),
            ))
        chart_layout(fig, 360, title=var)
        c1.plotly_chart(fig)

        dist = gap_report[var]["gap_length_distribution"]
        fig = go.Figure(go.Bar(
            x=list(dist.keys()), y=list(dist.values()),
            marker_color=COLORS["observed"],
        ))
        chart_layout(fig, 360, title="Gap lengths (records)",
                     yaxis_title="Gaps", hovermode="closest")
        c2.plotly_chart(fig)


# ──────────────────────────────────────────────────────────────
# Reanalysis
# ──────────────────────────────────────────────────────────────
with tab_rean:
    st.markdown(
        "Daily series from the [Open-Meteo Historical Weather API]"
        "(https://open-meteo.com/en/docs/historical-weather-api) (ERA5-based, "
        "no API key). Merged columns act as predictors for the ML methods and, "
        "once matched to a station variable, as a quantile-mapped fill source "
        "in the auto cascade."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        lat = st.number_input("Latitude (decimal degrees, WGS 84)", -90.0, 90.0,
                              DEFAULT_LAT, 0.01, format="%.4f")
        lon = st.number_input("Longitude (decimal degrees, WGS 84)", -180.0,
                              180.0, DEFAULT_LON, 0.01, format="%.4f")
        st.caption("EPSG:4326. Open-Meteo returns the nearest grid cell.")
        run_rean = st.button("Download and merge", type="primary",
                             icon=":material/cloud_download:", width="stretch")
        if s.rean_info is not None and st.button(
            "Remove reanalysis", icon=":material/delete:", width="stretch"
        ):
            engine.df = engine.df.drop(columns=s.rean_info["columns"])
            for key in ("rean_df", "rean_info", "rean_suggested", "rean_map"):
                s[key] = None
            clear_fill()
            st.rerun()
    with c2:
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=5, height=280,
               color=COLORS["missing"])

    if run_rean:
        dates = engine.df[date_col]
        with st.spinner(f"Downloading reanalysis for {lat:.4f}, {lon:.4f}…"):
            try:
                rean_df = fetch_reanalysis(
                    round(lat, 4), round(lon, 4),
                    dates.min().strftime("%Y-%m-%d"),
                    dates.max().strftime("%Y-%m-%d"),
                )
            except Exception as exc:
                st.error(f"Reanalysis download failed: {exc}",
                         icon=":material/cloud_off:")
            else:
                rean_cols = engine.merge_reanalysis(rean_df)
                suggested = suggest_reanalysis_mapping(station_cols, rean_cols)
                s.rean_df = rean_df
                s.rean_info = {
                    "source": "Open-Meteo Historical Weather API (ERA5)",
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "crs": "EPSG:4326",
                    "days": len(rean_df),
                    "columns": rean_cols,
                    "downloaded": datetime.now().isoformat(timespec="seconds"),
                }
                s.rean_suggested = suggested
                s.rean_map = dict(suggested)
                clear_fill()
                st.toast(f"Merged {len(rean_df):,} days of reanalysis.",
                         icon=":material/check_circle:")

    if s.rean_info is None:
        st.info("Optional step. Filling also works without reanalysis.",
                icon=":material/info:")
    else:
        ri = s.rean_info
        st.success(
            f"Merged {ri['days']:,} days ({len(ri['columns'])} variables) for "
            f"{ri['latitude']}, {ri['longitude']}.",
            icon=":material/check_circle:",
        )
        if (round(lat, 4), round(lon, 4)) != (ri["latitude"], ri["longitude"]):
            st.caption(":orange[:material/warning: Coordinates changed — download "
                       "again to use the new location.]")

        st.subheader("Match station variables")
        st.caption("Suggested from column names. Matched series are bias-"
                   "corrected by quantile mapping before use.")
        editor_df = pd.DataFrame({
            "Station variable": station_cols,
            "Reanalysis series": [
                s.rean_suggested.get(c, NO_MATCH) for c in station_cols
            ],
        })
        edited = st.data_editor(
            editor_df,
            hide_index=True,
            disabled=["Station variable"],
            key=f"rean_editor_{ri['downloaded']}",
            column_config={
                "Reanalysis series": st.column_config.SelectboxColumn(
                    options=[NO_MATCH] + ri["columns"], required=True
                ),
            },
        )
        new_map = {
            row["Station variable"]: row["Reanalysis series"]
            for _, row in edited.iterrows()
            if row["Reanalysis series"] != NO_MATCH
        }
        if new_map != s.rean_map:
            s.rean_map = new_map
            clear_fill()

        if s.rean_map:
            stats_rows = []
            for col, rcol in s.rean_map.items():
                both = engine.df[[col, rcol]].dropna()
                stats_rows.append({
                    "Station variable": col,
                    "Reanalysis series": rcol,
                    "Overlap (n)": len(both),
                    "Pearson r": both[col].corr(both[rcol]) if len(both) > 2 else np.nan,
                    "Mean difference": (both[col] - both[rcol]).mean(),
                })
            st.dataframe(
                pd.DataFrame(stats_rows), hide_index=True,
                column_config={
                    "Pearson r": st.column_config.NumberColumn(format="%.3f"),
                    "Mean difference": st.column_config.NumberColumn(
                        format="%.2f", help="Station minus reanalysis, before "
                                            "bias correction."),
                },
            )
            var = st.selectbox("Compare", list(s.rean_map), key="rean_var")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=engine.df[date_col], y=engine.df[s.rean_map[var]],
                mode="lines", name="Reanalysis (raw)",
                line=dict(color=COLORS["reanalysis"], width=1),
            ))
            fig.add_trace(go.Scatter(
                x=engine.df[date_col], y=engine.df[var], mode="lines",
                name="Station", line=dict(color=COLORS["observed"], width=1),
            ))
            chart_layout(fig, 340, title=f"{var} vs {s.rean_map[var]}")
            st.plotly_chart(fig)


# ──────────────────────────────────────────────────────────────
# Fill
# ──────────────────────────────────────────────────────────────
with tab_fill:
    if s.detection is None:
        st.info("Run gap detection in the **Gaps** tab first.",
                icon=":material/info:")
    else:
        rean_map = s.rean_map or {}
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
            method = c1.selectbox("Method", list(METHODS), format_func=METHODS.get,
                                  help=METHOD_HELP)
            use_rean = c2.toggle(
                "Use matched reanalysis",
                value=bool(rean_map),
                disabled=not rean_map or method != "auto",
                help="Adds quantile-mapped reanalysis to the auto cascade, "
                     "after spline interpolation. Requires matched variables "
                     "in the Reanalysis tab.",
            )
            run_fill = c3.button("Fill gaps", type="primary",
                                 icon=":material/auto_fix_high:", width="stretch")
            if method == "xgboost" and not HAS_XGBOOST:
                st.caption(":orange[XGBoost is not installed; Random Forest will "
                           "be used instead (`pip install climfill[ml]`).]")

        fill_params = {
            "method": method,
            "reanalysis_map": dict(rean_map) if use_rean and method == "auto" else {},
        }
        if run_fill:
            with st.status("Filling gaps…") as status:
                try:
                    engine.fill_gaps(
                        method=method,
                        reanalysis_map=fill_params["reanalysis_map"] or None,
                    )
                except Exception as exc:
                    clear_fill()
                    status.update(label="Gap filling failed", state="error")
                    st.error(str(exc))
                else:
                    s.fill_params = fill_params
                    s.validation = s.unc_params = None
                    status.update(label=f"Gaps filled ({METHODS[method]})",
                                  state="complete", expanded=False)

        if s.fill_params is None:
            st.info("Choose a method and press **Fill gaps**.",
                    icon=":material/info:")
        else:
            if s.fill_params != fill_params:
                st.caption(":orange[:material/warning: Settings changed since "
                           "the last run — fill again to apply them.]")
            report = engine.generate_report()
            gap_summary = report["gap_summary"]
            before = sum(g["original_missing"] for g in gap_summary.values())
            filled = sum(g["filled_count"] for g in gap_summary.values())
            remaining = before - filled

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Missing before", f"{before:,}", border=True)
            c2.metric("Filled", f"{filled:,}", border=True)
            c3.metric("Still missing", f"{remaining:,}", border=True,
                      help="Typically leading/trailing gaps or records where "
                           "every predictor is missing.")
            c4.metric("Fill rate", f"{filled / max(before, 1) * 100:.1f} %",
                      border=True)

            meta = engine.filler.fill_metadata
            method_rows, summary_rows = [], []
            for col in station_cols:
                gs = gap_summary.get(col)
                if not gs or gs["original_missing"] == 0:
                    continue
                fs = report["fill_summary"].get(col, {})
                used = fs.get("methods_used", {})
                summary_rows.append({
                    "Variable": col,
                    "Missing": gs["original_missing"],
                    "Filled": gs["filled_count"],
                    "Fill rate": gs["fill_rate_pct"],
                    "Mean confidence": fs.get("mean_confidence"),
                    "Methods": " · ".join(
                        f"{METHOD_NAMES.get(m, m)} {n}" for m, n in used.items()
                    ),
                })
                for m, n in used.items():
                    method_rows.append({"Variable": col, "Method": m, "Count": n})

            st.dataframe(
                pd.DataFrame(summary_rows), hide_index=True,
                column_config={
                    "Fill rate": st.column_config.ProgressColumn(
                        format="%.1f %%", min_value=0, max_value=100
                    ),
                    "Mean confidence": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Heuristic per-method score; cross-validated R² "
                             "for ML methods. See hold-out validation below "
                             "for measured accuracy.",
                    ),
                },
            )

            if method_rows:
                mdf = pd.DataFrame(method_rows)
                fig = go.Figure()
                for m in METHOD_COLORS:
                    part = mdf[mdf["Method"] == m]
                    if part.empty:
                        continue
                    fig.add_trace(go.Bar(
                        y=part["Variable"], x=part["Count"], orientation="h",
                        name=METHOD_NAMES[m], marker_color=METHOD_COLORS[m],
                    ))
                chart_layout(fig, 90 + 30 * mdf["Variable"].nunique(),
                             barmode="stack", title="Values filled by method",
                             hovermode="closest",
                             yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig)

            gap_vars = [r["Variable"] for r in summary_rows]
            if gap_vars:
                st.subheader("Inspect a variable")
                var = st.selectbox("Variable", gap_vars, key="fill_var",
                                   label_visibility="collapsed")
                dates = engine.filled_df[date_col]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=engine.filled_df[var], mode="lines",
                    name="Reconstructed", line=dict(color=COLORS["filled"], width=1),
                ))
                fig.add_trace(go.Scatter(
                    x=dates, y=engine.df[var], mode="lines", name="Observed",
                    line=dict(color=COLORS["observed"], width=1.2),
                ))
                unc = engine.uncertainty_results.get(var)
                if unc is not None and not unc.empty:
                    # One vertical segment per filled value, broken by None
                    ux, uy = [], []
                    for d, lo, hi in zip(dates.loc[unc.index],
                                         unc["ci_lower_95"], unc["ci_upper_95"]):
                        ux += [d, d, None]
                        uy += [lo, hi, None]
                    fig.add_trace(go.Scatter(
                        x=ux, y=uy, mode="lines", name="95 % interval",
                        line=dict(color=COLORS["filled"], width=4),
                        opacity=0.3, hoverinfo="skip",
                    ))
                col_meta = meta.get(var)
                if col_meta is not None:
                    for m, color in METHOD_COLORS.items():
                        mask = (col_meta["method"] == m).to_numpy()
                        if not mask.any():
                            continue
                        fig.add_trace(go.Scatter(
                            x=dates[mask], y=engine.filled_df.loc[mask, var],
                            mode="markers", name=METHOD_NAMES[m],
                            marker=dict(color=color, size=5),
                        ))
                chart_layout(fig, 420, title=var)
                st.plotly_chart(fig)

            # Hold-out validation
            with st.expander("Accuracy assessment (hold-out validation)",
                             icon=":material/fact_check:"):
                st.caption(
                    "Hides observed values as synthetic gaps, re-runs the "
                    "selected method and compares reconstructions with the "
                    "hidden observations. Metrics are in each variable's units."
                )
                c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
                gap_len = c1.select_slider(
                    "Synthetic gap length (records)", [1, 3, 7, 15, 30], value=1
                )
                fraction = c2.slider("Share of observed values hidden", 1, 20, 5,
                                     format="%d %%")
                if c3.button("Run validation", icon=":material/play_arrow:",
                             width="stretch"):
                    with st.spinner("Validating…"):
                        res = engine.validate(
                            method=s.fill_params["method"],
                            gap_length=gap_len,
                            fraction=fraction / 100,
                            reanalysis_map=s.fill_params["reanalysis_map"] or None,
                        )
                    s.validation = {
                        "method": s.fill_params["method"],
                        "gap_length": gap_len,
                        "fraction_pct": fraction,
                        "random_state": 42,
                        "results": res,
                    }
                if s.validation is not None:
                    v = s.validation
                    st.dataframe(
                        v["results"].rename(columns={
                            "variable": "Variable", "n": "n", "mae": "MAE",
                            "rmse": "RMSE", "bias": "Bias", "r2": "R²",
                        }),
                        hide_index=True,
                        column_config={
                            k: st.column_config.NumberColumn(format="%.3f")
                            for k in ("MAE", "RMSE", "Bias", "R²")
                        },
                    )
                    st.caption(
                        f"Method `{v['method']}`; gaps of {v['gap_length']} "
                        f"record(s) covering ≈ {v['fraction_pct']} % of observed "
                        f"values per variable; random seed {v['random_state']}. "
                        "Bias = mean(filled − observed)."
                    )

            # Bootstrap uncertainty
            with st.expander("Uncertainty (bootstrap 95 % intervals)",
                             icon=":material/ssid_chart:"):
                st.caption(
                    "Fits one Random Forest (50 trees) per bootstrap resample of "
                    "the observed records and reports the 2.5–97.5 percentile "
                    "range of predictions for each filled value. The interval "
                    "is independent of the method that produced the value. "
                    "Runtime grows with resamples × variables."
                )
                c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
                n_boot = c1.number_input("Resamples", 10, 200, 50, 10)
                unc_vars = c2.multiselect("Variables", gap_vars, default=gap_vars)
                if c3.button("Estimate", icon=":material/play_arrow:",
                             width="stretch", disabled=not unc_vars):
                    bar = st.progress(0.0, text="Starting…")

                    def _progress(col, done, total):
                        bar.progress(done / total, text=f"{col} ({done}/{total})")

                    engine.uncertainty_results = {}
                    engine.compute_uncertainty(
                        columns=unc_vars, n_bootstrap=int(n_boot),
                        progress_callback=_progress,
                    )
                    bar.empty()
                    s.unc_params = {"n_bootstrap": int(n_boot),
                                    "variables": list(unc_vars),
                                    "random_state": 42}
                    st.rerun()
                if s.unc_params is not None:
                    widths = [
                        {"Variable": c, "Values": len(u),
                         "Median CI width": float(u["ci_width"].median()),
                         "Max CI width": float(u["ci_width"].max())}
                        for c, u in engine.uncertainty_results.items()
                    ]
                    if widths:
                        st.dataframe(
                            pd.DataFrame(widths), hide_index=True,
                            column_config={
                                k: st.column_config.NumberColumn(format="%.3f")
                                for k in ("Median CI width", "Max CI width")
                            },
                        )
                        st.caption(f"{s.unc_params['n_bootstrap']} resamples; "
                                   "intervals are drawn on the chart above.")
                    else:
                        st.caption("Too few complete records to estimate "
                                   "intervals.")


# ──────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────
with tab_export:
    if s.fill_params is None:
        st.info("Fill gaps in the **Fill** tab first.", icon=":material/info:")
    else:
        has_unc = bool(engine.uncertainty_results)
        has_rean = s.rean_info is not None
        c1, c2, c3 = st.columns(3)
        include_flags = c1.toggle("Quality flags and methods", value=True)
        include_unc = c2.toggle("95 % intervals", value=has_unc, disabled=not has_unc)
        include_rean = c3.toggle("Reanalysis columns", value=False,
                                 disabled=not has_rean)

        export_df = engine.export(
            include_uncertainty=include_unc and has_unc,
            include_metadata=include_flags,
        )
        if not include_rean and has_rean:
            export_df = export_df.drop(columns=s.rean_info["columns"],
                                       errors="ignore")

        st.dataframe(export_df.head(200), height=320, hide_index=True,
                     column_config=date_column_config(export_df, date_col))
        st.caption(f"{export_df.shape[0]:,} rows × {export_df.shape[1]} columns. "
                   + (FLAG_LEGEND if include_flags else ""))

        report = engine.generate_report()
        report["climfill_version"] = __version__
        report["generated"] = datetime.now().isoformat(timespec="seconds")
        report["source_file"] = s.data_name
        report["settings"] = {
            "detection": s.detect_params,
            "fill": s.fill_params,
            "inserted_records": s.inserted_rows,
        }
        if has_rean:
            report["reanalysis"] = {
                k: v for k, v in s.rean_info.items() if k != "columns"
            }
        if s.validation is not None:
            v = s.validation
            report["validation"] = {
                **{k: v[k] for k in ("method", "gap_length", "fraction_pct",
                                     "random_state")},
                "results": clean_nan(v["results"].to_dict("records")),
            }
        if s.unc_params is not None:
            report["uncertainty"] = s.unc_params

        stem = Path(s.data_name or "station").stem
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button(
            "Download CSV", export_df.to_csv(index=False, float_format="%.6g"),
            f"{stem}_climfill_{stamp}.csv", "text/csv", type="primary",
            icon=":material/download:", on_click="ignore", width="stretch",
        )
        c2.download_button(
            "Download report (JSON)",
            json.dumps(report, indent=2, default=json_default),
            f"{stem}_climfill_report_{stamp}.json", "application/json",
            icon=":material/description:", on_click="ignore", width="stretch",
        )

render_progress()
