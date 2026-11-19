"""Accuracy, physical-limit and interval tests against withheld truth.

``generate_sample_data.py`` draws the complete series before it removes any
value, so the same seed without gaps gives the truth behind every gap in
``sample_station_data.csv``.
"""

import io
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climfill import ClimFillEngine, GapFiller, infer_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_sample_data import generate_sample_station  # noqa: E402

SAMPLE = ROOT / "sample_station_data.csv"


@pytest.fixture(scope="module")
def truth():
    df = generate_sample_station(gap_pct=0, n_sensor_failures=0, n_maintenance=0)
    values = df.drop(columns="Date")
    # The generator still plants ten sentinel codes in the gap-free copy
    return values.where(~values.isin([-999, -9999]))


@pytest.fixture(scope="module")
def filled():
    eng = ClimFillEngine()
    eng.load_csv(SAMPLE)
    eng.detect_gaps()
    eng.fill_gaps(method="auto")
    eng.compute_uncertainty()
    return eng


def _engine_from(df, **kwargs):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    eng = ClimFillEngine()
    eng.load_csv(buf, **kwargs)
    return eng


def _daily(n=400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    doy = dates.day_of_year.to_numpy()
    return pd.DataFrame({
        "Date": dates,
        "Temperature": 25 + 6 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1, n),
        "Rainfall": np.where(rng.random(n) < 0.3, rng.exponential(8, n), 0.0),
    })


# ── accuracy ────────────────────────────────────────────────────────────
def test_auto_fills_every_gap(filled):
    assert filled.filled_df[filled.station_columns()].isna().sum().sum() == 0


def test_auto_is_not_worse_than_monthly_climatology(filled, truth):
    month = filled.df["Date"].dt.month
    for col in filled.station_columns():
        cells = filled.df[col].isna() & truth[col].notna()
        baseline = filled.df[col].groupby(month).transform("mean")
        rmse = np.sqrt(np.mean((filled.filled_df.loc[cells, col] - truth.loc[cells, col]) ** 2))
        rmse_clim = np.sqrt(np.mean((baseline[cells] - truth.loc[cells, col]) ** 2))
        assert rmse <= 1.15 * rmse_clim, (col, rmse, rmse_clim)


def test_seasonal_method_is_not_worse_than_monthly_climatology(truth):
    eng = ClimFillEngine()
    eng.load_csv(SAMPLE)
    eng.fill_gaps(method="seasonal")
    month = eng.df["Date"].dt.month
    for col in ["Temperature_Mean", "Temperature_Max", "Temperature_Min", "Pressure"]:
        cells = eng.df[col].isna() & truth[col].notna()
        baseline = eng.df[col].groupby(month).transform("mean")
        rmse = np.sqrt(np.mean((eng.filled_df.loc[cells, col] - truth.loc[cells, col]) ** 2))
        rmse_clim = np.sqrt(np.mean((baseline[cells] - truth.loc[cells, col]) ** 2))
        assert rmse <= rmse_clim, (col, rmse, rmse_clim)


def test_single_long_gap_uses_one_method(filled):
    report = filled.gap_report["Temperature_Mean"]
    gap = max(report["gaps"], key=lambda g: g["length"])
    methods = filled.filler.fill_metadata["Temperature_Mean"]["method"]
    assert methods.iloc[gap["start_idx"]:gap["end_idx"] + 1].nunique() == 1


# ── physical limits ─────────────────────────────────────────────────────
def test_fills_respect_physical_limits(filled):
    out = filled.filled_df
    assert (out["Precipitation"] >= 0).all()
    assert out["Relative_Humidity"].between(0, 100).all()
    assert (out["Wind_Speed"] >= 0).all()
    assert (out["Solar_Radiation"] >= 0).all()


@pytest.mark.parametrize("method", ["spline", "seasonal", "random_forest"])
def test_precipitation_never_negative(method):
    eng = ClimFillEngine()
    eng.load_csv(SAMPLE)
    eng.fill_gaps(method=method, columns=["Precipitation"])
    assert (eng.filled_df["Precipitation"].dropna() >= 0).all()


def test_zero_inflated_variable_skips_spline_in_auto(filled):
    methods = filled.filler.fill_metadata["Precipitation"]["method"]
    assert "spline_interpolation" not in set(methods)
    was_missing = filled.df["Precipitation"].isna()
    values = filled.filled_df.loc[was_missing, "Precipitation"]
    wet_min = filled.df["Precipitation"][filled.df["Precipitation"] > 0].min()
    assert ((values == 0) | (values >= wet_min)).all()


def test_infer_bounds():
    assert infer_bounds("Rainfall_mm") == (0.0, None)
    assert infer_bounds("Relative_Humidity") == (0.0, 100.0)
    assert infer_bounds("RH") == (0.0, 100.0)
    assert infer_bounds("Temperature_Min") == (None, None)


def test_user_bounds_override():
    df = _daily()
    df.loc[100:102, "Temperature"] = np.nan
    eng = _engine_from(df)
    eng.fill_gaps(method="linear", bounds={"Temperature": (None, 10.0)})
    assert (eng.filled_df.loc[100:102, "Temperature"] == 10.0).all()


# ── intervals and confidence ────────────────────────────────────────────
def test_intervals_cover_withheld_truth(filled, truth):
    inside = total = 0
    for col, unc in filled.uncertainty_results.items():
        actual = truth.loc[unc.index, col]
        ok = actual.notna()
        hit = (actual[ok] >= unc.loc[ok, "ci_lower_95"]) & (
            actual[ok] <= unc.loc[ok, "ci_upper_95"]
        )
        assert hit.mean() >= 0.85, (col, hit.mean())
        inside += int(hit.sum())
        total += int(ok.sum())
    assert set(filled.uncertainty_results) == set(filled.station_columns())
    assert 0.90 <= inside / total <= 0.99


def test_filled_value_lies_inside_its_interval(filled):
    for unc in filled.uncertainty_results.values():
        assert (unc["ci_lower_95"] <= unc["filled_value"]).all()
        assert (unc["filled_value"] <= unc["ci_upper_95"]).all()
        assert (unc["n_residuals"] >= 10).all()


def test_intervals_are_reproducible(filled):
    eng = ClimFillEngine()
    eng.load_csv(SAMPLE)
    eng.fill_gaps(method="auto")
    again = eng.compute_uncertainty()
    for col, unc in filled.uncertainty_results.items():
        pd.testing.assert_frame_equal(unc, again[col])


def test_confidence_is_a_holdout_r2(filled):
    for col, meta in filled.filler.fill_metadata.items():
        conf = meta["confidence"][meta["flag"] == "F"]
        assert conf.between(0, 1).all(), col
    # Smooth, strongly seasonal temperature is reconstructed far better
    # than day-to-day precipitation
    report = filled.generate_report()["fill_summary"]
    assert report["Temperature_Mean"]["mean_confidence"] > 0.8
    assert report["Precipitation"]["mean_confidence"] < 0.5


def test_n_bootstrap_is_deprecated(filled):
    with pytest.warns(DeprecationWarning, match="n_bootstrap"):
        filled.compute_uncertainty(columns=["Pressure"], n_bootstrap=5)


# ── loading ─────────────────────────────────────────────────────────────
def test_absent_rows_become_gaps_and_are_filled():
    df = _daily().drop(index=range(100, 130))
    eng = _engine_from(df)
    assert len(eng.df) == 400
    report = eng.detect_gaps()["gap_report"]
    assert report["Temperature"]["missing_count"] == 30
    eng.fill_gaps(method="auto")
    assert eng.filled_df["Temperature"].notna().all()


def test_regularise_can_be_disabled_and_skips_irregular_series():
    df = _daily().drop(index=range(100, 130))
    assert len(_engine_from(df, regularise=False).df) == 370
    # Timestamps off a common grid are left as they are
    odd = _daily(n=40)
    odd.loc[10, "Date"] += pd.Timedelta(hours=5)
    odd = odd.drop(index=[20, 21])
    assert len(_engine_from(odd).df) == 38
    # Monthly series are not expanded
    monthly = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=24, freq="MS"),
        "Temperature": np.arange(24.0),
    }).drop(index=[5])
    assert len(_engine_from(monthly).df) == 23


def test_default_sentinels_keep_valid_large_values():
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=6),
        "Pressure_Pa": [100012.0, 99999.0, 100020.0, -999.0, 100002.0, 100008.0],
        "Radiation": [9999.0, 5000.0, -99.9, 5100.0, 5050.0, 5075.0],
    })
    eng = ClimFillEngine()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    info = eng.load_csv(buf)
    assert info["sentinels_replaced"] == {"Pressure_Pa": 1, "Radiation": 1}
    assert eng.df.loc[1, "Pressure_Pa"] == 99999.0
    assert eng.df.loc[0, "Radiation"] == 9999.0
    # A file that does use 9999 as its code can say so
    custom = _engine_from(df, sentinels=[-999, -99.9, 9999])
    assert pd.isna(custom.df.loc[0, "Radiation"])
    # ... and the choice survives detect_gaps()
    custom.detect_gaps()
    assert pd.isna(custom.df.loc[0, "Radiation"])


# ── options ─────────────────────────────────────────────────────────────
def test_fill_options_take_effect_or_raise():
    df = _daily()
    df.loc[100:102, "Temperature"] = np.nan
    eng = _engine_from(df)
    eng.fill_gaps(method="linear", max_gap=1)
    assert eng.filled_df.loc[100:102, "Temperature"].isna().all()
    eng.fill_gaps(method="linear", max_gap=3)
    assert eng.filled_df.loc[100:102, "Temperature"].notna().all()
    with pytest.raises(TypeError, match="max_gapp"):
        eng.fill_gaps(method="linear", max_gapp=3)
    with pytest.raises(TypeError, match="does not accept max_gap"):
        eng.fill_gaps(method="seasonal", max_gap=3)
    with pytest.raises(TypeError):
        eng.validate(method="auto", order=2)


def test_import_leaves_warning_filters_alone():
    # Third-party imports register specific filters of their own; the
    # package must not add a blanket "ignore everything" entry
    code = (
        "import warnings, climfill; "
        "assert not any(f[0] == 'ignore' and f[1] is None and f[2] is Warning "
        "and f[3] is None for f in warnings.filters)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_filling_raises_no_deprecation_warning():
    df = _daily()
    df.loc[100:140, "Temperature"] = np.nan
    eng = _engine_from(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", FutureWarning)
        eng.fill_gaps(method="random_forest")
    assert isinstance(eng.filler, GapFiller)
