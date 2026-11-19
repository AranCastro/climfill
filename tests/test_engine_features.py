"""Tests for reanalysis merging, validation, flags and other engine features."""

import io
import sys

import numpy as np
import pandas as pd
import pytest

import climfill.reanalysis as reanalysis
from climfill import ClimFillEngine, cli
from climfill.reanalysis import suggest_reanalysis_mapping


def _engine_from(df, **kwargs):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    eng = ClimFillEngine()
    eng.load_csv(buf, **kwargs)
    return eng


def _station(n=200, seed=0, date_name="Date"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    doy = dates.dayofyear.to_numpy()
    temp = 25 + 5 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        date_name: dates,
        "Temperature": temp,
        "Humidity": 70 - 2 * (temp - 25) + rng.normal(0, 1, n),
    })


def _reanalysis_for(df, date_name="Date"):
    return pd.DataFrame({
        "date": df[date_name],
        "reanalysis_temperature_2m_mean": df["Temperature"] - 1.5,
    })


def test_date_detected_when_numeric_column_comes_first():
    df = _station()[["Temperature", "Date", "Humidity"]]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    assert _engine_from(df).date_col == "Date"


def test_load_csv_sorts_records_chronologically():
    df = _station(n=30).iloc[::-1]
    eng = _engine_from(df)
    assert eng.df["Date"].is_monotonic_increasing


def test_zero_inflated_variable_screened_on_wet_days():
    rng = np.random.default_rng(1)
    n = 400
    rain = np.where(rng.random(n) < 0.8, 0.0, rng.exponential(5, n))
    rain[10] = 400.0
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "Rain": rain,
    })
    eng = _engine_from(df)
    out = eng.detect_gaps(outlier_threshold=3.0)["outliers"]
    assert 10 in out["Rain"]["indices"]
    assert out["Rain"]["count"] < 10


def test_outlier_threshold_and_columns_are_honoured():
    df = _station()
    df.loc[50, "Temperature"] = 60.0
    eng = _engine_from(df)
    assert 50 in eng.detect_gaps(outlier_threshold=1.5)["outliers"]["Temperature"]["indices"]
    assert eng.detect_gaps(outlier_threshold=50.0)["outliers"] == {}
    assert "Temperature" not in eng.detect_gaps(
        outlier_threshold=1.5, outlier_columns=["Humidity"]
    )["outliers"]


def test_export_marks_outliers_as_suspicious():
    df = _station()
    df.loc[50, "Temperature"] = 60.0
    df.loc[60:62, "Temperature"] = np.nan
    eng = _engine_from(df)
    eng.detect_gaps(outlier_threshold=3.0)
    eng.fill_gaps(method="linear")
    out = eng.export()
    assert out.loc[50, "Temperature_flag"] == "S"
    assert (out.loc[60:62, "Temperature_flag"] == "F").all()
    assert eng.generate_report()["outlier_summary"]["Temperature"] >= 1


def test_merge_reanalysis_keeps_station_date_named_date():
    df = _station(date_name="date")
    eng = _engine_from(df)
    rean = _reanalysis_for(df, "date")
    cols = eng.merge_reanalysis(rean)
    assert cols == ["reanalysis_temperature_2m_mean"]
    assert "date" in eng.df.columns and eng.date_col == "date"
    # Re-merging replaces instead of duplicating columns
    eng.merge_reanalysis(rean)
    assert list(eng.df.columns).count("reanalysis_temperature_2m_mean") == 1
    assert len(eng.df) == len(df)


def test_reanalysis_columns_are_not_fill_targets():
    df = _station()
    df.loc[20:25, "Temperature"] = np.nan
    eng = _engine_from(df)
    rean = _reanalysis_for(_station())
    rean.loc[5:8, "reanalysis_temperature_2m_mean"] = np.nan
    eng.merge_reanalysis(rean)
    eng.detect_gaps()
    assert "reanalysis_temperature_2m_mean" not in eng.gap_report
    eng.fill_gaps(method="auto")
    assert eng.filled_df["reanalysis_temperature_2m_mean"].isna().sum() == 4
    assert "reanalysis_temperature_2m_mean" not in eng.filler.fill_metadata


def test_auto_fill_uses_mapped_reanalysis_for_long_gaps():
    df = _station()
    df.loc[100:130, "Temperature"] = np.nan
    eng = _engine_from(df)
    eng.merge_reanalysis(_reanalysis_for(_station()))
    eng.detect_gaps()
    eng.fill_gaps(
        method="auto",
        reanalysis_map={"Temperature": "reanalysis_temperature_2m_mean"},
    )
    methods = eng.filler.fill_metadata["Temperature"]["method"].loc[100:130]
    assert (methods == "era5_bias_corrected").sum() > 0
    truth = _station().loc[100:130, "Temperature"]
    err = (eng.filled_df.loc[100:130, "Temperature"] - truth).abs().mean()
    assert err < 1.0


def test_insert_missing_timestamps_adds_empty_rows():
    df = _station(n=30).drop(index=[5, 6, 20])
    eng = _engine_from(df)
    assert eng.insert_missing_timestamps() == 3
    assert len(eng.df) == 30
    assert eng.df["Temperature"].isna().sum() == 3
    assert eng.insert_missing_timestamps() == 0
    assert eng.detect_gaps()["temporal_regularity"]["irregular_intervals"] == 0


def test_validate_reports_accuracy_metrics():
    df = _station(n=300)
    eng = _engine_from(df)
    eng.detect_gaps()
    res = eng.validate(method="linear", gap_length=1, fraction=0.05)
    assert list(res["variable"]) == ["Temperature", "Humidity"]
    assert (res["n"] > 0).all()
    assert (res["rmse"] >= res["mae"]).all()
    assert (res["r2"] > 0.5).all()
    # Same seed, same result
    pd.testing.assert_frame_equal(
        res, eng.validate(method="linear", gap_length=1, fraction=0.05)
    )
    # The engine's own data is left untouched
    assert eng.df["Temperature"].notna().all()


def test_uncertainty_progress_and_reset_on_refill():
    df = _station(n=120)
    df.loc[[30, 60, 90], "Temperature"] = np.nan
    eng = _engine_from(df)
    eng.detect_gaps()
    eng.fill_gaps(method="linear")
    calls = []
    first = eng.compute_uncertainty(
        n_bootstrap=3, progress_callback=lambda *a: calls.append(a)
    )
    assert calls == [("Temperature", 1, 1)]
    first = first["Temperature"].copy()
    eng.fill_gaps(method="linear")
    assert eng.uncertainty_results == {}
    second = eng.compute_uncertainty(n_bootstrap=3)["Temperature"]
    pd.testing.assert_frame_equal(first, second)


def test_suggest_reanalysis_mapping():
    rean = [
        "reanalysis_temperature_2m_mean",
        "reanalysis_temperature_2m_max",
        "reanalysis_precipitation_sum",
    ]
    mapping = suggest_reanalysis_mapping(
        ["Tmax", "Temperature_Mean", "Rainfall", "Station_ID", "Humidity"], rean
    )
    assert mapping == {
        "Tmax": "reanalysis_temperature_2m_max",
        "Temperature_Mean": "reanalysis_temperature_2m_mean",
        "Rainfall": "reanalysis_precipitation_sum",
    }


def test_open_meteo_http_error_is_readable(monkeypatch):
    import requests

    class _Resp:
        status_code = 400

        def raise_for_status(self):
            raise requests.HTTPError("400")

        def json(self):
            return {"error": True, "reason": "Latitude must be in range"}

    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match=r"HTTP 400\): Latitude must be in range"):
        reanalysis.get_open_meteo_data(100, 0, "2024-01-01", "2024-01-02")


def test_cli_reanalysis_requires_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "in.csv"
    _station(n=10).to_csv(path, index=False)
    monkeypatch.setattr(sys, "argv", ["climfill", str(path), "--reanalysis"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("method", ["random_forest", "xgboost"])
def test_ml_methods_fill_multi_day_gaps_and_outages(method):
    df = _station()
    df.loc[100:110, "Temperature"] = np.nan        # neighbours missing inside
    df.loc[150:154, ["Temperature", "Humidity"]] = np.nan  # full outage
    eng = _engine_from(df)
    eng.detect_gaps()
    eng.fill_gaps(method=method)
    assert eng.filled_df[["Temperature", "Humidity"]].isna().sum().sum() == 0
    flags = eng.filler.fill_metadata["Temperature"]["flag"]
    assert (flags.loc[100:110] == "F").all()
    assert (flags.loc[150:154] == "F").all()


def test_interpolation_leaves_long_and_edge_gaps_alone():
    df = _station(n=60)
    df.loc[[0, 1], "Temperature"] = np.nan      # leading edge
    df.loc[10:11, "Temperature"] = np.nan       # short interior gap
    df.loc[20:29, "Temperature"] = np.nan       # 10-record gap
    eng = _engine_from(df)
    eng.fill_gaps(method="linear")
    filled = eng.filled_df["Temperature"]
    assert filled.loc[10:11].notna().all()
    assert filled.loc[20:29].isna().all()
    assert filled.loc[[0, 1]].isna().all()
    eng.fill_gaps(method="spline")
    assert eng.filled_df["Temperature"].loc[20:29].isna().all()
