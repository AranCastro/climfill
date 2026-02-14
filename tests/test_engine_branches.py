import io

import numpy as np
import pandas as pd
import pytest

import climfill.engine as engine
from climfill.engine import ClimFillEngine, GapDetector, GapFiller


def _make_df(length=40, missing_idx=None):
    dates = pd.date_range("2021-01-01", periods=length, freq="D")
    data = pd.DataFrame(
        {
            "Date": dates,
            "Temp": np.linspace(0, 1, length),
            "Aux": np.linspace(5, 6, length),
        }
    )
    missing_idx = missing_idx or []
    data.loc[missing_idx, "Temp"] = np.nan
    return data


def test_temporal_regularity_without_date_column():
    df = pd.DataFrame({"value": [1, 2, 3]})
    det = GapDetector(df)
    assert det.check_temporal_regularity() is None


def test_temporal_regularity_single_value():
    df = pd.DataFrame({"Date": [pd.Timestamp("2020-01-01")], "Temp": [1.0]})
    det = GapDetector(df, date_col="Date")
    det.detect_date_column()
    assert det.check_temporal_regularity() is None


def test_record_fill_ignores_unknown_mask():
    df = _make_df(missing_idx=[1, 3])
    filler = GapFiller(df, "Date")
    filler._record_fill("Temp", mask="bad", method="x", confidence=0.0)
    meta = filler.fill_metadata["Temp"]
    assert not (meta["flag"] == "F").any()


def test_random_forest_fill_no_missing_short_circuits():
    df = _make_df()
    filler = GapFiller(df, "Date")
    result = filler.random_forest_fill("Temp", ["Aux"])
    pd.testing.assert_frame_equal(result, filler.filled_df)


def test_random_forest_fill_handles_cross_val_failure(monkeypatch):
    df = _make_df(length=60, missing_idx=[5, 10, 15, 20])
    filler = GapFiller(df, "Date")

    def _boom(*args, **kwargs):
        raise ValueError("cv fail")

    monkeypatch.setattr(engine, "cross_val_score", _boom)
    filled = filler.random_forest_fill("Temp", ["Aux"])
    assert filled["Temp"].isna().sum() < df["Temp"].isna().sum()


def test_detect_outliers_zscore_and_unknown():
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    values = [1.0] * 29 + [50.0]
    df = pd.DataFrame({"Date": dates, "Temp": values})
    det = GapDetector(df, date_col="Date")
    det.detect_date_column()
    z_out = det.detect_outliers(["Temp"], method="zscore", threshold=1.0)
    assert "Temp" in z_out
    assert det.detect_outliers(["Temp"], method="unknown") == {}


def test_record_fill_accepts_numpy_mask():
    df = _make_df(length=5, missing_idx=[1])
    filler = GapFiller(df, "Date")
    mask = np.array([False, True, False, False, False])
    filler._record_fill("Temp", mask, "manual", 0.9)
    assert filler.fill_metadata["Temp"]["flag"].iloc[1] == "F"


def test_linear_interpolation_skips_date_column():
    df = _make_df(length=6, missing_idx=[2])
    filler = GapFiller(df, "Date")
    result = filler.linear_interpolation(columns=["Date"])
    pd.testing.assert_frame_equal(result, filler.filled_df)


def test_spline_interpolation_falls_back(monkeypatch):
    df = _make_df(length=10, missing_idx=[2, 3])
    filler = GapFiller(df, "Date")
    original = pd.Series.interpolate

    def _maybe_fail(self, *args, **kwargs):
        if kwargs.get("method") == "spline":
            raise ValueError("bad spline")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "interpolate", _maybe_fail)
    filled = filler.spline_interpolation(["Temp"], max_gap=2, order=3)
    assert filled["Temp"].isna().sum() < df["Temp"].isna().sum()


def test_seasonal_interpolation_skips_when_full():
    df = _make_df(length=12)
    filler = GapFiller(df, "Date")
    result = filler.seasonal_interpolation(["Temp"])
    pd.testing.assert_series_equal(result["Temp"], df["Temp"])


def test_xgboost_fill_stubbed(monkeypatch):
    df = _make_df(length=80, missing_idx=[30])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    def _fake_cv(model, X, y, cv, scoring=None):
        return np.array([0.6, 0.7])

    class DummyXGB:
        def __init__(self, **kwargs):
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True
            self.mean_ = float(np.mean(y))

        def predict(self, X):
            return np.full(len(X), getattr(self, "mean_", 0.0))

    monkeypatch.setattr(engine, "HAS_XGBOOST", True)
    monkeypatch.setattr(engine, "XGBRegressor", DummyXGB)
    monkeypatch.setattr(engine, "cross_val_score", _fake_cv)

    eng = ClimFillEngine()
    eng.load_csv(buf)
    eng.fill_gaps(method="xgboost", columns=["Date", "Temp", "Aux"])
    assert eng.filled_df["Temp"].isna().sum() < df["Temp"].isna().sum()


def test_xgboost_fill_uses_random_forest_when_disabled(monkeypatch):
    df = _make_df(length=30, missing_idx=[5])
    filler = GapFiller(df, "Date")
    monkeypatch.setattr(engine, "HAS_XGBOOST", False)
    result = filler.xgboost_fill("Temp", predictor_cols=["Aux"])
    assert result["Temp"].isna().sum() <= df["Temp"].isna().sum()


def test_xgboost_fill_handles_cv_error(monkeypatch):
    df = _make_df(length=80, missing_idx=[30])
    filler = GapFiller(df, "Date")

    def _boom(*args, **kwargs):
        raise ValueError("cv error")

    monkeypatch.setattr(engine, "HAS_XGBOOST", True)
    monkeypatch.setattr(engine, "cross_val_score", _boom)

    class DummyXGB:
        def __init__(self, **kwargs):
            self.mean_ = 0.0

        def fit(self, X, y):
            self.mean_ = float(np.mean(y))

        def predict(self, X):
            return np.full(len(X), self.mean_)

    monkeypatch.setattr(engine, "XGBRegressor", DummyXGB)
    filled = filler.xgboost_fill("Temp", predictor_cols=["Aux"])
    assert filled["Temp"].isna().sum() < df["Temp"].isna().sum()


def test_era5_bias_correction_requires_overlap():
    df = _make_df(length=8, missing_idx=[1, 2])
    filler = GapFiller(df, "Date")
    filler.filled_df["reanalysis_temp"] = np.linspace(10, 11, len(df))
    result = filler.era5_bias_correction("Temp", "reanalysis_temp")
    assert result["Temp"].isna().sum() == df["Temp"].isna().sum()


def test_era5_bias_correction_returns_when_no_missing():
    df = _make_df(length=8)
    filler = GapFiller(df, "Date")
    filler.filled_df["reanalysis_temp"] = np.linspace(10, 11, len(df))
    result = filler.era5_bias_correction("Temp", "reanalysis_temp")
    pd.testing.assert_frame_equal(result, filler.filled_df)


def test_era5_bias_correction_quantile_mapping():
    df = _make_df(length=40, missing_idx=[5, 10, 15, 20])
    filler = GapFiller(df, "Date")
    filler.filled_df["reanalysis_temp"] = np.linspace(10, 12, len(df))
    result = filler.era5_bias_correction("Temp", "reanalysis_temp")
    assert result["Temp"].isna().sum() < df["Temp"].isna().sum()


def test_era5_bias_correction_bias_branch():
    df = _make_df(length=20, missing_idx=[5, 6])
    filler = GapFiller(df, "Date")
    filler.filled_df["reanalysis_temp"] = np.linspace(10, 11, len(df))
    result = filler.era5_bias_correction("Temp", "reanalysis_temp")
    assert result["Temp"].isna().sum() < df["Temp"].isna().sum()


def test_auto_fill_skips_non_target_columns():
    df = _make_df(length=20, missing_idx=[3, 4])
    filler = GapFiller(df, "Date")
    filler.auto_fill(columns=["Date", "Aux", "Temp"])
    assert "Temp" in filler.fill_metadata
    assert "Aux" not in filler.fill_metadata


def test_auto_fill_invokes_ml_stage(monkeypatch):
    df = _make_df(length=30, missing_idx=[10])
    filler = GapFiller(df, "Date")

    def _no_op(*args, **kwargs):
        return filler.filled_df

    def _fill_rf(target_col, predictor_cols=None, n_estimators=100):
        mask = filler.filled_df[target_col].isna()
        filler.filled_df.loc[mask, target_col] = 0.0
        filler._record_fill(target_col, mask, "rf_stub", 0.5)
        return filler.filled_df

    monkeypatch.setattr(filler, "linear_interpolation", _no_op)
    monkeypatch.setattr(filler, "spline_interpolation", _no_op)
    monkeypatch.setattr(filler, "seasonal_interpolation", _no_op)
    monkeypatch.setattr(engine, "HAS_XGBOOST", False)
    monkeypatch.setattr(filler, "random_forest_fill", _fill_rf)

    filled = filler.auto_fill(columns=["Temp", "Aux"])
    assert filled["Temp"].isna().sum() == 0
    assert filler.fill_metadata["Temp"]["method"].str.contains("rf_stub").any()


def test_bootstrap_uncertainty_short_circuits_no_missing():
    df = _make_df(length=10)
    filler = GapFiller(df, "Date")
    res = filler.bootstrap_uncertainty("Temp", n_bootstrap=5)
    assert res.empty


def test_bootstrap_uncertainty_short_circuits_with_little_data():
    df = _make_df(length=10, missing_idx=[1, 2])
    filler = GapFiller(df, "Date")
    res = filler.bootstrap_uncertainty("Temp", n_bootstrap=5)
    assert res.empty


def test_engine_compute_uncertainty_requires_fill():
    eng = ClimFillEngine()
    with pytest.raises(RuntimeError, match="Run fill_gaps"):
        eng.compute_uncertainty()


def test_export_requires_fill():
    eng = ClimFillEngine()
    with pytest.raises(RuntimeError, match="Run fill_gaps"):
        eng.export()


def test_generate_report_without_state():
    eng = ClimFillEngine()
    assert eng.generate_report() == {}
