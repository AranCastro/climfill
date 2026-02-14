"""ClimFill Test Suite — run: pytest tests/ -v"""

import numpy as np
import pandas as pd
import pytest
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climfill.engine import ClimFillEngine, GapDetector, GapFiller


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear
    temp = 25 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 2, n)
    precip = np.maximum(0, np.random.exponential(3, n))
    humidity = 60 + 15 * np.sin(2 * np.pi * (doy - 200) / 365) + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "Date": dates,
        "Temperature": np.round(temp, 1),
        "Precipitation": np.round(precip, 1),
        "Humidity": np.round(humidity, 1),
    })

    df.loc[10:12, "Temperature"] = np.nan
    df.loc[50:60, "Temperature"] = np.nan
    df.loc[200:240, "Temperature"] = np.nan
    df.loc[100:102, "Precipitation"] = np.nan
    df.loc[300:310, "Humidity"] = np.nan
    df.loc[500, "Temperature"] = -999
    df.loc[501, "Precipitation"] = -9999
    return df


@pytest.fixture
def engine_from_df(sample_df):
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False)
    buf.seek(0)
    eng = ClimFillEngine()
    eng.load_csv(buf)
    eng.detect_gaps()
    return eng


class TestGapDetector:
    def test_detect_date_column(self, sample_df):
        det = GapDetector(sample_df)
        assert det.detect_date_column() == "Date"

    def test_detect_date_column_explicit(self, sample_df):
        det = GapDetector(sample_df, date_col="Date")
        assert det.detect_date_column() == "Date"

    def test_no_date_column_raises(self):
        df = pd.DataFrame({"name": ["abc", "def"], "val": ["xyz", "uvw"]})
        det = GapDetector(df)
        with pytest.raises(ValueError, match="No date column"):
            det.detect_date_column()

    def test_replace_sentinels(self, sample_df):
        det = GapDetector(sample_df, date_col="Date")
        det.detect_date_column()
        replaced = det.replace_sentinels()
        assert len(replaced) > 0

    def test_analyze_gaps(self, sample_df):
        det = GapDetector(sample_df, date_col="Date")
        det.detect_date_column()
        det.replace_sentinels()
        report = det.analyze_gaps(["Temperature"])
        assert report["Temperature"]["missing_count"] > 0
        assert report["Temperature"]["n_gaps"] >= 3
        assert report["Temperature"]["max_gap_length"] == 41

    def test_gap_length_distribution(self, sample_df):
        det = GapDetector(sample_df, date_col="Date")
        det.detect_date_column()
        report = det.analyze_gaps(["Temperature"])
        dist = report["Temperature"]["gap_length_distribution"]
        assert ">30" in dist

    def test_temporal_regularity(self, sample_df):
        det = GapDetector(sample_df, date_col="Date")
        det.detect_date_column()
        result = det.check_temporal_regularity()
        assert result["regularity_pct"] == 100.0

    def test_outlier_detection(self, sample_df):
        sample_df.loc[100, "Temperature"] = 100.0
        det = GapDetector(sample_df, date_col="Date")
        det.detect_date_column()
        outliers = det.detect_outliers(["Temperature"])
        assert "Temperature" in outliers

    def test_no_gaps(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"Date": dates, "Temp": np.random.randn(100)})
        det = GapDetector(df, date_col="Date")
        det.detect_date_column()
        report = det.analyze_gaps(["Temp"])
        assert report["Temp"]["missing_count"] == 0
        assert report["Temp"]["n_gaps"] == 0


class TestGapFiller:
    def test_linear_interpolation(self, sample_df):
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        filler = GapFiller(sample_df, "Date")
        orig = sample_df["Temperature"].isna().sum()
        filler.linear_interpolation(["Temperature"], max_gap=3)
        assert filler.filled_df["Temperature"].isna().sum() < orig

    def test_spline_interpolation(self, sample_df):
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        filler = GapFiller(sample_df, "Date")
        orig = sample_df["Temperature"].isna().sum()
        filler.spline_interpolation(["Temperature"], max_gap=12)
        assert filler.filled_df["Temperature"].isna().sum() < orig

    def test_seasonal_interpolation(self, sample_df):
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        filler = GapFiller(sample_df, "Date")
        orig = sample_df["Temperature"].isna().sum()
        filler.seasonal_interpolation(["Temperature"])
        assert filler.filled_df["Temperature"].isna().sum() < orig

    def test_random_forest_fill(self, sample_df):
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        filler = GapFiller(sample_df, "Date")
        orig = sample_df["Temperature"].isna().sum()
        filler.random_forest_fill("Temperature", ["Precipitation", "Humidity"])
        assert filler.filled_df["Temperature"].isna().sum() <= orig

    def test_fill_metadata_tracking(self, sample_df):
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        filler = GapFiller(sample_df, "Date")
        filler.linear_interpolation(["Temperature"], max_gap=3)
        assert "Temperature" in filler.fill_metadata
        assert "F" in filler.fill_metadata["Temperature"]["flag"].values

    def test_auto_fill(self, sample_df):
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        sample_df.replace([-999, -9999], np.nan, inplace=True)
        filler = GapFiller(sample_df, "Date")
        filler.auto_fill(["Temperature", "Precipitation", "Humidity"])
        for col in ["Temperature", "Precipitation", "Humidity"]:
            assert filler.filled_df[col].isna().sum() <= sample_df[col].isna().sum()


class TestClimFillEngine:
    def test_load_csv(self, sample_df):
        buf = io.StringIO()
        sample_df.to_csv(buf, index=False)
        buf.seek(0)
        eng = ClimFillEngine()
        info = eng.load_csv(buf)
        assert info["shape"][0] > 0
        assert info["date_column"] == "Date"

    def test_detect_gaps(self, engine_from_df):
        result = engine_from_df.detect_gaps()
        assert "gap_report" in result
        assert "Temperature" in result["gap_report"]

    def test_fill_gaps_auto(self, engine_from_df):
        engine_from_df.fill_gaps(method="auto")
        assert engine_from_df.filled_df is not None
        assert engine_from_df.filled_df["Temperature"].isna().sum() < engine_from_df.df["Temperature"].isna().sum()

    def test_fill_gaps_linear(self, engine_from_df):
        engine_from_df.fill_gaps(method="linear")
        assert engine_from_df.filled_df is not None

    def test_fill_gaps_spline(self, engine_from_df):
        engine_from_df.fill_gaps(method="spline")
        assert engine_from_df.filled_df is not None

    def test_fill_gaps_seasonal(self, engine_from_df):
        engine_from_df.fill_gaps(method="seasonal")
        assert engine_from_df.filled_df is not None

    def test_fill_gaps_random_forest(self, engine_from_df):
        engine_from_df.fill_gaps(method="random_forest")
        assert engine_from_df.filled_df is not None

    def test_invalid_method(self, engine_from_df):
        with pytest.raises(ValueError, match="Unknown method"):
            engine_from_df.fill_gaps(method="invalid")

    def test_export(self, engine_from_df, tmp_path):
        engine_from_df.fill_gaps(method="auto")
        path = str(tmp_path / "out.csv")
        df = engine_from_df.export(filepath=path)
        assert os.path.exists(path)
        assert "Temperature_flag" in df.columns

    def test_export_no_flags(self, engine_from_df):
        engine_from_df.fill_gaps(method="linear")
        df = engine_from_df.export(include_metadata=False)
        assert not any(c.endswith("_flag") for c in df.columns)

    def test_generate_report(self, engine_from_df):
        engine_from_df.fill_gaps(method="auto")
        report = engine_from_df.generate_report()
        assert "gap_summary" in report
        assert "fill_summary" in report

    def test_uncertainty(self, engine_from_df):
        engine_from_df.fill_gaps(method="auto")
        results = engine_from_df.compute_uncertainty(n_bootstrap=10)
        assert isinstance(results, dict)

    def test_full_pipeline(self, sample_df, tmp_path):
        buf = io.StringIO()
        sample_df.to_csv(buf, index=False)
        buf.seek(0)

        eng = ClimFillEngine()
        eng.load_csv(buf)
        eng.detect_gaps()
        eng.fill_gaps(method="auto")
        eng.compute_uncertainty(n_bootstrap=10)
        path = str(tmp_path / "full.csv")
        df = eng.export(filepath=path, include_uncertainty=True)
        assert os.path.exists(path)
        assert df.shape[0] > 0


class TestEdgeCases:
    def test_all_missing_column(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"Date": dates, "Temp": [np.nan] * 100, "Precip": np.random.randn(100)})
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        eng = ClimFillEngine()
        eng.load_csv(buf)
        eng.detect_gaps()
        eng.fill_gaps(method="linear")  # Should not crash

    def test_no_missing_data(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"Date": dates, "Temp": np.random.randn(100) + 25})
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        eng = ClimFillEngine()
        eng.load_csv(buf)
        r = eng.detect_gaps()
        assert r["gap_report"]["Temp"]["missing_count"] == 0

    def test_single_variable(self):
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        temp = np.random.randn(365) + 25
        temp[10:15] = np.nan
        df = pd.DataFrame({"Date": dates, "Temp": temp})
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        eng = ClimFillEngine()
        eng.load_csv(buf)
        eng.detect_gaps()
        eng.fill_gaps(method="auto")
        assert eng.filled_df["Temp"].isna().sum() < 5

    def test_very_short_series(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "Temp": [25, np.nan, 27, 28, np.nan, 30, 31, np.nan, 33, 34],
        })
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        eng = ClimFillEngine()
        eng.load_csv(buf)
        eng.detect_gaps()
        eng.fill_gaps(method="linear")
        assert eng.filled_df["Temp"].isna().sum() == 0
