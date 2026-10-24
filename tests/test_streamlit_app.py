"""End-to-end tests of the Streamlit app using streamlit.testing.AppTest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

import climfill.reanalysis as reanalysis  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _start_with_sample():
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["source"] = "Sample"
    at.run()
    assert not at.exception
    return at


def test_landing_page_without_data():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Upload a station CSV" in i.value for i in at.info)
    assert len(at.tabs) == 0


def test_full_workflow_survives_reruns():
    at = _start_with_sample()
    assert at.metric[0].value == "1,826"

    _button(at, "Detect gaps").click().run()
    assert not at.exception
    assert at.session_state["detection"] is not None

    _button(at, "Fill gaps").click().run()
    assert not at.exception
    engine = at.session_state["engine"]
    assert engine.filled_df is not None

    # Interacting with any widget after filling must keep the results
    at.selectbox(key="fill_var").set_value("Pressure").run()
    assert not at.exception
    assert at.session_state["engine"].filled_df is not None

    _button(at, "Run validation").click().run()
    assert not at.exception
    results = at.session_state["validation"]["results"]
    assert set(results.columns) == {"variable", "n", "mae", "rmse", "bias", "r2"}
    assert (results["n"] > 0).all()

    unc_select = next(m for m in at.multiselect if m.label == "Variables")
    unc_select.set_value(["Pressure"])
    next(n for n in at.number_input if n.label == "Resamples").set_value(10)
    at.run()
    _button(at, "Estimate").click().run()
    assert not at.exception
    assert at.session_state["unc_params"]["variables"] == ["Pressure"]
    assert set(at.session_state["engine"].uncertainty_results) == {"Pressure"}

    labels = [d.label for d in at.get("download_button")]
    assert labels == ["Download CSV", "Download report (JSON)"]


def test_reanalysis_merge_and_mapping(monkeypatch):
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    rng = np.random.default_rng(0)
    fake = pd.DataFrame({
        "date": dates,
        "reanalysis_temperature_2m_mean": 27 + rng.normal(0, 1, len(dates)),
        "reanalysis_precipitation_sum": rng.exponential(2, len(dates)),
    })
    monkeypatch.setattr(reanalysis, "get_open_meteo_data", lambda **k: fake)

    at = _start_with_sample()
    _button(at, "Detect gaps").click().run()
    _button(at, "Download and merge").click().run()
    assert not at.exception
    assert at.session_state["rean_map"] == {
        "Temperature_Mean": "reanalysis_temperature_2m_mean",
        "Precipitation": "reanalysis_precipitation_sum",
    }

    _button(at, "Fill gaps").click().run()
    assert not at.exception
    engine = at.session_state["engine"]
    methods = engine.filler.fill_metadata["Temperature_Mean"]["method"]
    assert (methods == "era5_bias_corrected").any()
    # Reanalysis series are predictors, never fill targets
    assert not any(c.startswith("reanalysis_") for c in engine.filler.fill_metadata)

    _button(at, "Remove reanalysis").click().run()
    assert not at.exception
    assert not any(c.startswith("reanalysis_") for c in engine.df.columns)
    assert at.session_state["fill_params"] is None

