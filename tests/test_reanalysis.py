import pandas as pd
import pytest

import climfill.reanalysis as reanalysis


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_get_open_meteo_data_parses_and_prefixes(monkeypatch):
    payload = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_mean": [10.0, 11.0],
            "precipitation_sum": [0.1, 0.0],
        }
    }

    def _fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.get", _fake_get)

    df = reanalysis.get_open_meteo_data(
        lat=10.0,
        lon=20.0,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert set(df.columns) == {
        "date",
        "reanalysis_temperature_2m_mean",
        "reanalysis_precipitation_sum",
    }
    assert len(df) == 2


def test_get_open_meteo_data_raises_when_missing_daily(monkeypatch):
    def _fake_get(url, params=None, timeout=None):
        return _FakeResponse({"not_daily": {}})

    monkeypatch.setattr("requests.get", _fake_get)

    with pytest.raises(ValueError):
        reanalysis.get_open_meteo_data(
            lat=10.0,
            lon=20.0,
            start_date="2024-01-01",
            end_date="2024-01-02",
            variables=["temperature_2m_mean"],
        )
