import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from climfill import cli
import climfill.reanalysis as reanalysis


def _build_cli_df(with_gaps=True):
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Temperature": [10, 11, 12, 13, 14, 15, 16, 17],
            "Humidity": [50, 51, 52, 53, 54, 55, 56, 57],
        }
    )
    if with_gaps:
        df.loc[3:4, "Temperature"] = pd.NA
    return df


def test_cli_main_writes_output_and_report(tmp_path, monkeypatch, capsys):
    df = _build_cli_df()
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "filled.csv"
    report_path = tmp_path / "report.json"
    df.to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "climfill",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--method",
            "linear",
        ],
    )

    cli.main()

    assert output_path.exists()
    assert report_path.exists()
    report = json.loads(Path(report_path).read_text())
    assert "gap_summary" in report
    assert "Filling gaps" in capsys.readouterr().out


def test_cli_main_exits_when_no_gaps(tmp_path, monkeypatch):
    df = _build_cli_df(with_gaps=False)
    input_path = tmp_path / "clean.csv"
    df.to_csv(input_path, index=False)

    monkeypatch.setattr(sys, "argv", ["climfill", str(input_path), "--quiet"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0


def test_cli_reanalysis_failure_is_non_fatal(tmp_path, monkeypatch, capsys):
    df = _build_cli_df()
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "filled.csv"
    df.to_csv(input_path, index=False)

    def _fail_reanalysis(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(reanalysis, "get_open_meteo_data", _fail_reanalysis)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "climfill",
            str(input_path),
            "--output",
            str(output_path),
            "--method",
            "linear",
            "--reanalysis",
            "--lat",
            "10.0",
            "--lon",
            "20.0",
        ],
    )

    cli.main()

    assert output_path.exists()
    output = capsys.readouterr().out
    assert "Failed" in output or "Continuing without reanalysis" in output


def test_cli_reanalysis_success_with_uncertainty(tmp_path, monkeypatch):
    df = _build_cli_df()
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "filled.csv"
    df.to_csv(input_path, index=False)

    fake_rean = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=len(df), freq="D"),
            "reanalysis_temperature_2m_mean": range(len(df)),
        }
    )

    monkeypatch.setattr(reanalysis, "get_open_meteo_data", lambda *a, **k: fake_rean)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "climfill",
            str(input_path),
            "--output",
            str(output_path),
            "--method",
            "linear",
            "--lat",
            "0.0",
            "--lon",
            "0.0",
            "--reanalysis",
            "--uncertainty",
            "--bootstrap-n",
            "3",
        ],
    )

    cli.main()

    result_df = pd.read_csv(output_path)
    assert "reanalysis_temperature_2m_mean" in result_df.columns
