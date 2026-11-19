# 🌍 ClimFill — Automated Climate Station Data Gap-Filler

[![Tests](https://github.com/AranCastro/climfill/actions/workflows/tests.yml/badge.svg)](https://github.com/AranCastro/climfill/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18638725.svg)](https://doi.org/10.5281/zenodo.18638725)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

**Free, open-source tool for filling gaps in daily climate station records.**

> **Name.** This package is unrelated to *CLIMFILL v0.9*, the multivariate
> framework for gap filling gridded Earth observations by Bessenbacher et al.
> (2022, https://doi.org/10.5194/gmd-15-4569-2022). ClimFill, described here,
> works on single-station time series.

## Installation

Install from the repository (Python 3.9 or later, pandas 2.0 or later):

```bash
git clone https://github.com/AranCastro/climfill.git
cd climfill
pip install -e .            # Core
pip install -e ".[ml]"      # + XGBoost
pip install -e ".[app]"     # + Streamlit web UI
pip install -e ".[all]"     # Everything, including test tools
```

`constraints-tested.txt` lists the library versions with which the test suite
was last run; add `-c constraints-tested.txt` to reproduce that environment.

## Quick Start

### Python API

```python
from climfill import ClimFillEngine

engine = ClimFillEngine()
engine.load_csv("my_station.csv")
engine.detect_gaps()
engine.fill_gaps(method="auto")
engine.compute_uncertainty()
engine.export("filled.csv", include_uncertainty=True)
```

`load_csv` sorts the records by date, converts the codes -999, -9999 and
-99.9 to missing (`sentinels=[...]` replaces the list) and, for series with
a step of one day or less, inserts an empty row for every absent timestamp
so that gaps stored as missing rows are detected (`regularise=False`
disables this).

### Command Line

```bash
climfill station.csv --lat 10.78 --lon 79.13 --reanalysis --uncertainty
```

Coordinates are decimal degrees, WGS 84 (EPSG:4326). Without `--output` the
result is written beside the input as `<name>_filled<ext>`; the input file
is never overwritten. `--sentinels 9999,-999` sets the missing-value codes.

### Web Interface

```bash
pip install -e ".[app,ml]"
streamlit run streamlit_app.py
```

The app (Streamlit 1.51 or later) runs the workflow in five tabs: **Data**
(variable summary), **Gaps** (availability heatmap, gap statistics, outlier
screen, insertion of absent timestamps), **Reanalysis** (Open-Meteo download
for a WGS 84 / EPSG:4326 location, with editable station-to-reanalysis
variable matching), **Fill** (method choice, hold-out validation, bootstrap
intervals) and **Export** (CSV plus a JSON report of all settings). It follows
the system light/dark setting.

## Methods

| Method | Gap Length | Description |
|--------|-----------|-------------|
| Linear Interpolation | ≤ 3 records | Interior gaps only; longer gaps are left whole |
| Cubic Spline | ≤ 7 records | Interior gaps only |
| ERA5 Bias-Corrected | Any | Quantile-mapped reanalysis (needs a matched reanalysis series) |
| Seasonal Decomposition | Any | Annual-cycle climatology (mean and two harmonics; calendar-month means for records shorter than a year) plus the anomaly at the gap edges, damped with an e-folding distance of 7 records |
| Random Forest | Any | Multi-variable regression; refits without lag terms inside long gaps |
| XGBoost | Any | Gradient-boosted trees; same feature passes as Random Forest |

`auto` mode applies the stages in the order listed, per variable, and records
the method used for every filled value. Two rules depart from the plain order:

- An interpolation stage is skipped for a variable when, on observed values
  hidden as gaps of the same lengths, its root-mean-square error exceeds that
  of the seasonal stage.
- Zero-inflated variables (non-negative, at least 25 % zeros, e.g. daily
  precipitation) skip the spline and use regression before climatology.

### Physical limits

Filled values are limited to bounds inferred from the variable name:
precipitation, wind speed, solar radiation, pressure and evapotranspiration
are kept at or above zero, and relative humidity within 0–100 %. For a
zero-inflated variable, fills smaller than the smallest positive observation
are set to zero. `fill_gaps(bounds={"Column": (lower, upper)})` overrides the
inferred limits. Consistency between variables (e.g. Tmin ≤ Tmean ≤ Tmax) is
not enforced.

### Confidence and intervals

While filling, every stage hides a sample of observed values as gaps of the
lengths it is about to fill and reconstructs them.

- `confidence` is the R² of that hold-out, clamped to 0–1 (five-fold
  cross-validated R² for the regression methods; in-sample R² for
  bias-corrected reanalysis). It is empty when fewer than ten values could be
  held out.
- `compute_uncertainty()` gives each filled value a 95 % interval: the value
  minus the 97.5th and 2.5th percentiles of the hold-out residuals of the
  stage that produced it, limited to the physical bounds.

On the bundled synthetic sample (1,826 daily records, eight variables, 997
filled values, one realisation, seed 42) the intervals contained the withheld
true value for 91–98 % of filled values per variable. This figure has not
been established on observed station data.

### Accuracy assessment

```python
engine.validate(method="auto", gap_length=7, fraction=0.05)
```

hides about 5 % of the observed values per variable as synthetic gaps of the
given length, re-runs the method and returns n, MAE, RMSE, bias and R² per
variable (fixed random seed).

## Features

- **Gap detection** with sentinel replacement (-999, -9999, -99.9;
  configurable) and insertion of absent daily timestamps
- **Quality flags**: O (original), F (filled), S (suspicious: IQR or z-score
  outlier; zero-inflated variables such as precipitation are screened on
  non-zero values)
- **95 % intervals** from hold-out residuals of the method that produced
  each fill
- **Physical limits** on filled values
- **ERA5 reanalysis** via the Open-Meteo archive API (no API key; see the
  Open-Meteo terms for conditions of use)
- **CSV export** with flag, method and confidence for every value

## Citation

Castro, A. (2026). *ClimFill: Automated Climate Station Data Gap-Filler*
[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.18638725

Citation metadata is in [CITATION.cff](CITATION.cff).

## Contributing

Issues and pull requests are welcome. Priority areas: LSTM methods,
multi-station support, hourly data.

## License

MIT
