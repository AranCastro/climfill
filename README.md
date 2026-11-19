# 🌍 ClimFill — Automated Climate Station Data Gap-Filler

[![Tests](https://github.com/AranCastro/climfill/actions/workflows/tests.yml/badge.svg)](https://github.com/AranCastro/climfill/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18638725.svg)](https://doi.org/10.5281/zenodo.18638725)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

**Free, open-source tool for intelligently filling gaps in climate station data.**

## Installation

```bash
pip install climfill          # Core
pip install climfill[ml]      # + XGBoost
pip install climfill[app]     # + Streamlit web UI
pip install climfill[all]     # Everything

# From source
git clone https://github.com/AranCastro/climfill.git
cd climfill && pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from climfill import ClimFillEngine

engine = ClimFillEngine()
engine.load_csv("my_station.csv")
engine.detect_gaps()
engine.fill_gaps(method="auto")
engine.compute_uncertainty(n_bootstrap=50)
engine.export("filled.csv", include_uncertainty=True)
```

### Command Line

```bash
climfill station.csv --lat 10.78 --lon 79.13 --reanalysis --uncertainty
```

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
| Seasonal Decomposition | Any interior | Monthly climatology + interpolated anomaly |
| Random Forest | Any | Multi-variable regression; refits without lag terms inside long gaps |
| XGBoost | Any | Gradient-boosted trees; same feature passes as Random Forest |

`auto` mode applies the stages in the order listed, per variable, and records
the method used for every filled value.

### Accuracy assessment

```python
engine.validate(method="auto", gap_length=7, fraction=0.05)
```

hides about 5 % of the observed values per variable as synthetic gaps of the
given length, re-runs the method and returns n, MAE, RMSE, bias and R² per
variable (fixed random seed).

## Features

- **Smart gap detection** with sentinel replacement (-999, -9999)
- **Quality flags**: O (original), F (filled), S (suspicious: IQR or z-score
  outlier; zero-inflated variables such as precipitation are screened on
  non-zero values)
- **Bootstrap uncertainty** with 95% confidence intervals
- **Free reanalysis** via Open-Meteo (no API key needed)
- **Publication-ready** CSV export with full metadata

## Citation

Castro, A. (2026). *ClimFill: Automated Climate Station Data Gap-Filler*
[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.18638725

Citation metadata is in [CITATION.cff](CITATION.cff).

## Contributing

Issues and pull requests are welcome. Priority areas: LSTM methods,
multi-station support, hourly data.

## License

MIT
