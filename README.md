# 🌍 ClimFill — Automated Climate Station Data Gap-Filler

[![CI](https://github.com/yourusername/climfill/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/climfill/actions)
[![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen)]
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)]
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
git clone https://github.com/yourusername/climfill.git
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
streamlit run streamlit_app.py
```

## Methods

| Method | Gap Length | Description |
|--------|-----------|-------------|
| Linear Interpolation | 1–3 days | Boundary-constrained fill |
| Cubic Spline | 4–7 days | Smooth polynomial fit |
| Seasonal Decomposition | 8–30 days | Climatology + anomaly |
| Random Forest | Any | Multi-variable ML regression |
| XGBoost | Any | Gradient-boosted trees |
| ERA5 Bias-Corrected | Any | Quantile-mapped reanalysis |

`auto` mode cascades simple → complex, selecting the best method per gap.

## Features

- **Smart gap detection** with sentinel replacement (-999, -9999)
- **Quality flags**: O (original), F (filled), S (suspicious)
- **Bootstrap uncertainty** with 95% confidence intervals
- **Free reanalysis** via Open-Meteo (no API key needed)
- **Publication-ready** CSV export with full metadata

## Citation

```bibtex
@software{climfill2026,
  author    = {Castro, Aran},
  title     = {ClimFill: Automated Climate Station Data Gap-Filler},
  year      = {2026},
  version   = {1.0.0},
  publisher = {GitHub},
  url       = {https://github.com/dr-aran-castro/climfill}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Priority areas: LSTM methods, multi-station support, hourly data.

## License

MIT
