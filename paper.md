---
title: 'ClimFill: An Open-Source Tool for Automated Gap-Filling in Climate Station Time Series'
tags:
  - Python
  - climate science
  - gap-filling
  - time series
  - machine learning
  - reanalysis
authors:
  - name: Aran
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Earth Sciences, [University Name]
    index: 1
date: 2025
bibliography: paper.bib
---

# Summary

Gaps in climate station time series—caused by sensor failures, power outages, and maintenance—are a persistent challenge for Earth scientists. `ClimFill` is a free, open-source Python tool that automates detection, filling, and quality-flagging of missing data using a cascade of statistical methods, machine learning, and reanalysis data integration with built-in uncertainty quantification.

# Statement of Need

Climate time series are foundational to hydrology, agriculture, ecology, and climate research. Yet 30–50% of stations in developing regions have significant gaps [@Hunziker2017]. Current solutions are either too simplistic (linear interpolation), require programming expertise [@vanBuuren2011; @Stekhoven2012], or are prohibitively expensive. `ClimFill` bridges this gap with an accessible tool (web + CLI + Python API) that is scientifically rigorous and practically free.

# Methodology

`ClimFill` implements a hierarchical strategy:

1. **Gap Detection**: Identifies missing values including sentinel codes (-999, -9999), characterizes gap lengths, and flags outliers.
2. **Multi-Method Filling**: Linear interpolation (1–3 records), cubic spline (4–7), seasonal decomposition (8–30), and Random Forest/XGBoost regression for longer gaps using correlated variables and cyclical temporal features.
3. **Reanalysis Integration**: Auto-downloads gridded data from Open-Meteo (free) or ERA5 [@Hersbach2020], with quantile mapping bias correction [@Maraun2016].
4. **Uncertainty Quantification**: Bootstrap resampling generates 95% confidence intervals for each filled value.
5. **Quality Flagging**: Every value tagged as Original (`O`), Filled (`F`), or Suspicious (`S`) with method name and confidence score.

# Validation

Testing on synthetic 5-year daily data (8 variables, ~14% missing) achieved 99.6–100% fill rates with mean CI widths of ±0.63°C (temperature), ±3.47mm (precipitation), and ±0.31hPa (pressure).

# Acknowledgements

Built for the Earth Sciences community.

# References
