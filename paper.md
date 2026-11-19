---
title: "ClimFill: An Open-Source Tool for Automated Gap-Filling in Climate Station Time Series"
tags:
  - Python
  - climate science
  - gap-filling
  - time-series
  - machine-learning
  - reanalysis
authors:
  - name: Aran Castro
    orcid: 0000-0001-8038-606X
    affiliation: 1
affiliations:
  - name: Geospatial Campus, Nagercoil, India
    index: 1
date: 2026-02-14
bibliography: paper.bib
---

# Summary

Missing observations in climate station time series arise from sensor malfunction, power interruption, and maintenance outages, and manned station records carry further data-quality problems [@Hunziker2017]. Such gaps reduce the reliability of downstream analyses in hydrology, climatology, agriculture, and environmental modelling. *ClimFill* is an open-source Python package that detects, reconstructs, and flags missing values in daily meteorological station records using interpolation, climatology, regression, and optional bias-corrected reanalysis data. Each filled value is exported with the method that produced it, a hold-out skill score, and a 95% interval.

# Statement of Need

Station time series support water resource assessment, drought analysis, crop modelling, and studies of climate variability. Many observational records contain non-random gaps that bias trend estimates and reduce the sample available for statistical analysis. General-purpose imputation frameworks such as *mice* [@vanBuuren2011] and *missForest* [@Stekhoven2012] do not use temporal structure unless the user constructs lagged predictors, so their application to station series requires custom scripting. Bias correction of reanalysis data for infilling is usually implemented by hand for each study.

*ClimFill* combines these steps in one reproducible workflow that is accessible through a command-line interface, a Python API, and a web interface. The package is intended for researchers and practitioners who need a documented, method-aware reconstruction of single-station records.

# State of the Field

CLIMFILL v0.9 [@Bessenbacher2022] is a multivariate framework for gap filling gridded Earth observations; it combines spatial interpolation with a multivariate statistical step that accounts for the dependence between variables. Despite the similar name, the present package is independent of that framework and addresses a different data type: a single station record with no spatial neighbours. *mice* and *missForest* provide multivariate imputation without temporal predictors. The contribution of *ClimFill* is the combination, for station series, of gap-length-aware method selection, physical limits on filled values, per-value provenance, and intervals calibrated on withheld observations.

# Functionality

1. **Gap detection.** Missing-value codes (by default -999, -9999, and -99.9; configurable) are converted to missing, records are sorted by date, and absent timestamps of daily or sub-daily series are inserted as gaps. Gap lengths, temporal regularity, and statistical outliers are reported; zero-inflated variables are screened on their non-zero values.

2. **Gap-filling methods.**
   - Linear interpolation for interior gaps of up to three records
   - Spline interpolation for interior gaps of up to seven records
   - Bias-corrected reanalysis, where a matching reanalysis series is supplied
   - Annual-cycle climatology (mean and two harmonics) plus the anomaly at the gap edges, damped towards climatology inside long gaps
   - Random Forest or XGBoost regression on the other variables, lagged values, and cyclical calendar terms, refitted without lagged values inside multi-day gaps

   In the automatic mode the stages are applied in this order to each variable. An interpolation stage is skipped when its error on withheld observations exceeds that of the climatology stage. Zero-inflated variables such as daily precipitation are not splined, and regression precedes climatology for them.

3. **Physical limits.** Filled values are limited to bounds inferred from the variable name (for example, precipitation at or above zero and relative humidity within 0–100%), which the user may override.

4. **Reanalysis integration.** ERA5 daily series [@Hersbach2020] are retrieved from the Open-Meteo archive API for the station coordinates (decimal degrees, WGS 84, EPSG:4326) and bias-corrected by empirical quantile mapping [@Maraun2016] against the overlapping station observations.

5. **Skill and uncertainty.** While filling, each stage hides a sample of observed values as gaps of the lengths it is about to fill and reconstructs them. The R² of this hold-out is recorded as the confidence of the fill. The 95% interval of a filled value is the value minus the 97.5th and 2.5th percentiles of the hold-out residuals of the stage that produced it. A separate routine reports hold-out mean absolute error, root-mean-square error, bias, and R² for any method and gap length.

6. **Quality flags.** Each value is labelled Original, Filled, or Suspicious (statistical outlier), with the method recorded for every filled value.

# Quality Control

The package includes 110 automated tests executed through continuous integration (GitHub Actions) on Python 3.11 and 3.13, with line coverage of 96%. A subset of the tests compares filled values with withheld truth: the bundled sample series (1,826 daily records, eight variables) is generated with a fixed seed, so the complete series behind every gap is known. On this sample the automatic mode filled all 997 missing values, its root-mean-square error did not exceed 1.15 times that of calendar-month climatology for any variable, and the 95% intervals contained the true value for 91–98% of filled values per variable. These figures derive from one synthetic realisation and have not been established on observed station data.

# Limitations

Consistency between variables (for example, minimum temperature not exceeding maximum temperature) is not enforced. Reconstruction of daily precipitation from a single station has low skill, which the recorded confidence reflects. Bounds are inferred from column names and should be checked for variables with unconventional names or units.

# Availability and Installation

Source code is available at:
https://github.com/AranCastro/climfill

Archived release (v1.0.1):
https://doi.org/10.5281/zenodo.18638725

The functionality described here corresponds to version 1.1.0, which postdates the archived release.

Installation:

```bash
pip install -e .
```

# References
