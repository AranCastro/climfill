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

Missing observations in climate station time series are common due to sensor malfunction, power interruptions, and maintenance gaps. These discontinuities reduce the reliability of downstream analyses in hydrology, climatology, agriculture, and environmental modelling. *ClimFill* is an open-source Python package that automates detection, reconstruction, and quality-flagging of missing values in meteorological time series using statistical interpolation, machine learning, and optional reanalysis data integration with uncertainty estimation.

# Statement of Need

Climate time series underpin water resource assessments, drought analysis, crop modelling, and climate variability studies. However, many observational datasets contain non-random gaps that compromise statistical robustness and bias trend analysis. Existing gap-filling approaches range from simple interpolation to advanced imputation frameworks [@vanBuuren2011; @Stekhoven2012], yet these often require custom scripting or lack integrated uncertainty assessment. Additionally, bias-corrected reanalysis integration workflows are typically implemented manually.

*ClimFill* provides an integrated, reproducible, and automated framework accessible through a command-line interface, Python API, and web interface. The package is intended for researchers and practitioners who require transparent, method-aware, and quality-controlled reconstruction of climate station datasets.

# Functionality

ClimFill implements a hierarchical filling strategy:

1. **Gap Detection**  
   Automatically detects missing values including common sentinel codes (e.g., -999, -9999), quantifies gap lengths, evaluates temporal regularity, and identifies statistical outliers.

2. **Adaptive Gap-Filling Methods**  
   - Short gaps: linear interpolation  
   - Medium gaps: cubic spline interpolation  
   - Seasonal gaps: decomposition-based reconstruction  
   - Long gaps: Random Forest or XGBoost regression using correlated variables and cyclical temporal predictors  

3. **Reanalysis Integration**  
   Optional retrieval of gridded reanalysis data (e.g., ERA5 [@Hersbach2020]) via Open-Meteo API with quantile mapping bias correction [@Maraun2016].

4. **Uncertainty Quantification**  
   Bootstrap resampling provides confidence intervals for reconstructed values.

5. **Quality Flagging**  
   Each observation is labelled as Original, Filled, or Suspicious, with associated method metadata.

# Quality Control

The package includes 60+ automated unit tests executed through continuous integration (GitHub Actions). Test coverage exceeds 90%, ensuring robustness across edge cases including irregular temporal spacing, short time series, and machine learning fallback scenarios.

# Availability and Installation

Source code is available at:  
https://github.com/AranCastro/climfill  

Archived release (v1.0.1):  
https://doi.org/10.5281/zenodo.18638725  

Installation:

```bash
pip install -e .
