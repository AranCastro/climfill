# Changelog

## [1.1.0] - 2026-10-24

### Fixed
- Web app: results were lost and the app crashed (`'NoneType' object is not
  subscriptable`) on any widget interaction after filling, because the
  engine was rebuilt from the upload on every rerun.
- Web app: the outlier threshold slider had no effect; merged reanalysis was
  discarded on the next rerun; downloading twice duplicated columns.
- Date detection could pick a numeric column that preceded the date column.
- Records are sorted by date on load, so interpolation follows time order.
- Linear and spline interpolation filled the first 3/7 records of every gap,
  however long, and extrapolated trailing gaps; they now fill only interior
  gaps up to the stated length.
- Random Forest and XGBoost filled nothing inside multi-day gaps (lag
  features missing); they now refit without lag terms, then with calendar
  terms only.
- Reanalysis columns were filled as if they were station variables.
- Reanalysis merge dropped a station date column named `date`.
- IQR outlier screening flagged almost every rain day in zero-inflated
  precipitation; such variables are now screened on non-zero values.
- The S (suspicious) flag described in the documentation was never written.
- Stale uncertainty intervals survived a re-fill.
- Bootstrap resampling now uses a seeded generator (reproducible).
- Duplicate XGBoost import; broken README citation block and badges.

### Added
- Redesigned Streamlit app: tabbed workflow, persistent state, light/dark
  theme, availability heatmap, outlier and gap tables, station map,
  editable reanalysis matching, method breakdown, progress for uncertainty.
- `ClimFillEngine.validate()` hold-out accuracy assessment (n, MAE, RMSE,
  bias, R²), also in the app.
- Bias-corrected reanalysis stage in the `auto` cascade
  (`fill_gaps(reanalysis_map=...)`), used by the app and CLI.
- `ClimFillEngine.merge_reanalysis()`, `insert_missing_timestamps()`,
  `station_columns()`; `suggest_reanalysis_mapping()`.
- Readable Open-Meteo error messages (HTTP status and reason).
- CLI: `--reanalysis` now requires `--lat`/`--lon`.
- Streamlit app tests (AppTest); CI installs all extras on Python 3.11/3.13.

## [1.0.0] - 2025

### Added
- Core gap detection and filling engine
- 6 filling methods: linear, spline, seasonal, Random Forest, XGBoost, ERA5
- Automatic method selection
- Bootstrap uncertainty quantification
- Quality flags (O/F/S) with method tracking
- Open-Meteo reanalysis integration (free)
- Streamlit web application
- CLI interface
- Comprehensive test suite (34 tests)
- GitHub Actions CI
