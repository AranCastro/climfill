# Changelog

## [1.1.0] - unreleased

### Changed (may alter existing results)
- Uncertainty: `compute_uncertainty()` now derives 95 % intervals from the
  hold-out residuals of the method that produced each fill. The earlier
  bootstrap of Random Forest means contained the withheld true value for
  15–34 % of filled values on the bundled sample; the new intervals contain
  it for 91–98 %. `n_bootstrap`, `GapFiller.bootstrap_uncertainty()` and the
  CLI option `--bootstrap-n` are deprecated and have no effect.
- `confidence` is now the hold-out R² (0–1) of the method for that variable,
  in place of fixed constants and a cross-validated R² clamped to 0.3–0.95.
- Filled values are limited to physical bounds (`infer_bounds`, or
  `fill_gaps(bounds=...)`); precipitation can no longer be filled negative.
- Zero-inflated variables skip the spline in `auto` and use regression
  before climatology; fills below the smallest positive observation are
  set to zero.
- `auto` skips an interpolation stage for a variable when its hold-out error
  exceeds that of the seasonal stage.
- Seasonal method: annual-cycle climatology (mean and two harmonics) plus an
  anomaly damped away from the gap edges, in place of calendar-month means
  plus a linearly interpolated anomaly. It also fills leading and trailing
  gaps.
- `load_csv` inserts rows for absent timestamps of daily or sub-daily series
  (`regularise=False` disables this) and returns `rows_inserted`.
- Default missing-value codes are -999, -9999 and -99.9. The codes 9999 and
  99999 are no longer converted, because they are valid readings of some
  variables; pass `sentinels=[...]` or `--sentinels` where a file uses them.
- Open-Meteo requests now ask for `models=era5`
  (`get_open_meteo_data(model=None)` restores the provider default).
- CLI: a file without gaps is still written to the output path.
- Requires pandas 2.0 or later (the date parser already depended on it).

### Fixed
- CLI overwrote the input file when its name did not contain `.csv` and no
  `--output` was given; the default output name is now built from the file
  stem, and an output path equal to the input is refused.
- `fill_gaps(**kwargs)` discarded its keyword arguments; `max_gap`, `order`
  and `n_estimators` now take effect and any other keyword raises
  `TypeError`.
- Importing the package installed a global `warnings.filterwarnings("ignore")`.
- Deprecated pandas accessors (`dt.dayofyear`, `dt.dayofweek`).
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
- Accuracy tests against the withheld truth of the bundled sample
  (`tests/test_accuracy.py`): error against monthly climatology, physical
  limits, interval coverage.
- `constraints-tested.txt`; lint step and read-only token in CI.

## [1.0.1] - 2026-02-14

Zenodo DOI release (https://doi.org/10.5281/zenodo.18638725). The tag points
to the same commit as 1.0.0 and the code identifies itself as 1.0.0.

## [1.0.0] - 2026-02-14

### Added
- Core gap detection and filling engine
- 6 filling methods: linear, spline, seasonal, Random Forest, XGBoost, ERA5
- Automatic method selection
- Bootstrap uncertainty quantification
- Quality flags (O/F/S) with method tracking
- Open-Meteo reanalysis integration (free)
- Streamlit web application
- CLI interface
- Test suite (62 tests)
- GitHub Actions CI
