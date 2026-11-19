# Technical Audit of ClimFill (branch `main`, commit `8f07375`)

**Date of audit:** 19 November 2026
**Repository:** AranCastro/climfill
**Scope:** `climfill/` package, `streamlit_app.py`, command-line interface (CLI), tests, continuous integration (CI), packaging metadata, `README.md`, `paper.md`, `paper.bib`, `CITATION.cff`, Git tags and releases.

---

## 1. Summary

The test suite passes (62 of 62 tests, 98% line coverage), yet the tests assert only that the count of missing values decreases; none compares a filled value with a withheld observation. Execution against the bundled sample data, whose gap-free truth can be regenerated from `generate_sample_data.py` (seed 42), exposed six defects that affect the scientific validity of the output or cause data loss, and eleven further defects of high or medium severity. The published description of the method (README, `paper.md`) does not match the behaviour of the code in five places.

An open pull request dated 24 October 2026, [AranCastro/climfill#1](https://github.com/AranCastro/climfill/pull/1), already corrects seven of the findings below and partly corrects four. It has not been merged, so `main`, both release tags and the Zenodo archive still carry every defect. The same checks were re-run on that branch; the column "PR #1" in each table records the result.

No change was made to the repository during this audit.

## 2. Method

1. All 30 tracked files were read in full.
2. The test suite was executed in a clean virtual environment: Python 3.13.18, pandas 3.1.1, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1, pytest 9.2.0.
3. A verification script exercised each suspected defect. Accuracy was measured against the gap-free series regenerated from the sample generator (five years of daily data, 1,826 records, eight variables). The figures come from one synthetic realisation; they indicate the direction and size of an error and carry no significance test.
4. The web application was driven headlessly with `streamlit.testing.v1.AppTest` (Streamlit 1.67.0, Plotly 7.1.0).
5. The five entries of `paper.bib` and one missing reference were checked against the Crossref REST API by DOI.
6. Releases, tags, issues, pull requests and workflow runs were read from GitHub.

## 3. Findings

### 3.1 Critical: incorrect scientific output or loss of data

**Table 1.** Critical findings, with evidence from execution on the bundled sample data.

| ID | Location | Finding | Evidence | PR #1 |
|----|----------|---------|----------|-------|
| C1 | `engine.py:290`, `:307`, `:338` | `Series.interpolate(limit=n)` fills the first *n* records of **every** gap, whatever its length. In `auto` mode a single long gap is therefore filled by three methods in sequence, and the README statement "selecting the best method per gap" does not hold. Trailing gaps are extrapolated. | The longest gap in `Temperature_Mean` (31 days) was filled as linear (3), spline (7), seasonal (21). A ten-day trailing gap on a series ranging 15–25 received three repeats of the last value followed by spline extrapolation rising to 33.2. | Fixed |
| C2 | `engine.py:374–382`, `:443–448` | Random Forest and XGBoost require lag and lead values (±1, 2, 3, 7 days) of the target itself to be present. A missing value inside any gap longer than one day lacks them and is skipped. The methods described as suited to "long gaps" can fill only isolated single values. In `auto` mode the machine-learning stage contributed no fill at all, because the seasonal stage has no length limit. | `method="random_forest"` filled 35 of 116 missing `Temperature_Mean` values and between 25 and 36 values for each of the other seven variables. `auto` recorded only linear, spline and seasonal fills. | Fixed |
| C3 | `engine.py:558–606` | The "95% confidence interval" is the spread of bootstrap Random Forest means. It is computed from a model that did not produce the fill, uses already-filled values as lag predictors, and omits residual variance. It is neither an interval for the filled value nor a prediction interval. `np.random.choice` is unseeded, so results differ between runs. | Across the eight variables the true value lay inside the nominal 95% interval for 15–34% of filled cells; the exported filled value itself lay outside its own interval for 11–81% of cells. Two consecutive calls returned different intervals. | Open (truth inside interval for 6–31% of cells) |
| C4 | `engine.py:298–347` | No physical bounds are applied. Spline and anomaly interpolation are applied to precipitation, a zero-inflated, non-negative variable. | 20 of 104 filled precipitation values were negative, with a minimum of −56.0 mm. Root-mean-square error (RMSE) of the spline fills was 34.3 mm (n = 13) against 11.9 mm for a monthly-mean baseline; for the seasonal fills 21.4 mm (n = 34) against 4.1 mm. Dry days formed 71% of observations and 51% of fills. | Partly (29 of 104 negative, minimum −1.3 mm) |
| C5 | `streamlit_app.py:144–147` | The engine is rebuilt from the upload on every Streamlit rerun, while `session_state.filled` stays `True`. Any widget interaction after filling raises an exception, and reanalysis merged in Step 3 is discarded before Step 4. The export step cannot be completed in normal use. | After "Fill All Gaps", selecting another variable under "Compare Variable" raised `TypeError: 'NoneType' object is not subscriptable` at line 333. | Rewritten; its own AppTest suite passes |
| C6 | `cli.py:106` | The default output path is `input.replace(".csv", "_filled.csv")`. When the input name does not contain `.csv` (for example `.txt`, `.CSV`, `.dat`), the output path equals the input path and the original observations are overwritten without warning. | `climfill station.txt --method linear` left one file in the directory; its checksum changed and it held 33 columns in place of nine. | Open |

### 3.2 High: silent misbehaviour on common inputs

**Table 2.** High-severity findings.

| ID | Location | Finding | Evidence | PR #1 |
|----|----------|---------|----------|-------|
| H1 | `engine.py:43–51` | Date detection tries `pd.to_datetime` on every column in order. A numeric column placed before the date column is accepted as nanoseconds since 1970. The 80% threshold on line 46 is unreachable because parsing raises on the first failure. | With columns ordered `Temperature_Mean, Date, …` the detected date column was `Temperature_Mean`, range "1969-12-31 … 1970-01-01". A leading `Station_ID` column gave the same failure. | Fixed |
| H2 | `engine.py:622–643` | Gaps stored as absent rows are not detected, and the series is never reindexed to a regular calendar. Interpolation operates on row position, so unsorted input is filled incorrectly. | Deleting 30 consecutive dates left the reported missing count unchanged and the output 30 rows short. On a shuffled ten-day linear series the value for 5 January (true 4.0) was filled as 5.0. | Sorting fixed; reindexing available only through an explicit call |
| H3 | `cli.py:75–94`, `streamlit_app.py:261–288` | Reanalysis data are downloaded and merged but never used. `era5_bias_correction` is called from neither interface, and "ERA5 Bias-Corrected" in the README method table cannot be selected. The merged `reanalysis_*` columns become fill targets and are written to the CLI output. `--reanalysis` without `--lat`/`--lon` is ignored without a message. | Station fills were identical with and without merged reanalysis columns. | Addressed through `merge_reanalysis`; not independently verified |
| H4 | `engine.py:252–280` | The quality flag "S" (suspicious) is never written; outlier detection is reported on screen only. README and `paper.md` both state that three flags are produced. | Flags present after a full run: `F`, `O`. | Fixed (written on export) |
| H5 | `engine.py:82–88` | Interquartile-range screening on a zero-inflated variable has a near-zero range, so almost every wet day is reported as an outlier. | 409 of 432 wet days in `Precipitation` were flagged (Q1 = 0.0, Q3 = 0.1). | Fixed (four flagged) |
| H6 | `engine.py:401–410`, constants at `:295`, `:318`, `:345`, `:522` | The `confidence` column mixes fixed constants (0.70, 0.65, 0.75, 0.80) with a cross-validated R² that is clamped to [0.3, 0.95] and estimated with unshuffled K-fold on autocorrelated data containing lag predictors. A model with negative R² is reported at 0.3. The column has no stated definition and is exported as metadata. | Read from source. | Open |

### 3.3 Medium

**Table 3.** Medium-severity findings.

| ID | Location | Finding | PR #1 |
|----|----------|---------|-------|
| M1 | `engine.py:30` | The sentinel list contains 9999 and 99999 and is applied to every numeric column. A pressure of 99,999 Pa and a radiation total of 9,999 were both converted to missing in a test. The list is not configurable. | Open |
| M2 | `engine.py:18`, `reanalysis.py:10` | `warnings.filterwarnings("ignore")` runs at import and suppresses all warnings in the calling process. The pandas deprecation of `dt.dayofyear` and `dt.dayofweek` (lines 241–249) is hidden by it. | Open |
| M3 | `engine.py:660` | `fill_gaps(**kwargs)` discards its keyword arguments; `fill_gaps("linear", max_gap=1)` returned the same result as the default. | Open |
| M4 | `engine.py:322–347` | The method named "seasonal decomposition" is a monthly mean plus a linearly interpolated anomaly. It introduces a step at month boundaries and has no gap-length limit. On the sample data its RMSE exceeded the monthly-mean baseline for six of eight variables (for example `Temperature_Max`: 2.61 against 1.73, n = 38). | Open (`Temperature_Min`: 2.27 against 1.57, n = 94) |
| M5 | `pyproject.toml:22` | `pandas>=1.4.0` is declared, yet `format="mixed"` (line 45 of `engine.py`) requires pandas 2.0 or later. On pandas 1.x every file fails with "No date column detected". | Open |
| M6 | `streamlit_app.py:92` | The outlier threshold slider is not passed to the engine. Every `use_container_width` call (eight sites) logs "will be removed after 2025-12-31" under Streamlit 1.67.0. | Fixed |
| M7 | `cli.py:70–72` | When no gap is found the CLI exits before writing the output or the report that the user requested. | Open |
| M8 | `tests/` | Tests assert that the missing count falls and that no exception is raised. No test checks a filled value, a bound, or interval coverage, which is why C1–C4 passed CI. The CI job installs neither XGBoost nor Streamlit and runs one Python version. | CI extended; a `validate()` hold-out routine added |

### 3.4 Release, metadata and documentation

**Table 4.** Findings on release integrity and documentation.

| ID | Finding | PR #1 |
|----|---------|-------|
| R1 | Tags `v1.0.0` and `v1.0.1` point to the same commit (`0a18352`). `pyproject.toml` and `__version__` read 1.0.0 there, so the archive cited as v1.0.1 identifies itself as 1.0.0. `CHANGELOG.md` has no 1.0.1 entry, dates 1.0.0 as "2025" and reports 34 tests. | Version raised to 1.1.0; `CITATION.cff` still 1.0.1 |
| R2 | The name coincides with an existing, published gap-filling framework: CLIMFILL v0.9 (Bessenbacher et al., 2022). Neither `README.md` nor `paper.md` mentions it. A reviewer of a software paper will expect it under the state of the field, and users may confuse the two. | Open |
| R3 | `README.md`: badge and clone URLs use the placeholder `yourusername`; the CI badge points to `ci.yml`, whereas the workflow file is `tests.yml`; the test and coverage badges are static text; the `bibtex` fence in the Citation section is never closed and holds a Markdown badge in place of a BibTeX entry, so the Contributing and License sections render as code; `CONTRIBUTING.md` is linked and absent. | Fixed |
| R4 | `README.md` instructs `pip install climfill`. Publication of this package on PyPI was not confirmed (Section 5). Given R2, the name may belong to another project. | Open |
| R5 | `paper.md` ends inside an unclosed code fence, has no References heading and no state-of-the-field comparison with existing gap-filling software. Its functional claims C1, C2, H3 and H4 do not match the code. `Hunziker2017` is present in `paper.bib` and cited nowhere. | Open |
| R6 | `pyproject.toml` URLs use `yourusername`; the author field reads "Aran" in `pyproject.toml`, `__init__.py` and `LICENSE`, and "Aran Castro" in `CITATION.cff` and `paper.md`. | Fixed in `pyproject.toml` |
| R7 | Eight compiled `__pycache__/*.pyc` files are tracked and there is no `.gitignore`. `requirements.txt` carries lower bounds only, so the environment is not reproducible. `engine.py` imports XGBoost twice (lines 10–15 and 20–24); the linter reports six unused imports and two unused variables. | `.pyc` files removed, `.gitignore` added |

### 3.5 Low

- `.devcontainer/devcontainer.json` starts Streamlit with CORS and XSRF protection disabled. This is the stock Codespaces template and is acceptable for a private development container only.
- `tests.yml` triggers on both `push` and `pull_request`, which doubles each run on a pull-request branch; it declares no `permissions` block.
- `reanalysis.py` labels Open-Meteo archive output as ERA5. The request sets no `models` parameter, so the provider's default model selection applies; the data source written into the paper should name the model explicitly.
- `ERA5_VARIABLE_MAP` (`reanalysis.py:80–92`) is unused, and the module docstring refers to a CDS API path that is not implemented.
- `_record_fill` writes metadata in a per-row Python loop (`engine.py:277–280`).

## 4. Items checked and found sound

- No credentials, tokens or personal e-mail addresses are present in the tracked files or commit metadata.
- The only outbound request goes to a fixed Open-Meteo URL with a 60-second timeout and `raise_for_status`; user input reaches it only as numeric query parameters.
- `sample_station_data.csv` is reproduced exactly by `generate_sample_data.py` with seed 42.
- Integer columns containing sentinels are converted without error under pandas 3.1.1.
- The latest workflow run on `main` (14 February 2026) succeeded; there are no open issues.
- All five entries in `paper.bib` resolve at Crossref with matching journal, volume, issue, pages and year. Two cosmetic differences exist: the registered title of Maraun (2016) carries the subtitle "a Critical Review", and Stekhoven and Bühlmann is registered with an online year of 2011 against the print year 2012 given in the file.

## 5. Not verified

- Availability of the package on PyPI and resolution of the Zenodo DOI. Requests that would place the project name or DOI in an outbound query were not made.
- A live Open-Meteo download; only the mocked path and the merge logic were exercised.
- Behaviour on observed station data. All accuracy figures derive from the synthetic sample.
- The rewritten web application on the PR #1 branch was assessed through its own tests only.

## 6. Recommended order of work

1. Review and merge PR #1, which resolves C1, C2, C5, H1, H4, H5, M6 and R3.
2. Correct C6 before any further CLI use on files not named `*.csv`.
3. Replace the uncertainty routine (C3) with an interval derived from hold-out residuals of the method that produced each fill, and report its empirical coverage.
4. Add per-variable physical bounds and a separate treatment for precipitation (C4); restrict spline and anomaly interpolation to continuous variables.
5. Add hold-out accuracy tests against the regenerated sample truth so that CI fails when a method performs worse than monthly climatology (M4, M8).
6. Resolve the naming overlap (R2) and the version and tag inconsistency (R1), then issue a new tagged release and Zenodo version before `paper.md` is submitted.
7. Revise `paper.md` so that each functional claim matches the released code (R5).

## References

Bessenbacher, V., Seneviratne, S. I., & Gudmundsson, L. (2022). CLIMFILL v0.9: A framework for intelligently gap filling Earth observations. *Geoscientific Model Development, 15*(11), 4569–4596. https://doi.org/10.5194/gmd-15-4569-2022

Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society, 146*(730), 1999–2049. https://doi.org/10.1002/qj.3803

Maraun, D. (2016). Bias correcting climate change simulations: A critical review. *Current Climate Change Reports, 2*(4), 211–220. https://doi.org/10.1007/s40641-016-0050-x

Stekhoven, D. J., & Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics, 28*(1), 112–118. https://doi.org/10.1093/bioinformatics/btr597
