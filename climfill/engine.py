"""
ClimFill Engine — Core gap detection and filling logic.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import warnings

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None
    HAS_XGBOOST = False

REANALYSIS_PREFIX = "reanalysis_"


def _is_reanalysis(col):
    return str(col).startswith(REANALYSIS_PREFIX)


class GapDetector:
    """Detects and characterizes gaps in climate time series."""

    # Codes that cannot be a valid reading of any common station variable.
    # Positive codes such as 9999 or 99999 are omitted because they are
    # legitimate values of some variables (e.g. pressure in Pa); pass them
    # through ``sentinels`` when a file uses them.
    MISSING_SENTINELS = [-999, -9999, -99.9]

    def __init__(self, df, date_col=None, sentinels=None):
        self.df = df.copy()
        self.date_col = date_col
        self.sentinels = list(
            self.MISSING_SENTINELS if sentinels is None else sentinels
        )
        self.gap_report = {}

    def detect_date_column(self):
        """Auto-detect the datetime column."""
        if self.date_col and self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            return self.date_col

        # Numeric columns parse as epoch offsets, so try text columns first
        candidates = [
            c for c in self.df.columns
            if not pd.api.types.is_numeric_dtype(self.df[c])
        ] or list(self.df.columns)
        for col in candidates:
            try:
                parsed = pd.to_datetime(self.df[col], format="mixed")
                if parsed.notna().sum() > len(self.df) * 0.8:
                    self.df[col] = parsed
                    self.date_col = col
                    return col
            except (ValueError, TypeError):
                continue

        raise ValueError(
            "No date column detected. Please specify date_col parameter."
        )

    def replace_sentinels(self):
        """Replace the missing-value codes in ``self.sentinels`` with NaN."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        replaced = {}
        for col in numeric_cols:
            count = 0
            for sentinel in self.sentinels:
                mask = self.df[col] == sentinel
                count += mask.sum()
                self.df.loc[mask, col] = np.nan
            if count > 0:
                replaced[col] = int(count)
        return replaced

    def detect_outliers(self, columns=None, method="iqr", threshold=3.0):
        """Flag statistical outliers."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}
        for col in columns:
            series = self.df[col].dropna()
            # Zero-inflated variables (e.g. daily precipitation) are screened
            # on their non-zero values; fences on all values would flag
            # nearly every wet day
            if (series == 0).mean() >= 0.25:
                series = series[series != 0]
            if len(series) < 10:
                continue

            mask = pd.Series(False, index=self.df.index)
            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask.loc[series.index] = (series < lower) | (series > upper)
            elif method == "zscore":
                if series.std() == 0:
                    continue
                z = np.abs(stats.zscore(series))
                mask.loc[series.index] = z > threshold
            else:
                continue

            outlier_idx = self.df.index[mask & self.df[col].notna()].tolist()
            if len(outlier_idx) > 0:
                outliers[col] = {
                    "count": len(outlier_idx),
                    "indices": outlier_idx,
                    "values": self.df.loc[outlier_idx, col].tolist(),
                }
        return outliers

    def analyze_gaps(self, columns=None):
        """Comprehensive gap analysis for each variable."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        report = {}
        for col in columns:
            missing_mask = self.df[col].isna()
            total = len(self.df)
            n_missing = int(missing_mask.sum())

            if n_missing == 0:
                report[col] = {
                    "total_records": total,
                    "missing_count": 0,
                    "missing_pct": 0.0,
                    "n_gaps": 0,
                    "gaps": [],
                    "max_gap_length": 0,
                    "mean_gap_length": 0,
                    "gap_length_distribution": {
                        "1": 0, "2-5": 0, "6-15": 0, "16-30": 0, ">30": 0
                    },
                }
                continue

            # Find contiguous gap segments
            gaps = []
            in_gap = False
            gap_start = None
            for i in range(total):
                if missing_mask.iloc[i]:
                    if not in_gap:
                        gap_start = i
                        in_gap = True
                else:
                    if in_gap:
                        gap_end = i - 1
                        gap_len = i - gap_start
                        gap_info = {
                            "start_idx": gap_start,
                            "end_idx": gap_end,
                            "length": gap_len,
                        }
                        if self.date_col and self.date_col in self.df.columns:
                            gap_info["start_date"] = str(
                                self.df[self.date_col].iloc[gap_start]
                            )
                            gap_info["end_date"] = str(
                                self.df[self.date_col].iloc[gap_end]
                            )
                        gaps.append(gap_info)
                        in_gap = False

            # Handle gap at end of series
            if in_gap:
                gap_end = total - 1
                gap_len = total - gap_start
                gap_info = {
                    "start_idx": gap_start,
                    "end_idx": gap_end,
                    "length": gap_len,
                }
                if self.date_col and self.date_col in self.df.columns:
                    gap_info["start_date"] = str(
                        self.df[self.date_col].iloc[gap_start]
                    )
                    gap_info["end_date"] = str(
                        self.df[self.date_col].iloc[gap_end]
                    )
                gaps.append(gap_info)

            gap_lengths = [g["length"] for g in gaps]
            report[col] = {
                "total_records": total,
                "missing_count": n_missing,
                "missing_pct": round(n_missing / total * 100, 2),
                "n_gaps": len(gaps),
                "gaps": gaps,
                "max_gap_length": max(gap_lengths) if gap_lengths else 0,
                "mean_gap_length": (
                    round(float(np.mean(gap_lengths)), 1) if gap_lengths else 0
                ),
                "gap_length_distribution": {
                    "1": sum(1 for g in gap_lengths if g == 1),
                    "2-5": sum(1 for g in gap_lengths if 2 <= g <= 5),
                    "6-15": sum(1 for g in gap_lengths if 6 <= g <= 15),
                    "16-30": sum(1 for g in gap_lengths if 16 <= g <= 30),
                    ">30": sum(1 for g in gap_lengths if g > 30),
                },
            }

        self.gap_report = report
        return report

    def check_temporal_regularity(self):
        """Check if time series has regular intervals."""
        if self.date_col is None or self.date_col not in self.df.columns:
            return None

        dates = self.df[self.date_col].dropna().sort_values()
        diffs = dates.diff().dropna()
        if len(diffs) == 0:
            return None

        mode_diff = diffs.mode().iloc[0]
        irregular = int((diffs != mode_diff).sum())

        return {
            "expected_interval": str(mode_diff),
            "total_intervals": len(diffs),
            "irregular_intervals": irregular,
            "regularity_pct": round((1 - irregular / len(diffs)) * 100, 2),
        }


# Bounds inferred from the variable name: (keywords, lower, upper).
_BOUND_RULES = [
    (("precip", "rain", "prcp", "ppt", "snow"), 0.0, None),
    (("humid",), 0.0, 100.0),
    (("wind",), 0.0, None),
    (("solar", "radiation", "srad", "sunshine"), 0.0, None),
    (("press",), 0.0, None),
    (("evap", "et0"), 0.0, None),
]

MIN_RESIDUALS = 10


def infer_bounds(col):
    """Physical ``(lower, upper)`` limits suggested by a variable name.

    ``None`` means unbounded on that side. Names that match no rule return
    ``(None, None)``.
    """
    name = "".join(ch for ch in str(col).lower() if ch.isalnum())
    if name == "rh":
        return (0.0, 100.0)
    for keywords, lower, upper in _BOUND_RULES:
        if any(k in name for k in keywords):
            return (lower, upper)
    return (None, None)


def _is_zero_inflated(series):
    """Non-negative with at least a quarter of observed values equal to zero."""
    obs = series.dropna()
    return len(obs) > 0 and obs.min() >= 0 and (obs == 0).mean() >= 0.25


def _r2(truth, est):
    """Coefficient of determination clamped to [0, 1]; NaN when undefined."""
    truth, est = np.asarray(truth, float), np.asarray(est, float)
    if len(truth) < 2:
        return np.nan
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot == 0:
        return np.nan
    r2 = 1 - float(np.sum((est - truth) ** 2)) / ss_tot
    return float(min(1.0, max(0.0, r2)))


class GapFiller:
    """Multi-method gap filling with hold-out skill and interval estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Station data in chronological order.
    date_col : str
        Name of the datetime column.
    bounds : dict, optional
        ``{column: (lower, upper)}`` physical limits applied to every filled
        value; ``None`` leaves a side open. Columns not listed use
        :func:`infer_bounds`, and zero-inflated non-negative variables are
        bounded below by zero.
    random_state : int
        Seed of the generator that places hold-out gaps.

    Notes
    -----
    Every stage hides a sample of observed values, reconstructs them with the
    same method and keeps the residuals (filled minus observed). The
    ``confidence`` recorded for a fill is the R² of that hold-out, clamped to
    [0, 1] (NaN when fewer than ``MIN_RESIDUALS`` residuals are available),
    and :meth:`residual_intervals` turns the same residuals into intervals.
    """

    def __init__(self, df, date_col, bounds=None, random_state=42):
        self.df = df.copy()
        self.date_col = date_col
        self.filled_df = df.copy()
        self.fill_metadata = {}
        self.bounds = dict(bounds or {})
        self.random_state = random_state
        self.residual_pools = {}
        self._limits = {}

    def _get_numeric_cols(self):
        return [
            c
            for c in self.filled_df.select_dtypes(include=[np.number]).columns
            if c != self.date_col
        ]

    def _get_target_cols(self):
        """Station variables to fill; reanalysis columns are predictors only."""
        return [c for c in self._get_numeric_cols() if not _is_reanalysis(c)]

    def _has_dates(self):
        return bool(self.date_col) and self.date_col in self.filled_df.columns

    def _create_temporal_features(self, dates):
        """Create temporal features for ML models."""
        features = pd.DataFrame(index=dates.index)
        angle = 2 * np.pi * dates.dt.day_of_year / 365.25
        features["day_of_year_sin"] = np.sin(angle)
        features["day_of_year_cos"] = np.cos(angle)
        features["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
        features["year"] = dates.dt.year
        features["day_of_week"] = dates.dt.day_of_week
        return features

    # ── physical limits ─────────────────────────────────────────────────
    def limits(self, col):
        """``(lower, upper, wet_minimum)`` applied to fills of ``col``.

        ``wet_minimum`` is the smallest positive observation of a
        zero-inflated variable; smaller fills are set to zero.
        """
        if col not in self._limits:
            lower, upper = self.bounds.get(col, infer_bounds(col))
            wet_min = None
            observed = self.df[col].dropna()
            if _is_zero_inflated(observed):
                lower = 0.0 if lower is None else lower
                positive = observed[observed > 0]
                wet_min = float(positive.min()) if len(positive) else None
            self._limits[col] = (lower, upper, wet_min)
        return self._limits[col]

    def _constrain(self, col, values):
        lower, upper, wet_min = self.limits(col)
        values = pd.Series(values).clip(lower=lower, upper=upper)
        if wet_min is not None:
            values = values.mask(values < wet_min, 0.0)
        return values

    # ── metadata ────────────────────────────────────────────────────────
    def _init_metadata(self, col):
        """Initialize fill metadata for a column if not already done."""
        if col not in self.fill_metadata:
            self.fill_metadata[col] = {
                "method": pd.Series("original", index=self.df.index),
                "confidence": pd.Series(1.0, index=self.df.index),
                "flag": pd.Series("O", index=self.df.index),
                "pool": pd.Series("", index=self.df.index),
            }
            # Mark originally-missing values
            orig_missing = self.df[col].isna()
            self.fill_metadata[col]["flag"][orig_missing] = ""
            self.fill_metadata[col]["method"][orig_missing] = ""
            self.fill_metadata[col]["confidence"][orig_missing] = 0.0

    def _record_fill(self, col, mask, method, confidence, residuals=None,
                     pool=None):
        """Constrain filled values to the limits and record their metadata.

        ``residuals`` (filled minus observed, from a hold-out of the same
        method) are stored under ``pool`` for :meth:`residual_intervals`.
        """
        self._init_metadata(col)

        if isinstance(mask, pd.Series):
            indices = mask[mask].index
        elif isinstance(mask, np.ndarray):
            indices = self.filled_df.index[mask]
        else:
            return

        self.filled_df.loc[indices, col] = self._constrain(
            col, self.filled_df.loc[indices, col]
        )
        pool = pool or method
        meta = self.fill_metadata[col]
        meta["method"].loc[indices] = method
        meta["confidence"].loc[indices] = confidence
        meta["flag"].loc[indices] = "F"
        meta["pool"].loc[indices] = pool
        if residuals is not None and len(residuals) >= MIN_RESIDUALS:
            self.residual_pools.setdefault(col, {})[pool] = np.asarray(
                residuals, float
            )

    # ── hold-out skill ──────────────────────────────────────────────────
    @staticmethod
    def _run_lengths(mask):
        """Lengths of the contiguous True runs of a boolean Series."""
        if not mask.any():
            return np.array([], int)
        run_id = (mask != mask.shift()).cumsum()
        return mask[mask].groupby(run_id[mask]).size().to_numpy()

    def _holdout(self, col, fill_fn, lengths, frac=0.1, max_points=400):
        """Hide observed values as synthetic gaps and reconstruct them.

        Gap lengths follow ``lengths`` (those the stage is about to fill).
        Returns ``(residuals, r2)``; residuals are filled minus observed
        after the physical limits have been applied.
        """
        series = self.filled_df[col]
        observed = self.df[col].notna().to_numpy()
        n = len(series)
        lengths = np.asarray(lengths, int)
        if n < 5 or len(lengths) == 0 or observed.sum() < MIN_RESIDUALS:
            return None, np.nan
        reps = np.unique(
            np.round(np.quantile(lengths, [0, 0.5, 0.9, 1])).astype(int)
        )
        budget = min(max_points, int(observed.sum() * frac))
        rng = np.random.default_rng(self.random_state)
        taken = np.zeros(n, bool)
        for length in reps:
            if length + 2 > n:
                continue
            # A hidden gap needs an observed record on either side
            window = np.convolve(observed, np.ones(length + 2, int), "valid")
            starts = np.flatnonzero(window == length + 2) + 1
            rng.shuffle(starts)
            target = max(budget // len(reps), min(3 * length, budget))
            got = 0
            for st in starts:
                if taken[st - 1:st + length + 1].any():
                    continue
                taken[st:st + length] = True
                got += length
                if got >= target:
                    break
        if taken.sum() < MIN_RESIDUALS:
            return None, np.nan
        masked = series.copy()
        masked[taken] = np.nan
        est = self._constrain(col, fill_fn(masked)).to_numpy(float)[taken]
        truth = self.df[col].to_numpy(float)[taken]
        ok = ~np.isnan(est)
        if ok.sum() < MIN_RESIDUALS:
            return None, np.nan
        return est[ok] - truth[ok], _r2(truth[ok], est[ok])

    # ── interpolation ───────────────────────────────────────────────────
    @staticmethod
    def _bounded_gaps(series, max_gap):
        """Missing values in interior gaps of at most ``max_gap`` records.

        Longer gaps are left whole for later stages, and leading/trailing
        gaps are never extrapolated.
        """
        missing = series.isna()
        run_id = (missing != missing.shift()).cumsum()
        run_len = missing.groupby(run_id).transform("size")
        observed = series.notna()
        interior = observed.cummax() & observed[::-1].cummax()[::-1]
        return missing & (run_len <= max_gap) & interior

    @staticmethod
    def _linear_values(series):
        return series.interpolate(method="linear")

    @staticmethod
    def _spline_values(series, order=3):
        try:
            return series.interpolate(method="spline", order=order)
        except Exception:
            # Spline can fail on certain data; fall back to linear
            return series.interpolate(method="linear")

    def _interpolate(self, columns, max_gap, fill_fn, method,
                     only_if_better=False):
        """Fill interior gaps of up to ``max_gap`` records with ``fill_fn``.

        With ``only_if_better`` a variable is skipped when, on the same
        hidden observations, the interpolant has a larger root-mean-square
        error than climatology plus anomaly; its gaps then pass to the
        later stages. Day-to-day noise makes interpolation between two
        single readings the weaker estimate for some variables.
        """
        if columns is None:
            columns = self._get_target_cols()
        for col in columns:
            if col == self.date_col:
                continue
            mask = self._bounded_gaps(self.filled_df[col], max_gap)
            if not mask.any():
                continue
            lengths = self._run_lengths(mask)
            residuals, r2 = self._holdout(col, fill_fn, lengths)
            if only_if_better and residuals is not None and self._has_dates():
                reference, _ = self._holdout(col, self._seasonal_values, lengths)
                if reference is not None and (
                    np.mean(residuals ** 2) > np.mean(reference ** 2)
                ):
                    continue
            self.filled_df.loc[mask, col] = fill_fn(self.filled_df[col])[mask]
            self._record_fill(col, mask, method, r2, residuals)
        return self.filled_df

    def linear_interpolation(self, columns=None, max_gap=3,
                             only_if_better=False):
        """Linear interpolation for gaps of up to ``max_gap`` records."""
        return self._interpolate(
            columns, max_gap, self._linear_values, "linear_interpolation",
            only_if_better,
        )

    def spline_interpolation(self, columns=None, max_gap=7, order=3,
                             only_if_better=False):
        """Spline interpolation for gaps of up to ``max_gap`` records."""
        return self._interpolate(
            columns, max_gap,
            lambda s: self._spline_values(s, order), "spline_interpolation",
            only_if_better,
        )

    # ── climatology + anomaly ───────────────────────────────────────────
    def _climatology(self, series):
        """Expected value for each record from the annual cycle.

        With at least a year of data the cycle is a least-squares fit of the
        mean and the first two annual harmonics, which has no step at month
        boundaries. Shorter records use the mean of each calendar month.
        """
        dates = self.filled_df[self.date_col]
        observed = series.notna().to_numpy()
        if observed.sum() >= 60:
            span = (dates[observed].max() - dates[observed].min()).days
            if span >= 365:
                angle = (2 * np.pi * dates.dt.day_of_year / 365.25).to_numpy()
                design = np.column_stack([
                    np.ones(len(angle)), np.sin(angle), np.cos(angle),
                    np.sin(2 * angle), np.cos(2 * angle),
                ])
                coef, *_ = np.linalg.lstsq(
                    design[observed], series.to_numpy(float)[observed],
                    rcond=None,
                )
                return pd.Series(design @ coef, index=series.index)
        return series.groupby(dates.dt.month).transform("mean")

    def _seasonal_values(self, series, decay=7.0, window=7):
        """Climatology plus an anomaly that fades away from the gap edges.

        The anomaly at each gap edge is the mean departure from climatology
        of the adjacent observed records (``window``), interpolated across
        the gap and damped with an e-folding distance of ``decay`` records,
        so the interior of a long gap reverts to climatology.
        """
        observed = series.notna().to_numpy()
        if not observed.any():
            return pd.Series(np.nan, index=series.index)
        clim = self._climatology(series)
        anomaly = (series - clim).rolling(
            window, center=True, min_periods=1
        ).mean().where(series.notna())
        anomaly = anomaly.interpolate(method="linear", limit_direction="both")
        position = np.arange(len(series))
        obs_pos = position[observed]
        right = np.searchsorted(obs_pos, position).clip(0, len(obs_pos) - 1)
        left = (right - 1).clip(0)
        distance = np.minimum(
            np.abs(position - obs_pos[left]), np.abs(obs_pos[right] - position)
        )
        return clim + anomaly * np.exp(-distance / decay)

    def seasonal_interpolation(self, columns=None):
        """Annual-cycle climatology plus a damped anomaly, for any gap length."""
        if columns is None:
            columns = self._get_target_cols()
        if not self._has_dates():
            return self.filled_df
        for col in columns:
            if col == self.date_col:
                continue
            series = self.filled_df[col]
            missing_mask = series.isna()
            if not missing_mask.any():
                continue
            filled_values = self._seasonal_values(series)
            new_fills = missing_mask & filled_values.notna()
            if not new_fills.any():
                continue
            residuals, r2 = self._holdout(
                col, self._seasonal_values, self._run_lengths(missing_mask)
            )
            self.filled_df.loc[new_fills, col] = filled_values[new_fills]
            self._record_fill(
                col, new_fills, "seasonal_decomposition", r2, residuals
            )
        return self.filled_df

    # ── regression ──────────────────────────────────────────────────────
    def _ml_fill(self, target_col, predictor_cols, make_model, method):
        """Regression fill shared by the Random Forest and XGBoost methods.

        Pass 1 uses the values either side of a gap (lags/leads) as
        predictors. Inside multi-day gaps those neighbours are missing, so
        pass 2 refits without them, using the other variables and calendar
        terms. Pass 3 uses calendar terms alone, for records where the other
        variables are missing too (e.g. a full station outage).

        Confidence and residuals of each pass come from five-fold
        cross-validated predictions on contiguous blocks of the record.
        """
        if predictor_cols is None:
            predictor_cols = [
                c for c in self._get_numeric_cols() if c != target_col
            ]

        missing_mask = self.filled_df[target_col].isna()
        if not missing_mask.any():
            return self.filled_df
        observed = ~missing_mask

        base = pd.DataFrame(index=self.filled_df.index)
        for pc in predictor_cols:
            base[pc] = self.filled_df[pc]
        temp_feats = pd.DataFrame(index=self.filled_df.index)
        if self._has_dates():
            temp_feats = self._create_temporal_features(
                self.filled_df[self.date_col]
            )
            base = pd.concat([base, temp_feats], axis=1)

        lags = pd.DataFrame(index=self.filled_df.index)
        for lag in [1, 2, 3, 7]:
            lags[f"lag_{lag}"] = self.filled_df[target_col].shift(lag)
            lags[f"lead_{lag}"] = self.filled_df[target_col].shift(-lag)

        passes = (pd.concat([base, lags], axis=1), base, temp_feats)
        for number, features in enumerate(passes, start=1):
            if features.shape[1] == 0:
                continue
            complete = features.notna().all(axis=1)
            train_mask = observed & complete
            predict_mask = self.filled_df[target_col].isna() & complete

            if train_mask.sum() < 20 or predict_mask.sum() == 0:
                continue

            X_train = features.loc[train_mask]
            y_train = self.filled_df.loc[train_mask, target_col]
            X_predict = features.loc[predict_mask]

            model = make_model()
            model.fit(X_train, y_train)

            residuals, confidence = None, np.nan
            try:
                held = self._constrain(
                    target_col,
                    cross_val_predict(make_model(), X_train, y_train, cv=5),
                ).to_numpy(float)
                residuals = held - y_train.to_numpy(float)
                confidence = _r2(y_train, held)
            except Exception:
                pass

            self.filled_df.loc[predict_mask, target_col] = model.predict(X_predict)
            self._record_fill(
                target_col, predict_mask, method, confidence, residuals,
                pool=f"{method}:pass{number}",
            )
        return self.filled_df

    def random_forest_fill(self, target_col, predictor_cols=None, n_estimators=100):
        """Random Forest regression using correlated variables + temporal features."""
        return self._ml_fill(
            target_col,
            predictor_cols,
            lambda: RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
            "random_forest",
        )

    def xgboost_fill(self, target_col, predictor_cols=None):
        """XGBoost regression with temporal features."""
        if not HAS_XGBOOST:
            return self.random_forest_fill(target_col, predictor_cols)
        return self._ml_fill(
            target_col,
            predictor_cols,
            lambda: XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            ),
            "xgboost",
        )

    # ── reanalysis ──────────────────────────────────────────────────────
    def era5_bias_correction(self, target_col, era5_col):
        """Fill gaps using reanalysis with quantile mapping bias correction.

        Confidence and residuals are taken from the corrected series over
        the records where the station also reported (in-sample).
        """
        missing_mask = self.filled_df[target_col].isna()
        if not missing_mask.any() or era5_col not in self.filled_df.columns:
            return self.filled_df

        overlap = self.df[target_col].notna() & self.filled_df[era5_col].notna()

        if overlap.sum() < 10:
            return self.filled_df

        obs_vals = self.filled_df.loc[overlap, target_col]
        era5_vals = self.filled_df.loc[overlap, era5_col]

        if overlap.sum() < 30:
            # Mean and variance scaling
            scale = float(obs_vals.std() / era5_vals.std()) if era5_vals.std() > 0 else 1.0
            corrected = (self.filled_df[era5_col] - era5_vals.mean()) * scale + obs_vals.mean()
        else:
            # Quantile mapping
            percentiles = np.linspace(0, 100, min(100, int(overlap.sum())))
            obs_q = np.percentile(obs_vals, percentiles)
            era5_q = np.percentile(era5_vals, percentiles)
            corrected = pd.Series(
                np.interp(self.filled_df[era5_col], era5_q, obs_q),
                index=self.filled_df.index,
            )

        fill_mask = missing_mask & corrected.notna()
        if fill_mask.any():
            held = self._constrain(target_col, corrected[overlap]).to_numpy(float)
            truth = obs_vals.to_numpy(float)
            self.filled_df.loc[fill_mask, target_col] = corrected[fill_mask]
            self._record_fill(
                target_col, fill_mask, "era5_bias_corrected",
                _r2(truth, held), held - truth,
            )
        return self.filled_df

    # ── cascade ─────────────────────────────────────────────────────────
    def auto_fill(self, columns=None, reanalysis_map=None):
        """Automatic gap filling: cascade from simple to complex.

        Continuous variables: linear (gaps of up to 3 records), spline (up
        to 7), bias-corrected reanalysis, climatology plus anomaly, then
        regression for anything left. An interpolation stage is skipped for
        a variable when its hold-out error exceeds that of climatology plus
        anomaly.

        Zero-inflated variables such as daily precipitation skip the spline
        stage and place regression before climatology, because a smooth
        curve through wet and dry days has no physical meaning.

        Parameters
        ----------
        columns : list, optional
            Columns to fill. Defaults to all station (non-reanalysis) variables.
        reanalysis_map : dict, optional
            ``{station_col: reanalysis_col}``. When given, gaps left after
            interpolation are filled from the bias-corrected reanalysis
            series before the seasonal and ML stages.
        """
        if columns is None:
            columns = self._get_target_cols()
        reanalysis_map = reanalysis_map or {}

        def regression(col):
            other_cols = [c for c in columns if c != col and c != self.date_col]
            other_cols += [
                c for c in self._get_numeric_cols()
                if _is_reanalysis(c) and c not in other_cols
            ]
            if HAS_XGBOOST:
                self.xgboost_fill(col, other_cols)
            else:
                self.random_forest_fill(col, other_cols)

        for col in columns:
            if col == self.date_col:
                continue
            if not self.filled_df[col].isna().any():
                continue
            zero_inflated = _is_zero_inflated(self.df[col])

            stages = [
                lambda c: self.linear_interpolation(
                    [c], max_gap=3, only_if_better=True
                )
            ]
            if not zero_inflated:
                stages.append(
                    lambda c: self.spline_interpolation(
                        [c], max_gap=7, only_if_better=True
                    )
                )
            if col in reanalysis_map:
                stages.append(
                    lambda c: self.era5_bias_correction(c, reanalysis_map[c])
                )
            tail = [lambda c: self.seasonal_interpolation([c]), regression]
            stages += tail[::-1] if zero_inflated else tail

            for stage in stages:
                if not self.filled_df[col].isna().any():
                    break
                stage(col)

        return self.filled_df

    # ── intervals ───────────────────────────────────────────────────────
    def residual_intervals(self, target_col):
        """95 % intervals for the filled values of ``target_col``.

        For each fill the interval is the filled value minus the 97.5th and
        2.5th percentiles of the hold-out residuals of the stage that
        produced it, limited to the physical bounds. Fills from a stage with
        fewer than ``MIN_RESIDUALS`` residuals receive no interval.

        Returns
        -------
        pd.DataFrame
            Indexed like the filled records, with ``filled_value``,
            ``method``, ``n_residuals``, ``ci_lower_95``, ``ci_upper_95``
            and ``ci_width``. Empty when no interval can be given.
        """
        meta = self.fill_metadata.get(target_col)
        if meta is None:
            return pd.DataFrame()
        pools = self.residual_pools.get(target_col, {})
        lower_b, upper_b, _ = self.limits(target_col)
        filled = meta["flag"] == "F"
        parts = []
        for pool, labels in meta["pool"][filled].groupby(meta["pool"][filled]):
            residuals = pools.get(pool)
            if residuals is None:
                continue
            q_low, q_high = np.quantile(residuals, [0.025, 0.975])
            value = self.filled_df.loc[labels.index, target_col].astype(float)
            part = pd.DataFrame({
                "filled_value": value,
                "method": meta["method"].loc[labels.index],
                "n_residuals": len(residuals),
                "ci_lower_95": np.minimum(value, value - q_high),
                "ci_upper_95": np.maximum(value, value - q_low),
            })
            part["ci_lower_95"] = part["ci_lower_95"].clip(lower=lower_b)
            part["ci_upper_95"] = part["ci_upper_95"].clip(upper=upper_b)
            parts.append(part)
        if not parts:
            return pd.DataFrame()
        result = pd.concat(parts).sort_index()
        result["ci_width"] = result["ci_upper_95"] - result["ci_lower_95"]
        return result

    def bootstrap_uncertainty(self, target_col, n_bootstrap=None,
                              random_state=None):
        """Deprecated alias of :meth:`residual_intervals`."""
        warnings.warn(
            "bootstrap_uncertainty() is deprecated; intervals now come from "
            "hold-out residuals. Use residual_intervals().",
            DeprecationWarning, stacklevel=2,
        )
        return self.residual_intervals(target_col)


class ClimFillEngine:
    """Main interface for the ClimFill gap-filling pipeline."""

    METHODS = ("auto", "linear", "spline", "seasonal", "random_forest", "xgboost")

    def __init__(self):
        self.raw_df = None
        self.df = None
        self.date_col = None
        self.gap_report = None
        self.filler = None
        self.detector = None
        self.filled_df = None
        self.outliers = {}
        self.uncertainty_results = {}
        self.sentinels = None
        self.fill_bounds = None

    def load_csv(self, filepath_or_buffer, date_col=None, sentinels=None,
                 regularise=True):
        """Load climate station data from CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like
            CSV with one datetime column and numeric variables.
        date_col : str, optional
            Datetime column; detected when omitted.
        sentinels : list of float, optional
            Missing-value codes to convert to NaN. Defaults to
            ``GapDetector.MISSING_SENTINELS`` (-999, -9999, -99.9).
        regularise : bool
            Insert an empty row for every timestamp absent from a series
            whose usual interval is one day or shorter, so that gaps stored
            as missing rows are detected and filled. The count is returned
            as ``rows_inserted``.
        """
        self.raw_df = pd.read_csv(filepath_or_buffer)
        self.df = self.raw_df.copy()

        self.detector = GapDetector(self.df, date_col, sentinels)
        self.date_col = self.detector.detect_date_column()
        # Interpolation assumes chronological order
        self.detector.df = self.detector.df.sort_values(
            self.date_col, kind="stable"
        ).reset_index(drop=True)
        self.df = self.detector.df.copy()

        # Replace sentinel values
        replaced = self.detector.replace_sentinels()
        self.df = self.detector.df.copy()
        self.sentinels = list(self.detector.sentinels)

        inserted = 0
        if regularise:
            diffs = self.df[self.date_col].dropna().diff().dropna()
            diffs = diffs[diffs > pd.Timedelta(0)]
            if len(diffs) and diffs.mode().iloc[0] <= pd.Timedelta(days=1):
                step = diffs.mode().iloc[0]
                dates = pd.DatetimeIndex(self.df[self.date_col].dropna())
                grid = pd.date_range(dates.min(), dates.max(), freq=step)
                # Only when every record sits on the regular grid and the
                # record is not mostly absent
                if dates.isin(grid).all() and len(grid) <= 5 * len(dates):
                    inserted = self.insert_missing_timestamps(step)

        return {
            "rows_inserted": inserted,
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "date_column": self.date_col,
            "date_range": (
                f"{self.df[self.date_col].min()} to {self.df[self.date_col].max()}"
            ),
            "sentinels_replaced": replaced,
        }

    def station_columns(self):
        """Numeric station variables (excludes the date and reanalysis columns)."""
        return [
            c
            for c in self.df.select_dtypes(include=[np.number]).columns
            if c != self.date_col and not _is_reanalysis(c)
        ]

    def detect_gaps(self, check_outliers=True, outlier_method="iqr",
                    outlier_threshold=3.0, outlier_columns=None):
        """Run gap detection and analysis.

        ``outlier_columns`` limits the outlier screen to those variables
        (default: all station variables).
        """
        self.detector = GapDetector(
            self.df, self.date_col, self.sentinels
        )
        self.detector.detect_date_column()
        self.df = self.detector.df.copy()
        columns = self.station_columns()
        self.gap_report = self.detector.analyze_gaps(columns)

        result = {
            "gap_report": self.gap_report,
            "temporal_regularity": self.detector.check_temporal_regularity(),
        }
        self.outliers = {}
        if check_outliers:
            self.outliers = self.detector.detect_outliers(
                columns if outlier_columns is None else list(outlier_columns),
                method=outlier_method,
                threshold=outlier_threshold,
            )
            result["outliers"] = self.outliers
        return result

    def insert_missing_timestamps(self, freq=None):
        """Add empty rows for timestamps absent from a regular series.

        Records missing from the file are invisible to gap detection, so
        they are inserted as all-NaN rows. ``freq`` defaults to the most
        common interval. Returns the number of rows inserted.
        """
        dates = self.df[self.date_col].dropna()
        if freq is None:
            diffs = dates.sort_values().diff().dropna()
            if diffs.empty:
                return 0
            freq = diffs.mode().iloc[0]
        full = pd.date_range(dates.min(), dates.max(), freq=freq)
        absent = full.difference(pd.DatetimeIndex(dates))
        if len(absent) == 0:
            return 0
        extra = pd.DataFrame({self.date_col: absent})
        self.df = (
            pd.concat([self.df, extra], ignore_index=True)
            .sort_values(self.date_col, kind="stable")
            .reset_index(drop=True)
        )
        return len(absent)

    def merge_reanalysis(self, rean_df):
        """Join daily reanalysis columns onto the station data by calendar date.

        ``rean_df`` must carry a ``date`` column and ``reanalysis_``-prefixed
        variables, as returned by :func:`get_open_meteo_data`. Re-merging
        replaces previously merged reanalysis columns.
        """
        rean_cols = [c for c in rean_df.columns if _is_reanalysis(c)]
        rean = rean_df[rean_cols].copy()
        rean["_merge_date"] = pd.to_datetime(rean_df["date"]).dt.normalize()
        rean = rean.drop_duplicates("_merge_date")

        base = self.df.drop(
            columns=[c for c in self.df.columns if _is_reanalysis(c)]
        )
        base["_merge_date"] = base[self.date_col].dt.normalize()
        merged = base.merge(rean, on="_merge_date", how="left")
        self.df = merged.drop(columns=["_merge_date"])
        return rean_cols

    # Keyword options accepted by fill_gaps() for each method
    FILL_OPTIONS = {
        "linear": ("max_gap",),
        "spline": ("max_gap", "order"),
        "random_forest": ("n_estimators",),
    }

    def _run_fill(self, filler, method, columns, reanalysis_map, options=None):
        options = options or {}
        if method == "auto":
            return filler.auto_fill(columns, reanalysis_map=reanalysis_map)
        if method == "linear":
            return filler.linear_interpolation(columns, **options)
        if method == "spline":
            return filler.spline_interpolation(columns, **options)
        if method == "seasonal":
            return filler.seasonal_interpolation(columns)
        if method in ("random_forest", "xgboost"):
            if columns is None:
                columns = filler._get_target_cols()
            for col in columns:
                if col == self.date_col:
                    continue
                if method == "xgboost":
                    filler.xgboost_fill(col)
                else:
                    filler.random_forest_fill(col, **options)
            return filler.filled_df
        raise ValueError(
            f"Unknown method: {method}. Choose from: {', '.join(self.METHODS)}"
        )

    def _check_options(self, method, options):
        allowed = self.FILL_OPTIONS.get(method, ())
        unknown = sorted(set(options) - set(allowed))
        if unknown:
            raise TypeError(
                f"fill_gaps(method={method!r}) does not accept "
                f"{', '.join(unknown)}; accepted options: "
                f"{', '.join(allowed) or 'none'}"
            )

    def fill_gaps(self, method="auto", columns=None, reanalysis_map=None,
                  bounds=None, **options):
        """Fill gaps using the specified method.

        Parameters
        ----------
        method : str
            One of ``ClimFillEngine.METHODS``.
        columns : list, optional
            Variables to fill (default: all station variables).
        reanalysis_map : dict, optional
            ``{station_col: reanalysis_col}``; enables the bias-corrected
            reanalysis stage of the ``auto`` cascade.
        bounds : dict, optional
            ``{column: (lower, upper)}`` physical limits for filled values,
            overriding those inferred by :func:`infer_bounds`.
        **options
            ``max_gap`` (linear, spline), ``order`` (spline) or
            ``n_estimators`` (random_forest). Any other keyword raises
            ``TypeError``.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method: {method}. Choose from: {', '.join(self.METHODS)}"
            )
        self._check_options(method, options)
        self.filler = GapFiller(self.df, self.date_col, bounds=bounds)
        self.fill_bounds = bounds
        self.uncertainty_results = {}
        self.filled_df = self._run_fill(
            self.filler, method, columns, reanalysis_map, options
        )
        return self.filled_df

    def compute_uncertainty(self, columns=None, n_bootstrap=None,
                            progress_callback=None):
        """95 % intervals for the filled values, from hold-out residuals.

        Each interval is built from the residuals that the method which
        produced the fill left on observed values hidden during
        :meth:`fill_gaps` (see :meth:`GapFiller.residual_intervals`).

        ``n_bootstrap`` is deprecated and has no effect: the earlier
        bootstrap of Random Forest means did not describe the error of the
        filled value. ``progress_callback(col, done, total)`` is called
        after each variable.
        """
        if self.filler is None:
            raise RuntimeError("Run fill_gaps() first.")
        if n_bootstrap is not None:
            warnings.warn(
                "n_bootstrap is deprecated and ignored; intervals are derived "
                "from hold-out residuals.",
                DeprecationWarning, stacklevel=2,
            )

        if columns is None:
            columns = self.station_columns()
        columns = [c for c in columns if self.df[c].isna().sum() > 0]

        for i, col in enumerate(columns, start=1):
            result = self.filler.residual_intervals(col)
            if not result.empty:
                self.uncertainty_results[col] = result
            if progress_callback is not None:
                progress_callback(col, i, len(columns))

        return self.uncertainty_results

    def validate(self, method="auto", gap_length=1, fraction=0.05,
                 columns=None, reanalysis_map=None, random_state=42,
                 bounds=None, **options):
        """Hold-out accuracy assessment of a fill method.

        Observed values are hidden as synthetic gaps of ``gap_length``
        records (about ``fraction`` of the observed values per variable),
        the method is re-run, and the reconstructed values are compared
        with the hidden observations.

        Returns
        -------
        pd.DataFrame
            One row per variable: n, MAE, RMSE, bias (filled - observed)
            and R^2.
        """
        if columns is None:
            columns = self.station_columns()
        rng = np.random.default_rng(random_state)
        masked = self.df.copy()
        held_out = {}

        for col in columns:
            observed = self.df[col].notna().to_numpy()
            # Starts where a full window of gap_length observed values fits
            window = np.convolve(observed, np.ones(gap_length, int), "valid")
            starts = np.flatnonzero(window == gap_length)
            target = max(1, int(observed.sum() * fraction / gap_length))
            rng.shuffle(starts)
            taken = np.zeros(len(observed), bool)
            chosen = []
            for st_ in starts:
                # Keep one observed record either side so gaps stay separate
                lo, hi = max(st_ - 1, 0), st_ + gap_length + 1
                if taken[lo:hi].any():
                    continue
                taken[st_:st_ + gap_length] = True
                chosen.append(st_)
                if len(chosen) >= target:
                    break
            idx = self.df.index[taken]
            if len(idx) == 0:
                continue
            held_out[col] = idx
            masked.loc[idx, col] = np.nan

        self._check_options(method, options)
        filler = GapFiller(masked, self.date_col, bounds=bounds)
        filled = self._run_fill(filler, method, columns, reanalysis_map, options)

        rows = []
        for col, idx in held_out.items():
            obs = self.df.loc[idx, col]
            est = filled.loc[idx, col]
            ok = est.notna()
            obs, est = obs[ok].to_numpy(float), est[ok].to_numpy(float)
            row = {"variable": col, "n": int(ok.sum())}
            if len(obs) > 0:
                err = est - obs
                ss_tot = float(np.sum((obs - obs.mean()) ** 2))
                row.update(
                    mae=float(np.mean(np.abs(err))),
                    rmse=float(np.sqrt(np.mean(err ** 2))),
                    bias=float(np.mean(err)),
                    r2=(1 - float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else np.nan,
                )
            rows.append(row)
        return pd.DataFrame(
            rows, columns=["variable", "n", "mae", "rmse", "bias", "r2"]
        )

    def export(self, filepath=None, include_uncertainty=True, include_metadata=True):
        """Export filled dataset with quality flags and metadata."""
        if self.filled_df is None:
            raise RuntimeError("Run fill_gaps() first.")

        export_df = self.filled_df.copy()

        if include_metadata and self.filler and self.filler.fill_metadata:
            for col, meta in self.filler.fill_metadata.items():
                export_df[f"{col}_flag"] = meta["flag"]
                export_df[f"{col}_method"] = meta["method"]
                export_df[f"{col}_confidence"] = meta["confidence"]

        if include_metadata and self.outliers:
            # Statistical outliers in the original record are flagged "S"
            for col, info in self.outliers.items():
                flag_col = f"{col}_flag"
                if flag_col not in export_df.columns:
                    export_df[flag_col] = np.where(
                        self.df[col].notna(), "O", ""
                    )
                export_df.loc[info["indices"], flag_col] = "S"

        if include_uncertainty and self.uncertainty_results:
            for col, unc_df in self.uncertainty_results.items():
                if not unc_df.empty:
                    export_df[f"{col}_ci_lower"] = np.nan
                    export_df[f"{col}_ci_upper"] = np.nan
                    export_df[f"{col}_ci_width"] = np.nan
                    export_df.loc[unc_df.index, f"{col}_ci_lower"] = unc_df[
                        "ci_lower_95"
                    ].values
                    export_df.loc[unc_df.index, f"{col}_ci_upper"] = unc_df[
                        "ci_upper_95"
                    ].values
                    export_df.loc[unc_df.index, f"{col}_ci_width"] = unc_df[
                        "ci_width"
                    ].values

        if filepath:
            export_df.to_csv(filepath, index=False)

        return export_df

    def generate_report(self):
        """Generate a summary report."""
        if self.gap_report is None or self.filler is None:
            return {}

        report = {
            "dataset_info": {
                "total_records": len(self.df),
                "date_range": (
                    f"{self.df[self.date_col].min()} to "
                    f"{self.df[self.date_col].max()}"
                ),
            },
            "gap_summary": {},
            "fill_summary": {},
        }

        for col in self.gap_report:
            orig_missing = self.gap_report[col]["missing_count"]
            remaining = (
                int(self.filled_df[col].isna().sum())
                if col in self.filled_df.columns
                else 0
            )
            filled_count = orig_missing - remaining

            report["gap_summary"][col] = {
                "original_missing": orig_missing,
                "original_missing_pct": self.gap_report[col]["missing_pct"],
                "remaining_missing": remaining,
                "filled_count": filled_count,
                "fill_rate_pct": round(
                    filled_count / max(orig_missing, 1) * 100, 1
                ),
            }

            if col in self.filler.fill_metadata:
                meta = self.filler.fill_metadata[col]
                is_filled = meta["flag"] == "F"
                methods_used = meta["method"][is_filled].value_counts().to_dict()
                mean_conf = meta["confidence"][is_filled].mean()
                mean_conf = (
                    None if pd.isna(mean_conf) else round(float(mean_conf), 3)
                )
                report["fill_summary"][col] = {
                    "methods_used": methods_used,
                    "mean_confidence": mean_conf,
                }

        if self.outliers:
            report["outlier_summary"] = {
                col: info["count"] for col, info in self.outliers.items()
            }

        return report
