"""
ClimFill Engine — Core gap detection and filling logic.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None
    HAS_XGBOOST = False
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class GapDetector:
    """Detects and characterizes gaps in climate time series."""

    MISSING_SENTINELS = [-999, -9999, -99.9, -999.0, -9999.0, 9999, 99999]

    def __init__(self, df, date_col=None):
        self.df = df.copy()
        self.date_col = date_col
        self.gap_report = {}

    def detect_date_column(self):
        """Auto-detect the datetime column."""
        if self.date_col and self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            return self.date_col

        for col in self.df.columns:
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
        """Replace common missing value sentinels with NaN."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        replaced = {}
        for col in numeric_cols:
            count = 0
            for sentinel in self.MISSING_SENTINELS:
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
            if len(series) < 10:
                continue

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask = (self.df[col] < lower) | (self.df[col] > upper)
            elif method == "zscore":
                z = np.abs(stats.zscore(series))
                mask = pd.Series(False, index=self.df.index)
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


class GapFiller:
    """Multi-method gap filling with uncertainty quantification."""

    def __init__(self, df, date_col):
        self.df = df.copy()
        self.date_col = date_col
        self.filled_df = df.copy()
        self.fill_metadata = {}

    def _get_numeric_cols(self):
        return [
            c
            for c in self.filled_df.select_dtypes(include=[np.number]).columns
            if c != self.date_col
        ]

    def _create_temporal_features(self, dates):
        """Create temporal features for ML models."""
        features = pd.DataFrame(index=dates.index)
        features["day_of_year_sin"] = np.sin(
            2 * np.pi * dates.dt.dayofyear / 365.25
        )
        features["day_of_year_cos"] = np.cos(
            2 * np.pi * dates.dt.dayofyear / 365.25
        )
        features["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
        features["year"] = dates.dt.year
        features["day_of_week"] = dates.dt.dayofweek
        return features

    def _init_metadata(self, col):
        """Initialize fill metadata for a column if not already done."""
        if col not in self.fill_metadata:
            self.fill_metadata[col] = {
                "method": pd.Series("original", index=self.df.index),
                "confidence": pd.Series(1.0, index=self.df.index),
                "flag": pd.Series("O", index=self.df.index),
            }
            # Mark originally-missing values
            orig_missing = self.df[col].isna()
            self.fill_metadata[col]["flag"][orig_missing] = ""
            self.fill_metadata[col]["method"][orig_missing] = ""
            self.fill_metadata[col]["confidence"][orig_missing] = 0.0

    def _record_fill(self, col, mask, method, confidence):
        """Record metadata for filled values."""
        self._init_metadata(col)

        if isinstance(mask, pd.Series):
            indices = mask[mask].index
        elif isinstance(mask, np.ndarray):
            indices = self.filled_df.index[mask]
        else:
            return

        for idx in indices:
            self.fill_metadata[col]["method"].at[idx] = method
            self.fill_metadata[col]["confidence"].at[idx] = confidence
            self.fill_metadata[col]["flag"].at[idx] = "F"

    def linear_interpolation(self, columns=None, max_gap=3):
        """Linear interpolation for short gaps."""
        if columns is None:
            columns = self._get_numeric_cols()
        for col in columns:
            if col == self.date_col:
                continue
            missing_before = self.filled_df[col].isna()
            self.filled_df[col] = self.filled_df[col].interpolate(
                method="linear", limit=max_gap
            )
            filled_mask = missing_before & self.filled_df[col].notna()
            if filled_mask.any():
                self._record_fill(col, filled_mask, "linear_interpolation", 0.7)
        return self.filled_df

    def spline_interpolation(self, columns=None, max_gap=7, order=3):
        """Spline interpolation for medium gaps."""
        if columns is None:
            columns = self._get_numeric_cols()
        for col in columns:
            if col == self.date_col:
                continue
            missing_before = self.filled_df[col].isna()
            try:
                self.filled_df[col] = self.filled_df[col].interpolate(
                    method="spline", order=order, limit=max_gap
                )
            except Exception:
                # Spline can fail on certain data; fall back to linear
                self.filled_df[col] = self.filled_df[col].interpolate(
                    method="linear", limit=max_gap
                )
            filled_mask = missing_before & self.filled_df[col].notna()
            if filled_mask.any():
                self._record_fill(
                    col, filled_mask, "spline_interpolation", 0.65
                )
        return self.filled_df

    def seasonal_interpolation(self, columns=None):
        """Seasonal decomposition-based filling."""
        if columns is None:
            columns = self._get_numeric_cols()
        for col in columns:
            if col == self.date_col:
                continue
            series = self.filled_df[col].copy()
            missing_mask = series.isna()
            if not missing_mask.any():
                continue
            if self.date_col and self.date_col in self.filled_df.columns:
                month = self.filled_df[self.date_col].dt.month
                # Monthly climatology from available data
                monthly_mean = series.groupby(month).transform("mean")
                anomaly = series - monthly_mean
                anomaly_filled = anomaly.interpolate(method="linear")
                filled_values = monthly_mean + anomaly_filled
                new_fills = missing_mask & filled_values.notna()
                series[new_fills] = filled_values[new_fills]
                self.filled_df[col] = series
                if new_fills.any():
                    self._record_fill(
                        col, new_fills, "seasonal_decomposition", 0.75
                    )
        return self.filled_df

    def random_forest_fill(self, target_col, predictor_cols=None, n_estimators=100):
        """Random Forest regression using correlated variables + temporal features."""
        if predictor_cols is None:
            predictor_cols = [
                c
                for c in self._get_numeric_cols()
                if c != target_col
            ]

        missing_mask = self.filled_df[target_col].isna()
        if not missing_mask.any():
            return self.filled_df

        # Build features
        features = pd.DataFrame(index=self.filled_df.index)
        for pc in predictor_cols:
            features[pc] = self.filled_df[pc]

        if self.date_col and self.date_col in self.filled_df.columns:
            temp_feats = self._create_temporal_features(
                self.filled_df[self.date_col]
            )
            features = pd.concat([features, temp_feats], axis=1)

        # Lag features
        for lag in [1, 2, 3, 7]:
            lag_fwd = self.filled_df[target_col].shift(lag)
            lag_bwd = self.filled_df[target_col].shift(-lag)
            features[f"lag_{lag}"] = lag_fwd
            features[f"lead_{lag}"] = lag_bwd

        # Train/predict split
        train_mask = (~missing_mask) & features.notna().all(axis=1)
        predict_mask = missing_mask & features.notna().all(axis=1)

        if train_mask.sum() < 20 or predict_mask.sum() == 0:
            return self.filled_df

        X_train = features.loc[train_mask]
        y_train = self.filled_df.loc[train_mask, target_col]
        X_predict = features.loc[predict_mask]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        try:
            n_cv = min(5, int(train_mask.sum()))
            if n_cv >= 2:
                cv_scores = cross_val_score(
                    rf, X_train, y_train, cv=n_cv, scoring="r2"
                )
                confidence = max(0.3, min(0.95, float(np.mean(cv_scores))))
            else:
                confidence = 0.5
        except Exception:
            confidence = 0.5

        predictions = rf.predict(X_predict)
        self.filled_df.loc[predict_mask, target_col] = predictions
        self._record_fill(
            target_col, predict_mask, "random_forest", round(confidence, 3)
        )
        return self.filled_df

    def xgboost_fill(self, target_col, predictor_cols=None):
        """XGBoost regression with temporal features."""
        if not HAS_XGBOOST:
            return self.random_forest_fill(target_col, predictor_cols)

        if predictor_cols is None:
            predictor_cols = [
                c for c in self._get_numeric_cols() if c != target_col
            ]

        missing_mask = self.filled_df[target_col].isna()
        if not missing_mask.any():
            return self.filled_df

        features = pd.DataFrame(index=self.filled_df.index)
        for pc in predictor_cols:
            features[pc] = self.filled_df[pc]

        if self.date_col and self.date_col in self.filled_df.columns:
            temp_feats = self._create_temporal_features(
                self.filled_df[self.date_col]
            )
            features = pd.concat([features, temp_feats], axis=1)

        for lag in [1, 2, 3, 7]:
            features[f"lag_{lag}"] = self.filled_df[target_col].shift(lag)
            features[f"lead_{lag}"] = self.filled_df[target_col].shift(-lag)

        train_mask = (~missing_mask) & features.notna().all(axis=1)
        predict_mask = missing_mask & features.notna().all(axis=1)

        if train_mask.sum() < 20 or predict_mask.sum() == 0:
            return self.filled_df

        X_train = features.loc[train_mask]
        y_train = self.filled_df.loc[train_mask, target_col]
        X_predict = features.loc[predict_mask]

        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)

        try:
            n_cv = min(5, int(train_mask.sum()))
            if n_cv >= 2:
                cv_scores = cross_val_score(
                    xgb, X_train, y_train, cv=n_cv, scoring="r2"
                )
                confidence = max(0.3, min(0.95, float(np.mean(cv_scores))))
            else:
                confidence = 0.5
        except Exception:
            confidence = 0.5

        predictions = xgb.predict(X_predict)
        self.filled_df.loc[predict_mask, target_col] = predictions
        self._record_fill(
            target_col, predict_mask, "xgboost", round(confidence, 3)
        )
        return self.filled_df

    def era5_bias_correction(self, target_col, era5_col):
        """Fill gaps using reanalysis with quantile mapping bias correction."""
        missing_mask = self.filled_df[target_col].isna()
        if not missing_mask.any() or era5_col not in self.filled_df.columns:
            return self.filled_df

        overlap = (
            self.filled_df[target_col].notna()
            & self.filled_df[era5_col].notna()
        )

        if overlap.sum() < 10:
            return self.filled_df

        obs_vals = self.filled_df.loc[overlap, target_col]
        era5_vals = self.filled_df.loc[overlap, era5_col]

        if overlap.sum() < 30:
            # Simple bias correction
            bias = float(obs_vals.mean() - era5_vals.mean())
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
        self.filled_df.loc[fill_mask, target_col] = corrected[fill_mask]
        if fill_mask.any():
            self._record_fill(
                target_col, fill_mask, "era5_bias_corrected", 0.8
            )
        return self.filled_df

    def auto_fill(self, columns=None):
        """Automatic gap filling: cascade from simple to complex."""
        if columns is None:
            columns = self._get_numeric_cols()

        for col in columns:
            if col == self.date_col:
                continue
            if not self.filled_df[col].isna().any():
                continue

            # Step 1: Short gaps — linear
            self.linear_interpolation([col], max_gap=3)

            # Step 2: Medium gaps — spline
            if self.filled_df[col].isna().any():
                self.spline_interpolation([col], max_gap=7)

            # Step 3: Seasonal patterns
            if self.filled_df[col].isna().any():
                self.seasonal_interpolation([col])

            # Step 4: ML for remaining
            if self.filled_df[col].isna().any():
                other_cols = [c for c in columns if c != col and c != self.date_col]
                if HAS_XGBOOST:
                    self.xgboost_fill(col, other_cols)
                else:
                    self.random_forest_fill(col, other_cols)

        return self.filled_df

    def bootstrap_uncertainty(self, target_col, n_bootstrap=50):
        """Calculate uncertainty via bootstrap resampling."""
        original_missing = self.df[target_col].isna()
        if not original_missing.any():
            return pd.DataFrame()

        predictor_cols = [c for c in self._get_numeric_cols() if c != target_col]

        features = pd.DataFrame(index=self.filled_df.index)
        for pc in predictor_cols:
            features[pc] = self.filled_df[pc]

        if self.date_col and self.date_col in self.filled_df.columns:
            temp_feats = self._create_temporal_features(
                self.filled_df[self.date_col]
            )
            features = pd.concat([features, temp_feats], axis=1)

        for lag in [1, 2, 3, 7]:
            features[f"lag_{lag}"] = self.filled_df[target_col].shift(lag)
            features[f"lead_{lag}"] = self.filled_df[target_col].shift(-lag)

        train_mask = (~original_missing) & features.notna().all(axis=1)
        predict_mask = original_missing & features.notna().all(axis=1)

        if train_mask.sum() < 20 or predict_mask.sum() == 0:
            return pd.DataFrame()

        X_train = features.loc[train_mask].values
        y_train = self.filled_df.loc[train_mask, target_col].values
        X_predict = features.loc[predict_mask].values

        predictions = np.zeros((n_bootstrap, int(predict_mask.sum())))
        for i in range(n_bootstrap):
            idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            model = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=i, n_jobs=-1
            )
            model.fit(X_train[idx], y_train[idx])
            predictions[i] = model.predict(X_predict)

        result = pd.DataFrame(index=self.filled_df.index[predict_mask])
        result["filled_value"] = self.filled_df.loc[predict_mask, target_col].values
        result["mean_prediction"] = np.mean(predictions, axis=0)
        result["std_prediction"] = np.std(predictions, axis=0)
        result["ci_lower_95"] = np.percentile(predictions, 2.5, axis=0)
        result["ci_upper_95"] = np.percentile(predictions, 97.5, axis=0)
        result["ci_width"] = result["ci_upper_95"] - result["ci_lower_95"]
        return result


class ClimFillEngine:
    """Main interface for the ClimFill gap-filling pipeline."""

    def __init__(self):
        self.raw_df = None
        self.df = None
        self.date_col = None
        self.gap_report = None
        self.filler = None
        self.detector = None
        self.filled_df = None
        self.uncertainty_results = {}

    def load_csv(self, filepath_or_buffer, date_col=None):
        """Load climate station data from CSV."""
        self.raw_df = pd.read_csv(filepath_or_buffer)
        self.df = self.raw_df.copy()

        self.detector = GapDetector(self.df, date_col)
        self.date_col = self.detector.detect_date_column()
        self.df = self.detector.df.copy()

        # Replace sentinel values
        replaced = self.detector.replace_sentinels()
        self.df = self.detector.df.copy()

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "date_column": self.date_col,
            "date_range": (
                f"{self.df[self.date_col].min()} to {self.df[self.date_col].max()}"
            ),
            "sentinels_replaced": replaced,
        }

    def detect_gaps(self, check_outliers=True):
        """Run gap detection and analysis."""
        self.detector = GapDetector(self.df, self.date_col)
        self.detector.detect_date_column()
        self.df = self.detector.df.copy()
        self.gap_report = self.detector.analyze_gaps()

        result = {
            "gap_report": self.gap_report,
            "temporal_regularity": self.detector.check_temporal_regularity(),
        }
        if check_outliers:
            result["outliers"] = self.detector.detect_outliers()
        return result

    def fill_gaps(self, method="auto", columns=None, **kwargs):
        """Fill gaps using the specified method."""
        self.filler = GapFiller(self.df, self.date_col)

        if method == "auto":
            self.filled_df = self.filler.auto_fill(columns)
        elif method == "linear":
            self.filled_df = self.filler.linear_interpolation(columns)
        elif method == "spline":
            self.filled_df = self.filler.spline_interpolation(columns)
        elif method == "seasonal":
            self.filled_df = self.filler.seasonal_interpolation(columns)
        elif method == "random_forest":
            self.filled_df = self._fill_ml(columns, "random_forest")
        elif method == "xgboost":
            self.filled_df = self._fill_ml(columns, "xgboost")
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from: auto, linear, spline, seasonal, random_forest, xgboost"
            )
        return self.filled_df

    def _fill_ml(self, columns, method):
        if columns is None:
            columns = self.filler._get_numeric_cols()
        for col in columns:
            if col == self.date_col:
                continue
            if method == "xgboost":
                self.filler.xgboost_fill(col)
            else:
                self.filler.random_forest_fill(col)
        return self.filler.filled_df

    def compute_uncertainty(self, columns=None, n_bootstrap=50):
        """Compute uncertainty estimates for filled values."""
        if self.filler is None:
            raise RuntimeError("Run fill_gaps() first.")

        if columns is None:
            columns = [
                c
                for c in self.df.select_dtypes(include=[np.number]).columns
                if c != self.date_col
            ]

        for col in columns:
            if self.df[col].isna().sum() > 0:
                result = self.filler.bootstrap_uncertainty(col, n_bootstrap)
                if not result.empty:
                    self.uncertainty_results[col] = result

        return self.uncertainty_results

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
                mean_conf = (
                    round(float(meta["confidence"][is_filled].mean()), 3)
                    if is_filled.any()
                    else None
                )
                report["fill_summary"][col] = {
                    "methods_used": methods_used,
                    "mean_confidence": mean_conf,
                }

        return report
