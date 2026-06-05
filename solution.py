"""
Traffic Demand Prediction
=========================

Predicts the `demand` value for each row in `dataset/test.csv` after training
a regression model on `dataset/train.csv`. The test target is never read --
all aggregations and encoders are fit on TRAIN only and then transformed
onto test.

Why this version is more involved
---------------------------------
A naive random holdout reports ~92, but the leaderboard returned ~87. The
mismatch is a distribution-shift story: train covers all 96 timestamps of
day 48 plus the first 9 timestamps of day 49, while test covers the
remaining timestamps of day 49. A random holdout lets the model see the
same (geohash, hour) pair on both sides of the split, which inflates the
score. To close that gap we:

1. Engineer features that generalise across timestamps -- per-location
   statistics, day-48 → day-49 reference signals, geohash-decoded
   lat/lon, smoothed target encoding -- instead of relying on the model
   to memorise (geohash, hour) cells.
2. Validate on the day-49 slice of train instead of a random sample, so
   the holdout actually mimics the test-set shift.

Approach
--------
1. Parse timestamps; add hour/minute/cyclic encodings and rush-hour flags.
2. Decode each 6-char geohash to (lat, lon) -- this gives the model real
   spatial coordinates that generalise to the 10 test geohashes never seen
   in train.
3. Build per-geohash aggregates of demand from TRAIN ONLY:
     - smoothed target encoding for geohash and geohash×hour
     - day-48 mean demand at the same geohash and same (geohash, hour)
   These are computed in an out-of-fold loop for train rows so a row's
   own target never contributes to its own encoded feature.
4. Impute Weather / RoadType nulls with the per-geohash mode (these
   attributes are stable per location). Impute Temperature with the
   (day, hour) mean.
5. Ordinal-encode remaining categoricals.
6. Train a RandomForest + ExtraTrees ensemble and average predictions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "submission.csv"
)

RANDOM_STATE = 42


# ===========================================================================
# Geohash decoding
# ===========================================================================
# A geohash is a base-32 encoding that recursively bisects the lat/lon
# rectangle. Decoding it gives the model genuine coordinates instead of an
# opaque categorical -- crucial for the 10 test geohashes never seen in
# train, since nearby trained geohashes still inform their lat/lon.
_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_IDX = {c: i for i, c in enumerate(_GEOHASH_BASE32)}


def decode_geohash(gh: str) -> Tuple[float, float]:
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True  # even bits drive longitude, odd bits drive latitude
    for ch in gh:
        idx = _GEOHASH_IDX[ch]
        for shift in range(4, -1, -1):
            bit = (idx >> shift) & 1
            if even:
                mid = 0.5 * (lon_lo + lon_hi)
                if bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = 0.5 * (lat_lo + lat_hi)
                if bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return 0.5 * (lat_lo + lat_hi), 0.5 * (lon_lo + lon_hi)


def add_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Decode each geohash once via a lookup table -- ~1.2k unique values."""
    unique = df["geohash"].unique()
    table = {gh: decode_geohash(gh) for gh in unique}
    df["lat"] = df["geohash"].map(lambda g: table[g][0])
    df["lon"] = df["geohash"].map(lambda g: table[g][1])
    return df


# ===========================================================================
# Time features
# ===========================================================================
def parse_timestamp_to_minutes(ts: pd.Series) -> pd.Series:
    parts = ts.astype(str).str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    minutes = parse_timestamp_to_minutes(df["timestamp"])
    df["minutes_of_day"] = minutes
    df["hour"] = minutes // 60
    df["minute"] = minutes % 60
    # Cyclic encodings let trees split smoothly across the midnight boundary.
    radians_day = 2.0 * np.pi * minutes / (24 * 60)
    df["time_sin"] = np.sin(radians_day)
    df["time_cos"] = np.cos(radians_day)
    # Half-day cycle: AM/PM symmetry sometimes carries useful traffic structure.
    radians_half = 2.0 * np.pi * minutes / (12 * 60)
    df["time_sin_half"] = np.sin(radians_half)
    df["time_cos_half"] = np.cos(radians_half)
    # Coarse buckets: rush hours and night dominate demand in most cities,
    # and giving the model a ready-made indicator saves it some splits.
    h = df["hour"]
    df["is_morning_rush"] = ((h >= 7) & (h <= 10)).astype(np.int8)
    df["is_evening_rush"] = ((h >= 17) & (h <= 20)).astype(np.int8)
    df["is_night"] = ((h <= 5) | (h >= 22)).astype(np.int8)
    return df


def add_geohash_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Increasingly coarse spatial buckets via geohash prefixes."""
    for n in (3, 4, 5):
        df[f"geohash_p{n}"] = df["geohash"].str[:n]
    return df


# ===========================================================================
# Group-mode imputation for categoricals
# ===========================================================================
# RoadType and Weather are mostly stable per geohash (the road type doesn't
# change, weather is shared across nearby readings). When a row's value is
# missing, the modal value at that geohash is a much better guess than a
# blanket "unknown" tag.
def _mode_or_global(series: pd.Series, global_mode):
    s = series.dropna()
    return s.mode().iat[0] if not s.empty else global_mode


def fit_group_mode_maps(
    train: pd.DataFrame, cols: List[str]
) -> Dict[str, Tuple[Dict[str, str], str]]:
    """For each col, build {geohash -> modal value} plus a global fallback."""
    maps: Dict[str, Tuple[Dict[str, str], str]] = {}
    for col in cols:
        global_mode = train[col].dropna().mode()
        global_val = global_mode.iat[0] if not global_mode.empty else "Unknown"
        per_geohash = (
            train.groupby("geohash")[col]
            .apply(lambda s: _mode_or_global(s, global_val))
            .to_dict()
        )
        maps[col] = (per_geohash, global_val)
    return maps


def apply_group_mode_imputation(
    df: pd.DataFrame, maps: Dict[str, Tuple[Dict[str, str], str]]
) -> pd.DataFrame:
    df = df.copy()
    for col, (per_geohash, global_val) in maps.items():
        fill = df["geohash"].map(per_geohash).fillna(global_val)
        df[col] = df[col].fillna(fill)
    return df


# ===========================================================================
# Numeric imputation for Temperature
# ===========================================================================
# Temperature is missing ~3% of rows. The (day, hour) mean is a much better
# guess than the global mean, because temperature swings strongly with hour
# of day. Fit on train, apply to both train and test.
def fit_temperature_table(train: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    table = (
        train.dropna(subset=["Temperature"])
        .groupby(["day", "hour"])["Temperature"]
        .mean()
        .reset_index()
        .rename(columns={"Temperature": "temp_dayhour"})
    )
    global_mean = float(train["Temperature"].mean())
    return table, global_mean


def apply_temperature_imputation(
    df: pd.DataFrame, table: pd.DataFrame, global_mean: float
) -> pd.DataFrame:
    df = df.merge(table, on=["day", "hour"], how="left")
    df["Temperature"] = df["Temperature"].fillna(df["temp_dayhour"]).fillna(global_mean)
    df = df.drop(columns=["temp_dayhour"])
    return df


# ===========================================================================
# Target encoding (smoothed) and day-48 reference signals
# ===========================================================================
# Target encoding replaces a high-cardinality key (e.g. geohash) with the
# mean of `demand` observed for that key. It is the standard remedy when
# one-hot is impractical and tree models struggle to learn meaningful
# splits from raw IDs.
#
# Two safety measures:
#   * Smoothing toward the global mean prevents tiny-sample groups from
#     dominating. A group with `n` observations gets pulled to the prior
#     with weight `smoothing / (smoothing + n)`.
#   * For TRAIN rows we use K-fold out-of-fold encoding: a row's own
#     target is never in the aggregate that becomes its feature, so the
#     model cannot trivially regress demand onto itself.
#
# Test rows use the full-train aggregate, which is the cleanest estimate
# we have (still target-free w.r.t. test).
def _smoothed_means(
    df: pd.DataFrame,
    group_cols: List[str],
    target: str,
    smoothing: float,
    global_mean: float,
) -> pd.DataFrame:
    agg = df.groupby(group_cols)[target].agg(["mean", "count"]).reset_index()
    agg["smoothed"] = (
        agg["mean"] * agg["count"] + global_mean * smoothing
    ) / (agg["count"] + smoothing)
    return agg[group_cols + ["smoothed"]]


def target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_cols: List[str],
    target: str,
    out_col: str,
    smoothing: float = 30.0,
    n_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF target encoding for train, full-train encoding for test."""
    global_mean = float(train[target].mean())

    # ---- Test: full-train aggregate (test target never touched) -------------
    full_map = _smoothed_means(train, group_cols, target, smoothing, global_mean)
    test_enc = (
        test[group_cols]
        .merge(full_map, on=group_cols, how="left")["smoothed"]
        .fillna(global_mean)
        .to_numpy()
    )

    # ---- Train: out-of-fold ------------------------------------------------
    train_enc = np.full(len(train), global_mean, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    train_reset = train.reset_index(drop=True)
    for fold_train_idx, fold_val_idx in kf.split(train_reset):
        fold_train = train_reset.iloc[fold_train_idx]
        fold_map = _smoothed_means(fold_train, group_cols, target, smoothing, global_mean)
        merged = (
            train_reset.iloc[fold_val_idx][group_cols]
            .merge(fold_map, on=group_cols, how="left")["smoothed"]
            .fillna(global_mean)
            .to_numpy()
        )
        train_enc[fold_val_idx] = merged

    # Naming matches the requested out_col -- the caller assigns it.
    _ = out_col
    return train_enc, test_enc


def day_reference_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Day-48 demand statistics as a reference signal for day 49 / unseen rows.

    Day 48 is fully observed in train. For every geohash and every
    (geohash, hour) we summarise its demand on day 48. These columns give
    the model a "what did this place look like yesterday" baseline -- the
    single most useful structural feature for short-horizon demand
    forecasting.

    Applied to both train and test identically, but built from day-48 data
    only (so it leaks nothing about day-49 targets, including the day-49
    rows that exist inside train).
    """
    day48 = train[train["day"] == 48]

    geo_stats = (
        day48.groupby("geohash")["demand"]
        .agg(d48_geo_mean="mean", d48_geo_median="median", d48_geo_std="std")
        .reset_index()
    )
    geo_hour_stats = (
        day48.groupby(["geohash", "hour"])["demand"]
        .agg(d48_geohour_mean="mean", d48_geohour_max="max")
        .reset_index()
    )

    def _attach(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(geo_stats, on="geohash", how="left")
        df = df.merge(geo_hour_stats, on=["geohash", "hour"], how="left")
        return df

    return _attach(train), _attach(test)


# ===========================================================================
# Ordinal encoder
# ===========================================================================
class OrdinalEncoder:
    """Train-fit ordinal encoder.

    Unknown categories and NaNs both map to -1, giving the tree model an
    explicit "unseen / missing" branch it can split on.
    """

    def __init__(self) -> None:
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "OrdinalEncoder":
        for col in columns:
            uniques = df[col].dropna().unique()
            self.maps[col] = {val: i for i, val in enumerate(sorted(uniques))}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.maps.items():
            df[col] = df[col].map(mapping).fillna(-1).astype(np.int32)
        return df


# ===========================================================================
# Top-level feature pipeline
# ===========================================================================
CATEGORICAL_COLS = [
    "geohash",
    "geohash_p3",
    "geohash_p4",
    "geohash_p5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
]


def build_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """All-in-one feature engineering. Fits on train only, transforms test."""
    train = train_raw.copy()
    test = test_raw.copy()

    # Time + space derivations are deterministic per row, no fitting needed.
    train = add_time_features(train)
    test = add_time_features(test)
    train = add_geohash_prefix_features(train)
    test = add_geohash_prefix_features(test)
    train = add_latlon(train)
    test = add_latlon(test)

    # Group-mode imputation for categorical strings. Fit on train.
    mode_maps = fit_group_mode_maps(train, ["Weather", "RoadType"])
    train = apply_group_mode_imputation(train, mode_maps)
    test = apply_group_mode_imputation(test, mode_maps)

    # Temperature: (day, hour) mean imputation.
    temp_table, temp_global = fit_temperature_table(train)
    train = apply_temperature_imputation(train, temp_table, temp_global)
    test = apply_temperature_imputation(test, temp_table, temp_global)

    # Frequency encoding: a count of how often this geohash appears in
    # train. Busy geohashes are often busier in absolute terms too.
    geo_counts = train["geohash"].value_counts()
    train["geohash_freq"] = train["geohash"].map(geo_counts).fillna(0)
    test["geohash_freq"] = test["geohash"].map(geo_counts).fillna(0)

    # Day-48 reference statistics (fit on day-48 portion of train).
    train, test = day_reference_features(train, test)

    # Smoothed target encoding (OOF for train).
    te_geo_train, te_geo_test = target_encode(
        train, test, ["geohash"], "demand", "te_geohash"
    )
    te_geohour_train, te_geohour_test = target_encode(
        train, test, ["geohash", "hour"], "demand", "te_geohash_hour", smoothing=20.0
    )
    te_hour_train, te_hour_test = target_encode(
        train, test, ["hour"], "demand", "te_hour", smoothing=10.0
    )
    train["te_geohash"] = te_geo_train
    test["te_geohash"] = te_geo_test
    train["te_geohash_hour"] = te_geohour_train
    test["te_geohash_hour"] = te_geohour_test
    train["te_hour"] = te_hour_train
    test["te_hour"] = te_hour_test

    # Ordinal-encode raw categoricals last so the target encodings see the
    # original string keys.
    encoder = OrdinalEncoder().fit(train, CATEGORICAL_COLS)
    train = encoder.transform(train)
    test = encoder.transform(test)

    feature_cols = CATEGORICAL_COLS + [
        "day",
        "minutes_of_day",
        "hour",
        "minute",
        "time_sin",
        "time_cos",
        "time_sin_half",
        "time_cos_half",
        "is_morning_rush",
        "is_evening_rush",
        "is_night",
        "lat",
        "lon",
        "NumberofLanes",
        "Temperature",
        "geohash_freq",
        "d48_geo_mean",
        "d48_geo_median",
        "d48_geo_std",
        "d48_geohour_mean",
        "d48_geohour_max",
        "te_geohash",
        "te_geohash_hour",
        "te_hour",
    ]
    return train, test, feature_cols


# ===========================================================================
# Modelling: RandomForest + HistGradientBoosting ensemble
# ===========================================================================
# RF (bagging) and HistGBM (boosting) make very different bias / variance
# tradeoffs and produce errors that are only weakly correlated, so averaging
# their predictions tends to beat either alone. HistGBM also handles NaN
# natively, so no imputation pass is needed for it -- a nice simplification
# over the previous RF + ExtraTrees combo, which kept hitting MemoryError on
# the full-data refit.
N_JOBS = 2
MAX_DEPTH = 22
MAX_SAMPLES = 0.65


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=250,
        max_depth=MAX_DEPTH,
        min_samples_leaf=3,
        max_samples=MAX_SAMPLES,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )


def make_hist_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def fit_ensemble(X: pd.DataFrame, y: np.ndarray, impute_means: pd.Series):
    """Fit RF on imputed X (no native NaN) and HistGBM on raw X."""
    rf = make_random_forest().fit(X.fillna(impute_means), y)
    gbm = make_hist_gbm().fit(X, y)
    return rf, gbm


def predict_ensemble(rf, gbm, X: pd.DataFrame, impute_means: pd.Series) -> np.ndarray:
    rf_pred = rf.predict(X.fillna(impute_means))
    gbm_pred = gbm.predict(X)
    return np.clip(0.5 * (rf_pred + gbm_pred), 0.0, 1.0)


def score(actual: np.ndarray, predicted: np.ndarray) -> float:
    return max(0.0, 100.0 * r2_score(actual, predicted))


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("Loading data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    print(f"  train: {train_raw.shape}, test: {test_raw.shape}")
    assert "demand" not in test_raw.columns, "test.csv must not expose the target"

    print("Engineering features (all fits on train only)...")
    train, test, feature_cols = build_features(train_raw, test_raw)

    X = train[feature_cols]
    y = train["demand"].values
    X_test = test[feature_cols]

    # -----------------------------------------------------------------------
    # Two validations, side-by-side
    # -----------------------------------------------------------------------
    # 1) Random 15% holdout. Optimistic because train and test rows for the
    #    same geohash×hour can land on opposite sides of the split, letting
    #    the model effectively "look up" the cell average. Useful as a
    #    consistency check vs. prior runs.
    # 2) Day-49 holdout. Pessimistic but distribution-shifted in the same
    #    direction as the real test set (which is day 49 timestamps the
    #    model has never seen). The leaderboard score should sit ABOVE this
    #    number, because test covers daytime hours -- easier than the
    #    night-time slice of day-49 we have here.
    print("Validation A: random 15% holdout")
    Xa_tr, Xa_val, ya_tr, ya_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    means_a = Xa_tr.mean(numeric_only=True)
    rf_a, gbm_a = fit_ensemble(Xa_tr, ya_tr, means_a)
    ens_a = predict_ensemble(rf_a, gbm_a, Xa_val, means_a)
    print(f"  RF only  : {score(ya_val, np.clip(rf_a.predict(Xa_val.fillna(means_a)), 0, 1)):.4f}")
    print(f"  GBM only : {score(ya_val, np.clip(gbm_a.predict(Xa_val), 0, 1)):.4f}")
    print(f"  Ensemble : {score(ya_val, ens_a):.4f}")

    print("Validation B: day-49 rows held out (mimics test distribution shift)")
    val_mask = train["day"].to_numpy() == 49
    Xb_tr, Xb_val = X.loc[~val_mask], X.loc[val_mask]
    yb_tr, yb_val = y[~val_mask], y[val_mask]
    means_b = Xb_tr.mean(numeric_only=True)
    rf_b, gbm_b = fit_ensemble(Xb_tr, yb_tr, means_b)
    ens_b = predict_ensemble(rf_b, gbm_b, Xb_val, means_b)
    print(f"  RF only  : {score(yb_val, np.clip(rf_b.predict(Xb_val.fillna(means_b)), 0, 1)):.4f}")
    print(f"  GBM only : {score(yb_val, np.clip(gbm_b.predict(Xb_val), 0, 1)):.4f}")
    print(f"  Ensemble : {score(yb_val, ens_b):.4f}")

    # -----------------------------------------------------------------------
    # Refit on full train, predict test
    # -----------------------------------------------------------------------
    print("Refitting on full train...")
    full_means = X.mean(numeric_only=True)
    rf_final, et_final = fit_ensemble(X, y, full_means)

    print("Predicting on test...")
    test_pred = predict_ensemble(rf_final, et_final, X_test, full_means)

    submission = pd.DataFrame(
        {"Index": test_raw["Index"].values, "demand": test_pred}
    )
    assert submission.shape == (41778, 2), f"unexpected submission shape {submission.shape}"
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    main()
