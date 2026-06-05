"""
Traffic Demand Prediction
=========================

Predicts the `demand` value for each row in `dataset/test.csv` after training
a regression model on `dataset/train.csv`. The test file's target column is
never read, so the model has no opportunity to peek at it.

Approach overview
-----------------
1. Load train and test side-by-side. The test set is used ONLY for its
   feature columns -- never for the target (which it doesn't contain anyway).
2. Engineer features:
   - Parse the `timestamp` ("H:M") into minutes-since-midnight plus cyclic
     sin/cos encodings, which let a tree model represent time-of-day patterns
     cleanly (rush hour, late night, etc.).
   - Split `geohash` into prefix levels of increasing precision. The 6-char
     geohash is high-cardinality (~1.2k unique values), so we let the model
     learn coarse spatial buckets via the shorter prefixes and fine-grained
     location via the full code.
   - Ordinal-encode all categorical strings; missing categories get a
     dedicated -1 code so the tree can learn a "value missing" split.
   - Numeric NAs (Temperature) are left as NaN -- XGBoost handles missing
     values natively, which is strictly better than imputing a constant.
3. Train an XGBoost gradient-boosted regressor. Tree boosters are a strong
   default for tabular data of this size with mixed feature types.
4. Evaluate with a held-out slice of the TRAIN data (no test leakage) using
   the competition metric: max(0, 100 * R^2).
5. Refit on the full train set and write predictions to `submission.csv`.

The categorical encoder and feature pipeline are fit on TRAIN ONLY and then
applied to TEST. This is what guarantees there is no information leakage
from the test set into the model.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def parse_timestamp_to_minutes(ts: pd.Series) -> pd.Series:
    """Convert "H:M" strings into minutes-since-midnight as an int."""
    # str.split avoids a slow per-row apply; 'expand=True' yields two columns.
    parts = ts.astype(str).str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add minutes-of-day plus cyclic encodings.

    Cyclic encodings (sin/cos over the 24h period) give the model a smooth
    notion of "11pm is close to midnight", which a raw minute count cannot
    convey by itself.
    """
    minutes = parse_timestamp_to_minutes(df["timestamp"])
    df["minutes_of_day"] = minutes
    df["hour"] = minutes // 60
    radians = 2.0 * np.pi * minutes / (24 * 60)
    df["time_sin"] = np.sin(radians)
    df["time_cos"] = np.cos(radians)
    return df


def add_geohash_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Geohash prefixes act as increasingly coarse spatial buckets.

    A geohash like "qp02zt" shares its first N characters with everything in
    the same spatial cell at that precision. Exposing prefixes lets the
    model learn "this whole area is busy" without seeing every full code.
    """
    for n in (3, 4, 5):
        df[f"geohash_p{n}"] = df["geohash"].str[:n]
    return df


class OrdinalEncoder:
    """Train-fit ordinal encoder for categorical strings.

    Each known category maps to a non-negative integer. Unknown categories
    encountered at predict time and NaNs both map to -1, giving the tree
    model an explicit "unseen/missing" branch it can split on.
    """

    def __init__(self) -> None:
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "OrdinalEncoder":
        for col in columns:
            uniques = df[col].dropna().unique()
            # Sort for determinism across runs.
            self.maps[col] = {val: i for i, val in enumerate(sorted(uniques))}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.maps.items():
            df[col] = df[col].map(mapping).fillna(-1).astype(np.int32)
        return df


def build_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Apply identical feature engineering to train and test.

    The encoder is FIT on train only; test is merely transformed. This is
    the guard that prevents test data from influencing the model.
    """
    train = add_time_features(train_raw.copy())
    test = add_time_features(test_raw.copy())

    train = add_geohash_prefix_features(train)
    test = add_geohash_prefix_features(test)

    categorical_cols = [
        "geohash",
        "geohash_p3",
        "geohash_p4",
        "geohash_p5",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
    ]
    encoder = OrdinalEncoder().fit(train, categorical_cols)
    train = encoder.transform(train)
    test = encoder.transform(test)

    feature_cols = categorical_cols + [
        "day",
        "minutes_of_day",
        "hour",
        "time_sin",
        "time_cos",
        "NumberofLanes",
        "Temperature",
    ]
    return train, test, feature_cols


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
# The final model is a simple average of a RandomForest and an ExtraTrees
# regressor. Holdout comparison (see compare_models.py) showed both as the
# two strongest candidates; averaging them reduces variance further because
# the two algorithms make different randomization choices when splitting
# (RF picks the best split among a random feature subset; ExtraTrees picks
# a random split threshold). Their errors are partially decorrelated, so
# the mean prediction tends to beat either model alone.


# Parallelism is intentionally bounded. With `n_jobs=-1` the forests spawn
# one worker per CPU, each holding a full copy of the working arrays; on
# this machine that overflowed RAM during the full-data refit. n_jobs=2
# plus a hard depth cap and bagged subsampling (max_samples) keeps the
# per-tree memory footprint well under the available budget.
N_JOBS = 2
MAX_DEPTH = 24            # caps the worst-case tree size
MAX_SAMPLES = 0.7         # each tree sees 70% of training rows


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        max_samples=MAX_SAMPLES,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )


def make_extra_trees() -> ExtraTreesRegressor:
    """ExtraTrees uses random split thresholds for additional diversity.

    `bootstrap=True` is required for `max_samples` to take effect on
    ExtraTrees (unlike RandomForest, which bootstraps by default).
    """
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        bootstrap=True,
        max_samples=MAX_SAMPLES,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )


def fit_ensemble(X: pd.DataFrame, y: np.ndarray, impute_means: pd.Series):
    """Fit RF on raw X (NaNs allowed) and ExtraTrees on mean-imputed X.

    ExtraTrees does not accept NaN in this sklearn version, so we impute
    using means computed from the TRAINING data only (passed in via
    `impute_means`). The same means must be reused at predict time -- that
    is what keeps the test set blind to its own statistics.
    """
    rf = make_random_forest().fit(X, y)
    et = make_extra_trees().fit(X.fillna(impute_means), y)
    return rf, et


def predict_ensemble(rf, et, X: pd.DataFrame, impute_means: pd.Series) -> np.ndarray:
    """Average RF and ExtraTrees predictions, clipped to the valid range.

    A simple 50/50 mean is a strong baseline ensemble: it assumes neither
    model dominates and lets their errors (which come from different
    randomization mechanisms) partially cancel. If a single model is
    clearly stronger, weighting could help -- but that adds a hyperparameter
    chosen from the holdout, which the user did not ask for.
    """
    rf_pred = rf.predict(X)
    et_pred = et.predict(X.fillna(impute_means))
    return np.clip(0.5 * (rf_pred + et_pred), 0.0, 1.0)


def score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Competition metric: clipped 100 * R^2."""
    return max(0.0, 100.0 * r2_score(actual, predicted))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    print(f"  train: {train_raw.shape}, test: {test_raw.shape}")

    # Sanity check: confirm the test file does NOT contain the target.
    # If it did, we would refuse to read it during training anyway.
    assert "demand" not in test_raw.columns, "test.csv must not expose the target"

    print("Engineering features (encoder fit on train only)...")
    train, test, feature_cols = build_features(train_raw, test_raw)

    X = train[feature_cols]
    y = train["demand"].values
    X_test = test[feature_cols]

    # -----------------------------------------------------------------------
    # Hold-out validation
    # -----------------------------------------------------------------------
    # We split the TRAIN set only -- the real test set stays untouched.
    # A random split is reasonable here because train and test share the same
    # period (day 49 appears in both, with non-overlapping timestamps within
    # the day), so within-train randomness reflects the deployment setting.
    print("Validating on a held-out slice of train...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    # Imputation means come from the training split only so the holdout
    # mirrors the train/test relationship (no leakage from validation rows).
    val_means = X_tr.mean(numeric_only=True)
    rf_val, et_val = fit_ensemble(X_tr, y_tr, val_means)
    rf_only = np.clip(rf_val.predict(X_val), 0.0, 1.0)
    et_only = np.clip(et_val.predict(X_val.fillna(val_means)), 0.0, 1.0)
    ens_pred = predict_ensemble(rf_val, et_val, X_val, val_means)
    print(f"  RandomForest only : {score(y_val, rf_only):.4f}")
    print(f"  ExtraTrees   only : {score(y_val, et_only):.4f}")
    print(f"  Ensemble (mean)   : {score(y_val, ens_pred):.4f}")

    # -----------------------------------------------------------------------
    # Final fit on full training data, then predict on test
    # -----------------------------------------------------------------------
    print("Refitting ensemble on full training data...")
    full_means = X.mean(numeric_only=True)
    rf_final, et_final = fit_ensemble(X, y, full_means)

    print("Predicting on test...")
    test_pred = predict_ensemble(rf_final, et_final, X_test, full_means)

    submission = pd.DataFrame(
        {"Index": test_raw["Index"].values, "demand": test_pred}
    )
    # Guard against silently shipping a wrong-sized submission.
    assert submission.shape == (41778, 2), f"unexpected submission shape {submission.shape}"
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    main()
