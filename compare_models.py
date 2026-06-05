"""
Model comparison for Traffic Demand Prediction
==============================================

Trains several regressors on the SAME train/holdout split (built only from
`dataset/train.csv`) and prints the competition metric `max(0, 100 * R^2)`
side-by-side. The test set is never read here -- this script is purely for
choosing a model family before locking in `solution.py`.

Models evaluated:
- Ridge ............. linear baseline; establishes the floor.
- DecisionTree ...... single tree; shows how much boosting/bagging buys us.
- RandomForest ...... bagged trees; robust, low-variance reference point.
- ExtraTrees ........ randomized splits; often a touch faster than RF.
- HistGradientBoost . sklearn's histogram-based GBM; native NaN handling.
- XGBoost ........... gradient boosting; current production choice.

The feature pipeline is the SAME one used in solution.py (imported, not
duplicated), so differences in score are attributable to the model.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from solution import (
    RANDOM_STATE,
    TEST_PATH,
    TRAIN_PATH,
    build_features,
    score,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry is a zero-arg factory so we get a fresh, unfitted estimator per
# run. Hyperparameters are kept modest and comparable -- the goal here is
# fair comparison of model families, not per-model tuning.
MODEL_FACTORIES: Dict[str, Callable[[], object]] = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "DecisionTree": lambda: DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE
    ),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "ExtraTrees": lambda: ExtraTreesRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoost": lambda: HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def prepare_split() -> tuple:
    """Load train, engineer features, return a fixed 85/15 holdout split.

    The split seed matches solution.py so the XGBoost number here lines up
    with what the production script prints.
    """
    train_raw = pd.read_csv(TRAIN_PATH)
    # We still load test so the feature builder can be exercised the same
    # way as in production -- but we never use test labels (it has none).
    test_raw = pd.read_csv(TEST_PATH)
    train, _test, feature_cols = build_features(train_raw, test_raw)

    X = train[feature_cols]
    y = train["demand"].values
    return train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)


def evaluate(name: str, factory: Callable[[], object], split) -> Dict[str, float]:
    """Fit `factory()` on the training split, score on the holdout."""
    X_tr, X_val, y_tr, y_val = split

    # Some sklearn estimators (Ridge, ExtraTrees) cannot consume NaNs.
    # Fill with the train-column mean ONLY for those, so the comparison
    # stays apples-to-apples; XGBoost/HistGBM/RandomForest get raw NaNs.
    needs_imputation = name in {"Ridge", "ExtraTrees"}
    if needs_imputation:
        means = X_tr.mean(numeric_only=True)
        X_tr_used = X_tr.fillna(means)
        X_val_used = X_val.fillna(means)
    else:
        X_tr_used, X_val_used = X_tr, X_val

    model = factory()
    t0 = time.perf_counter()
    model.fit(X_tr_used, y_tr)
    fit_secs = time.perf_counter() - t0

    pred = np.clip(model.predict(X_val_used), 0.0, 1.0)
    return {
        "model": name,
        "score": score(y_val, pred),
        "raw_r2": r2_score(y_val, pred),
        "fit_secs": fit_secs,
    }


def main() -> None:
    print("Preparing data (encoder fit on train only)...")
    split = prepare_split()
    print(f"  train rows: {len(split[0])}, holdout rows: {len(split[1])}")
    print()

    results = []
    for name, factory in MODEL_FACTORIES.items():
        print(f"Training {name}...")
        results.append(evaluate(name, factory, split))

    table = pd.DataFrame(results).sort_values("score", ascending=False)
    table["score"] = table["score"].round(4)
    table["raw_r2"] = table["raw_r2"].round(4)
    table["fit_secs"] = table["fit_secs"].round(2)

    print()
    print("Holdout results (metric = max(0, 100 * R^2)):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
