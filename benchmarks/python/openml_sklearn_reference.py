#!/usr/bin/env python3
"""Sklearn baseline for OpenML ids 46898 / 45046 / 1597 (same as MLBenchmarks :sberbank, :allstate_claims, :creditcard).

  pip install -r benchmarks/python/requirements.txt
  python benchmarks/python/openml_sklearn_reference.py sberbank

Note: this is a single-config baseline, not a hyperparam sweep — numbers should not be
compared directly against the best-of-N results from the Julia benchmark drivers.
Seed matches Julia's `seed=123` but Julia uses Xoshiro and NumPy uses Mersenne Twister,
so row-level partitions differ; fractions and aggregate behavior are the parity target.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

OPENML_IDS = {
    "sberbank": 46898,
    "allstate_claims": 45046,
    "creditcard": 1597,
}


def mlbenchmarks_style_split(
    n_rows: int, seed: int = 123, eval_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_rows)
    eval_cut = int(np.floor((1.0 - eval_frac - test_frac) * n_rows))
    test_cut = int(np.floor((1.0 - test_frac) * n_rows))
    return idx[:eval_cut], idx[eval_cut:test_cut], idx[test_cut:]


def load_allstate():
    Xy = fetch_openml(data_id=OPENML_IDS["allstate_claims"], as_frame=True, parser="auto")
    df = Xy.frame
    y = df["loss"].astype(float).values
    X = df.drop(columns=["loss"])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = (y - y.mean()) / y.std()
    return X, y


def load_sberbank():
    Xy = fetch_openml(data_id=OPENML_IDS["sberbank"], as_frame=True, parser="auto")
    df = Xy.frame
    y_raw = pd.to_numeric(df["price_doc"], errors="coerce").astype(float).values
    X = df.drop(columns=["price_doc"])
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = np.nanmedian(X[c].values)
        X[c] = X[c].fillna(med)
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
    y = (y_raw - y_raw.mean()) / y_raw.std()
    return X, y


def load_creditcard():
    Xy = fetch_openml(data_id=OPENML_IDS["creditcard"], as_frame=True, parser="auto")
    df = Xy.frame
    target_col = "Class" if "Class" in df.columns else "class"
    y = pd.to_numeric(df[target_col], errors="coerce").astype(int).values
    X = df.drop(columns=[target_col])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = np.nanmedian(X[c].values)
        X[c] = X[c].fillna(med)
    return X, y


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", default="sberbank", choices=list(OPENML_IDS.keys()))
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    if args.dataset == "allstate_claims":
        X_df, y = load_allstate()
        task = "regression"
    elif args.dataset == "sberbank":
        X_df, y = load_sberbank()
        task = "regression"
    else:
        X_df, y = load_creditcard()
        task = "classification"

    train_i, eval_i, test_i = mlbenchmarks_style_split(len(y), seed=args.seed)
    fit_i = np.concatenate([train_i, eval_i])
    X_fit, X_test = X_df.iloc[fit_i], X_df.iloc[test_i]
    y_fit, y_test = y[fit_i], y[test_i]
    val_fraction = len(eval_i) / len(fit_i)

    pre = build_preprocess(X_df)
    common = dict(
        max_depth=8,
        learning_rate=0.1,
        max_iter=400,
        early_stopping=True,
        validation_fraction=val_fraction,
        n_iter_no_change=10,
        random_state=args.seed,
    )
    if task == "regression":
        pipe = Pipeline([("prep", pre), ("gb", HistGradientBoostingRegressor(**common))])
        t0 = time.perf_counter()
        pipe.fit(X_fit, y_fit)
        fit_s = time.perf_counter() - t0
        pred = pipe.predict(X_test)
        score = mean_squared_error(y_test, pred)
        n_iter = pipe.named_steps["gb"].n_iter_
        print(
            f"dataset={args.dataset} fit_wall_time_s={fit_s:.4f} n_iter={n_iter} "
            f"test_mse={score:.6f}"
        )
    else:
        pipe = Pipeline([("prep", pre), ("gb", HistGradientBoostingClassifier(**common))])
        t0 = time.perf_counter()
        pipe.fit(X_fit, y_fit)
        fit_s = time.perf_counter() - t0
        proba = pipe.predict_proba(X_test)[:, 1]
        ll = log_loss(y_test, proba)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        n_iter = pipe.named_steps["gb"].n_iter_
        print(
            f"dataset={args.dataset} fit_wall_time_s={fit_s:.4f} n_iter={n_iter} "
            f"test_log_loss={ll:.6f} test_roc_auc={auc:.6f}"
        )


if __name__ == "__main__":
    main()
