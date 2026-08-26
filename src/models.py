"""Stages 5-6 - models.

Order is deliberate: empirical baseline -> logistic -> boosted trees. Each must
beat the previous one on purged CV or it does not ship. A simpler model you can
explain is worth more than a marginal AUC bump you cannot.

Capacity is kept deliberately low (depth 3-4, heavy regularisation). The binding
constraint in this problem is signal-to-noise, not model capacity; extra capacity
buys memorised noise, not edge. Twelve LightGBM configurations were tried and all
twelve had NEGATIVE Brier skill - worse calibrated than a constant. XGBoost and
the GPU path were dropped for the same reason: at this data scale a single fit is
small, the GPU never helped, and an unused dependency is a liability.

Calibration is not optional. A model whose probability output is not reliable is
useless for sizing, whatever its AUC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_JOBS, RANDOM_SEED
from .validation import PurgedSplit, average_uniqueness, evaluate

@dataclass
class ModelSpec:
    name: str
    kind: str                      # 'logit' | 'lgbm' | 'rf'
    params: dict = field(default_factory=dict)
    calibrate: bool = True

    def build(self):
        if self.kind == "logit":
            p = dict(C=0.05, max_iter=2000, solver="lbfgs")
            p.update(self.params)
            return Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(**p))])
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            p = dict(n_estimators=200, max_depth=None, min_samples_leaf=2,
                     n_jobs=N_JOBS, random_state=RANDOM_SEED)
            p.update(self.params)
            return RandomForestClassifier(**p)
        if self.kind == "lgbm":
            import lightgbm as lgb
            p = dict(n_estimators=400, learning_rate=0.03, max_depth=4, num_leaves=15,
                     min_child_samples=200, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                     n_jobs=N_JOBS, random_state=RANDOM_SEED, verbosity=-1)
            p.update(self.params)
            return lgb.LGBMClassifier(**p)
        raise ValueError(f"unknown kind {self.kind}")


def _clean(X: pd.DataFrame, medians: pd.Series | None = None):
    """Median-impute and clip. Trees tolerate raw NaNs, logistic does not, and a
    single infinite ratio can silently destroy a StandardScaler."""
    X = X.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    # A column that is entirely NaN within this slice has a NaN median, so the
    # fill above leaves it NaN and the fit dies. Happens on family subsets where
    # a feature never applies. Zero is the right filler post-standardisation.
    X = X.fillna(0.0)
    return X.clip(lower=X.quantile(0.001), upper=X.quantile(0.999), axis=1), medians


def fit_predict(spec: ModelSpec, X_tr: pd.DataFrame, y_tr: np.ndarray,
                w_tr: np.ndarray, X_te: pd.DataFrame,
                t0_tr: np.ndarray | None = None) -> np.ndarray:
    """Fit with an inner time-ordered calibration holdout, predict on test."""
    X_tr_c, med = _clean(X_tr)
    X_te_c, _ = _clean(X_te, med)

    if not spec.calibrate or len(X_tr_c) < 2000:
        m = spec.build()
        m.fit(X_tr_c, y_tr, **_sw(spec, w_tr))
        return m.predict_proba(X_te_c)[:, 1]

    # inner split is chronological, never random - a random inner split would
    # leak the same way an unpurged outer split does
    order = np.argsort(t0_tr) if t0_tr is not None else np.arange(len(X_tr_c))
    cut = int(0.85 * len(order))
    fit_i, cal_i = order[:cut], order[cut:]
    if len(cal_i) < 500 or len(np.unique(y_tr[cal_i])) < 2:
        fit_i, cal_i = order, order

    m = spec.build()
    m.fit(X_tr_c.iloc[fit_i], y_tr[fit_i], **_sw(spec, w_tr[fit_i]))
    p_cal = m.predict_proba(X_tr_c.iloc[cal_i])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_cal, y_tr[cal_i], sample_weight=w_tr[cal_i])

    raw = m.predict_proba(X_te_c)[:, 1]
    return np.clip(iso.predict(raw), 1e-6, 1 - 1e-6)


def _sw(spec: ModelSpec, w: np.ndarray) -> dict:
    if spec.kind == "logit":
        return {"lr__sample_weight": w}
    return {"sample_weight": w}


def run_cv(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, r: np.ndarray,
           t0: np.ndarray, t1: np.ndarray, splits: list[PurgedSplit],
           use_weights: bool = True, verbose: bool = False) -> dict:
    """Run a model across purged splits; return pooled + per-split metrics."""
    w_all = average_uniqueness(t0, t1) if use_weights else np.ones(len(y))
    w_all = w_all / w_all.mean()

    oof_p, oof_i, per_split = [], [], []
    for k, sp in enumerate(splits):
        if len(sp.train) < 1000 or len(sp.test) < 200:
            continue
        if len(np.unique(y[sp.train])) < 2:
            continue
        p = fit_predict(spec, X.iloc[sp.train], y[sp.train], w_all[sp.train],
                        X.iloc[sp.test], t0[sp.train])
        oof_p.append(p)
        oof_i.append(sp.test)
        try:
            per_split.append(evaluate(p, y[sp.test], r[sp.test], w_all[sp.test]))
        except Exception:
            pass
        if verbose:
            print(f"    split {k:>2}  n_tr={len(sp.train):>7,}  n_te={len(sp.test):>6,}"
                  f"  auc={per_split[-1]['auc']:.4f}  rtop={per_split[-1]['r_top_decile']:+.3f}")

    if not oof_p:
        return {"n": 0}
    p = np.concatenate(oof_p)
    i = np.concatenate(oof_i)
    res = evaluate(p, y[i], r[i], w_all[i])
    res["model"] = spec.name

    if per_split:
        df = pd.DataFrame(per_split)
        for col in ("auc", "brier_skill", "r_top_decile", "r_spread"):
            res[f"{col}_split_mean"] = float(df[col].mean())
            res[f"{col}_split_std"] = float(df[col].std())
        # share of splits where the top decile was actually profitable
        res["pct_splits_top_positive"] = float((df["r_top_decile"] > 0).mean())
    res["_oof_p"] = p
    res["_oof_i"] = i
    return res
