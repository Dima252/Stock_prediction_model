"""Does a different TARGET help? Dev-only, purged CV.

Everything so far predicts P(target hit first). But the decision rule is on
expectancy, and those are not the same object: a timeout that drifts to -0.4R and
a clean stop at -1.0R are both "not a win", yet they are very different trades.

Three formulations, same features, same splits:
  A. classify  win        - target hit before stop            (what we have)
  B. classify  r_net > 0  - simply "did this trade make money"
  C. regress   r_net      - expectancy directly

DISCIPLINE: this runs on DEV ONLY. The 2021-2025 holdout has already been spent
once. Every extra look at it costs statistical validity, so a variant that wins
here does NOT get a free re-test - that trade-off is stated, not hidden.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import EMBARGO_PCT, N_CV_SPLITS, N_JOBS, PROCESSED, RANDOM_SEED, REPORTS, VALID_END
from src.features import FEATURE_COLS
from src.models import ModelSpec, _clean, fit_predict
from src.validation import (average_uniqueness, block_bootstrap_ci,
                            expectancy_by_decile, purged_kfold)

pd.set_option("display.width", 240)


def design(df):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].astype(np.float32).copy()
    for s in sorted(df["setup"].unique()):
        X[f"setup_{s}"] = (df["setup"] == s).astype(np.int8)
    return X


def run_regression(X, target, r, t0, t1, splits, w, kind="ridge"):
    """Rank trades by predicted r_net instead of predicted P(win)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    ps, idx = [], []
    for sp in splits:
        if len(sp.train) < 1000 or len(sp.test) < 200:
            continue
        Xtr, med = _clean(X.iloc[sp.train])
        Xte, _ = _clean(X.iloc[sp.test], med)
        if kind == "ridge":
            m = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1000.0))])
            m.fit(Xtr, target[sp.train], m__sample_weight=w[sp.train])
        else:
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                  num_leaves=7, min_child_samples=400,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                  reg_lambda=5.0, n_jobs=N_JOBS,
                                  random_state=RANDOM_SEED, verbosity=-1)
            m.fit(Xtr, target[sp.train], sample_weight=w[sp.train])
        ps.append(m.predict(Xte)); idx.append(sp.test)
    p = np.concatenate(ps); i = np.concatenate(idx)
    return p, i


def summarise(name, p, i, r, dates):
    dec = expectancy_by_decile(p, r[i])
    top = r[i][p >= np.quantile(p, 0.9)]
    td = dates[i][p >= np.quantile(p, 0.9)]
    bot = r[i][p <= np.quantile(p, 0.1)]
    bd = dates[i][p <= np.quantile(p, 0.1)]
    tlo, thi, _ = block_bootstrap_ci(top, td, "M")
    blo, bhi, _ = block_bootstrap_ci(bot, bd, "M")
    return {"variant": name, "n": len(i),
            "R_top_decile": float(top.mean()), "top_ci_lo": tlo, "top_ci_hi": thi,
            "R_bot_decile": float(bot.mean()), "bot_ci_lo": blo, "bot_ci_hi": bhi,
            "spread": float(top.mean() - bot.mean()),
            "monotone_deciles": int((dec.r_mean.diff().dropna() > 0).sum())}


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[~ev.setup.str.startswith("any_liquid")].sort_values("date").reset_index(drop=True)
    dev = ev[ev.date <= VALID_END].reset_index(drop=True)

    dates_ax = np.sort(dev["date"].unique())
    pos = pd.Series(np.arange(len(dates_ax)), index=dates_ax)
    t0 = pos.reindex(dev["date"].values).to_numpy().astype(int)
    t1 = pos.reindex(dev["exit_date"].values).fillna(len(dates_ax) - 1).to_numpy().astype(int)

    X = design(dev)
    r = dev["r_net"].to_numpy(float)
    dates = dev["date"].to_numpy()
    w = average_uniqueness(t0, t1); w /= w.mean()
    splits = purged_kfold(t0, t1, N_CV_SPLITS, EMBARGO_PCT)

    print(f"=== dev {len(dev):,} events, {N_CV_SPLITS} purged splits ===")
    print(f"take every event: {r.mean():+.4f}R\n")

    rows = []
    spec = ModelSpec("logit", "logit", {"C": 0.01})

    print("A. classify win (target before stop) ...")
    t = time.time()
    ps, idx = [], []
    y = dev["win"].to_numpy(int)
    for sp in splits:
        if len(sp.train) < 1000 or len(sp.test) < 200:
            continue
        ps.append(fit_predict(spec, X.iloc[sp.train], y[sp.train], w[sp.train],
                              X.iloc[sp.test], t0[sp.train]))
        idx.append(sp.test)
    rows.append(summarise("A logit P(win)", np.concatenate(ps),
                          np.concatenate(idx), r, dates))
    print(f"   ({time.time()-t:.0f}s)")

    print("B. classify r_net > 0 (did it make money) ...")
    t = time.time()
    ps, idx = [], []
    y2 = (r > 0).astype(int)
    for sp in splits:
        if len(sp.train) < 1000 or len(sp.test) < 200:
            continue
        ps.append(fit_predict(spec, X.iloc[sp.train], y2[sp.train], w[sp.train],
                              X.iloc[sp.test], t0[sp.train]))
        idx.append(sp.test)
    rows.append(summarise("B logit P(r>0)", np.concatenate(ps),
                          np.concatenate(idx), r, dates))
    print(f"   ({time.time()-t:.0f}s)")

    print("C. regress r_net directly (ridge) ...")
    t = time.time()
    p, i = run_regression(X, r, r, t0, t1, splits, w, "ridge")
    rows.append(summarise("C ridge E[R]", p, i, r, dates))
    print(f"   ({time.time()-t:.0f}s)")

    print("D. regress r_net directly (lgbm) ...")
    t = time.time()
    p, i = run_regression(X, r, r, t0, t1, splits, w, "lgbm")
    rows.append(summarise("D lgbm E[R]", p, i, r, dates))
    print(f"   ({time.time()-t:.0f}s)")

    out = pd.DataFrame(rows)
    print("\n=== TARGET VARIANTS (dev, purged CV, month-block CIs) ===")
    print(out.round(4).to_string(index=False))
    print("\nR_bot_decile is the number to watch: a strongly negative bottom decile")
    print("with a CI clear of zero is a working VETO, which is what the holdout")
    print("said this feature set actually supports.")
    out.to_csv(REPORTS / "stage9_target_variants.csv", index=False)
    print(f"\nwrote {REPORTS/'stage9_target_variants.csv'}")


if __name__ == "__main__":
    main()
