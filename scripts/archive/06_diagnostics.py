"""Does the model earn its place?

A model that ranks trades well but only rediscovers "setup X is the good setup"
has added nothing a one-line rule could not. This script forces that
comparison, and asks what the model is actually keying on.

Run after 04_train.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import EMBARGO_PCT, N_CV_SPLITS, PROCESSED, REPORTS, VALID_END
from src.features import FEATURE_COLS
from src.models import ModelSpec, _clean, run_cv
from src.validation import average_uniqueness, block_bootstrap_ci, purged_kfold

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def strategy_row(name: str, r: np.ndarray, dates: np.ndarray) -> dict:
    lo, hi, nb = block_bootstrap_ci(r, dates, "M") if len(r) > 50 else (np.nan,) * 3
    return {"strategy": name, "n": len(r), "mean_R": float(r.mean()),
            "ci_lo": lo, "ci_hi": hi, "clears_0": bool(lo > 0) if lo == lo else False,
            "total_R": float(r.sum())}


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[~ev.setup.str.startswith("any_liquid")].sort_values("date").reset_index(drop=True)
    dev = ev[ev.date <= VALID_END].reset_index(drop=True)

    dates_all = np.sort(dev["date"].unique())
    pos = pd.Series(np.arange(len(dates_all)), index=dates_all)
    t0 = pos.reindex(dev["date"].values).to_numpy().astype(int)
    t1 = pos.reindex(dev["exit_date"].values).fillna(pos.iloc[-1]).to_numpy().astype(int)

    feats = [c for c in FEATURE_COLS if c in dev.columns]
    X = dev[feats].astype(np.float32).copy()
    for s in sorted(dev["setup"].unique()):
        X[f"setup_{s}"] = (dev["setup"] == s).astype(np.int8)
    y = dev["win"].to_numpy(int)
    r = dev["r_net"].to_numpy(float)
    d = dev["date"].to_numpy()

    print("=== fitting winning model (logistic) on purged CV ===")
    splits = purged_kfold(t0, t1, N_CV_SPLITS, EMBARGO_PCT)
    spec = ModelSpec(name="logit", kind="logit", params={"C": 0.05})
    res = run_cv(spec, X, y, r, t0, t1, splits)
    p, idx = res["_oof_p"], res["_oof_i"]
    sub = dev.iloc[idx].reset_index(drop=True)
    rr, dd = r[idx], d[idx]
    print(f"OOF predictions for {len(idx):,} events, auc={res['auc']:.4f}\n")

    # ------------------------------------------------ what is in the top decile?
    thr = np.quantile(p, 0.9)
    top = p >= thr
    print("=== composition of the model's top decile vs the whole event pool ===")
    comp = pd.DataFrame({
        "all_events": sub["setup"].value_counts(normalize=True),
        "top_decile": sub.loc[top, "setup"].value_counts(normalize=True),
    }).fillna(0)
    comp["lift"] = comp["top_decile"] / comp["all_events"].replace(0, np.nan)
    print(comp.round(3).to_string(), "\n")

    # ------------------------------------------------------- the real comparison
    print("=== THE COMPARISON THAT MATTERS ===")
    print("model top-decile vs simple rules any beginner could write\n")
    rows = [strategy_row("model: top decile", rr[top], dd[top]),
            strategy_row("model: top 5%", rr[p >= np.quantile(p, 0.95)],
                         dd[p >= np.quantile(p, 0.95)]),
            strategy_row("take EVERY event", rr, dd)]
    for s in sorted(sub["setup"].unique()):
        m = (sub["setup"] == s).to_numpy()
        rows.append(strategy_row(f"rule: always {s}", rr[m], dd[m]))
    # best single rule + the model applied inside it
    for s in sorted(sub["setup"].unique()):
        m = (sub["setup"] == s).to_numpy()
        if m.sum() < 2000:
            continue
        thr_s = np.quantile(p[m], 0.7)
        mm = m & (p >= thr_s)
        rows.append(strategy_row(f"rule {s} + model top 30%", rr[mm], dd[mm]))
    comp2 = pd.DataFrame(rows).sort_values("mean_R", ascending=False)
    print(comp2.round(4).to_string(index=False))

    print("\nif 'model: top decile' does not clearly beat the best 'rule:' row,")
    print("the model is rediscovering the setup ranking, not adding information.\n")

    # -------------------------------------------------------- what drives it?
    print("=== standardised logistic coefficients (what the model keys on) ===")
    Xc, med = _clean(X)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xc)
    lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Xc), y)
    coef = (pd.Series(lr.coef_[0], index=Xc.columns)
              .sort_values(key=np.abs, ascending=False).head(20))
    print(coef.round(4).to_string())

    # ------------------------------------------------------------ by regime
    print("\n=== top-decile mean R by market regime ===")
    reg = pd.DataFrame({"r": rr, "top": top,
                        "bull": sub["spy_above_200"].to_numpy(),
                        "year": pd.to_datetime(dd).year})
    print(reg[reg.top].groupby("bull")["r"].agg(["size", "mean"]).round(4).to_string())
    print("\n=== top-decile mean R by year ===")
    yr = reg[reg.top].groupby("year")["r"].agg(["size", "mean"]).round(4)
    yr["positive"] = yr["mean"] > 0
    print(yr.to_string())
    print(f"\npositive in {yr.positive.mean()*100:.0f}% of years")

    comp2.to_csv(REPORTS / "stage6_strategy_comparison.csv", index=False)
    print(f"\nwrote {REPORTS/'stage6_strategy_comparison.csv'}")


if __name__ == "__main__":
    main()
