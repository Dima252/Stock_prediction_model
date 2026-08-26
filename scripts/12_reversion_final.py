"""Final reversion model: dev search, portfolio simulation, then ONE holdout read.

Why a portfolio simulation and not just "top decile": the top decile of 100k
events is not a strategy you can trade. On a given day the reversion book fires
on dozens of names and you can hold maybe 5-20. The realistic question is "of
today's candidates, which K do I take?", and the honest benchmark is K names
picked at random from the SAME day's candidates - which controls for the market,
the setup mix, and the day, all at once.

Protocol: everything is chosen on dev. The holdout is read once, at the end.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBARGO_PCT, N_CV_SPLITS, PROCESSED, REPORTS, VALID_END
from src.features import FEATURE_COLS
from src.models import ModelSpec, fit_predict
from src.setups import REVERSION_SETUPS
from src.validation import average_uniqueness, evaluate, purged_kfold

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)


def design(df, cols=None):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].astype(np.float32).copy()
    for s in sorted(REVERSION_SETUPS):
        X[f"setup_{s}"] = (df["setup"] == s).astype(np.int8)
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)


def axis(df):
    ax = np.sort(df["date"].unique())
    pos = pd.Series(np.arange(len(ax)), index=ax)
    return (pos.reindex(df["date"].values).to_numpy().astype(int),
            pos.reindex(df["exit_date"].values).fillna(len(ax) - 1).to_numpy().astype(int))


def boot_months(d: pd.Series, seed=7):
    d = d.dropna()
    if len(d) < 12:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bm = d.to_numpy()[rng.integers(0, len(d), (4000, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(bm, .025)), float(np.quantile(bm, .975))


def topk_portfolio(df, score, k, seed=7):
    """Take the K highest-scoring candidates each day; benchmark = K at random
    from the same day. Returns (per-trade R, monthly edge series)."""
    rng = np.random.default_rng(seed)
    d = df[["date", "r_net"]].copy()
    d["s"] = score
    picked, rand = [], []
    for dt, g in d.groupby("date", sort=False):
        if len(g) == 0:
            continue
        kk = min(k, len(g))
        picked.append(g.nlargest(kk, "s")[["date", "r_net"]])
        rand.append(g.iloc[rng.choice(len(g), kk, replace=False)][["date", "r_net"]])
    P = pd.concat(picked); Rn = pd.concat(rand)
    mp = P.groupby(pd.PeriodIndex(P.date, freq="M"))["r_net"].mean()
    mr = Rn.groupby(pd.PeriodIndex(Rn.date, freq="M"))["r_net"].mean()
    return P["r_net"], (mp - mr).dropna()


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[ev.setup.isin(REVERSION_SETUPS)].sort_values("date").reset_index(drop=True)
    dev = ev[ev.date <= VALID_END].reset_index(drop=True)
    te = ev[ev.date > VALID_END].reset_index(drop=True)

    t0, t1 = axis(dev)
    X = design(dev)
    y = dev["win"].to_numpy(int)
    r = dev["r_net"].to_numpy(float)
    w = average_uniqueness(t0, t1); w /= w.mean()
    splits = purged_kfold(t0, t1, N_CV_SPLITS, EMBARGO_PCT)

    print(f"=== reversion family: dev {len(dev):,} / holdout {len(te):,} ===")
    print(f"exit: close > SMA5, 1R stop, 10d cap   mean hold {dev.hold_days.mean():.2f}d")
    print(f"take every event (dev): {r.mean():+.4f}R   base rate {y.mean():.4f}\n")

    # -------------------------------------------------- dev: model selection
    cands = {
        "logit(C=0.01)": ModelSpec("logit", "logit", {"C": 0.01}),
        "logit(C=0.1)": ModelSpec("logit", "logit", {"C": 0.1}),
        "lgbm d3/l7/m400": ModelSpec("lgbm", "lgbm", {"max_depth": 3, "num_leaves": 7,
                                                      "min_child_samples": 400}),
        "lgbm d4/l15/m200": ModelSpec("lgbm", "lgbm", {"max_depth": 4, "num_leaves": 15,
                                                       "min_child_samples": 200}),
    }
    oof = {}
    print("=== dev, purged CV ===")
    for name, spec in cands.items():
        t = time.time()
        ps, idx = [], []
        for sp in splits:
            if len(sp.train) < 1000 or len(sp.test) < 200:
                continue
            ps.append(fit_predict(spec, X.iloc[sp.train], y[sp.train], w[sp.train],
                                  X.iloc[sp.test], t0[sp.train]))
            idx.append(sp.test)
        p, i = np.concatenate(ps), np.concatenate(idx)
        oof[name] = (p, i)
        m = evaluate(p, y[i], r[i])
        sub = dev.iloc[i].reset_index(drop=True)
        _, ed = topk_portfolio(sub, p, 10)
        e, lo, hi = boot_months(ed)
        print(f"  {name:<18} auc={m['auc']:.4f} brier_skill={m['brier_skill']:+.5f}"
              f"   top10/day edge vs random10 {e:+.4f} [{lo:+.4f},{hi:+.4f}]  ({time.time()-t:.0f}s)")

    # rank-average ensemble of the two families of model
    i_common = oof["logit(C=0.01)"][1]
    ens = (pd.Series(oof["logit(C=0.01)"][0]).rank(pct=True).to_numpy()
           + pd.Series(oof["lgbm d3/l7/m400"][0]).rank(pct=True).to_numpy()) / 2
    sub = dev.iloc[i_common].reset_index(drop=True)
    _, ed = topk_portfolio(sub, ens, 10)
    e, lo, hi = boot_months(ed)
    print(f"  {'ensemble(rank avg)':<18} top10/day edge vs random10 "
          f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]")

    print("\n=== dev: how many positions per day? ===")
    for k in (3, 5, 10, 20, 50):
        p, i = oof["logit(C=0.01)"]
        sub = dev.iloc[i].reset_index(drop=True)
        rr, ed = topk_portfolio(sub, p, k)
        e, lo, hi = boot_months(ed)
        print(f"  top {k:>2}/day  R/trade={rr.mean():+.4f}  n={len(rr):,}  "
              f"edge vs random {e:+.4f} [{lo:+.4f},{hi:+.4f}]")

    # ------------------------------------------------------- FINAL HOLDOUT
    print("\n" + "=" * 74)
    print("=== FINAL HOLDOUT - one read, chosen config: logit(C=0.01), top10/day ===")
    print("=" * 74)
    Xte = design(te, X.columns)
    p_te = fit_predict(cands["logit(C=0.01)"], X, y, w, Xte, t0)
    rte = te["r_net"].to_numpy(float)
    print(f"holdout {te.date.min().date()} .. {te.date.max().date()}, {len(te):,} events")
    print(f"take every event: {rte.mean():+.4f}R\n")

    rows = []
    for k in (3, 5, 10, 20):
        rr, ed = topk_portfolio(te, p_te, k)
        e, lo, hi = boot_months(ed)
        yr = pd.DataFrame({"r": rr.to_numpy(), "y": pd.to_datetime(
            te.loc[rr.index, "date"]).dt.year.to_numpy()})
        pos_yr = yr.groupby("y")["r"].mean()
        print(f"  top {k:>2}/day   R/trade={rr.mean():+.4f}   n={len(rr):,}   "
              f"edge vs random{k} {e:+.4f} [{lo:+.4f},{hi:+.4f}]   "
              f"{(pos_yr > 0).mean()*100:.0f}% of years +")
        rows.append({"k": k, "R": rr.mean(), "n": len(rr), "edge": e,
                     "lo": lo, "hi": hi, "pct_years_pos": float((pos_yr > 0).mean())})
        if k == 10:
            print("     by year:", {int(a): round(b, 4) for a, b in pos_yr.items()})

    pd.DataFrame(rows).to_csv(REPORTS / "stage12_final_holdout.csv", index=False)
    print(f"\nwrote {REPORTS/'stage12_final_holdout.csv'}")


if __name__ == "__main__":
    main()
