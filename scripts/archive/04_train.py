"""Stages 5-6 - the model ladder, on purged CV.

Ladder (each rung must beat the one below it or it does not ship):
    0. empirical histogram   (stage 3, the thing to beat)
    1. regularised logistic  (does the feature set carry linear information?)
    2. LightGBM              (do interactions add anything?)

Protocol
--------
* dev  = everything up to VALID_END. All model selection happens here.
* test = after VALID_END. Touched ONCE, at the very end, and never tuned against.
* Every dev comparison uses purged k-fold with embargo + uniqueness weights.
* The winner additionally gets CPCV, which returns a DISTRIBUTION of out-of-sample
  results instead of one fragile point estimate.
* Trials are counted. Every configuration tried inflates the best result you see.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.baseline import baseline_predictions
from src.config import (EMBARGO_PCT, CPCV_N_GROUPS, CPCV_N_TEST, N_CV_SPLITS,
                        PROCESSED, RANDOM_SEED, REPORTS, VALID_END)
from src.features import FEATURE_COLS
from src.models import ModelSpec, run_cv
from src.validation import (average_uniqueness, block_bootstrap_ci, bootstrap_ci,
                            calibration_table, combinatorial_purged_cv,
                            concurrency_report, evaluate, expectancy_by_decile,
                            purged_kfold)


def report_top_decile(p: np.ndarray, r: np.ndarray, dates: np.ndarray,
                      label: str) -> None:
    """Top-decile expectancy with an interval that respects the correlation.

    The event-level CI is printed only to show how badly it misleads: it treats
    370k overlapping, market-correlated events as independent draws and comes out
    ~4x too narrow. The month-block interval is the one to believe.
    """
    m = p >= np.quantile(p, 0.9)
    top, td = r[m], dates[m]
    n_lo, n_hi = bootstrap_ci(top)
    b_lo, b_hi, n_blocks = block_bootstrap_ci(top, td, "M")
    print(f"\n{label}: top-decile mean R = {top.mean():+.4f}  (n={len(top):,})")
    print(f"   event-level 95% CI  [{n_lo:+.4f}, {n_hi:+.4f}]   <- TOO NARROW, ignore")
    print(f"   month-block 95% CI  [{b_lo:+.4f}, {b_hi:+.4f}]   ({n_blocks} blocks)")
    verdict = "clears zero" if b_lo > 0 else "DOES NOT clear zero"
    print(f"   -> {verdict}")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

HIST_COLS = ["rsi14", "dist_sma200_atr", "vix_rank_2y", "rv_rank_2y"]


def build_design(ev: pd.DataFrame):
    """Design matrix + integer label spans on a shared trading-day axis."""
    dates = np.sort(ev["date"].unique())
    pos = pd.Series(np.arange(len(dates)), index=dates)
    t0 = pos.reindex(ev["date"].values).to_numpy()
    t1 = pos.reindex(ev["exit_date"].values).to_numpy()
    bad = ~np.isfinite(t1)
    if bad.any():                       # exit past the end of the panel
        t1 = np.where(bad, t0, t1)
    t0, t1 = t0.astype(int), t1.astype(int)

    feats = [c for c in FEATURE_COLS if c in ev.columns]
    X = ev[feats].astype(np.float32).copy()
    for s in sorted(ev["setup"].unique()):
        X[f"setup_{s}"] = (ev["setup"] == s).astype(np.int8)
    y = ev["win"].to_numpy(int)
    r = ev["r_net"].to_numpy(float)
    return X, y, r, t0, t1, feats


def run_cv_hist(ev: pd.DataFrame, y, r, t0, t1, splits) -> dict:
    """The stage-3 histogram scored through the identical purged splits."""
    w = average_uniqueness(t0, t1); w /= w.mean()
    ps, idx, per_split = [], [], []
    for sp in splits:
        if len(sp.train) < 1000 or len(sp.test) < 200:
            continue
        p = baseline_predictions(ev.iloc[sp.train], ev.iloc[sp.test],
                                 HIST_COLS, n_bins=4, prior_n=200)
        ps.append(p); idx.append(sp.test)
        try:
            per_split.append(evaluate(p, y[sp.test], r[sp.test], w[sp.test]))
        except Exception:
            pass
    if not ps:
        return {"n": 0}
    p = np.concatenate(ps); i = np.concatenate(idx)
    out = evaluate(p, y[i], r[i], w[i])
    out["model"] = "histogram"
    # same per-split summary the ML rungs get, so the comparison is like-for-like
    if per_split:
        df = pd.DataFrame(per_split)
        for col in ("auc", "brier_skill", "r_top_decile", "r_spread"):
            out[f"{col}_split_mean"] = float(df[col].mean())
            out[f"{col}_split_std"] = float(df[col].std())
        out["pct_splits_top_positive"] = float((df["r_top_decile"] > 0).mean())
    out["_oof_p"], out["_oof_i"] = p, i
    return out


def show(res: dict) -> None:
    if res.get("n", 0) == 0:
        print("   (no valid splits)")
        return
    print(f"  {res['model']:<22} n={res['n']:>7,}  auc={res['auc']:.4f}  "
          f"brier_skill={res['brier_skill']:+.5f}  cal_mae={res['cal_mae']:.4f}")
    print(f"  {'':<22} R top decile={res['r_top_decile']:+.4f}  "
          f"bottom={res['r_bot_decile']:+.4f}  spread={res['r_spread']:+.4f}  "
          f"(all events R={res['r_mean_all']:+.4f})")
    if "auc_split_mean" in res:
        print(f"  {'':<22} across splits: auc={res['auc_split_mean']:.4f}"
              f"+-{res['auc_split_std']:.4f}   "
              f"R_top={res['r_top_decile_split_mean']:+.4f}"
              f"+-{res['r_top_decile_split_std']:.4f}   "
              f"{res['pct_splits_top_positive']*100:.0f}% of splits positive")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="hyperparameter search")
    ap.add_argument("--cpcv", action="store_true", help="combinatorial purged CV")
    ap.add_argument("--test", action="store_true", help="FINAL holdout - once only")
    ap.add_argument("--setup", default=None, help="restrict to one setup")
    args = ap.parse_args()

    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[~ev.setup.str.startswith("any_liquid")].reset_index(drop=True)
    if args.setup:
        ev = ev[ev.setup == args.setup].reset_index(drop=True)
    ev = ev.sort_values("date").reset_index(drop=True)

    dev = ev[ev.date <= VALID_END].reset_index(drop=True)
    print(f"=== dev: {len(dev):,} events ({dev.date.min().date()} .. "
          f"{dev.date.max().date()}) ===")
    print(f"    test held out: {len(ev) - len(dev):,} events after {VALID_END}\n")

    X, y, r, t0, t1, feats = build_design(dev)
    print(f"design: {X.shape[0]:,} x {X.shape[1]} features")
    cr = concurrency_report(t0, t1, dev["date"].to_numpy())
    print(f"mean concurrency  : {cr['mean_concurrency']:,.0f} labels open at once")
    print(f"mean label span   : {cr['mean_span_days']:.1f} trading days")
    print(f"avg uniqueness    : {cr['mean_uniqueness']:.4f}  "
          f"(classical eff-N ~{cr['uniqueness_eff_n']:,.0f})")
    print(f"calendar months   : {cr['n_months']}   trading days: {cr['n_trading_days']:,}")
    print("  NOTE: classical eff-N counts different tickers on the same day as")
    print("  concurrent, so it UNDERstates independence; the raw event count wildly")
    print("  OVERstates it. Month-block bootstrap is used for every interval below.")
    print(f"base rate P(target first): {y.mean():.4f}")
    print(f"mean R (all dev events)  : {r.mean():+.4f}\n")

    splits = purged_kfold(t0, t1, N_CV_SPLITS, EMBARGO_PCT)
    print(f"purged k-fold: {len(splits)} splits, "
          f"mean train {np.mean([len(s.train) for s in splits]):,.0f}, "
          f"mean test {np.mean([len(s.test) for s in splits]):,.0f}\n")

    results, trials = {}, 0
    specs: dict[str, ModelSpec] = {}          # so the winner can be rebuilt later

    print("=== rung 0: empirical histogram ===")
    t = time.time(); res = run_cv_hist(dev, y, r, t0, t1, splits)
    results["histogram"] = res; show(res); print(f"  ({time.time()-t:.0f}s)\n")

    print("=== rung 1: regularised logistic ===")
    for C in ([0.005, 0.01, 0.05, 0.2, 1.0] if args.sweep else [0.05]):
        t = time.time()
        spec = ModelSpec(name=f"logit(C={C})", kind="logit", params={"C": C})
        res = run_cv(spec, X, y, r, t0, t1, splits); trials += 1
        results[spec.name] = res; specs[spec.name] = spec
        show(res); print(f"  ({time.time()-t:.0f}s)")
    print()

    print("=== rung 2: LightGBM ===")
    grid = [{}]
    if args.sweep:
        grid = []
        for depth, leaves in ((3, 7), (4, 15), (5, 31)):
            for mcs in (100, 400):
                for lr, n in ((0.02, 600), (0.05, 300)):
                    grid.append({"max_depth": depth, "num_leaves": leaves,
                                 "min_child_samples": mcs, "learning_rate": lr,
                                 "n_estimators": n})
    for p in grid:
        t = time.time()
        tag = "lgbm" if not p else ("lgbm d%d/l%d/m%d/lr%.2f" %
                                    (p["max_depth"], p["num_leaves"],
                                     p["min_child_samples"], p["learning_rate"]))
        spec = ModelSpec(name=tag, kind="lgbm", params=p)
        res = run_cv(spec, X, y, r, t0, t1, splits); trials += 1
        results[spec.name] = res; specs[spec.name] = spec
        show(res); print(f"  ({time.time()-t:.0f}s)")
    print()

    # ------------------------------------------------------------- leaderboard
    rows = [{k: v for k, v in x.items() if not k.startswith("_")}
            for x in results.values() if x.get("n", 0) > 0]
    lb = pd.DataFrame(rows).sort_values("r_top_decile", ascending=False)
    keep = ["model", "n", "auc", "brier_skill", "cal_mae", "r_top_decile",
            "r_spread", "pct_splits_top_positive"]
    print("=== leaderboard (dev, purged CV) ===")
    print(lb[[c for c in keep if c in lb.columns]].to_string(index=False))
    print(f"\ntrials run this session: {trials}  "
          f"(every trial inflates the best number you see)")
    lb.to_csv(REPORTS / "stage6_leaderboard.csv", index=False)

    best_name = lb.iloc[0]["model"]
    best = results[best_name]
    print(f"\n=== best on dev: {best_name} ===")
    print("\ncalibration (dev, out-of-fold):")
    print(calibration_table(best["_oof_p"], y[best["_oof_i"]]).to_string(index=False))
    print("\nexpectancy by predicted-probability decile:")
    dec = expectancy_by_decile(best["_oof_p"], r[best["_oof_i"]])
    print(dec.to_string(index=False))
    report_top_decile(best["_oof_p"], r[best["_oof_i"]],
                      dev["date"].to_numpy()[best["_oof_i"]], "dev (out-of-fold)")

    # -------------------------------------------------------------------- CPCV
    if args.cpcv:
        print(f"\n=== CPCV on dev ({CPCV_N_GROUPS} groups, {CPCV_N_TEST} test) ===")
        print("a DISTRIBUTION of out-of-sample results, not one point estimate")
        cs = combinatorial_purged_cv(t0, t1, CPCV_N_GROUPS, CPCV_N_TEST, EMBARGO_PCT)
        print(f"{len(cs)} combinatorial splits")
        t = time.time()
        if best_name == "histogram":
            res = run_cv_hist(dev, y, r, t0, t1, cs)
        else:
            res = run_cv(specs[best_name], X, y, r, t0, t1, cs)
        show(res)
        print(f"  ({time.time()-t:.0f}s)")
        if "r_top_decile_split_mean" in res:
            print(f"\n  across {len(cs)} paths: top-decile R = "
                  f"{res['r_top_decile_split_mean']:+.4f} +- "
                  f"{res['r_top_decile_split_std']:.4f}, "
                  f"{res['pct_splits_top_positive']*100:.0f}% positive")

    # ------------------------------------------------------- FINAL HOLDOUT
    if args.test:
        print("\n" + "=" * 70)
        print("=== FINAL HOLDOUT - fit on all dev, evaluate ONCE on test ===")
        print("=== this number is the honest one. do not tune against it. ===")
        print("=" * 70)
        te = ev[ev.date > VALID_END].reset_index(drop=True)
        Xte, yte, rte, t0te, t1te, _ = build_design(te)
        Xte = Xte.reindex(columns=X.columns, fill_value=0.0)
        print(f"test: {len(te):,} events "
              f"({te.date.min().date()} .. {te.date.max().date()})")
        print(f"test base rate {yte.mean():.4f}, mean R {rte.mean():+.4f}\n")

        if best_name == "histogram":
            p_te = baseline_predictions(dev, te, HIST_COLS, n_bins=4, prior_n=200)
        else:
            from src.models import fit_predict
            w = average_uniqueness(t0, t1); w /= w.mean()
            p_te = fit_predict(specs[best_name], X, y, w, Xte, t0)

        res_te = evaluate(p_te, yte, rte)
        res_te["model"] = f"{best_name} [TEST]"
        show(res_te)
        print("\ntest calibration:")
        print(calibration_table(p_te, yte).to_string(index=False))
        print("\ntest expectancy by decile:")
        print(expectancy_by_decile(p_te, rte).to_string(index=False))
        report_top_decile(p_te, rte, te["date"].to_numpy(), "TEST")
        b_lo, b_hi, _ = block_bootstrap_ci(rte, te["date"].to_numpy(), "M")
        print(f"\nvs taking EVERY event in test: {rte.mean():+.4f} "
              f"month-block CI [{b_lo:+.4f}, {b_hi:+.4f}]")
        print("   the model only earns its place if the top decile clearly beats this")
        json.dump({k: v for k, v in res_te.items() if not k.startswith("_")},
                  open(REPORTS / "stage8_holdout.json", "w"), indent=2, default=float)

    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in results.items()},
              open(REPORTS / "stage6_results.json", "w"), indent=2, default=float)
    print(f"\nwrote {REPORTS/'stage6_leaderboard.csv'}")


if __name__ == "__main__":
    main()
