"""Per-family models, then the honest holdout.

One model across both exit schemes was a mistake: under the trailing-stop scheme
"win" means "made money" and R ranges over roughly [-1, +10], while under the
7-day bracket "win" means "hit +1.5R first" and R lives in [-1, +1.5]. Pooling
them asks one logistic to fit two different problems on two different scales.

So: separate model per family, and each judged against the only competitor that
matters - taking every event in that family without a model.

The final comparison uses the PAIRED MONTHLY EDGE (strategy month-mean minus
control month-mean, same month), which differences out the market-wide monthly
factor that inflates every naive interval in this problem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import baseline_predictions
from src.config import EMBARGO_PCT, N_CV_SPLITS, PROCESSED, REPORTS, VALID_END
from src.features import FEATURE_COLS
from src.models import ModelSpec, fit_predict, run_cv
from src.validation import (average_uniqueness, block_bootstrap_ci,
                            evaluate, purged_kfold)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

HIST_COLS = ["rsi14", "dist_sma200_atr", "vix_rank_2y", "rv_rank_2y"]
FAMILIES = {
    "reversion": (["connors_rsi2", "pullback_3day", "rs_pullback_50ma"],
                  "any_liquid_reversion"),
    "trend": (["minervini_vcp", "oneil_breakout", "darvas_box", "weinstein_stage2"],
              "any_liquid_trend"),
}


def design(df, setups, cols=None):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].astype(np.float32).copy()
    for s in sorted(setups):
        X[f"setup_{s}"] = (df["setup"] == s).astype(np.int8)
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)


def axis(df):
    ax = np.sort(df["date"].unique())
    pos = pd.Series(np.arange(len(ax)), index=ax)
    t0 = pos.reindex(df["date"].values).to_numpy().astype(int)
    t1 = pos.reindex(df["exit_date"].values).fillna(len(ax) - 1).to_numpy().astype(int)
    return t0, t1


def paired_edge(r_a, d_a, r_b, d_b, seed=7):
    """Monthly mean of A minus monthly mean of B, bootstrapped over months."""
    a = pd.Series(r_a).groupby(pd.PeriodIndex(pd.to_datetime(d_a), freq="M")).mean()
    b = pd.Series(r_b).groupby(pd.PeriodIndex(pd.to_datetime(d_b), freq="M")).mean()
    d = (a - b).dropna()
    if len(d) < 12:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (4000, len(d)))
    bm = d.to_numpy()[idx].mean(axis=1)
    return (float(d.mean()), float(np.quantile(bm, .025)),
            float(np.quantile(bm, .975)), int(len(d)))


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    out_rows = []

    for fam, (setups, ctrl) in FAMILIES.items():
        sub = ev[ev.setup.isin(setups)].sort_values("date").reset_index(drop=True)
        ctl = ev[ev.setup == ctrl].sort_values("date").reset_index(drop=True)
        dev = sub[sub.date <= VALID_END].reset_index(drop=True)
        te = sub[sub.date > VALID_END].reset_index(drop=True)
        ctl_te = ctl[ctl.date > VALID_END]

        print("=" * 78)
        print(f"=== FAMILY: {fam}   dev {len(dev):,} / test {len(te):,} events ===")
        print("=" * 78)

        t0, t1 = axis(dev)
        X = design(dev, setups)
        y = dev["win"].to_numpy(int)
        r = dev["r_net"].to_numpy(float)
        d = dev["date"].to_numpy()
        w = average_uniqueness(t0, t1); w /= w.mean()
        splits = purged_kfold(t0, t1, N_CV_SPLITS, EMBARGO_PCT)
        print(f"base rate {y.mean():.4f}   take-every-event {r.mean():+.4f}R\n")

        # ---- dev: does any model beat taking every event in the family?
        cands = {
            "histogram": None,
            "logit(C=0.01)": ModelSpec("logit", "logit", {"C": 0.01}),
            "lgbm d3/l7/m400": ModelSpec("lgbm", "lgbm", {"max_depth": 3,
                                                          "num_leaves": 7,
                                                          "min_child_samples": 400}),
        }
        dev_res = {}
        for name, spec in cands.items():
            if spec is None:
                ps, idx = [], []
                for sp in splits:
                    if len(sp.train) < 1000 or len(sp.test) < 200:
                        continue
                    ps.append(baseline_predictions(dev.iloc[sp.train], dev.iloc[sp.test],
                                                   HIST_COLS, 4, 200))
                    idx.append(sp.test)
                p, i = np.concatenate(ps), np.concatenate(idx)
                m = evaluate(p, y[i], r[i])
            else:
                res = run_cv(spec, X, y, r, t0, t1, splits)
                p, i, m = res["_oof_p"], res["_oof_i"], res
            dev_res[name] = (p, i)
            top = p >= np.quantile(p, 0.9)
            e, lo, hi, nb = paired_edge(r[i][top], d[i][top], r[i], d[i])
            print(f"  {name:<18} auc={m['auc']:.4f} brier_skill={m['brier_skill']:+.5f}"
                  f"  R_top={m['r_top_decile']:+.4f}")
            print(f"  {'':<18} paired edge of top decile vs all family events: "
                  f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]")

        # ---- holdout, once
        print(f"\n  --- HOLDOUT {te.date.min().date()} .. {te.date.max().date()} ---")
        Xte = design(te, setups, X.columns)
        rte, dte = te["r_net"].to_numpy(float), te["date"].to_numpy()

        e, lo, hi, nb = paired_edge(rte, dte, ctl_te["r_net"].to_numpy(),
                                    ctl_te["date"].to_numpy())
        print(f"  rule: take every {fam} event     R={rte.mean():+.4f}   "
              f"paired edge vs control {e:+.4f} [{lo:+.4f},{hi:+.4f}] ({nb} months)")
        out_rows.append({"family": fam, "strategy": f"take every {fam} event",
                         "n": len(te), "R": rte.mean(), "edge": e,
                         "edge_lo": lo, "edge_hi": hi})

        for name, spec in cands.items():
            if spec is None:
                p_te = baseline_predictions(dev, te, HIST_COLS, 4, 200)
            else:
                p_te = fit_predict(spec, X, y, w, Xte, t0)
            top = p_te >= np.quantile(p_te, 0.9)
            # Isotonic collapses to a coarse step function when there is no
            # signal to calibrate. The 90th percentile then lands inside a big
            # flat block and "top decile" quietly selects everything, which
            # scores as a perfect zero edge rather than as the failure it is.
            frac = top.mean()
            if frac > 0.25:
                print(f"  model {name:<18} DEGENERATE: only "
                      f"{len(np.unique(p_te))} distinct probabilities, "
                      f"top-decile cut selects {frac*100:.0f}% of events -> no "
                      f"usable ranking, row skipped")
                continue
            e, lo, hi, nb = paired_edge(rte[top], dte[top], rte, dte)
            blo, bhi, _ = block_bootstrap_ci(rte[top], dte[top], "M")
            print(f"  model {name:<18} top-decile R={rte[top].mean():+.4f} "
                  f"[{blo:+.4f},{bhi:+.4f}]   edge vs family {e:+.4f} [{lo:+.4f},{hi:+.4f}]")
            out_rows.append({"family": fam, "strategy": f"model {name} top decile",
                             "n": int(top.sum()), "R": float(rte[top].mean()),
                             "edge": e, "edge_lo": lo, "edge_hi": hi})

        for s in sorted(setups):
            m = (te["setup"] == s).to_numpy()
            if m.sum() < 200:
                continue
            e, lo, hi, nb = paired_edge(rte[m], dte[m], ctl_te["r_net"].to_numpy(),
                                        ctl_te["date"].to_numpy())
            print(f"  rule: always {s:<18} R={rte[m].mean():+.4f}  "
                  f"edge vs control {e:+.4f} [{lo:+.4f},{hi:+.4f}]  n={int(m.sum()):,}")
            out_rows.append({"family": fam, "strategy": f"rule always {s}",
                             "n": int(m.sum()), "R": float(rte[m].mean()),
                             "edge": e, "edge_lo": lo, "edge_hi": hi})
        print()

    res = pd.DataFrame(out_rows)
    res["clears_0"] = res.edge_lo > 0
    res.to_csv(REPORTS / "stage10_family_holdout.csv", index=False)
    print("=" * 78)
    print("=== HOLDOUT SUMMARY (paired monthly edge, 95% CI) ===")
    print(res.sort_values("edge", ascending=False).round(4).to_string(index=False))
    print(f"\nwrote {REPORTS/'stage10_family_holdout.csv'}")


if __name__ == "__main__":
    main()
