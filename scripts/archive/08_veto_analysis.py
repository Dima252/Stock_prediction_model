"""Is the model useful as a VETO rather than a green light?

The holdout showed the logistic's top decile at ~0R but its BOTTOM decile at
-0.14R, against -0.03R for the event pool. That asymmetry is the interesting
result: the model may not find good trades, but it may reliably identify bad
ones - which is exactly what a "should I take this trade?" tool needs.

This tests that directly, on the holdout, with month-block intervals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED, REPORTS, VALID_END
from src.features import FEATURE_COLS
from src.setups import REVERSION_SETUPS, TREND_SETUPS

SETUP_ORDER = sorted(TREND_SETUPS + REVERSION_SETUPS)
from src.models import ModelSpec, fit_predict
from src.validation import average_uniqueness, block_bootstrap_ci

pd.set_option("display.width", 240)


def design(df, cols=None):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].astype(np.float32).copy()
    for s in SETUP_ORDER:
        X[f"setup_{s}"] = (df["setup"] == s).astype(np.int8)
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)


def ci(r, d):
    lo, hi, _ = block_bootstrap_ci(r, d, "M") if len(r) > 50 else (np.nan,) * 3
    return lo, hi


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[~ev.setup.str.startswith("any_liquid")].sort_values("date").reset_index(drop=True)
    dev = ev[ev.date <= VALID_END].reset_index(drop=True)
    te = ev[ev.date > VALID_END].reset_index(drop=True)

    dates = np.sort(dev["date"].unique())
    pos = pd.Series(np.arange(len(dates)), index=dates)
    t0 = pos.reindex(dev["date"].values).to_numpy().astype(int)
    t1 = pos.reindex(dev["exit_date"].values).fillna(len(dates) - 1).to_numpy().astype(int)

    X = design(dev); Xte = design(te, X.columns)
    y, r = dev["win"].to_numpy(int), dev["r_net"].to_numpy(float)
    rte, dte = te["r_net"].to_numpy(float), te["date"].to_numpy()
    w = average_uniqueness(t0, t1); w /= w.mean()

    spec = ModelSpec("logit", "logit", {"C": 0.01})
    print("fitting logit(C=0.01) on dev, scoring holdout ...")
    p = fit_predict(spec, X, y, w, Xte, t0)

    print(f"\n=== holdout: {len(te):,} events, "
          f"{te.date.min().date()} .. {te.date.max().date()} ===")
    print(f"baseline: take every event -> {rte.mean():+.4f}R\n")

    print("=== mean R by model-confidence decile (holdout) ===")
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    rows = []
    for b in range(10):
        m = (q == b).to_numpy()
        lo, hi = ci(rte[m], dte[m])
        rows.append({"decile": b, "n": int(m.sum()), "p_mean": float(p[m].mean()),
                     "mean_R": float(rte[m].mean()), "ci_lo": lo, "ci_hi": hi})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== VETO TEST: skip the worst X%, take everything else ===")
    print("if vetoing genuinely helps, mean R should rise as the veto widens\n")
    rows = []
    for cut in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        thr = np.quantile(p, cut) if cut > 0 else -np.inf
        m = p > thr
        lo, hi = ci(rte[m], dte[m])
        rows.append({"veto_worst": f"{cut*100:.0f}%", "n_kept": int(m.sum()),
                     "mean_R": float(rte[m].mean()), "ci_lo": lo, "ci_hi": hi,
                     "total_R": float(rte[m].sum())})
    veto = pd.DataFrame(rows)
    print(veto.round(4).to_string(index=False))

    print("\n=== the vetoed trades themselves ===")
    for cut in (0.1, 0.2, 0.3):
        m = p <= np.quantile(p, cut)
        lo, hi = ci(rte[m], dte[m])
        print(f"  worst {cut*100:>2.0f}% by model: n={int(m.sum()):>6,}  "
              f"mean R={rte[m].mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")

    print("\n=== veto applied WITHIN each setup ===")
    print("does the model add information beyond just ranking the setups?\n")
    rows = []
    for s in sorted(te["setup"].unique()):
        ms = (te["setup"] == s).to_numpy()
        if ms.sum() < 1000:
            continue
        thr = np.quantile(p[ms], 0.3)
        keep = ms & (p > thr)
        lo0, hi0 = ci(rte[ms], dte[ms])
        lo1, hi1 = ci(rte[keep], dte[keep])
        rows.append({"setup": s, "n_all": int(ms.sum()), "R_all": float(rte[ms].mean()),
                     "n_kept": int(keep.sum()), "R_after_veto": float(rte[keep].mean()),
                     "delta": float(rte[keep].mean() - rte[ms].mean()),
                     "kept_ci_lo": lo1, "kept_ci_hi": hi1})
    within = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(within.round(4).to_string(index=False))
    print("\npositive `delta` = the veto improved that setup; this is the only")
    print("evidence that the model knows something the setup label does not.")

    veto.to_csv(REPORTS / "stage8_veto.csv", index=False)
    within.to_csv(REPORTS / "stage8_veto_within_setup.csv", index=False)
    print(f"\nwrote {REPORTS/'stage8_veto.csv'}")


if __name__ == "__main__":
    main()
