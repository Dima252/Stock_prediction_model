"""Stage 3 - the empirical baseline, and the first kill gate.

Gate: does ANY setup beat its unconditional base rate by more than costs, with a
bootstrap CI clear of zero, and not only in one regime?

If nothing clears this, stop. Do not write a line of ML - go back and reconsider
the setups. That is the cheap fix, and finding it out here costs a week rather
than two months.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import baseline_oos, setup_summary, yearly_stability
from src.config import PROCESSED, REPORTS, TRAIN_END
from src.setups import REVERSION_SETUPS, TREND_SETUPS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    print(f"=== {len(ev):,} events, {ev.date.min().date()} .. {ev.date.max().date()} ===\n")

    # ---------------------------------------------------------------- outcomes
    print("=== outcome mix (all events) ===")
    mix = ev["outcome"].value_counts(normalize=True).rename(
        index={1: "target", -1: "stop", 0: "timeout"})
    print(mix.to_string(), "\n")

    # --------------------------------------------------------- setup vs control
    print("=== per-setup stats vs unconditional base rate ===")
    print("(edge_r is the only column that matters: E[R] above the control)\n")
    control_map = ({k: "any_liquid_trend" for k in TREND_SETUPS}
                   | {k: "any_liquid_reversion" for k in REVERSION_SETUPS})
    s = setup_summary(ev, control_map)
    cols = ["setup", "control", "n", "win_rate", "r_mean", "r_ci_lo", "r_ci_hi",
            "edge_r", "ci_clears_zero", "beats_control", "hold_days"]
    print(s[[c for c in cols if c in s.columns]].to_string(index=False), "\n")

    # ------------------------------------------------------------- yearly drift
    print("=== mean R by year and setup (regime stability) ===")
    print("a setup that only worked in one or two years is an artefact\n")
    t, n = yearly_stability(ev)
    print(t.to_string(), "\n")

    print("=== share of years with positive mean R ===")
    pos = (t > 0).mean().sort_values(ascending=False)
    print(pos.to_string(), "\n")

    # --------------------------------------------- out-of-sample bucket lookup
    print("=== out-of-sample histogram test ===")
    print("bucket edges + stats learned on train only, applied to test.")
    print("if r_train and r_test are uncorrelated the buckets carry no signal.\n")
    tr = ev[ev.date <= TRAIN_END]
    te = ev[ev.date > TRAIN_END]
    print(f"train {len(tr):,} events (<= {TRAIN_END}), test {len(te):,} events\n")

    by = ["rsi14", "dist_sma200_atr", "vix_rank_2y", "rv_rank_2y"]
    for setup in ev.setup.unique():
        a, b = tr[tr.setup == setup], te[te.setup == setup]
        if len(a) < 3000 or len(b) < 1000:
            continue
        j = baseline_oos(a, b, by, n_bins=3, min_n=150)
        if len(j) < 6:
            continue
        rho = np.corrcoef(j.r_train, j.r_test)[0, 1]
        print(f"  {setup:<20} buckets={len(j):>3}  corr(r_train, r_test)={rho:+.3f}"
              f"   spread_train={j.r_train.max()-j.r_train.min():.3f}"
              f"  spread_test={j.r_test.max()-j.r_test.min():.3f}")

    # ------------------------------------------------------------------- gate
    print("\n=== KILL GATE 1 ===")
    real = s[~s.setup.str.startswith("any_liquid")]
    stable = pos[pos >= 0.6].index.tolist()
    passed = [r.setup for _, r in real.iterrows()
              if r.r_ci_lo > 0 and r.beats_control and r.setup in stable]

    for name in ("any_liquid_trend", "any_liquid_reversion"):
        if name in s.setup.values:
            print(f"  control {name:<22} R={float(s.loc[s.setup==name,'r_mean'].iloc[0]):+.4f}")
    print()
    for _, row in real.iterrows():
        flags = []
        if row.r_ci_lo > 0:
            flags.append("CI>0")
        if row.beats_control:
            flags.append("beats control")
        if row.setup in stable:
            flags.append("stable")
        mark = "PASS" if row.setup in passed else "----"
        print(f"  {mark}  {row.setup:<20} R={row.r_mean:+.4f} "
              f"[{row.r_ci_lo:+.4f},{row.r_ci_hi:+.4f}] "
              f"edge={row.edge_r:+.4f}  {' '.join(flags)}")

    if passed:
        print(f"\n  GATE PASSED by: {', '.join(passed)}")
        print("  -> proceed to stage 4/5/6")
    else:
        print("\n  GATE FAILED - no setup beats its base rate after costs with a")
        print("  CI clear of zero and stability across years.")
        print("  -> revisit setup definitions before modelling")

    s.to_csv(REPORTS / "stage3_setup_summary.csv", index=False)
    t.to_csv(REPORTS / "stage3_yearly_r.csv")
    print(f"\nwrote {REPORTS/'stage3_setup_summary.csv'}")


if __name__ == "__main__":
    main()
