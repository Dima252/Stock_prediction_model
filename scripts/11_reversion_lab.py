"""Reversion lab - exit sweep + model search. DEV ONLY (1999-2020).

The 2021-25 holdout has already been read twice. Everything here is decided on
dev with purged CV; the holdout is spent once, at the end, on one configuration
chosen before looking. Trials are counted and printed, because the maximum of N
noisy estimates is biased upward by roughly sigma*sqrt(2 ln N) and pretending
otherwise is how these projects fool themselves.

Statistic throughout: PAIRED MONTHLY EDGE vs the same-month control, which
differences out the market-wide monthly factor.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (BarrierConfig, N_JOBS, PROCESSED, RAW, REPORTS,
                        SECTOR_ETFS, VALID_END)
from src.features import (XS_RANK_COLS, add_cross_sectional_ranks,
                          add_market_context, build_ticker_features)
from src.labeling import label_panel
from src.setups import REVERSION_SETUPS, generate_events

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

CONTROL = "any_liquid"

EXITS = {
    "bracket_1.5R_7d":  (BarrierConfig(1.5, 1.0, 7), None),
    "bracket_1.0R_5d":  (BarrierConfig(1.0, 1.0, 5), None),
    "bracket_2.0R_10d": (BarrierConfig(2.0, 1.0, 10), None),
    "bracket_1.5R_3d":  (BarrierConfig(1.5, 1.0, 3), None),
    "signal_sma5":      (BarrierConfig(0.0, 1.0, 10), "exit_above_sma5"),
    "signal_sma5_wide": (BarrierConfig(0.0, 2.0, 10), "exit_above_sma5"),
    "signal_sma5_fast": (BarrierConfig(0.0, 1.0, 5), "exit_above_sma5"),
    # 2.0R/10d won the first sweep, but it was the WIDEST cell in that grid -
    # so it may be the edge of the grid rather than an optimum. Extend outward
    # until the edge stops improving, otherwise the "best" exit is an artefact
    # of where the search happened to stop.
    "bracket_2.0R_15d": (BarrierConfig(2.0, 1.0, 15), None),
    "bracket_2.5R_15d": (BarrierConfig(2.5, 1.0, 15), None),
    "bracket_3.0R_20d": (BarrierConfig(3.0, 1.0, 20), None),
    "bracket_4.0R_30d": (BarrierConfig(4.0, 1.0, 30), None),
    "trail_3atr_20d":   (BarrierConfig(0.0, 1.0, 20, trail_atr=3.0), None),
}


def paired_edge(r_a, d_a, r_b, d_b, seed=7):
    a = pd.Series(r_a).groupby(pd.PeriodIndex(pd.to_datetime(d_a), freq="M")).mean()
    b = pd.Series(r_b).groupby(pd.PeriodIndex(pd.to_datetime(d_b), freq="M")).mean()
    d = (a - b).dropna()
    if len(d) < 12:
        return np.nan, np.nan, np.nan, 0.0
    rng = np.random.default_rng(seed)
    bm = d.to_numpy()[rng.integers(0, len(d), (4000, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.quantile(bm, .025)),
            float(np.quantile(bm, .975)), float((d > 0).mean()))


def build_panel():
    from joblib import Parallel, delayed
    px = pd.read_parquet(RAW / "prices_liq.parquet")
    parts = Parallel(n_jobs=N_JOBS, batch_size=8)(
        delayed(build_ticker_features)(g) for _, g in px.groupby("ticker", sort=False))
    f = pd.concat(parts, ignore_index=True)
    ctx = pd.read_parquet(RAW / "context.parquet")
    cw = ctx.pivot(index="date", columns="ticker", values="close").sort_index()
    s2e = {v: k for k, v in SECTOR_ETFS.items()}
    smap = (pd.read_parquet(PROCESSED / "universe.parquet")
              .set_index("ticker")["sector"].map(s2e).to_dict())
    f = add_market_context(f, cw, smap)
    f = add_cross_sectional_ranks(f, XS_RANK_COLS)
    return f, generate_events(f)


def main() -> None:
    print("=== building panel ===")
    t = time.time()
    f, events = build_panel()
    keep = REVERSION_SETUPS + [CONTROL]
    events = events[events.setup.isin(keep)]
    ohlc = f[["date", "ticker", "open", "high", "low", "close", "atr14",
              "exit_above_sma5"]]
    bars = events[["ticker", "date"]].drop_duplicates()
    print(f"panel ready in {time.time()-t:.0f}s; {len(bars):,} unique bars\n")

    rows, trials = [], 0
    for ename, (cfg, sig) in EXITS.items():
        t = time.time()
        lab = label_panel(ohlc, bars, cfg, n_jobs=N_JOBS, exit_sig_col=sig)
        ev = events.merge(lab, on=["ticker", "date"], how="inner")
        ev = ev[ev.date <= VALID_END]
        ctl = ev[ev.setup == CONTROL]
        for s in REVERSION_SETUPS:
            g = ev[ev.setup == s]
            if len(g) < 500:
                continue
            e, lo, hi, pm = paired_edge(g.r_net.to_numpy(), g.date.to_numpy(),
                                        ctl.r_net.to_numpy(), ctl.date.to_numpy())
            rows.append({"exit": ename, "setup": s, "n": len(g),
                         "R": float(g.r_net.mean()), "edge": e, "lo": lo, "hi": hi,
                         "months_pos": pm, "hold": float(g.hold_days.mean())})
            trials += 1
        print(f"  {ename:<18} ({time.time()-t:.0f}s)  "
              f"control R={ctl.r_net.mean():+.4f}")

    df = pd.DataFrame(rows)
    df["clears"] = df.lo > 0
    df.to_csv(REPORTS / "stage11_exit_sweep.csv", index=False)

    print(f"\n=== EXIT x SETUP, dev only, paired monthly edge ({trials} cells) ===")
    piv = df.pivot_table(index="setup", columns="exit", values="edge")
    print(piv.round(4).to_string())

    print("\n=== which exit is best on average across setups? ===")
    by_exit = (df.groupby("exit")
                 .agg(mean_edge=("edge", "mean"), median_edge=("edge", "median"),
                      pct_clearing=("clears", "mean"), mean_hold=("hold", "mean"))
                 .sort_values("mean_edge", ascending=False))
    print(by_exit.round(4).to_string())

    print("\n=== top 15 individual cells ===")
    top = df.sort_values("edge", ascending=False).head(15)
    print(top[["setup", "exit", "n", "R", "edge", "lo", "hi", "months_pos",
               "clears"]].round(4).to_string(index=False))

    print(f"\ntrials: {trials}. The best cell above is the maximum of {trials} noisy")
    print("estimates and is therefore biased upward; judge the EXIT columns and the")
    print("setup ROWS, which average over many cells, rather than any single cell.")
    print(f"\nwrote {REPORTS/'stage11_exit_sweep.csv'}")


if __name__ == "__main__":
    main()
