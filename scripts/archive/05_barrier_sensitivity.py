"""Robustness: does the result survive a different trade bracket?

The barriers are trading parameters, not model hyperparameters - but if an edge
exists ONLY at 2R/1R over 10 days and vanishes at 1.5R or 20 days, it is a
fitting artefact rather than a property of the market. A real effect should show
a smooth, sensible gradient across brackets.

Relabelling is cheap (~10s per config), so we sweep the whole grid.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import N_JOBS, PROCESSED, RAW, SECTOR_ETFS
from src.features import (XS_RANK_COLS, add_cross_sectional_ranks,
                          add_market_context, build_ticker_features)
from src.config import BarrierConfig
from src.labeling import label_panel
from src.setups import generate_events
from src.validation import block_bootstrap_ci

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

TARGETS = [1.0, 1.5, 2.0, 3.0]
STOPS = [0.75, 1.0, 1.5]
HOLDS = [5, 10, 20]


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    from joblib import Parallel, delayed
    px = pd.read_parquet(RAW / "prices_liq.parquet")
    groups = [g for _, g in px.groupby("ticker", sort=False)]
    parts = Parallel(n_jobs=N_JOBS, batch_size=8)(
        delayed(build_ticker_features)(g) for g in groups)
    f = pd.concat(parts, ignore_index=True)
    ctx = pd.read_parquet(RAW / "context.parquet")
    ctx_wide = ctx.pivot(index="date", columns="ticker", values="close").sort_index()
    s2e = {v: k for k, v in SECTOR_ETFS.items()}
    sector_map = (pd.read_parquet(PROCESSED / "universe.parquet")
                    .set_index("ticker")["sector"].map(s2e).to_dict())
    f = add_market_context(f, ctx_wide, sector_map)
    f = add_cross_sectional_ranks(f, XS_RANK_COLS)
    events = generate_events(f)
    return f, events


def main() -> None:
    print("=== building panel once ===")
    t = time.time()
    f, events = build_panel()
    bars = events[["ticker", "date"]].drop_duplicates()
    ohlc = f[["date", "ticker", "open", "high", "low", "close", "atr14"]]
    print(f"panel + events ready in {time.time()-t:.0f}s\n")

    rows = []
    grid = list(itertools.product(TARGETS, STOPS, HOLDS))
    print(f"=== sweeping {len(grid)} barrier configurations ===\n")
    for i, (tg, st, hold) in enumerate(grid, 1):
        cfg = BarrierConfig(target_atr=tg, stop_atr=st, max_hold_days=hold)
        t = time.time()
        lab = label_panel(ohlc, bars, cfg, n_jobs=N_JOBS)
        ev = events.merge(lab, on=["ticker", "date"], how="inner")
        ctrl = ev[ev.setup == "any_liquid"]["r_net"].mean()
        for setup, g in ev.groupby("setup", sort=False):
            lo, hi, nb = block_bootstrap_ci(g["r_net"].to_numpy(),
                                            g["date"].to_numpy(), "M")
            rows.append({"target_atr": tg, "stop_atr": st, "hold": hold,
                         "rr": tg / st, "setup": setup, "n": len(g),
                         "win_rate": float((g["outcome"] == 1).mean()),
                         "r_mean": float(g["r_net"].mean()),
                         "r_ci_lo": lo, "r_ci_hi": hi,
                         "edge_vs_control": float(g["r_net"].mean() - ctrl)})
        print(f"  [{i:>2}/{len(grid)}] target={tg} stop={st} hold={hold:>2}  "
              f"({time.time()-t:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent.parent.parent / "reports"
              / "stage5_barrier_sensitivity.csv", index=False)

    print("\n=== EDGE vs CONTROL by bracket (mean R above any_liquid) ===")
    print("positive = the setup beats simply buying a random liquid stock\n")
    for setup in sorted(df.setup.unique()):
        if setup == "any_liquid":
            continue
        sub = df[df.setup == setup]
        piv = sub.pivot_table(index=["target_atr", "stop_atr"], columns="hold",
                              values="edge_vs_control")
        share = float((sub.edge_vs_control > 0).mean())
        print(f"--- {setup}  (positive in {share*100:.0f}% of {len(sub)} brackets) ---")
        print(piv.round(4).to_string(), "\n")

    print("=== ROBUSTNESS SUMMARY ===")
    summ = (df[df.setup != "any_liquid"]
            .groupby("setup")
            .agg(brackets=("edge_vs_control", "size"),
                 pct_positive=("edge_vs_control", lambda s: float((s > 0).mean())),
                 median_edge=("edge_vs_control", "median"),
                 worst=("edge_vs_control", "min"),
                 best=("edge_vs_control", "max"))
            .sort_values("median_edge", ascending=False))
    print(summ.round(4).to_string())
    print("\na setup positive in ~all brackets is a property of the market;")
    print("one positive in only a few is a property of your parameter choice.")


if __name__ == "__main__":
    main()
