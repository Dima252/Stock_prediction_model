"""Stage 2 - features, setup detection, triple-barrier labels -> event table.

Output: data/processed/events.parquet, one row per (ticker, date, setup) with all
features as of the decision bar and the realised outcome of the trade.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (MOMENTUM_EXIT, N_JOBS, PROCESSED, RAW, REVERSION_EXIT,
                        REVERSION_EXIT_SIGNAL, SECTOR_ETFS)
from src.features import (FEATURE_COLS, XS_RANK_COLS,
                          add_cross_sectional_ranks, add_market_context,
                          build_ticker_features)
from src.labeling import label_panel
from src.setups import REVERSION_SETUPS, TREND_SETUPS, generate_events

SECTOR_TO_ETF = {v: k for k, v in SECTOR_ETFS.items()}

KEEP = (["date", "ticker", "open", "high", "low", "close", "volume", "liquid",
         "atr14", "ret20", "ret60", "exit_above_sma5"] + FEATURE_COLS)


def main() -> None:
    t_start = time.time()
    print("=== load prices ===")
    px = pd.read_parquet(RAW / "prices_liq.parquet")
    print(f"{len(px):,} rows, {px.ticker.nunique()} tickers")

    print("\n=== per-ticker features (parallel) ===")
    from joblib import Parallel, delayed
    groups = [g for _, g in px.groupby("ticker", sort=False)]
    del px
    gc.collect()

    parts = Parallel(n_jobs=N_JOBS, batch_size=8, verbose=1)(
        delayed(build_ticker_features)(g) for g in groups)
    del groups
    gc.collect()
    f = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"feature panel: {len(f):,} rows x {f.shape[1]} cols  "
          f"({time.time()-t_start:.0f}s)")

    print("\n=== market context + relative strength ===")
    ctx = pd.read_parquet(RAW / "context.parquet")
    ctx_wide = ctx.pivot(index="date", columns="ticker", values="close").sort_index()
    sector_map = (pd.read_parquet(PROCESSED / "universe.parquet")
                    .set_index("ticker")["sector"].map(SECTOR_TO_ETF).to_dict())
    f = add_market_context(f, ctx_wide, sector_map)
    f = add_cross_sectional_ranks(f, XS_RANK_COLS)

    # keep only what is needed downstream, and shrink
    keep = [c for c in dict.fromkeys(KEEP) if c in f.columns]
    f = f[keep]
    for c in f.columns:
        if f[c].dtype == np.float64:
            f[c] = f[c].astype(np.float32)
    gc.collect()
    print(f"panel ready: {len(f):,} rows x {f.shape[1]} cols  "
          f"({f.memory_usage(deep=True).sum()/1e9:.2f} GB)")

    print("\n=== setup detection ===")
    events = generate_events(f)
    print(f"total {len(events):,} events")

    print("\n=== triple-barrier labelling (exit matched to setup family) ===")
    # Trend setups get a trailing stop and no profit target; reversion setups get
    # a tight fast bracket. Judging a breakout on a 2R/10-day cap tests the cap.
    # The control is labelled under BOTH schemes so each family is compared
    # against a control that was exited the same way it was.
    ohlc = f[["date", "ticker", "open", "high", "low", "close", "atr14",
              "exit_above_sma5"]]
    exit_groups = {
        "trend": (MOMENTUM_EXIT, TREND_SETUPS + ["any_liquid"], None),
        "reversion": (REVERSION_EXIT, REVERSION_SETUPS + ["any_liquid"],
                      REVERSION_EXIT_SIGNAL),
    }
    parts, t0 = [], time.time()
    for gname, (cfg, setups, sig) in exit_groups.items():
        ev_g = events[events.setup.isin(setups)]
        bars_g = ev_g[["ticker", "date"]].drop_duplicates()
        print(f"  {gname:<10} target={cfg.target_atr} stop={cfg.stop_atr} "
              f"hold={cfg.max_hold_days} trail={cfg.trail_atr} sig={sig}  "
              f"({len(bars_g):,} bars)")
        lab = label_panel(ohlc, bars_g, cfg, n_jobs=N_JOBS, exit_sig_col=sig)
        m = ev_g.merge(lab, on=["ticker", "date"], how="inner")
        m["exit_scheme"] = gname
        m.loc[m.setup == "any_liquid", "setup"] = f"any_liquid_{gname}"
        parts.append(m)
    ev = pd.concat(parts, ignore_index=True)
    print(f"labelled {len(ev):,} events in {time.time()-t0:.0f}s")

    print("\n=== join features ===")
    feat_cols = [c for c in FEATURE_COLS if c in f.columns]
    ev = ev.merge(f[["date", "ticker"] + feat_cols], on=["date", "ticker"], how="left")

    # a feature row that is all-NaN means the warm-up window was not satisfied
    n_before = len(ev)
    ev = ev.dropna(subset=["dist_sma200_atr", "ret60_z", "rv_rank_2y"])
    print(f"dropped {n_before - len(ev):,} events inside indicator warm-up")

    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    ev = ev.sort_values(["date", "ticker"]).reset_index(drop=True)
    ev.to_parquet(PROCESSED / "events.parquet", index=False)

    print(f"\n=== event table: {len(ev):,} rows -> events.parquet ===")
    summ = ev.groupby("setup").agg(n=("r_net", "size"), win=("win", "mean"),
                                   r=("r_net", "mean"), hold=("hold_days", "mean"))
    print(summ.sort_values("r", ascending=False).round(4).to_string())
    print(f"\ndate range {ev.date.min().date()} .. {ev.date.max().date()}")
    print(f"total wall time {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
