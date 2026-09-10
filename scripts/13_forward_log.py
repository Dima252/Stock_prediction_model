"""Daily forward-test logger for `market_drop`. PAPER ONLY - places no orders.

Run once per day after the US close:

    ./.venv/Scripts/python.exe scripts/13_forward_log.py

It fetches the trailing window, scans every completed session it has not seen,
appends new signals to data/live/journal.csv, advances open positions, and prints
today's candidates plus a running live-vs-backtest comparison.

    --report-only   print the journal report without fetching
    --no-fetch      reuse the cached price file (for testing)
    --backfill N    on a first run, also log the last N calendar days of signals
                    so there is something to compare against immediately

The comparison against the backtest is the point of the whole exercise. Expect
live to come in below the study: the study's +0.094 same-day edge is an optimistic
point estimate that does not price in the full search behind it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED, REPORTS
from src.live import (JOURNAL, PRICE_CACHE, build_panel,
                      drop_incomplete_last_bar, fetch_recent, load_journal,
                      resolve, save_journal, scan)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# Signals dated on or before this are inside the study data and are NOT a
# forward test, however they were logged. They are kept in the journal (useful
# as a consistency check) but excluded from the forward statistics.
STUDY_END = "2025-12-30"

# Backtest reference numbers for the holdout period. Live is compared to these.
BACKTEST = {"r_net": 0.1127, "win_rate": None, "hold_days": 2.25,
            "same_day_edge": 0.0936, "edge_lo": 0.0324, "edge_hi": 0.1570}


def report(j: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("=== FORWARD-TEST JOURNAL ===")
    print("=" * 74)
    if j.empty:
        print("journal is empty - nothing logged yet")
        return

    closed = j[j.status == "CLOSED"]
    openp = j[j.status == "OPEN"]
    pend = j[j.status == "PENDING_ENTRY"]
    span = (f"{pd.to_datetime(j.signal_date).min().date()} .. "
            f"{pd.to_datetime(j.signal_date).max().date()}")
    print(f"signals logged : {len(j):,}   ({span})")
    print(f"  closed       : {len(closed):,}")
    print(f"  open         : {len(openp):,}")
    print(f"  pending entry: {len(pend):,}")

    if not openp.empty:
        print("\n--- currently open ---")
        cols = ["signal_date", "ticker", "entry_date", "entry_px", "stop_px",
                "hold_days"]
        print(openp[cols].to_string(index=False))

    if len(closed) == 0:
        print("\nno closed trades yet - nothing to compare")
        return

    # ---- forward-only: signals at or before the study end are not a forward test
    study_end = pd.Timestamp(STUDY_END)
    pre = closed[pd.to_datetime(closed.signal_date) <= study_end]
    fwd = closed[pd.to_datetime(closed.signal_date) > study_end]
    if len(pre):
        print(f"\n  ({len(pre):,} closed trades are dated on or before the study end "
              f"{STUDY_END}\n   and are EXCLUDED from the forward statistics below.)")
    if fwd.empty:
        print("\nno genuinely forward trades yet")
        return

    r = fwd["r_net"].astype(float)
    days = fwd.groupby("signal_date")["r_net"].mean()

    print("\n--- forward closed-trade performance (paper) ---")
    print(f"  trades            : {len(r):,}")
    print(f"  DISTINCT SIGNAL DAYS: {len(days)}   <- the real sample size")
    print(f"  win rate          : {(r > 0).mean():.3f}")
    print(f"  mean hold         : {fwd['hold_days'].astype(float).mean():.2f} sessions")
    print(f"  outcome mix       : {fwd.outcome.value_counts().to_dict()}")

    # THE UNIT MATTERS. This setup fires in clusters - a single oversold day can
    # produce 180 trades that share one market bounce. Averaging over trades
    # treats them as 180 independent observations and reports an interval several
    # times too narrow. Average over DAYS.
    print("\n  mean R, by unit of observation:")
    print(f"    per trade : {r.mean():+.4f}   <- MISLEADING, trades are clustered")
    lo = hi = np.nan
    if len(days) >= 8:
        rng = np.random.default_rng(7)
        bm = days.to_numpy()[rng.integers(0, len(days), (4000, len(days)))].mean(axis=1)
        lo, hi = float(np.quantile(bm, .025)), float(np.quantile(bm, .975))
        print(f"    per day   : {days.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   <- USE THIS")
    else:
        print(f"    per day   : {days.mean():+.4f}   (too few days for an interval)")

    biggest = fwd.signal_date.value_counts()
    share = biggest.head(3).sum() / len(fwd)
    print(f"\n  concentration: the 3 busiest days hold {share*100:.0f}% of all trades")

    print("\n--- live vs backtest ---")
    print(f"  backtest holdout R/trade : {BACKTEST['r_net']:+.4f}")
    print(f"  live R/day               : {days.mean():+.4f}")
    print(f"  backtest mean hold       : {BACKTEST['hold_days']:.2f} sessions")
    print(f"  live mean hold           : {fwd['hold_days'].astype(float).mean():.2f}")

    if len(days) < 40:
        print(f"\n  VERDICT: {len(days)} signal days is far too few to judge.")
        print("  The setup fires on ~47 active days a year, so a fair read needs")
        print("  roughly a full year. Do not act on these numbers yet.")
    elif lo == lo and lo > 0:
        print("\n  VERDICT: day-level CI clears zero. Encouraging, still early.")
    else:
        print("\n  VERDICT: day-level CI does NOT clear zero - no confirmed edge yet.")

    if len(closed) >= 30:
        c = closed.copy()
        c["month"] = pd.to_datetime(c.signal_date).dt.to_period("M")
        m = c.groupby("month")["r_net"].agg(["size", "mean"]).round(4)
        print("\n--- by month ---")
        print(m.to_string())

    closed.to_csv(REPORTS / "forward_closed_trades.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--backfill", type=int, default=0,
                    help="also log signals from the last N calendar days")
    args = ap.parse_args()

    j = load_journal()
    if args.report_only:
        report(j)
        return

    if args.no_fetch and PRICE_CACHE.exists():
        px = pd.read_parquet(PRICE_CACHE)
        print(f"using cached prices ({len(px):,} rows)")
    else:
        print("=== fetching recent prices ===")
        uni = pd.read_parquet(PROCESSED / "universe.parquet")
        px = fetch_recent(uni.ticker.tolist())
        px.to_parquet(PRICE_CACHE, index=False)
        print(f"fetched {len(px):,} rows, {px.ticker.nunique()} symbols")

    px = drop_incomplete_last_bar(px)
    if px.empty:
        print("no complete sessions available")
        return

    print("\n=== building features ===")
    panel = build_panel(px)
    latest = panel["date"].max()
    print(f"latest complete session: {pd.Timestamp(latest).date()}")

    # scan every session after the last one already logged
    since = None
    if not j.empty:
        since = pd.to_datetime(j.signal_date).max()
    elif args.backfill:
        since = pd.Timestamp(dt.date.today() - dt.timedelta(days=args.backfill))
    else:
        since = pd.Timestamp(latest) - pd.Timedelta(days=1)

    print(f"\n=== scanning sessions after {pd.Timestamp(since).date()} ===")
    new = scan(panel, j, since=since)
    if len(new):
        print(f"{len(new)} new signal(s)")
        j = pd.concat([j, new], ignore_index=True)
    else:
        print("no new signals")

    print("\n=== resolving open positions ===")
    j = resolve(j, panel)
    save_journal(j)
    print(f"journal saved -> {JOURNAL}")

    todays = j[pd.to_datetime(j.signal_date) == pd.Timestamp(latest)]
    print(f"\n=== candidates from {pd.Timestamp(latest).date()} "
          f"(enter at next open) ===")
    if todays.empty:
        print("  none - the tape was not broadly oversold enough today.")
        print("  This setup is idle most days by construction (~47 active days/yr).")
    else:
        cols = ["ticker", "close_at_signal", "rsi2", "oversold_breadth",
                "resid_ret5_atr", "entry_px", "stop_px", "status"]
        print(todays[cols].round(4).to_string(index=False))

    report(j)


if __name__ == "__main__":
    main()
