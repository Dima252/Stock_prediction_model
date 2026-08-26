"""The logger's core claim: the live path reproduces the study exactly.

If live signals or live outcomes differ from what the backtest recorded for the
same (date, ticker), then any live/backtest gap you later observe is confounded
with a code difference, and the whole forward test proves nothing.

This replays the live code path over a historical window and diffs it against
events.parquet.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED, RAW
from src.live import (SETUP_NAME, build_panel, drop_incomplete_last_bar,
                      resolve, scan, session_is_final)

WARMUP_BARS = 320          # skip while rolling windows are still filling
TOL = 1e-4


def _load_window(n_sessions: int = 700) -> pd.DataFrame:
    """Last N sessions of equities + context, in the shape fetch_recent returns."""
    eq = pd.read_parquet(RAW / "prices_liq.parquet")
    ctx = pd.read_parquet(RAW / "context.parquet")
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    px = pd.concat([eq[cols], ctx[cols]], ignore_index=True)
    keep = np.sort(px["date"].unique())[-n_sessions:]
    return px[px["date"].isin(keep)].sort_values(["ticker", "date"]).reset_index(drop=True)


def test_session_completeness_guard():
    y = pd.Timestamp("2024-05-01")
    now = dt.datetime(2024, 5, 2, 12, 0, tzinfo=dt.timezone.utc)
    assert session_is_final(y, now), "a prior session must count as final"
    today_open = dt.datetime(2024, 5, 2, 14, 0, tzinfo=dt.timezone.utc)
    assert not session_is_final(pd.Timestamp("2024-05-02"), today_open), \
        "today's bar must NOT be final while the market is open"
    today_closed = dt.datetime(2024, 5, 2, 21, 30, tzinfo=dt.timezone.utc)
    assert session_is_final(pd.Timestamp("2024-05-02"), today_closed)
    assert not session_is_final(pd.Timestamp("2024-05-03"), today_closed), \
        "a future-dated bar can never be final"


def test_incomplete_bar_is_dropped():
    px = pd.DataFrame({"date": pd.to_datetime(["2024-05-01", "2024-05-02"]),
                       "ticker": ["A", "A"], "open": [1.0, 1.0], "high": [1.0, 1.0],
                       "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1]})
    mid_session = dt.datetime(2024, 5, 2, 14, 0, tzinfo=dt.timezone.utc)
    out = drop_incomplete_last_bar(px, mid_session)
    assert len(out) == 1 and out["date"].max() == pd.Timestamp("2024-05-01")


def test_live_signals_match_backtest():
    px = _load_window()
    panel = build_panel(px)
    sessions = np.sort(panel["date"].unique())
    start = pd.Timestamp(sessions[WARMUP_BARS])

    live = scan(panel, pd.DataFrame(), since=start)
    bt = pd.read_parquet(PROCESSED / "events.parquet")
    bt = bt[(bt.setup == SETUP_NAME) & (bt.date > start)]

    live_keys = set(zip(pd.to_datetime(live.signal_date), live.ticker))
    bt_keys = set(zip(pd.to_datetime(bt.date), bt.ticker))
    only_live, only_bt = live_keys - bt_keys, bt_keys - live_keys

    print(f"  live={len(live_keys):,} backtest={len(bt_keys):,} "
          f"only_live={len(only_live)} only_backtest={len(only_bt)}")
    overlap = len(live_keys & bt_keys)
    assert overlap > 100, f"only {overlap} shared signals - window too small"
    disagree = (len(only_live) + len(only_bt)) / max(len(bt_keys), 1)
    assert disagree < 0.02, (
        f"{disagree:.1%} of signals disagree between live and backtest "
        f"(only_live={len(only_live)}, only_bt={len(only_bt)})")


def test_live_outcomes_match_backtest():
    px = _load_window()
    panel = build_panel(px)
    sessions = np.sort(panel["date"].unique())
    start = pd.Timestamp(sessions[WARMUP_BARS])
    # leave room at the end so trades actually resolve inside the window
    end = pd.Timestamp(sessions[-15])

    live = scan(panel, pd.DataFrame(), since=start)
    live = live[pd.to_datetime(live.signal_date) <= end]
    live = resolve(live, panel)
    live = live[live.status == "CLOSED"]

    bt = pd.read_parquet(PROCESSED / "events.parquet")
    bt = bt[bt.setup == SETUP_NAME][["date", "ticker", "r_net", "hold_days"]]

    m = live.merge(bt, left_on=["signal_date", "ticker"],
                   right_on=["date", "ticker"], suffixes=("_live", "_bt"))
    assert len(m) > 100, f"only {len(m)} matched closed trades"

    dr = (m["r_net_live"].astype(float) - m["r_net_bt"].astype(float)).abs()
    dh = (m["hold_days_live"].astype(float) - m["hold_days_bt"].astype(float)).abs()
    bad = int((dr > TOL).sum())
    print(f"  matched={len(m):,}  max|dR|={dr.max():.2e}  "
          f"max|d hold|={dh.max():.0f}  mismatches={bad}")
    assert bad == 0, f"{bad} trades differ in realised R between live and backtest"
    assert dh.max() == 0, "holding periods differ between live and backtest"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
