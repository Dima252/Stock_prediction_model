"""Stage 1 - triple-barrier labeling.

Conventions
-----------
* Decision is made on the CLOSE of bar t. Entry is the OPEN of bar t+1.
  Labelling from the close of t would be a one-bar look-ahead leak.
* Barriers are volatility-scaled off ATR14 measured at the close of t (the last
  fully-observable bar), so they are comparable across tickers and regimes.
* 1R == the stop distance == stop_atr * ATR14. Target of 2.0/1.0 ATR => +2R.
* GAPS ARE HONOURED. If a bar opens beyond a barrier the fill is the open, not
  the barrier level, so a gap through the stop produces a loss WORSE than -1R.
  Ignoring this is the most common way backtests flatter themselves.
* SAME-BAR AMBIGUITY. Daily bars cannot resolve whether the high or the low came
  first. We assume the STOP hit first. Pessimistic, honest, and errs in the
  direction you want to err.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BARRIERS, COST_BPS_ROUNDTRIP, BarrierConfig

# outcome codes
TARGET, STOP, TIMEOUT = 1, -1, 0


def true_range(high, low, close) -> np.ndarray:
    prev_close = np.concatenate([[np.nan], close[:-1]])
    a = high - low
    b = np.abs(high - prev_close)
    c = np.abs(low - prev_close)
    return np.nanmax(np.vstack([a, b, c]), axis=0)


def atr(high, low, close, n: int = 14) -> np.ndarray:
    """Wilder ATR, computed causally (value at i uses bars <= i)."""
    tr = true_range(high, low, close)
    out = np.full_like(tr, np.nan, dtype=float)
    if len(tr) <= n:
        return out
    seed = np.nanmean(tr[1:n + 1])
    out[n] = seed
    prev = seed
    for i in range(n + 1, len(tr)):
        prev = (prev * (n - 1) + tr[i]) / n
        out[i] = prev
    return out


def label_events_single(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    atr_arr: np.ndarray, event_idx: np.ndarray,
    cfg: BarrierConfig = BARRIERS, cost_bps: float = COST_BPS_ROUNDTRIP,
    exit_sig: np.ndarray | None = None,
) -> pd.DataFrame:
    """Label one ticker's events. `event_idx` are positions of decision bars (t).

    `exit_sig` is an optional causal per-bar boolean: when it is True at the CLOSE
    of a held bar, the trade exits at that close. This is how the reversion
    literature actually exits - Connors closes when price recovers above its
    5-day average, not at a fixed price target. A price target and a recovery
    signal are different trades and they label differently.
    """
    n = len(c)
    rows = []
    for t in event_idx:
        e = t + 1                                  # entry bar
        if e >= n:
            continue
        a = atr_arr[t]                             # ATR known at close of t
        if not np.isfinite(a) or a <= 0:
            continue
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            continue

        one_r = cfg.stop_atr * a
        use_target = cfg.target_atr > 0
        upper = entry + cfg.target_atr * a if use_target else np.inf
        lower = entry - cfg.stop_atr * a
        trail = cfg.trail_atr * a if cfg.trail_atr > 0 else 0.0
        last = min(e + cfg.max_hold_days - 1, n - 1)

        outcome, exit_px, exit_i = TIMEOUT, c[last], last
        run_max = entry            # highest high seen so far, for the trail
        for i in range(e, last + 1):
            # trailing stop ratchets up only; it never loosens below the initial
            stop_lvl = max(lower, run_max - trail) if trail > 0 else lower

            # --- gaps first: a bar that OPENS beyond a barrier fills at the open
            if o[i] <= stop_lvl:
                outcome, exit_px, exit_i = STOP, o[i], i
                break
            if use_target and o[i] >= upper:
                outcome, exit_px, exit_i = TARGET, o[i], i
                break
            hit_dn = l[i] <= stop_lvl
            hit_up = use_target and h[i] >= upper
            if hit_dn:                             # covers the ambiguous case:
                outcome, exit_px, exit_i = STOP, stop_lvl, i   # stop assumed first
                break
            if hit_up:
                outcome, exit_px, exit_i = TARGET, upper, i
                break
            # signal exit is evaluated on the CLOSE, after the intrabar barriers
            if exit_sig is not None and exit_sig[i]:
                outcome, exit_px, exit_i = TARGET if c[i] > entry else TIMEOUT, c[i], i
                break
            # only bars that survived intact extend the trail
            if h[i] > run_max:
                run_max = h[i]

        gross_r = (exit_px - entry) / one_r
        cost_r = (entry * cost_bps / 1e4) / one_r  # round-trip slippage+commission
        rows.append((t, e, exit_i, exit_i - e + 1, outcome, entry, exit_px,
                     a, one_r, gross_r, gross_r - cost_r))

    if not rows:
        return pd.DataFrame(columns=["t", "entry_i", "exit_i", "hold_days", "outcome",
                                     "entry_px", "exit_px", "atr", "one_r",
                                     "r_gross", "r_net"])
    return pd.DataFrame(rows, columns=["t", "entry_i", "exit_i", "hold_days", "outcome",
                                       "entry_px", "exit_px", "atr", "one_r",
                                       "r_gross", "r_net"])


def label_panel(px: pd.DataFrame, bars: pd.DataFrame,
                cfg: BarrierConfig = BARRIERS, n_jobs: int = 8,
                exit_sig_col: str | None = None) -> pd.DataFrame:
    """Label decision bars across the whole panel, parallel by ticker.

    px   : long panel with date, ticker, open/high/low/close and atr14
    bars : UNIQUE [ticker, date] decision bars.

    Setups are deliberately NOT handled here. One bar can fire several setups at
    once (a 20d breakout is frequently also a 52w-high momentum event); the trade
    outcome is a property of the bar, not of the setup that spotted it. Label each
    bar once and join setups back afterwards - correct, and cheaper.
    """
    from joblib import Parallel, delayed

    px = px.sort_values(["ticker", "date"])
    bars = bars[["ticker", "date"]].drop_duplicates()
    ev_by_tk = {tk: g for tk, g in bars.groupby("ticker", sort=False)}

    def _one(tk: str, g: pd.DataFrame):
        ge = ev_by_tk.get(tk)
        if ge is None or ge.empty:
            return None
        g = g.reset_index(drop=True)
        pos = pd.Series(g.index.values, index=g["date"].values)
        idx = pos.reindex(ge["date"].values).dropna()
        if idx.empty:
            return None
        ev_idx = idx.values.astype(int)
        sig = (g[exit_sig_col].to_numpy(bool)
               if exit_sig_col and exit_sig_col in g else None)
        lab = label_events_single(
            g["open"].to_numpy(float), g["high"].to_numpy(float),
            g["low"].to_numpy(float), g["close"].to_numpy(float),
            g["atr14"].to_numpy(float), ev_idx, cfg, exit_sig=sig)
        if lab.empty:
            return None
        lab["ticker"] = tk
        lab["date"] = g["date"].to_numpy()[lab["t"].to_numpy()]
        lab["entry_date"] = g["date"].to_numpy()[lab["entry_i"].to_numpy()]
        lab["exit_date"] = g["date"].to_numpy()[lab["exit_i"].to_numpy()]
        return lab

    parts = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_one)(tk, g) for tk, g in px.groupby("ticker", sort=False))
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # With no profit target (trend/trailing-stop scheme) NO event can ever be
    # TARGET, so "hit the target first" is not a usable label there - it would be
    # all zeros. Fall back to the only sensible definition: did it make money.
    if cfg.target_atr > 0:
        out["win"] = (out["outcome"] == TARGET).astype(int)
    else:
        out["win"] = (out["r_net"] > 0).astype(int)
    out["win_net"] = (out["r_net"] > 0).astype(int)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
