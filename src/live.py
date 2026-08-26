"""Forward-test logger for `market_drop`. Paper only - it never places an order.

Design commitments
------------------
1. **Same code path as the backtest.** Signals come from `src.setups.s_market_drop`
   and outcomes from `src.labeling.label_events_single` - the identical functions
   the study used. If the logger had its own copy of the rules, a live/backtest
   discrepancy would be indistinguishable from a difference in the code.

2. **Never act on an incomplete bar.** Intraday, the vendor returns a partial bar
   for today. `rsi2 < 10` computed on a partial bar is not the signal that will
   exist at the close, and acting on it would silently manufacture look-ahead.
   The last bar is dropped unless the session has closed.

3. **Decide on the close, enter at the next open.** Same convention as the study.
   A signal logged tonight is filled at tomorrow's open, and the journal records
   both dates so the lag is auditable.

4. **Idempotent.** Re-running on the same day never duplicates a signal, and
   catch-up after missed days is automatic.

5. **A position is only closed when the data proves it closed.** A trade whose
   labeller run ends on the last available bar is still OPEN, not a timeout - the
   distinction matters and is handled explicitly.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .config import (CONTEXT_SYMBOLS, DATA, MIN_DOLLAR_VOLUME, MIN_PRICE,
                     PROCESSED, REVERSION_EXIT, REVERSION_EXIT_SIGNAL)
from .data import repair_ohlc
from .features import (XS_RANK_COLS, add_cross_sectional_ranks,
                       add_market_context, build_ticker_features)
from .labeling import STOP, TARGET, TIMEOUT, label_events_single
from .setups import SETUPS

LIVE = DATA / "live"
LIVE.mkdir(parents=True, exist_ok=True)
JOURNAL = LIVE / "journal.csv"
PRICE_CACHE = LIVE / "prices_recent.parquet"

SETUP_NAME = "market_drop"
LOOKBACK_DAYS = 900          # enough for the 200d SMA, 52w windows and 60d beta

JOURNAL_COLS = [
    "signal_date", "ticker", "setup", "status", "entry_date", "entry_px",
    "atr14", "one_r", "stop_px", "max_hold_days", "exit_rule",
    "exit_date", "exit_px", "hold_days", "outcome", "r_gross", "r_net",
    "rsi2", "oversold_breadth", "resid_ret5_atr", "dist_sma200_atr",
    "close_at_signal", "logged_at",
]

# US market closes 16:00 ET == 20:00/21:00 UTC depending on DST. Use 21:00 UTC as
# the conservative cutoff: before it, today's bar is treated as incomplete.
CLOSE_CUTOFF_UTC = dt.time(21, 0)


# ------------------------------------------------------------------ data layer
def session_is_final(bar_date: pd.Timestamp, now_utc: dt.datetime | None = None) -> bool:
    """Is the bar dated `bar_date` a completed session?"""
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    d = pd.Timestamp(bar_date).date()
    if d < now.date():
        return True
    if d > now.date():
        return False
    return now.time() >= CLOSE_CUTOFF_UTC


def drop_incomplete_last_bar(px: pd.DataFrame,
                             now_utc: dt.datetime | None = None) -> pd.DataFrame:
    """Remove today's bar while the session is still open."""
    if px.empty:
        return px
    last = px["date"].max()
    if not session_is_final(last, now_utc):
        print(f"  dropping incomplete session {pd.Timestamp(last).date()} "
              f"(market still open)")
        return px[px["date"] < last]
    return px


def fetch_recent(tickers: list[str], lookback_days: int = LOOKBACK_DAYS,
                 batch: int = 150) -> pd.DataFrame:
    """Download the trailing window for the whole universe plus context symbols."""
    import yfinance as yf

    start = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
    frames = []
    syms = sorted(set(tickers) | set(CONTEXT_SYMBOLS))
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        raw = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                          threads=True, group_by="ticker", timeout=60)
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            for tk in chunk:
                if tk not in raw.columns.get_level_values(0):
                    continue
                s = raw[tk].dropna(how="all").copy()
                if s.empty:
                    continue
                s["ticker"] = tk
                frames.append(s.reset_index())
        else:
            s = raw.dropna(how="all").copy()
            s["ticker"] = chunk[0]
            frames.append(s.reset_index())
        print(f"  fetched {min(i+batch, len(syms))}/{len(syms)}")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    return df.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_panel(px: pd.DataFrame) -> pd.DataFrame:
    """Features for the whole universe - identical functions to the study."""
    from joblib import Parallel, delayed
    from .config import N_JOBS, SECTOR_ETFS

    eq = px[~px.ticker.isin(CONTEXT_SYMBOLS)].copy()
    ctx = px[px.ticker.isin(CONTEXT_SYMBOLS)].copy()

    eq = repair_ohlc(eq)
    eq["dollar_volume"] = eq["close"] * eq["volume"]
    g = eq.groupby("ticker", sort=False)["dollar_volume"]
    eq["dv20"] = g.transform(lambda s: s.rolling(20, min_periods=15).median())
    eq["liquid"] = (eq["dv20"] >= MIN_DOLLAR_VOLUME) & (eq["close"] >= MIN_PRICE)

    parts = Parallel(n_jobs=N_JOBS, batch_size=8)(
        delayed(build_ticker_features)(gg) for _, gg in eq.groupby("ticker", sort=False))
    f = pd.concat(parts, ignore_index=True)

    cw = ctx.pivot(index="date", columns="ticker", values="close").sort_index()
    s2e = {v: k for k, v in SECTOR_ETFS.items()}
    smap = (pd.read_parquet(PROCESSED / "universe.parquet")
              .set_index("ticker")["sector"].map(s2e).to_dict())
    f = add_market_context(f, cw, smap)
    return add_cross_sectional_ranks(f, XS_RANK_COLS)


# ----------------------------------------------------------------- the journal
def load_journal() -> pd.DataFrame:
    if JOURNAL.exists():
        j = pd.read_csv(JOURNAL, parse_dates=["signal_date", "entry_date", "exit_date"])
        for c in JOURNAL_COLS:
            if c not in j.columns:
                j[c] = np.nan
        return j[JOURNAL_COLS]
    return pd.DataFrame(columns=JOURNAL_COLS)


def save_journal(j: pd.DataFrame) -> None:
    j = j.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    j.to_csv(JOURNAL, index=False)


def scan(panel: pd.DataFrame, journal: pd.DataFrame,
         since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Detect new signals on every completed session not yet scanned.

    Catch-up is automatic: whatever sessions exist after the last logged signal
    date are all scanned, so a missed day does not silently vanish.
    """
    mask = SETUPS[SETUP_NAME](panel).fillna(False).to_numpy()
    liq = panel["liquid"].to_numpy() if "liquid" in panel else np.ones(len(panel), bool)
    cand = panel.loc[mask & liq].copy()
    if since is not None:
        cand = cand[cand["date"] > since]
    if cand.empty:
        return pd.DataFrame(columns=JOURNAL_COLS)

    # entry is the NEXT session's open; a signal on the latest bar has no entry
    # bar yet and is held back until it does
    nxt = (panel[["date", "ticker", "open"]]
           .sort_values(["ticker", "date"]))
    nxt["entry_date"] = nxt.groupby("ticker", sort=False)["date"].shift(-1)
    nxt["entry_px"] = nxt.groupby("ticker", sort=False)["open"].shift(-1)
    cand = cand.merge(nxt[["date", "ticker", "entry_date", "entry_px"]],
                      on=["date", "ticker"], how="left")

    rows = []
    for _, r in cand.iterrows():
        one_r = REVERSION_EXIT.stop_atr * r["atr14"]
        if not np.isfinite(one_r) or one_r <= 0:
            continue
        pending = not np.isfinite(r.get("entry_px", np.nan))
        rows.append({
            "signal_date": r["date"], "ticker": r["ticker"], "setup": SETUP_NAME,
            "status": "PENDING_ENTRY" if pending else "OPEN",
            "entry_date": r["entry_date"], "entry_px": r["entry_px"],
            "atr14": r["atr14"], "one_r": one_r,
            "stop_px": (r["entry_px"] - one_r) if not pending else np.nan,
            "max_hold_days": REVERSION_EXIT.max_hold_days,
            "exit_rule": "close > SMA5, or stop, or 10-session cap",
            "exit_date": pd.NaT, "exit_px": np.nan, "hold_days": np.nan,
            "outcome": np.nan, "r_gross": np.nan, "r_net": np.nan,
            "rsi2": r.get("rsi2"), "oversold_breadth": r.get("oversold_breadth"),
            "resid_ret5_atr": r.get("resid_ret5_atr"),
            "dist_sma200_atr": r.get("dist_sma200_atr"),
            "close_at_signal": r["close"],
            "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
        })
    new = pd.DataFrame(rows, columns=JOURNAL_COLS)
    if new.empty or journal.empty:
        return new
    seen = set(zip(pd.to_datetime(journal.signal_date), journal.ticker))
    keep = [k not in seen for k in zip(pd.to_datetime(new.signal_date), new.ticker)]
    return new[keep]


def resolve(journal: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Advance every unresolved position using the study's own labeller."""
    if journal.empty:
        return journal
    last_bar = panel["date"].max()
    by_tk = {tk: g.sort_values("date").reset_index(drop=True)
             for tk, g in panel.groupby("ticker", sort=False)}

    j = journal.copy()
    # pandas 3.0 refuses to put a string into an all-NaN float column, so the
    # text columns are coerced up front rather than at first assignment.
    for c in ("status", "outcome", "exit_rule"):
        if c in j:
            j[c] = j[c].astype(object)
    for c in ("entry_date", "exit_date"):
        if c in j:
            j[c] = pd.to_datetime(j[c], errors="coerce")
    for i, row in j.iterrows():
        if row["status"] == "CLOSED":
            continue
        g = by_tk.get(row["ticker"])
        if g is None:
            continue
        pos = g.index[g["date"] == pd.Timestamp(row["signal_date"])]
        if len(pos) == 0:
            continue
        t = int(pos[0])
        if t + 1 >= len(g):                      # entry bar not printed yet
            j.at[i, "status"] = "PENDING_ENTRY"
            continue

        sig = (g[REVERSION_EXIT_SIGNAL].to_numpy(bool)
               if REVERSION_EXIT_SIGNAL in g else None)
        lab = label_events_single(
            g["open"].to_numpy(float), g["high"].to_numpy(float),
            g["low"].to_numpy(float), g["close"].to_numpy(float),
            g["atr14"].to_numpy(float), np.array([t]), REVERSION_EXIT,
            exit_sig=sig)
        if lab.empty:
            continue
        L = lab.iloc[0]

        # A run that ends on the final available bar without hitting a barrier is
        # STILL OPEN - the data simply has not caught up. Recording that as a
        # timeout would fabricate a closed trade at today's price.
        ran_out = (int(L["exit_i"]) >= len(g) - 1
                   and L["outcome"] == TIMEOUT
                   and int(L["hold_days"]) < REVERSION_EXIT.max_hold_days)
        j.at[i, "entry_date"] = g["date"].iloc[int(L["entry_i"])]
        j.at[i, "entry_px"] = L["entry_px"]
        j.at[i, "one_r"] = L["one_r"]
        j.at[i, "stop_px"] = L["entry_px"] - L["one_r"]
        if ran_out:
            j.at[i, "status"] = "OPEN"
            j.at[i, "hold_days"] = int(L["hold_days"])
            continue
        j.at[i, "status"] = "CLOSED"
        j.at[i, "exit_date"] = g["date"].iloc[int(L["exit_i"])]
        j.at[i, "exit_px"] = L["exit_px"]
        j.at[i, "hold_days"] = int(L["hold_days"])
        j.at[i, "outcome"] = {TARGET: "WIN", STOP: "STOP",
                              TIMEOUT: "TIMEOUT"}[int(L["outcome"])]
        j.at[i, "r_gross"] = L["r_gross"]
        j.at[i, "r_net"] = L["r_net"]
    return j
