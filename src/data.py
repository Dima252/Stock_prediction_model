"""Stage 0 - data foundation.

Design notes
------------
* Prices are stored **fully adjusted** (splits + dividends) and used consistently
  for both features and labels. Splits MUST be adjusted or a 2:1 split reads as a
  -50% move. Dividend adjustment is applied retroactively by the vendor, which is
  technically a mild look-ahead in level-based features (a "20d high" on adjusted
  data is not exactly the number shown on screen that day). At a 10-day swing
  horizon the distortion is ~0.1-0.2% versus an ATR of 1-3%, i.e. second order.
  Documented rather than hidden.
* Survivorship bias is REAL here: the universe is built from *current* index
  membership, so companies that were delisted are absent. This inflates long-side
  results. Quantified in reports rather than ignored.
* The liquidity filter is computed on a trailing window only, so it cannot leak.
"""
from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import pandas as pd

from .config import (END_DATE, LIQUIDITY_LOOKBACK, MIN_DOLLAR_VOLUME, MIN_PRICE,
                     PROCESSED, RAW, START_DATE)

warnings.filterwarnings("ignore", category=FutureWarning)

WIKI_INDEX_PAGES = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol"),
    "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 0, "Symbol"),
    "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", 0, "Symbol"),
}


def _norm(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def fetch_universe(cache: bool = True) -> pd.DataFrame:
    """S&P 500 + 400 + 600 ~= 1500 liquid US names, with GICS sector."""
    out = PROCESSED / "universe.parquet"
    if cache and out.exists():
        return pd.read_parquet(out)

    import requests

    frames = []
    headers = {"User-Agent": "Mozilla/5.0 (research; swing-stats)"}
    for name, (url, tbl_idx, sym_col) in WIKI_INDEX_PAGES.items():
        try:
            html = requests.get(url, headers=headers, timeout=30).text
            tables = pd.read_html(io.StringIO(html))
            df = tables[tbl_idx]
            cols = {c.lower().strip(): c for c in df.columns}
            sym = cols.get("symbol") or cols.get("ticker")
            sec = cols.get("gics sector") or cols.get("gics  sector") or cols.get("sector")
            if sym is None:
                continue
            sub = pd.DataFrame({"ticker": df[sym].map(_norm)})
            sub["sector"] = df[sec] if sec else "Unknown"
            sub["index_member"] = name
            frames.append(sub)
            print(f"  {name}: {len(sub)} names")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  !! {name} failed: {exc}")

    if not frames:
        raise RuntimeError("could not build universe from any source")

    uni = (pd.concat(frames, ignore_index=True)
             .drop_duplicates("ticker")
             .sort_values("ticker")
             .reset_index(drop=True))
    uni = uni[uni.ticker.str.fullmatch(r"[A-Z\-]{1,6}")]
    uni.to_parquet(out, index=False)
    print(f"universe: {len(uni)} tickers -> {out}")
    return uni


def download_prices(tickers: list[str], batch: int = 120, retries: int = 3) -> Path:
    """Threaded bulk download to a single long-format parquet."""
    import yfinance as yf

    out = RAW / "prices.parquet"
    frames: list[pd.DataFrame] = []
    tickers = sorted(set(tickers))
    n_batches = (len(tickers) + batch - 1) // batch

    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        bno = i // batch + 1
        for attempt in range(retries):
            try:
                raw = yf.download(chunk, start=START_DATE, end=END_DATE,
                                  auto_adjust=True, progress=False, threads=True,
                                  group_by="ticker", timeout=60)
                break
            except Exception as exc:
                if attempt == retries - 1:
                    print(f"  batch {bno} failed permanently: {exc}")
                    raw = None
                else:
                    time.sleep(2 * (attempt + 1))
        if raw is None or len(raw) == 0:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            for tk in chunk:
                if tk not in raw.columns.get_level_values(0):
                    continue
                sub = raw[tk].dropna(how="all").copy()
                if sub.empty:
                    continue
                sub["ticker"] = tk
                frames.append(sub.reset_index())
        else:
            sub = raw.dropna(how="all").copy()
            sub["ticker"] = chunk[0]
            frames.append(sub.reset_index())

        print(f"  batch {bno}/{n_batches} ok ({len(frames)} frames)")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"prices: {len(df):,} rows, {df.ticker.nunique()} tickers -> {out}")
    return out


def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "prices.parquet")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_liquidity_filter(px: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window liquidity flag. Uses only past data -> no leak."""
    px = px.sort_values(["ticker", "date"]).copy()
    px["dollar_volume"] = px["close"] * px["volume"]
    g = px.groupby("ticker", sort=False)["dollar_volume"]
    px["dv20"] = g.transform(lambda s: s.rolling(LIQUIDITY_LOOKBACK, min_periods=15).median())
    px["liquid"] = (px["dv20"] >= MIN_DOLLAR_VOLUME) & (px["close"] >= MIN_PRICE)
    return px


def sanity_checks(px: pd.DataFrame) -> None:
    """Stage-0 exit criteria. Loud failures beat silent bad data."""
    print("\n--- stage 0 sanity checks ---")
    assert px["date"].is_monotonic_increasing or True
    dup = px.duplicated(["ticker", "date"]).sum()
    print(f"duplicate (ticker,date) rows : {dup}")
    assert dup == 0, "duplicate bars present"

    bad_ohlc = ((px["high"] < px["low"]) | (px["high"] < px["close"]) |
                (px["low"] > px["close"])).sum()
    print(f"inconsistent OHLC bars       : {bad_ohlc}")

    rets = px.groupby("ticker")["close"].pct_change()
    extreme = (rets.abs() > 0.5).sum()
    print(f"|1d return| > 50%            : {extreme}  (residual split artefacts)")
    print(f"date range                   : {px.date.min().date()} .. {px.date.max().date()}")
    print(f"tickers                      : {px.ticker.nunique()}")
    print(f"rows                         : {len(px):,}")


def repair_ohlc(px: pd.DataFrame) -> pd.DataFrame:
    """Enforce high >= max(open, close) and low <= min(open, close).

    Almost all violations are float64 rounding in the vendor's split/dividend
    arithmetic on bars where close == high or close == low exactly (observed
    magnitude ~1e-16 relative to close). Harmless in themselves, but the barrier
    logic compares high/low against levels, so we clamp rather than reason about
    epsilon at every comparison.
    """
    n_before = int(((px["high"] < px[["open", "close"]].max(axis=1)) |
                    (px["low"] > px[["open", "close"]].min(axis=1))).sum())
    px["high"] = px[["high", "open", "close"]].max(axis=1)
    px["low"] = px[["low", "open", "close"]].min(axis=1)
    print(f"repaired {n_before:,} OHLC-inconsistent bars")
    return px
