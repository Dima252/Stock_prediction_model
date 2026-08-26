"""Stage 0 - build the universe and download prices.

Survivorship bias is explicit here: the universe comes from CURRENT index
membership, so names that were delisted (the ones that blew up) are absent. This
inflates long-side results. It is reported, not hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONTEXT_SYMBOLS, END_DATE, RAW, START_DATE
from src.data import add_liquidity_filter, download_prices, fetch_universe, sanity_checks


def main() -> None:
    print("=== universe ===")
    uni = fetch_universe()
    tickers = uni.ticker.tolist()
    print(f"total {len(tickers)} tickers\n")

    print("=== context symbols ===")
    import yfinance as yf
    ctx = yf.download(CONTEXT_SYMBOLS, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True, group_by="ticker")
    frames = []
    for tk in CONTEXT_SYMBOLS:
        if isinstance(ctx.columns, pd.MultiIndex) and tk in ctx.columns.get_level_values(0):
            sub = ctx[tk].dropna(how="all").copy()
            if sub.empty:
                continue
            sub["ticker"] = tk
            frames.append(sub.reset_index())
    c = pd.concat(frames, ignore_index=True)
    c.columns = [str(x).lower().replace(" ", "_") for x in c.columns]
    c = c.rename(columns={"index": "date"})
    c["date"] = pd.to_datetime(c["date"]).dt.tz_localize(None).dt.normalize()
    c = c[["date", "ticker", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    c.to_parquet(RAW / "context.parquet", index=False)
    print(f"context: {len(c):,} rows, {c.ticker.nunique()} symbols\n")

    print("=== equity prices ===")
    download_prices(tickers)

    px = pd.read_parquet(RAW / "prices.parquet")
    sanity_checks(px)

    px = add_liquidity_filter(px)
    liq = px.groupby("date")["liquid"].sum()
    print(f"\nliquid names per day: median {liq.median():.0f}, "
          f"min {liq.min():.0f}, max {liq.max():.0f}")
    px.to_parquet(RAW / "prices_liq.parquet", index=False)
    print(f"-> {RAW / 'prices_liq.parquet'}")


if __name__ == "__main__":
    main()
