"""Feature engineering. Every feature is causal (uses bars <= t only) and
stationary (a ratio, z-score, or percentile rank - never a raw price or volume).

Earnings: 25 years of true earnings dates for ~1500 names is not available free,
so we DETECT earnings-like events causally - an abnormal overnight gap coinciding
with a volume spike - and derive "days since" / "expected days until next" from a
~63 trading-day cadence. It is a proxy and is labelled as one, but it captures the
thing that matters: an earnings print landing inside the holding window turns the
outcome distribution into a coin flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labeling import atr, true_range

EPS = 1e-9


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile of the current value within its own trailing window."""
    return s.rolling(window, min_periods=max(5, window // 4)).rank(pct=True)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(5, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 4)).std()
    return (s - m) / (sd + EPS)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.Series(true_range(h.to_numpy(float), l.to_numpy(float), c.to_numpy(float)),
                   index=c.index)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=c.index).ewm(alpha=1 / n, adjust=False).mean() / (atr_n + EPS)
    mdi = 100 * pd.Series(minus, index=c.index).ewm(alpha=1 / n, adjust=False).mean() / (atr_n + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _earnings_proxy(g: pd.DataFrame) -> pd.DataFrame:
    """Causally detect earnings-like events and derive distance-to-next."""
    gap = (g["open"] / g["close"].shift(1) - 1).abs()
    typ_gap = gap.rolling(120, min_periods=40).median().clip(lower=0.002)
    volr = g["volume"] / (g["volume"].rolling(20, min_periods=10).mean() + EPS)
    is_ev = ((gap > 4 * typ_gap) & (volr > 1.8)).fillna(False).to_numpy()

    idx = np.arange(len(g), dtype=float)
    last = np.where(is_ev, idx, np.nan)
    last = pd.Series(last).ffill().to_numpy()
    since = idx - last                       # nan until the first detected event
    since = np.clip(since, 0, 200)

    g["days_since_earn"] = since
    # next print expected ~63 trading days after the last one
    g["days_to_earn"] = np.clip(63 - (since % 63), 0, 63)
    g["earn_in_window"] = ((g["days_to_earn"] <= 10) & np.isfinite(since)).astype(np.int8)
    return g


def build_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    """All per-ticker features. `g` is one ticker's bars, sorted by date."""
    g = g.sort_values("date").reset_index(drop=True)
    o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
    hn, ln, cn = h.to_numpy(float), l.to_numpy(float), c.to_numpy(float)

    a14 = pd.Series(atr(hn, ln, cn, 14), index=g.index)
    a50 = pd.Series(atr(hn, ln, cn, 50), index=g.index)
    g["atr14"] = a14
    g["atr_pct"] = a14 / (c + EPS)

    # ---- trend / location (distances in ATR units => comparable across names)
    for w in (20, 50, 200):
        sma = c.rolling(w, min_periods=w // 2).mean()
        g[f"dist_sma{w}_atr"] = (c - sma) / (a14 + EPS)
    g["sma20_50_atr"] = (c.rolling(20, min_periods=10).mean()
                         - c.rolling(50, min_periods=25).mean()) / (a14 + EPS)
    hi52 = h.rolling(252, min_periods=100).max()
    lo52 = l.rolling(252, min_periods=100).min()
    g["pct_52w_range"] = (c - lo52) / (hi52 - lo52 + EPS)
    g["dist_52w_high_atr"] = (c - hi52) / (a14 + EPS)
    g["adx14"] = _adx(h, l, c, 14)

    # ---- momentum
    for w in (5, 20, 60):
        g[f"ret{w}"] = c.pct_change(w)
        g[f"ret{w}_z"] = _zscore(g[f"ret{w}"], 252)
    g["rsi14"] = _rsi(c, 14)
    g["ret1_z"] = _zscore(c.pct_change(), 252)

    # ---- volatility
    ret1 = c.pct_change()
    rv20 = ret1.rolling(20, min_periods=10).std()
    g["rv20"] = rv20
    g["rv_rank_2y"] = _rolling_rank(rv20, 504)
    g["atr_expansion"] = a14 / (a50 + EPS)
    bb_sd = c.rolling(20, min_periods=10).std()
    g["bb_width"] = (4 * bb_sd) / (c.rolling(20, min_periods=10).mean() + EPS)
    g["bb_width_rank"] = _rolling_rank(g["bb_width"], 252)
    g["vol_of_vol"] = rv20.rolling(60, min_periods=30).std() / (rv20 + EPS)

    # ---- volume
    g["vol_ratio20"] = v / (v.rolling(20, min_periods=10).mean() + EPS)
    g["vol_ratio_rank"] = _rolling_rank(g["vol_ratio20"], 252)
    dv = c * v
    g["dollar_vol_rank"] = _rolling_rank(dv, 252)
    obv = (np.sign(ret1.fillna(0)) * v).cumsum()
    g["obv_slope20"] = (obv - obv.shift(20)) / (v.rolling(20, min_periods=10).mean() * 20 + EPS)

    # ---- structure
    hh20 = h.rolling(20, min_periods=10).max()
    ll20 = l.rolling(20, min_periods=10).min()
    g["dist_hh20_atr"] = (c - hh20) / (a14 + EPS)
    g["dist_ll20_atr"] = (c - ll20) / (a14 + EPS)
    g["range20_atr"] = (hh20 - ll20) / (a14 + EPS)
    # bars since the 20d high was set (0 == the high is today)
    g["days_since_hh20"] = h.rolling(20, min_periods=10).apply(
        lambda x: float(len(x) - 1 - np.argmax(x)), raw=True)
    g["close_loc_bar"] = (c - l) / (h - l + EPS)
    g["gap_atr"] = (o - c.shift(1)) / (a14 + EPS)

    g = _earnings_proxy(g)
    g = _practitioner_features(g, o, h, l, c, v, a14, hi52, lo52)
    return _reversion_features(g, o, h, l, c, v, a14)


def _reversion_features(g, o, h, l, c, v, a14):
    """Inputs specific to mean reversion - depth, exhaustion, and capitulation."""
    # Internal Bar Strength: where in its own range the bar closed. One of the
    # most robust short-horizon reversion signals in the literature.
    g["ibs"] = (c - l) / (h - l + EPS)
    g["ibs_2d"] = g["ibs"].rolling(2, min_periods=2).mean()

    # Bollinger %b and how far below the lower band, in ATR
    ma20 = c.rolling(20, min_periods=10).mean()
    sd20 = c.rolling(20, min_periods=10).std()
    upper, lower = ma20 + 2 * sd20, ma20 - 2 * sd20
    g["bb_pctb"] = (c - lower) / (upper - lower + EPS)
    g["dist_bb_lower_atr"] = (c - lower) / (a14 + EPS)

    # Connors' cumulative RSI(2): sum over the last 2-3 days, catches a grind
    # down as well as a single spike
    g["cum_rsi2_2"] = g["rsi2"].rolling(2, min_periods=2).sum()
    g["cum_rsi2_3"] = g["rsi2"].rolling(3, min_periods=3).sum()

    # depth of the drop, volatility-normalised
    g["ret5_atr"] = (c - c.shift(5)) / (a14 + EPS)
    g["ret3_atr"] = (c - c.shift(3)) / (a14 + EPS)
    g["ret1_atr"] = (c - c.shift(1)) / (a14 + EPS)

    # capitulation: is the selling on heavy volume, or is it a quiet drift?
    dn = (c.diff() < 0)
    vr = v / (v.rolling(20, min_periods=10).mean() + EPS)
    g["dn_vol_ratio"] = vr.where(dn).rolling(5, min_periods=1).mean()
    g["down_days_2"] = dn.astype(int).rolling(2, min_periods=2).sum()

    # how stretched below the short average
    g["dist_sma10_atr"] = (c - c.rolling(10, min_periods=5).mean()) / (a14 + EPS)
    # Connors' exit condition, precomputed as a causal per-bar boolean
    g["exit_above_sma5"] = (g["dist_sma5_atr"] > 0).astype(np.int8)
    return g


def _practitioner_features(g, o, h, l, c, v, a14, hi52, lo52):
    """Features required by the documented rule sets of working traders.

    Sources: Minervini's Trend Template and VCP, O'Neil's CANSLIM base/RS rating,
    Weinstein's stage analysis, Darvas boxes, Connors' RSI(2) work. Each of these
    people published specific, checkable criteria; these are the inputs to them.
    """
    # ---- moving-average structure (Minervini trend template, Weinstein stages)
    sma50 = c.rolling(50, min_periods=25).mean()
    sma150 = c.rolling(150, min_periods=75).mean()
    sma200 = c.rolling(200, min_periods=100).mean()
    g["sma150_slope20"] = (sma150 / sma150.shift(20) - 1)
    g["sma200_slope20"] = (sma200 / sma200.shift(20) - 1)
    g["ma_stack"] = ((c > sma50) & (sma50 > sma150)
                     & (sma150 > sma200)).astype(np.int8)
    g["dist_sma150_atr"] = (c - sma150) / (a14 + EPS)
    g["above_52w_low_pct"] = c / (lo52 + EPS) - 1.0
    g["below_52w_high_pct"] = 1.0 - c / (hi52 + EPS)

    # ---- O'Neil relative-strength rating inputs (blended, then ranked x-section)
    for w in (126, 189, 252):
        g[f"ret{w}"] = c.pct_change(w)
    g["rs_score"] = (0.4 * g["ret60"].fillna(0) + 0.2 * g["ret126"].fillna(0)
                     + 0.2 * g["ret189"].fillna(0) + 0.2 * g["ret252"].fillna(0))

    # ---- base / consolidation geometry (O'Neil bases, Darvas boxes)
    for w in (35, 50):
        bh = h.rolling(w, min_periods=w // 2).max()
        bl = l.rolling(w, min_periods=w // 2).min()
        g[f"base{w}_depth"] = (bh - bl) / (bh + EPS)
        g[f"dist_base{w}_top_atr"] = (c - bh) / (a14 + EPS)
    # bars since the 52w high was last touched == how long the base has built
    at_high = (h >= hi52 * 0.999).astype(float)
    idx = pd.Series(np.arange(len(g)), index=g.index)
    last_hi = idx.where(at_high > 0).ffill()
    g["base_length"] = (idx - last_hi).clip(0, 250)

    # ---- volatility contraction pattern (Minervini VCP)
    r5 = (h.rolling(5, min_periods=3).max() - l.rolling(5, min_periods=3).min()) / (c + EPS)
    r10 = (h.rolling(10, min_periods=5).max() - l.rolling(10, min_periods=5).min()) / (c + EPS)
    r20 = (h.rolling(20, min_periods=10).max() - l.rolling(20, min_periods=10).min()) / (c + EPS)
    g["contraction_5_20"] = r5 / (r20 + EPS)      # <1 means recent range is tighter
    g["contraction_10_20"] = r10 / (r20 + EPS)
    g["vcp"] = ((r5 < r10) & (r10 < r20)).astype(np.int8)   # successive contractions
    v20 = v.rolling(20, min_periods=10).mean()
    v50 = v.rolling(50, min_periods=25).mean()
    g["vol_dryup"] = v20 / (v50 + EPS)            # <1 means volume drying up in base

    # ---- Connors mean-reversion inputs
    g["rsi2"] = _rsi(c, 2)
    g["rsi4"] = _rsi(c, 4)
    sma5 = c.rolling(5, min_periods=3).mean()
    g["dist_sma5_atr"] = (c - sma5) / (a14 + EPS)
    down = (c.diff() < 0).astype(int)
    g["down_days_3"] = down.rolling(3, min_periods=3).sum()
    g["down_days_5"] = down.rolling(5, min_periods=5).sum()
    return g


def build_market_frame(ctx_wide: pd.DataFrame) -> pd.DataFrame:
    """Regime features from index/ETF closes. One row per date."""
    spy = ctx_wide["SPY"]
    m = pd.DataFrame(index=ctx_wide.index)
    m["spy_ret20"] = spy.pct_change(20)
    m["spy_ret60"] = spy.pct_change(60)
    sma200 = spy.rolling(200, min_periods=100).mean()
    m["spy_above_200"] = (spy > sma200).astype(np.int8)
    m["spy_dist_200_pct"] = spy / (sma200 + EPS) - 1
    spy_rv = spy.pct_change().rolling(20, min_periods=10).std()
    m["spy_rv20"] = spy_rv
    m["spy_rv_rank"] = _rolling_rank(spy_rv, 504)
    # O'Neil's "M": three quarters of stocks follow the general market, so only
    # buy when the market itself is in a confirmed uptrend.
    sma50s = spy.rolling(50, min_periods=25).mean()
    m["spy_above_50"] = (spy > sma50s).astype(np.int8)
    m["spy_uptrend"] = ((spy > sma50s) & (spy > sma200)).astype(np.int8)
    m["spy_dd_from_high"] = spy / spy.rolling(252, min_periods=100).max() - 1.0

    if "^VIX" in ctx_wide:
        vix = ctx_wide["^VIX"]
        m["vix"] = vix
        m["vix_rank_2y"] = _rolling_rank(vix, 504)
        m["vix_chg5"] = vix.pct_change(5)
        m["vix_term"] = vix / (vix.rolling(60, min_periods=30).mean() + EPS)
    if "HYG" in ctx_wide and "LQD" in ctx_wide:      # credit stress
        m["credit_ratio20"] = (ctx_wide["HYG"] / ctx_wide["LQD"]).pct_change(20)
    if "IWM" in ctx_wide:                            # small-vs-large risk appetite
        m["iwm_spy_ret20"] = ctx_wide["IWM"].pct_change(20) - m["spy_ret20"]
    if "TLT" in ctx_wide:
        m["tlt_ret20"] = ctx_wide["TLT"].pct_change(20)
    return m


def add_market_context(px: pd.DataFrame, ctx_wide: pd.DataFrame,
                       sector_etf_map: dict[str, str]) -> pd.DataFrame:
    """Attach regime + relative-strength features.

    Most swing setups are beta in disguise. Without regime context a model looks
    brilliant on 2013-2019 and detonates in 2022.
    """
    m = build_market_frame(ctx_wide)
    px = px.merge(m.reset_index().rename(columns={"index": "date"}), on="date", how="left")

    # relative strength vs SPY
    px["rs_spy20"] = px["ret20"] - px["spy_ret20"]
    px["rs_spy60"] = px["ret60"] - px["spy_ret60"]

    # relative strength vs own sector ETF
    etf_r20 = ctx_wide.pct_change(20)
    stacked = etf_r20.stack().rename("etf_r20").reset_index()
    stacked.columns = ["date", "sector_etf", "etf_r20"]
    px["sector_etf"] = px["ticker"].map(sector_etf_map)
    px = px.merge(stacked, on=["date", "sector_etf"], how="left")
    px["rs_sector20"] = px["ret20"] - px["etf_r20"]

    # ---- idiosyncratic vs market-wide drop ------------------------------
    # A stock down 5% because the whole market is down is a different trade from
    # one down 5% on its own. Beta-adjust the recent move to separate them.
    spy_r1 = ctx_wide["SPY"].pct_change().rename("spy_r1").reset_index()
    px = px.merge(spy_r1, on="date", how="left")
    px = px.sort_values(["ticker", "date"])
    g = px.groupby("ticker", sort=False)
    px["_r1"] = g["close"].pct_change()
    cov = g.apply(lambda d: d["_r1"].rolling(60, min_periods=30).cov(d["spy_r1"]),
                  include_groups=False).reset_index(level=0, drop=True)
    var = px.groupby("ticker", sort=False)["spy_r1"].transform(
        lambda s: s.rolling(60, min_periods=30).var())
    px["beta60"] = (cov / (var + EPS)).clip(-3, 4)
    spy_r5 = ctx_wide["SPY"].pct_change(5).rename("spy_r5").reset_index()
    px = px.merge(spy_r5, on="date", how="left")
    px["resid_ret5"] = px["ret5"] - px["beta60"] * px["spy_r5"]
    px["resid_ret5_atr"] = px["resid_ret5"] * px["close"] / (px["atr14"] + EPS)

    # breadth of oversold-ness: is everything oversold (a market event) or just
    # this name (an idiosyncratic one)?
    osb = (px.assign(_os=(px["rsi2"] < 10).astype(float))
             .groupby("date")["_os"].mean().rename("oversold_breadth").reset_index())
    px = px.merge(osb, on="date", how="left")

    # breadth: share of the universe above its own 200d SMA, that day
    breadth = (px.assign(_ab=(px["dist_sma200_atr"] > 0).astype(float))
                 .groupby("date")["_ab"].mean().rename("breadth_200").reset_index())
    px = px.merge(breadth, on="date", how="left")
    px = px.sort_values(["ticker", "date"])
    px["breadth_chg20"] = px.groupby("ticker", sort=False)["breadth_200"].diff(20)
    return px


def add_cross_sectional_ranks(px: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Percentile rank within each date - makes a feature relative to what the
    rest of the market is doing that day rather than to an absolute level."""
    for c in cols:
        if c in px.columns:
            px[f"xs_{c}"] = px.groupby("date", sort=False)[c].rank(pct=True)
    return px


MARKET_COLS = ["spy_ret20", "spy_ret60", "spy_above_200", "spy_above_50", "spy_uptrend",
               "spy_dd_from_high", "spy_dist_200_pct", "spy_rv20",
               "spy_rv_rank", "vix", "vix_rank_2y", "vix_chg5", "vix_term",
               "credit_ratio20", "iwm_spy_ret20", "tlt_ret20", "breadth_200",
               "breadth_chg20"]

STOCK_COLS = ["atr_pct", "dist_sma20_atr", "dist_sma50_atr", "dist_sma200_atr",
              "sma20_50_atr", "pct_52w_range", "dist_52w_high_atr", "adx14",
              "ret5_z", "ret20_z", "ret60_z", "rsi14", "ret1_z",
              "rv_rank_2y", "atr_expansion", "bb_width_rank", "vol_of_vol",
              "vol_ratio20", "vol_ratio_rank", "dollar_vol_rank", "obv_slope20",
              "dist_hh20_atr", "dist_ll20_atr", "range20_atr", "days_since_hh20",
              "close_loc_bar", "gap_atr", "days_since_earn", "days_to_earn",
              "earn_in_window", "rs_spy20", "rs_spy60", "rs_sector20"]

REVERSION_COLS = [
    "ibs", "ibs_2d", "bb_pctb", "dist_bb_lower_atr", "cum_rsi2_2", "cum_rsi2_3",
    "ret5_atr", "ret3_atr", "ret1_atr", "dn_vol_ratio", "down_days_2",
    "dist_sma10_atr", "beta60", "resid_ret5", "resid_ret5_atr", "oversold_breadth",
]

PRACTITIONER_COLS = [
    "sma150_slope20", "sma200_slope20", "ma_stack", "dist_sma150_atr",
    "above_52w_low_pct", "below_52w_high_pct", "rs_score",
    "base35_depth", "dist_base35_top_atr", "base50_depth", "dist_base50_top_atr",
    "base_length", "contraction_5_20", "contraction_10_20", "vcp", "vol_dryup",
    "rsi2", "rsi4", "dist_sma5_atr", "down_days_3", "down_days_5",
]

# rs_score ranked across the universe each day IS O'Neil's RS Rating.
XS_RANK_COLS = ["ret20_z", "ret60_z", "rs_spy20", "rs_sector20", "atr_pct",
                "dist_sma200_atr", "vol_ratio20", "adx14", "rs_score",
                "resid_ret5_atr", "ibs"]

FEATURE_COLS = (STOCK_COLS + PRACTITIONER_COLS + REVERSION_COLS + MARKET_COLS
                + [f"xs_{c}" for c in XS_RANK_COLS])
