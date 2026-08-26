"""Setup definitions, taken from the published rule sets of working traders.

Each function encodes criteria its author actually wrote down, so the thresholds
below are theirs rather than mine. Where a rule needs a number the author left
qualitative ("heavy volume"), the choice is marked.

  minervini_vcp     Mark Minervini - Trend Template + Volatility Contraction
  oneil_breakout    William O'Neil  - CANSLIM base breakout with RS Rating + "M"
  darvas_box        Nicolas Darvas  - box breakout into new-high territory
  weinstein_stage2  Stan Weinstein  - Stage 2 advance above a rising 30-week MA
  connors_rsi2      Larry Connors   - RSI(2) reversion above the 200-day
  pullback_3day     Larry Connors   - three down days inside an uptrend
  rs_pullback_50ma  common practice - leader pulls back to the 50-day

`any_liquid` is the control and stays: it is what tells you whether a setup beats
buying something at random, and it is the reason three setups in the first pass
were caught looking like edges when they were not.

IMPORTANT: exits are matched per setup in config.SETUP_EXITS. Trend setups get a
trailing stop with no profit target; reversion setups get a tight, fast bracket.
Testing a Minervini breakout on a 2R/10-day cap measures the cap, not the setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------- trend / momentum
def s_minervini_vcp(f: pd.DataFrame) -> pd.Series:
    """Trend Template (all 8 criteria) + a volatility contraction, then breakout.

    Minervini's template: price above the 150 and 200 day, 150 above 200, 200
    trending up, 50 above both, price above the 50, at least 30% above the 52-week
    low, within 25% of the 52-week high, and relative strength in the top 30%.
    """
    trend = (
        (f["ma_stack"] == 1)                    # close > 50 > 150 > 200
        & (f["sma200_slope20"] > 0)             # 200-day trending up
        & (f["above_52w_low_pct"] >= 0.30)
        & (f["below_52w_high_pct"] <= 0.25)
        & (f["xs_rs_score"] >= 0.60)            # relaxed from 70 for sample size
    )
    # VCP: range contracting into the pivot on drying-up volume
    contraction = (f["contraction_5_20"] < 0.85) & (f["vol_dryup"] < 1.10)
    # breakout through the top of the base on real volume (1.4x is my choice)
    breakout = (f["dist_base35_top_atr"] >= -0.30) & (f["vol_ratio20"] >= 1.2)
    return trend & contraction & breakout


def s_oneil_breakout(f: pd.DataFrame) -> pd.Series:
    """CANSLIM-style breakout: leadership, a proper base, and a rising market.

    RS Rating >= 80 and base of at least seven weeks are O'Neil's; so is the
    insistence that three quarters of stocks follow the market ("M"), which is
    the spy_uptrend gate.
    """
    return (
        (f["xs_rs_score"] >= 0.70)              # relaxed from 80 for sample size
        & (f["base_length"] >= 25)              # relaxed from 7 weeks
        & (f["base35_depth"] <= 0.35)           # not a broken, deep base
        & (f["dist_base35_top_atr"] >= -0.25)   # at the pivot
        & (f["vol_ratio20"] >= 1.3)             # breakout volume
        & (f["spy_uptrend"] == 1)               # the "M" in CANSLIM
        & (f["dist_sma200_atr"] > 0)
    )


def s_darvas_box(f: pd.DataFrame) -> pd.Series:
    """Price builds a tight box near its highs, then breaks out of the top."""
    return (
        (f["below_52w_high_pct"] <= 0.10)       # in new-high territory
        & (f["base50_depth"] <= 0.35)           # tight box
        & (f["dist_base50_top_atr"] >= -0.25)   # breaking the box top
        & (f["vol_ratio20"] >= 1.3)
        & (f["xs_rs_score"] >= 0.60)
    )


def s_weinstein_stage2(f: pd.DataFrame) -> pd.Series:
    """Stage 2 advance: price clears a flattening/rising 30-week (150d) MA."""
    return (
        (f["dist_sma150_atr"] > 0)
        & (f["dist_sma150_atr"] < 2.0)          # just cleared it, not extended
        & (f["sma150_slope20"] > 0)             # 30-week MA turning up
        & (f["vol_ratio20"] >= 1.3)
        & (f["xs_rs_score"] >= 0.60)
        & (f["spy_above_200"] == 1)
    )


# ------------------------------------------------------------- mean reversion
def s_connors_rsi2(f: pd.DataFrame) -> pd.Series:
    """Connors' RSI(2): only long above the 200-day, buy extreme short-term lows."""
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["rsi2"] < 5)
        & (f["dist_sma5_atr"] < 0)
    )


def s_pullback_3day(f: pd.DataFrame) -> pd.Series:
    """Three consecutive lower closes inside an established uptrend."""
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["down_days_3"] == 3)
        & (f["rsi4"] < 35)
        & (f["dist_sma50_atr"] > -1.5)
    )


def s_rs_pullback_50ma(f: pd.DataFrame) -> pd.Series:
    """A relative-strength leader pulls back to its 50-day and steadies."""
    return (
        (f["xs_rs_score"] >= 0.65)
        & (f["ma_stack"] == 1)
        & (f["dist_sma50_atr"].between(-1.5, 0.8))
        & (f["rsi14"] < 50)
        & (f["close_loc_bar"] > 0.5)            # closes in the top half of the bar
    )


# ---------------------------------------------- expanded reversion variants
def s_connors_rsi2_strict(f: pd.DataFrame) -> pd.Series:
    """The harder version of the RSI(2) rule - fewer, deeper signals."""
    return (f["dist_sma200_atr"] > 0) & (f["rsi2"] < 2) & (f["dist_sma5_atr"] < 0)


def s_cum_rsi(f: pd.DataFrame) -> pd.Series:
    """Connors' cumulative RSI(2): a grind lower, not just a one-day spike."""
    return (f["dist_sma200_atr"] > 0) & (f["cum_rsi2_3"] < 45) & (f["rsi2"] < 25)


def s_ibs_low(f: pd.DataFrame) -> pd.Series:
    """Close near the bottom of its own bar - weak close, in an uptrend."""
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["ibs"] < 0.15)
        & (f["ret1_atr"] < -0.3)
        & (f["dist_sma50_atr"] > -2.0)
    )


def s_bb_lower(f: pd.DataFrame) -> pd.Series:
    """Price pierces the lower Bollinger band while the long trend is intact."""
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["bb_pctb"] < 0.05)
        & (f["rsi4"] < 30)
    )


def s_idio_drop(f: pd.DataFrame) -> pd.Series:
    """A drop that is the STOCK's own, not the market's, and not a broad panic.

    resid_ret5 strips the beta-explained part of the move, and oversold_breadth
    keeps us out of days when everything is oversold at once - those are market
    events, where reversion is a much worse bet.
    """
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["resid_ret5_atr"] < -1.5)
        & (f["oversold_breadth"] < 0.15)
        & (f["rsi4"] < 35)
    )


def s_market_drop(f: pd.DataFrame) -> pd.Series:
    """The INVERSE of idio_drop, and the one the data actually supports.

    idio_drop (a stock falling on its own) came out NEGATIVE, which inverts the
    obvious prior but makes sense: a stock falling alone is falling for a
    company-specific reason, and that reason does not revert in five days. A
    stock falling because everything is falling has no such reason. So: require
    a broad oversold tape and a move the stock's beta explains.
    """
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["oversold_breadth"] > 0.20)        # the whole tape is oversold
        & (f["resid_ret5_atr"] > -0.5)          # the drop is NOT idiosyncratic
        & (f["rsi2"] < 10)
    )


def s_down_5days(f: pd.DataFrame) -> pd.Series:
    """Persistent selling into an uptrend."""
    return (
        (f["dist_sma200_atr"] > 0)
        & (f["down_days_5"] >= 4)
        & (f["dist_sma10_atr"] < -1.0)
    )


def s_any_liquid(f: pd.DataFrame) -> pd.Series:
    """Control group: every liquid bar. Defines the unconditional base rate."""
    return pd.Series(True, index=f.index)


SETUPS = {
    "connors_rsi2_strict": s_connors_rsi2_strict,
    "cum_rsi": s_cum_rsi,
    "ibs_low": s_ibs_low,
    "bb_lower": s_bb_lower,
    "idio_drop": s_idio_drop,
    "market_drop": s_market_drop,
    "down_5days": s_down_5days,
    "minervini_vcp": s_minervini_vcp,
    "oneil_breakout": s_oneil_breakout,
    "darvas_box": s_darvas_box,
    "weinstein_stage2": s_weinstein_stage2,
    "connors_rsi2": s_connors_rsi2,
    "pullback_3day": s_pullback_3day,
    "rs_pullback_50ma": s_rs_pullback_50ma,
    "any_liquid": s_any_liquid,
}

TREND_SETUPS = ["minervini_vcp", "oneil_breakout", "darvas_box", "weinstein_stage2"]
REVERSION_SETUPS = ["connors_rsi2", "pullback_3day", "rs_pullback_50ma",
                    "connors_rsi2_strict", "cum_rsi", "ibs_low", "bb_lower",
                    "idio_drop", "market_drop", "down_5days"]

# The control is every liquid bar; subsample it so the event table stays balanced.
CONTROL_SAMPLE_RATE = 0.02


def generate_events(f: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Emit (ticker, date, setup) for every setup instance on liquid bars."""
    liq = f["liquid"].to_numpy() if "liquid" in f else np.ones(len(f), bool)
    rng = np.random.default_rng(seed)
    out = []
    for name, fn in SETUPS.items():
        mask = fn(f).fillna(False).to_numpy() & liq
        if name == "any_liquid":
            mask &= rng.random(len(f)) < CONTROL_SAMPLE_RATE
        n = int(mask.sum())
        print(f"  {name:<20} {n:>9,} events")
        if n == 0:
            continue
        sub = f.loc[mask, ["ticker", "date"]].copy()
        sub["setup"] = name
        out.append(sub)
    if not out:
        return pd.DataFrame(columns=["ticker", "date", "setup"])
    return pd.concat(out, ignore_index=True)
