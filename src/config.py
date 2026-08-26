"""Central configuration. All tunables live here so experiments are reproducible."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

for _p in (RAW, PROCESSED, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data window
START_DATE = "1999-01-01"
END_DATE = "2025-12-31"

# Market-context symbols pulled alongside the equity universe.
BENCHMARK = "SPY"
VOL_INDEX = "^VIX"
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLE": "Energy", "XLI": "Industrials", "XLY": "Cons Discretionary",
    "XLP": "Cons Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}
CONTEXT_SYMBOLS = [BENCHMARK, VOL_INDEX, "IWM", "QQQ", "TLT", "HYG", "LQD"] + list(SECTOR_ETFS)

# ------------------------------------------------------------------- labeling
@dataclass(frozen=True)
class BarrierConfig:
    """Triple-barrier parameters. These are *trading* parameters, not model
    hyperparameters - they define the trade you are actually evaluating.

    `trail_atr > 0` switches the stop to a Chandelier trailing stop (highest high
    since entry, minus trail_atr * ATR). `target_atr <= 0` removes the profit
    target entirely so the trade is only ever closed by the trail or by time.

    That combination matters more than it looks. Trend strategies earn their
    return from a small number of very large winners; a fixed 2R target truncates
    exactly the tail they depend on. Judging a momentum entry on a 2R/10-day
    bracket is testing the exit, not the entry.
    """
    target_atr: float = 2.0      # upper barrier in ATR14 units; <=0 disables it
    stop_atr: float = 1.0        # initial stop in ATR14 units
    max_hold_days: int = 10      # vertical barrier (trading days)
    trail_atr: float = 0.0       # >0 -> chandelier trailing stop
    ambiguous_bar: str = "stop"  # same-bar touch of both -> assume stop hit first

    @property
    def r_target(self) -> float:
        return self.target_atr / self.stop_atr if self.target_atr > 0 else float("nan")


BARRIERS = BarrierConfig()

# Exits matched to the nature of each family of setup, rather than one bracket
# imposed on everything.
MOMENTUM_EXIT = BarrierConfig(target_atr=0.0, stop_atr=1.5,
                              max_hold_days=60, trail_atr=3.0)
# Chosen on dev by EDGE PER DAY HELD across 12 exit schemes x 10 setups, not by
# edge per trade. Wider brackets score higher per trade but hold ~4x longer, so
# they are worse per unit of capital - they drift into buy-and-hold. This is
# Connors' own exit: close when price recovers above its 5-day average.
REVERSION_EXIT = BarrierConfig(target_atr=0.0, stop_atr=1.0, max_hold_days=10)
REVERSION_EXIT_SIGNAL = "exit_above_sma5"

SETUP_EXITS = {
    "minervini_vcp": MOMENTUM_EXIT,
    "oneil_breakout": MOMENTUM_EXIT,
    "darvas_box": MOMENTUM_EXIT,
    "weinstein_stage2": MOMENTUM_EXIT,
    "connors_rsi2": REVERSION_EXIT,
    "pullback_3day": REVERSION_EXIT,
    "rs_pullback_50ma": REVERSION_EXIT,
}

# ------------------------------------------------------------------ liquidity
# Applied point-in-time (trailing window only) so it can never leak.
MIN_DOLLAR_VOLUME = 5_000_000.0   # 20d median dollar volume
MIN_PRICE = 5.0
LIQUIDITY_LOOKBACK = 20

# ----------------------------------------------------------------- validation
EMBARGO_PCT = 0.01
N_CV_SPLITS = 8
CPCV_N_GROUPS = 10
CPCV_N_TEST = 2

# Chronological holdout. The test set is touched exactly once, at the end.
TRAIN_END = "2016-12-31"
VALID_END = "2020-12-31"   # validation: 2017-2020
# test: 2021-01-01 .. END_DATE

# ---------------------------------------------------------------------- costs
# Round-trip cost in R units is computed per-event from ATR; this is in bps.
COST_BPS_ROUNDTRIP = 8.0

RANDOM_SEED = 7
N_JOBS = 16
