"""Stage 1 exit criteria: synthetic paths where the answer is known by construction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BarrierConfig
from src.labeling import STOP, TARGET, TIMEOUT, atr, label_events_single

CFG = BarrierConfig(target_atr=2.0, stop_atr=1.0, max_hold_days=10)
FLAT_ATR = 1.0  # 1 price unit == 1R, makes assertions readable


def _run(bars, t=0, cfg=CFG, cost_bps=0.0, exit_sig=None):
    """bars: list of (o,h,l,c) starting at the DECISION bar t=0."""
    a = np.array(bars, dtype=float)
    o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    atr_arr = np.full(len(c), FLAT_ATR)
    return label_events_single(o, h, l, c, atr_arr, np.array([t]), cfg, cost_bps,
                               exit_sig=exit_sig)


def test_entry_is_next_open_not_decision_close():
    # decision bar closes at 100, next bar OPENS at 105 -> entry must be 105
    bars = [(99, 101, 98, 100), (105, 105, 105, 105)] + [(105, 105, 105, 105)] * 9
    r = _run(bars)
    assert r.entry_px.iloc[0] == 105.0, "entry must be the open of t+1"
    assert r.entry_i.iloc[0] == 1


def test_clean_target_hit_is_exactly_plus_2r():
    bars = [(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102.5, 100, 102)]
    bars += [(102, 102, 102, 102)] * 8
    r = _run(bars)
    assert r.outcome.iloc[0] == TARGET
    assert np.isclose(r.r_gross.iloc[0], 2.0), r.r_gross.iloc[0]


def test_clean_stop_hit_is_exactly_minus_1r():
    bars = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 100.2, 98.9, 99)]
    bars += [(99, 99, 99, 99)] * 8
    r = _run(bars)
    assert r.outcome.iloc[0] == STOP
    assert np.isclose(r.r_gross.iloc[0], -1.0), r.r_gross.iloc[0]


def test_gap_through_stop_is_worse_than_minus_1r():
    # entry 100, stop 99; bar 2 opens at 95 -> fill at 95, i.e. -5R
    bars = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (95, 96, 94, 95)]
    bars += [(95, 95, 95, 95)] * 8
    r = _run(bars)
    assert r.outcome.iloc[0] == STOP
    assert np.isclose(r.r_gross.iloc[0], -5.0), r.r_gross.iloc[0]
    assert r.r_gross.iloc[0] < -1.0, "gap risk must be honoured"


def test_gap_through_target_is_better_than_plus_2r():
    bars = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (108, 109, 107, 108)]
    bars += [(108, 108, 108, 108)] * 8
    r = _run(bars)
    assert r.outcome.iloc[0] == TARGET
    assert np.isclose(r.r_gross.iloc[0], 8.0), r.r_gross.iloc[0]


def test_same_bar_both_barriers_resolves_pessimistically():
    # entry 100, upper 102, lower 99; one bar spans 98..103
    bars = [(100, 100, 100, 100), (100, 103, 98, 101)] + [(101, 101, 101, 101)] * 9
    r = _run(bars)
    assert r.outcome.iloc[0] == STOP, "ambiguous bar must assume the stop hit first"
    assert np.isclose(r.r_gross.iloc[0], -1.0)


def test_timeout_exits_at_close_of_last_held_bar():
    bars = [(100, 100, 100, 100)] + [(100, 100.9, 99.2, 100.4)] * 12
    r = _run(bars)
    assert r.outcome.iloc[0] == TIMEOUT
    assert r.hold_days.iloc[0] == CFG.max_hold_days
    assert np.isclose(r.r_gross.iloc[0], 0.4), r.r_gross.iloc[0]


def test_costs_reduce_r_monotonically():
    bars = [(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102.5, 100, 102)]
    bars += [(102, 102, 102, 102)] * 8
    free = _run(bars, cost_bps=0.0).r_net.iloc[0]
    paid = _run(bars, cost_bps=8.0).r_net.iloc[0]
    assert paid < free
    assert np.isclose(free - paid, (100 * 8e-4) / 1.0)


def test_trailing_stop_ratchets_up_and_locks_in_profit():
    # entry 100, trail 3 ATR. Price runs to 110, then falls back.
    # trail should sit at 110-3 = 107, so the exit is a PROFIT, not -1R.
    cfg = BarrierConfig(target_atr=0.0, stop_atr=1.0, max_hold_days=60, trail_atr=3.0)
    bars = [(100, 100, 100, 100), (100, 102, 99.5, 101), (101, 110, 100.5, 109)]
    bars += [(109, 109, 106, 106)]              # low 106 < 107 -> trail hit
    bars += [(106, 106, 106, 106)] * 10
    r = _run(bars, cfg=cfg)
    assert r.outcome.iloc[0] == STOP
    assert np.isclose(r.r_gross.iloc[0], 7.0), r.r_gross.iloc[0]
    assert r.r_gross.iloc[0] > 0, "a trailing stop must be able to exit in profit"


def test_trailing_stop_never_loosens_below_initial_stop():
    # price never rises, so the trail must not sit below the initial 99 stop
    cfg = BarrierConfig(target_atr=0.0, stop_atr=1.0, max_hold_days=60, trail_atr=3.0)
    bars = [(100, 100, 100, 100), (100, 100.2, 99.6, 99.8), (99.8, 99.9, 98.5, 98.7)]
    bars += [(98.7, 98.7, 98.7, 98.7)] * 10
    r = _run(bars, cfg=cfg)
    assert r.outcome.iloc[0] == STOP
    assert np.isclose(r.r_gross.iloc[0], -1.0), r.r_gross.iloc[0]


def test_no_target_means_trend_can_run_past_2r():
    # with target disabled the trade must NOT be capped at +2R
    cfg = BarrierConfig(target_atr=0.0, stop_atr=1.0, max_hold_days=20, trail_atr=3.0)
    bars = [(100, 100, 100, 100)]
    bars += [(100 + i, 101 + i, 99.5 + i, 100.5 + i) for i in range(19)]
    r = _run(bars, cfg=cfg)
    assert r.r_gross.iloc[0] > 2.0, f"trend capped at {r.r_gross.iloc[0]}"


def test_trail_uses_only_highs_from_completed_bars():
    # a bar that spikes to 110 and reverses to 105 in the SAME bar must not
    # retroactively set the trail to 107 and claim an exit at 107
    cfg = BarrierConfig(target_atr=0.0, stop_atr=1.0, max_hold_days=20, trail_atr=3.0)
    bars = [(100, 100, 100, 100), (100, 110, 105, 106)]
    bars += [(106, 106, 106, 106)] * 10
    r = _run(bars, cfg=cfg)
    # Entering bar 1 the trail is max(99, 100-3) = 99, so bar 1's low of 105
    # cannot stop it out. Using bar 1's OWN high would put the trail at 107 and
    # fabricate an exit inside that bar - the classic intrabar look-ahead.
    assert r.exit_i.iloc[0] != 1, "trail used the current bar's own high"
    # Bar 1's high legitimately sets the trail to 107 for bar 2, which opens
    # at 106 -> a real trailing exit, in profit.
    assert r.exit_i.iloc[0] == 2
    assert np.isclose(r.r_gross.iloc[0], 6.0), r.r_gross.iloc[0]


def test_signal_exit_closes_at_that_bars_close():
    cfg = BarrierConfig(target_atr=5.0, stop_atr=1.0, max_hold_days=10)
    bars = [(100, 100, 100, 100), (100, 100.8, 99.5, 100.2), (100.2, 101.5, 100, 101.3)]
    bars += [(101, 101, 101, 101)] * 8
    sig = np.zeros(len(bars), bool); sig[2] = True     # recovery signal on bar 2
    r = _run(bars, cfg=cfg, exit_sig=sig)
    assert r.exit_i.iloc[0] == 2
    assert np.isclose(r.exit_px.iloc[0], 101.3), r.exit_px.iloc[0]
    assert np.isclose(r.r_gross.iloc[0], 1.3), r.r_gross.iloc[0]


def test_stop_takes_precedence_over_signal_on_same_bar():
    # the stop is an intrabar event; the signal is evaluated at the close, so a
    # bar that breaches the stop must stop out even if it closes on a signal
    cfg = BarrierConfig(target_atr=5.0, stop_atr=1.0, max_hold_days=10)
    bars = [(100, 100, 100, 100), (100, 101, 98.5, 100.9)]
    bars += [(100.9, 100.9, 100.9, 100.9)] * 9
    sig = np.ones(len(bars), bool)
    r = _run(bars, cfg=cfg, exit_sig=sig)
    assert r.outcome.iloc[0] == STOP
    assert np.isclose(r.r_gross.iloc[0], -1.0), r.r_gross.iloc[0]


def test_atr_is_causal():
    rng = np.random.default_rng(0)
    n = 200
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    h, l = c + 1, c - 1
    full = atr(h, l, c, 14)
    # recomputing on a prefix must not change earlier values
    k = 120
    pre = atr(h[:k], l[:k], c[:k], 14)
    assert np.allclose(full[:k][~np.isnan(pre)], pre[~np.isnan(pre)]), "ATR leaks future"


def test_no_event_past_end_of_series():
    bars = [(100, 100, 100, 100)]
    r = _run(bars, t=0)
    assert r.empty, "an event on the last bar has no entry bar and must be dropped"


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
