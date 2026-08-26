# Swing-trade statistics engine

A conditional-probability tool for swing traders: *given a setup that looks like
this, what has actually happened next?* It grades a trade you have already found
rather than screening for candidates.

**Read [FINDINGS.md](FINDINGS.md) before trusting any number here.** One-line
summary: the machine-learning layer did not survive honest testing and is not part
of the recommendation. One four-line rule did.

---

## Coming back to this after months away? Read only this section.

**What the project concluded:** a single setup, `market_drop`, survived a
same-day-controlled holdout test. No model. Everything else washed out.

**What is running:** `scripts/13_forward_log.py` paper-trades that rule forward
into `data/live/journal.csv`. It places no orders.

**What to do:**

```bash
cd Stock_prediction_model
./.venv/Scripts/python.exe scripts/13_forward_log.py --report-only
```

**How to read it:** look at **mean R per DAY**, not per trade, and at the count of
**distinct signal days**, not trades. The setup fires in clusters — one oversold
day can produce 180 trades that all share a single market bounce, and averaging
over trades treats that as 180 independent observations. The report prints both
and labels the per-trade figure `MISLEADING` for exactly this reason.

**The bar:** 40+ distinct signal days with a day-level CI clear of zero. At ~47
active days a year that is roughly a year. Below that the report refuses to draw a
conclusion, and so should you.

**If the logger has not been running,** register it (see *Keeping it running*) and
accept that the clock starts now.

---

## The rule being tested

```python
(dist_sma200_atr > 0)          # long-term uptrend intact
& (oversold_breadth > 0.20)    # the whole tape is oversold, not just this name
& (resid_ret5_atr > -0.5)      # the drop is NOT idiosyncratic
& (rsi2 < 10)                  # short-term washout
```

Entry at the next open. Exit on `close > SMA5`, a 1×ATR stop, or 10 sessions,
whichever comes first. Defined in [`src/setups.py`](src/setups.py) as
`s_market_drop`; the exit is `REVERSION_EXIT` in `src/config.py`.

Backtest holdout: **+0.113R per trade, same-day edge +0.094 [+0.032, +0.157]**.
Treat that as optimistic — it does not price in the search behind it.

## Setup

```bash
py -3.13 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Python 3.13 specifically — 3.14 has no wheels for several dependencies.

## Keeping it running

`run_daily_log.bat` wraps the daily call and appends to `data/live/run.log`.
Register it once, ~30 min after the US close:

```cmd
schtasks /create /tn "swing-forward-log" /tr "%CD%\run_daily_log.bat" ^
         /sc weekly /d MON,TUE,WED,THU,FRI /st 22:30
```

Missed runs cost nothing — the scan catches up on the next one. But if nothing is
scheduled, nothing is logged.

```bash
scripts/13_forward_log.py                 # daily: fetch, scan, resolve, report
scripts/13_forward_log.py --report-only   # just the report, no network
scripts/13_forward_log.py --backfill 250  # seed history on a first run
```

Four things the logger does deliberately:

1. **Reuses the study's own functions** for both signal and outcome. A private
   copy of the rules would make any live/backtest gap indistinguishable from a
   code difference. `tests/test_live_matches_backtest.py` replays the live path
   over history and asserts an exact match — currently 3,056 trades, max |ΔR| 5e-06.
2. **Refuses to act on an incomplete bar.** Run it intraday and today's partial
   bar is dropped: `rsi2 < 10` on a half-formed bar is not the signal that will
   exist at the close.
3. **Signals on the close, enters at the next open**, recording both dates so the
   lag is auditable rather than assumed.
4. **Only closes a position when the data proves it closed.** A trade whose
   labeller run merely runs out of bars stays OPEN, not a timeout at today's price.

## Rebuilding the study from scratch

```bash
./.venv/Scripts/python.exe scripts/00_null_test.py       # ~2 min, validates the harness
./.venv/Scripts/python.exe scripts/01_download.py        # ~15 min, 1500 names x 25y
./.venv/Scripts/python.exe scripts/02_build_events.py    # ~1 min
./.venv/Scripts/python.exe scripts/03_baseline.py        # ~1 min, kill gate 1
./.venv/Scripts/python.exe scripts/11_reversion_lab.py   # ~8 min, exit sweep
./.venv/Scripts/python.exe scripts/12_reversion_final.py # ~6 min, model + holdout
```

| Script | Purpose |
|---|---|
| `00_null_test.py` | Proves the CV harness finds no edge in pure noise |
| `01_download.py` | Universe (S&P 500+400+600) + 25y daily OHLCV |
| `02_build_events.py` | Features, setups, triple-barrier labels |
| `03_baseline.py` | Empirical baseline + **kill gate 1** |
| `10_family_models.py` | Trend vs reversion, per-family models |
| `11_reversion_lab.py` | 12 exit schemes × 10 setups |
| `12_reversion_final.py` | Final model, portfolio sim, holdout |
| `13_forward_log.py` | **The daily logger** |

`scripts/archive/` holds six scripts from superseded lines of investigation. They
still run and they back specific claims in FINDINGS.md — see the README there.

## Layout

```
src/config.py       all tunables (barriers, exits, costs, CV, dates)
src/data.py         ingestion, liquidity filter, OHLC repair
src/labeling.py     triple-barrier labeller  <- the core primitive
src/features.py     causal, stationary features (99 of them)
src/setups.py       setup definitions  <- edit these to test your own rules
src/baseline.py     empirical conditional distribution (no ML)
src/validation.py   purged CV, CPCV, block bootstrap, null test
src/models.py       logistic / LightGBM + isotonic calibration
src/live.py         forward-test logger (paper only)
tests/              synthetic-path tests + live-vs-backtest equivalence
```

## Five things that will stop you fooling yourself

All five were learned the hard way here, and each one changed a conclusion.

1. **`tests/` must pass.** `test_labeling.py` (16 tests) guards the primitive
   everything inherits: gap fills, same-bar ambiguity resolved pessimistically,
   trailing stops that never use the current bar's own high.
2. **`00_null_test.py` must report AUC ≈ 0.50.** It also shows an *unpurged*
   splitter inventing AUC 0.561 and +0.65R **out of pure noise** — squarely in the
   range people report as a discovered edge.
3. **Never use event-level confidence intervals.** Events overlap and share a
   market factor; the naive interval is ~4× too narrow. Use `block_bootstrap_ci`,
   or better, a paired same-day comparison.
4. **Always benchmark against the same day.** Comparing a strategy to a *pool*
   lets a model that concentrates its picks in favourable months score an "edge"
   that is really market timing. This inflated an earlier result here by ~15×.
5. **Judge exits by return per day held, not per trade.** Widening an exit raises
   R per trade monotonically while the trade quietly becomes buy-and-hold. Ranked
   per unit of capital, the ordering of 12 exit schemes completely inverts.

## Testing your own setups

Replace the functions in [`src/setups.py`](src/setups.py) — each is one boolean
over the feature frame — keep `any_liquid` as the control, then rerun
`02_build_events.py` and `03_baseline.py`. About 90 seconds end to end.

The control is the whole point: it tells you whether a setup beats buying
something at random. Three of the five setups in the first pass looked profitable
and were worse than the control.
