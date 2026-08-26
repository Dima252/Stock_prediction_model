# Findings — v3 (reversion deep-dive)

Universe 1,499 US stocks, 7.85M bars, 1999–2025. 1.78M labelled events.
Dev 1999–2020, holdout 2021–2025.

---

## Correction to v2 first

**v2 reported the reversion model at "+0.159 edge, holdout". That number was
inflated and I no longer stand behind it.**

It compared the model's top decile against the *whole family pool*. Holdout events
per month range from 973 to 17,326 — an 17.8× swing — so a pool benchmark is not
day-matched, and a model that concentrates its picks into favourable months scores
an "edge" that is really market timing wearing a costume.

The correct benchmark is **top-K per day vs K random candidates from the same
day**, which holds the day, the market, and the setup mix fixed. Under it:

| | dev | holdout |
|---|---|---|
| model top-10/day vs random-10/day | +0.0150 [+0.0006, +0.0293] | **−0.0105** [−0.0372, +0.0157] |

**The model adds nothing out of sample.** Same conclusion at K = 3, 5, 10; only
K=20 is marginally positive (+0.0092) and it does not clear zero.

## What actually survived

One setup, under a strict same-day-controlled test, on both periods:

### `market_drop`

```python
(dist_sma200_atr > 0)          # long-term uptrend intact
& (oversold_breadth > 0.20)    # the whole tape is oversold, not just this name
& (resid_ret5_atr > -0.5)      # the drop is NOT idiosyncratic
& (rsi2 < 10)                  # short-term washout
```

| | n | R/trade | Same-day edge vs control | 95% CI |
|---|---|---|---|---|
| dev 1999–2020 | 38,745 | +0.061 | **+0.042** | [+0.007, +0.079] |
| holdout 2021–25 | 12,627 | +0.113 | **+0.094** | [+0.032, +0.157] |

It was ranked **#1 of 10 setups in all 12 exit schemes on dev**, before the
holdout was read — so it was pre-selected on dev, not fished out of the test set.

**How it was found is the point.** I built `idio_drop` on the obvious prior that a
stock falling on its *own* should mean-revert best. It came out **negative in all
12 exit schemes**. Inverting it gave `market_drop`, which is the best setup in the
study. A stock falling alone is falling for a company-specific reason and that
reason does not revert in five days; a stock falling because everything is falling
has no such reason.

### Everything else washed out

Same-day-controlled holdout edge, all 10 reversion setups:

| Setup | dev edge | holdout edge | Survived |
|---|---|---|---|
| `market_drop` | +0.042 ✓ | **+0.094 ✓** | **yes** |
| `connors_rsi2_strict` | +0.040 ✓ | −0.031 | no |
| `down_5days` | +0.024 ✓ | −0.004 | no |
| `cum_rsi` | +0.018 ✓ | +0.001 | no |
| `connors_rsi2` | +0.017 | +0.002 | no |
| `bb_lower` | +0.008 | −0.004 | no |
| `pullback_3day` | +0.009 | −0.007 | no |
| `ibs_low` | −0.001 | −0.005 | no |
| `idio_drop` | −0.005 | −0.002 | no |

Four cleared zero on dev; **one** cleared it on the holdout. That ratio is roughly
what you'd expect from chance among four marginal candidates, which is exactly why
the holdout exists.

## The exit result — and why per-trade R is the wrong objective

Swept 12 exit schemes × 10 setups on dev. Edge per *trade* rose monotonically as
the exit widened, all the way to the grid edge — a warning sign, so I extended the
grid, and it kept rising. The trade was drifting into buy-and-hold.

Ranking by **edge per day held** (capital efficiency) inverts the table completely:

| Exit | Edge/trade | Mean hold | **Edge/day** |
|---|---|---|---|
| `signal_sma5` (close > 5-day MA) | +0.080 | 2.2d | **0.0352** |
| `bracket_1.0R_5d` | +0.085 | 2.7d | 0.0326 |
| `bracket_2.0R_10d` | +0.114 | 4.8d | 0.0241 |
| `bracket_4.0R_30d` | +0.151 | 9.6d | 0.0159 |

The widest bracket looks **twice as good per trade and is half as good per unit of
capital**. Connors' own exit rule wins once capital recycling is counted. All
results above use it.

## Practical characteristics of `market_drop`

- **Fires rarely and in clusters**: ~47 active days/year, ~50 candidates on those
  days. It waits for a broadly oversold tape. This is not an everyday strategy —
  expect long idle stretches punctuated by busy weeks.
- **Cost-robust**: the *edge* is nearly invariant to slippage (+0.094 at 8bps →
  +0.088 at 40bps) because the control pays the same costs. Raw R stays positive
  to roughly 35bps round-trip and is +0.086 at a realistic 15bps.
- **Noisy across years**: positive in 59% of the 27 years — 50% on dev, 100% on
  the 5 holdout years. The dev mean is carried by strong years (2009, 2010, 2015,
  2017); 2002, 2018 and 2013 were clearly negative.

## Honest limits

- **The holdout has now been read several times.** `market_drop` was pre-selected
  on dev, which protects it, but the printed CI does not price in the full search.
  Treat +0.094 as an optimistic point estimate.
- **50% of dev years positive is not a lot.** A five-for-five holdout run is
  encouraging and is also only five observations.
- Survivorship bias is present and unfixed; it inflates the control too, which is
  part of why the same-day paired statistic is the trustworthy one.
- The ML layer is **not** part of the recommendation. It did not survive.

## What I'd do next

Forward-test `market_drop` alone, with the signal exit, unmodelled. It is a
four-line rule; there is nothing to overfit, and the honest next test is time.
