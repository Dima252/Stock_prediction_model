"""Stage 4 - validation harness. Built BEFORE any model.

Overlapping labels are the thing that kills these projects. With a 10-day horizon
and daily events, consecutive samples share ~90% of their outcome path. Naive CV
therefore inflates the effective sample size roughly 10x and reports edge that is
not there. Three defences, all implemented here:

1. PURGING   - drop training samples whose label window overlaps the test window.
2. EMBARGO   - additionally drop training samples that start just after the test
               window, since serial correlation leaks backwards too.
3. UNIQUENESS WEIGHTS - down-weight samples whose outcome path is shared with
               many concurrent samples (Lopez de Prado average uniqueness).

`null_test` is the harness's own exit criterion: fed pure noise it must report no
edge. If it finds edge in random numbers, every number it produces afterwards is
fiction.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# --------------------------------------------------------------------- weights
def average_uniqueness(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Average uniqueness of each label given overlapping concurrent labels.

    t0/t1 are integer positions on a shared time axis (event bar, exit bar).
    """
    n_t = int(max(t1.max(), t0.max())) + 2
    diff = np.zeros(n_t + 1, dtype=np.float64)
    np.add.at(diff, t0, 1.0)
    np.add.at(diff, t1 + 1, -1.0)
    conc = np.cumsum(diff)[:n_t]
    conc[conc < 1] = 1.0
    inv = 1.0 / conc
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    span = (t1 - t0 + 1).astype(np.float64)
    return (csum[t1 + 1] - csum[t0]) / span


# ------------------------------------------------------------------- splitters
@dataclass
class PurgedSplit:
    train: np.ndarray
    test: np.ndarray
    test_span: tuple[int, int]


def purged_kfold(t0: np.ndarray, t1: np.ndarray, n_splits: int = 8,
                 embargo_pct: float = 0.01) -> list[PurgedSplit]:
    """Contiguous-in-time test folds with purge + embargo."""
    order = np.argsort(t0, kind="mergesort")
    folds = np.array_split(order, n_splits)
    horizon = int(np.ceil(embargo_pct * (t0.max() - t0.min() + 1)))
    out: list[PurgedSplit] = []
    for f in folds:
        if len(f) == 0:
            continue
        lo, hi = t0[f].min(), t0[f].max()
        test_end = max(hi, t1[f].max())
        # purge: any train label whose [t0,t1] overlaps the test window
        overlaps = (t1 >= lo) & (t0 <= test_end)
        # embargo: train labels starting just after the test window
        embargoed = (t0 > test_end) & (t0 <= test_end + horizon)
        train = np.where(~overlaps & ~embargoed)[0]
        out.append(PurgedSplit(train=train, test=np.sort(f), test_span=(int(lo), int(hi))))
    return out


def combinatorial_purged_cv(t0: np.ndarray, t1: np.ndarray, n_groups: int = 10,
                            n_test: int = 2, embargo_pct: float = 0.01
                            ) -> list[PurgedSplit]:
    """CPCV - C(n_groups, n_test) train/test splits instead of n_splits.

    Returns a DISTRIBUTION of out-of-sample results rather than one fragile point
    estimate. C(10,2) = 45 splits; this is where surplus CPU actually buys you
    something.
    """
    order = np.argsort(t0, kind="mergesort")
    groups = np.array_split(order, n_groups)
    horizon = int(np.ceil(embargo_pct * (t0.max() - t0.min() + 1)))
    out: list[PurgedSplit] = []
    for combo in combinations(range(n_groups), n_test):
        test = np.sort(np.concatenate([groups[i] for i in combo]))
        if len(test) == 0:
            continue
        mask_overlap = np.zeros(len(t0), dtype=bool)
        mask_embargo = np.zeros(len(t0), dtype=bool)
        for i in combo:
            g = groups[i]
            lo, hi = t0[g].min(), max(t0[g].max(), t1[g].max())
            mask_overlap |= (t1 >= lo) & (t0 <= hi)
            mask_embargo |= (t0 > hi) & (t0 <= hi + horizon)
        train = np.where(~mask_overlap & ~mask_embargo)[0]
        out.append(PurgedSplit(train=train, test=test,
                               test_span=(int(t0[test].min()), int(t0[test].max()))))
    return out


# --------------------------------------------------------------------- metrics
def expectancy_by_decile(p: np.ndarray, r: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """The number that actually matters: E[R] as a function of model confidence.

    A monotone, rising curve is a usable model. A flat one means the probability
    carries no economic information no matter what the AUC says.
    """
    q = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    df = pd.DataFrame({"bin": q, "p": p, "r": r})
    g = df.groupby("bin").agg(n=("r", "size"), p_mean=("p", "mean"),
                              r_mean=("r", "mean"), r_med=("r", "median"),
                              win_rate=("r", lambda s: float((s > 0).mean())))
    g["r_se"] = df.groupby("bin")["r"].sem()
    return g.reset_index()


def calibration_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    df = pd.DataFrame({"bin": q, "p": p, "y": y})
    g = df.groupby("bin").agg(n=("y", "size"), p_pred=("p", "mean"), p_actual=("y", "mean"))
    g["gap"] = g["p_pred"] - g["p_actual"]
    return g.reset_index()


def evaluate(p: np.ndarray, y: np.ndarray, r: np.ndarray,
             w: np.ndarray | None = None) -> dict:
    """Calibration first, discrimination second, economics last but loudest."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    r = np.asarray(r, float)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "logloss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else np.nan,
        "r_mean_all": float(r.mean()),
    }
    # Brier skill vs always predicting the base rate
    base = np.full_like(p, y.mean())
    out["brier_skill"] = 1.0 - out["brier"] / float(brier_score_loss(y, base, sample_weight=w))

    dec = expectancy_by_decile(p, r)
    out["r_top_decile"] = float(dec.r_mean.iloc[-1])
    out["r_bot_decile"] = float(dec.r_mean.iloc[0])
    out["r_spread"] = out["r_top_decile"] - out["r_bot_decile"]
    out["n_top_decile"] = int(dec.n.iloc[-1])

    cal = calibration_table(p, y)
    out["cal_mae"] = float(cal.gap.abs().mean())
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ------------------------------------------------------------------- null test
def null_test(cv_runner, n_features: int = 20, n_samples: int = 20000,
              horizon: int = 10, seed: int = 0) -> dict:
    """Harness exit criterion. Pure-noise features, shuffled labels.

    `cv_runner(X, y, r, t0, t1) -> dict` must return metrics. A correct harness
    reports auc ~= 0.50, brier_skill ~= 0 and r_spread ~= 0. Anything else means
    the splitter leaks and every downstream number is fiction.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)),
                     columns=[f"noise_{i}" for i in range(n_features)])
    t0 = np.sort(rng.integers(0, n_samples, n_samples))
    t1 = t0 + horizon
    y = rng.integers(0, 2, n_samples)
    r = np.where(y == 1, 2.0, -1.0) + rng.normal(0, 0.1, n_samples)
    return cv_runner(X, y, r, t0, t1)


# ------------------------------------------------- honest confidence intervals
def block_bootstrap_ci(x: np.ndarray, dates: np.ndarray, freq: str = "M",
                       n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 7) -> tuple[float, float, int]:
    """Resample whole CALENDAR BLOCKS, not individual events.

    Why this and not `bootstrap_ci`: events are not independent. On any given day
    hundreds of positions are open across the panel, all sharing one market
    factor, and each label spans ~5-10 days. An event-level bootstrap assumes
    370k independent observations when the true independent count is closer to
    the number of months in the sample. It therefore reports intervals that are
    far too narrow - the single easiest way to convince yourself of an edge that
    is not there.

    Returns (lo, hi, n_blocks).
    """
    x = np.asarray(x, float)
    keys = pd.PeriodIndex(pd.to_datetime(dates), freq=freq)
    groups = pd.Series(x).groupby(keys.astype(str))
    blocks = [g.to_numpy() for _, g in groups]
    n = len(blocks)
    if n < 5:
        return (np.nan, np.nan, n)
    sums = np.array([b.sum() for b in blocks])
    cnts = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)), n)


def concurrency_report(t0: np.ndarray, t1: np.ndarray,
                       dates: np.ndarray | None = None) -> dict:
    """Describe how badly overlapped the labels are.

    NOTE on interpreting average uniqueness across a wide panel: the classical
    measure counts every simultaneously-open label as concurrent, including two
    different tickers on the same day. Those are correlated (shared market
    factor) but NOT identical, so uniqueness understates the true independent
    sample size, while the raw event count wildly overstates it. The truth sits
    between; the number of calendar blocks is the more useful practical guide.
    """
    n_t = int(max(t1.max(), t0.max())) + 2
    diff = np.zeros(n_t + 1)
    np.add.at(diff, t0, 1.0)
    np.add.at(diff, t1 + 1, -1.0)
    conc = np.cumsum(diff)[:n_t]
    u = average_uniqueness(t0, t1)
    out = {
        "n_events": int(len(t0)),
        "mean_concurrency": float(conc[conc > 0].mean()),
        "max_concurrency": int(conc.max()),
        "mean_uniqueness": float(u.mean()),
        "uniqueness_eff_n": float(u.sum()),
        "mean_span_days": float((t1 - t0 + 1).mean()),
    }
    if dates is not None:
        d = pd.to_datetime(dates)
        out["n_months"] = int(pd.PeriodIndex(d, freq="M").nunique())
        out["n_trading_days"] = int(pd.Series(d).nunique())
    return out
