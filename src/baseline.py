"""Stage 3 - the empirical baseline. No ML.

This is the honest v1 and the bar every model must clear. If a gradient-boosted
tree cannot beat a bucketed histogram on purged CV, there is no model - only a
memorised noise pattern.

Everything here reports n alongside the estimate. A 62% hit rate on 19 samples is
not a finding, and the tool must never let that read like one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import bootstrap_ci


def _agg(g: pd.DataFrame) -> pd.Series:
    r = g["r_net"].to_numpy(float)
    lo, hi = bootstrap_ci(r) if len(r) >= 30 else (np.nan, np.nan)
    return pd.Series({
        "n": len(g),
        # "did it make money" - the only win definition that works under BOTH the
        # target bracket and the trailing-stop scheme (which has no target, so
        # outcome==TARGET is impossible there and would read as a 0% win rate).
        "win_rate": float((r > 0).mean()),
        "target_rate": float((g["outcome"] == 1).mean()),
        "stop_rate": float((g["outcome"] == -1).mean()),
        "timeout_rate": float((g["outcome"] == 0).mean()),
        "r_mean": float(r.mean()),
        "r_median": float(np.median(r)),
        "r_ci_lo": lo,
        "r_ci_hi": hi,
        "r_p05": float(np.quantile(r, 0.05)),
        "r_p95": float(np.quantile(r, 0.95)),
        "hold_days": float(g["hold_days"].mean()),
    })


def setup_summary(ev: pd.DataFrame, control_map: dict[str, str] | None = None,
                  control: str = "any_liquid") -> pd.DataFrame:
    """Per-setup stats with the relevant base rate alongside.

    The `edge_*` columns are the whole point: an absolute win rate means nothing
    without the number it has to beat.

    `control_map` maps each setup to ITS OWN control. That matters once exits
    differ by setup family - a 60-day trailing-stop scheme banks the market's
    drift and scores ~+0.11R on randomly chosen stocks, while a 7-day bracket
    scores ~+0.02R. Comparing a trend setup against the reversion control would
    flatter it by the difference between the two exit schemes.
    """
    rows = ev.groupby("setup", sort=False).apply(_agg, include_groups=False).reset_index()
    lookup = rows.set_index("setup")

    def _base(setup: str, col: str) -> float:
        ctrl = (control_map or {}).get(setup, control)
        return float(lookup.loc[ctrl, col]) if ctrl in lookup.index else np.nan

    rows["control"] = rows["setup"].map(lambda s: (control_map or {}).get(s, control))
    rows["edge_win"] = rows.apply(lambda r: r.win_rate - _base(r.setup, "win_rate"), axis=1)
    rows["edge_r"] = rows.apply(lambda r: r.r_mean - _base(r.setup, "r_mean"), axis=1)
    rows["ci_clears_zero"] = rows["r_ci_lo"] > 0
    rows["beats_control"] = rows["edge_r"] > 0
    return rows.sort_values("edge_r", ascending=False).reset_index(drop=True)


def yearly_stability(ev: pd.DataFrame) -> pd.DataFrame:
    """A setup that only worked in 2020 is a regime artefact, not an edge."""
    e = ev.copy()
    e["year"] = pd.to_datetime(e["date"]).dt.year
    t = e.pivot_table(index="year", columns="setup", values="r_net", aggfunc="mean")
    n = e.pivot_table(index="year", columns="setup", values="r_net", aggfunc="size")
    return t, n


def fit_buckets(train: pd.DataFrame, by: list[str], n_bins: int = 4) -> dict:
    """Learn bucket edges on train only, so the lookup can be scored out-of-sample."""
    edges = {}
    for c in by:
        q = np.linspace(0, 1, n_bins + 1)
        edges[c] = np.unique(np.nanquantile(train[c].to_numpy(float), q))
    return edges


def apply_buckets(df: pd.DataFrame, edges: dict) -> pd.Series:
    """Map rows to a bucket id using train-derived edges."""
    codes = []
    for c, e in edges.items():
        v = df[c].to_numpy(float)
        codes.append(np.clip(np.digitize(v, e[1:-1]), 0, max(len(e) - 2, 0)))
    key = np.zeros(len(df), dtype=np.int64)
    for c in codes:
        key = key * 16 + c
    return pd.Series(key, index=df.index)


def baseline_oos(train: pd.DataFrame, test: pd.DataFrame, by: list[str],
                 n_bins: int = 4, min_n: int = 200) -> pd.DataFrame:
    """Score the histogram out-of-sample: bucket stats from train, applied to test.

    Returns the per-bucket train estimate next to the realised test outcome. If
    the two columns are uncorrelated, the buckets carry no forward information.
    """
    edges = fit_buckets(train, by, n_bins)
    tr_key = apply_buckets(train, edges)
    te_key = apply_buckets(test, edges)

    tr_stats = (train.assign(_k=tr_key)
                     .groupby("_k")
                     .agg(n_train=("r_net", "size"),
                          r_train=("r_net", "mean"),
                          p_train=("win", "mean")))
    te_stats = (test.assign(_k=te_key)
                    .groupby("_k")
                    .agg(n_test=("r_net", "size"),
                         r_test=("r_net", "mean"),
                         p_test=("win", "mean")))
    j = tr_stats.join(te_stats, how="inner")
    return j[(j.n_train >= min_n) & (j.n_test >= 50)].sort_values("r_train", ascending=False)


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, by: list[str],
                         n_bins: int = 4, prior_n: int = 200) -> np.ndarray:
    """Per-row P(win) from the histogram, shrunk toward the train base rate.

    Shrinkage matters: a bucket with 12 observations should not be allowed to
    assert 0.75. Weight = n / (n + prior_n).
    """
    edges = fit_buckets(train, by, n_bins)
    tr_key = apply_buckets(train, edges)
    te_key = apply_buckets(test, edges)
    g = train.assign(_k=tr_key).groupby("_k")["win"].agg(["size", "mean"])
    prior = float(train["win"].mean())
    w = g["size"] / (g["size"] + prior_n)
    shrunk = w * g["mean"] + (1 - w) * prior
    return te_key.map(shrunk).fillna(prior).to_numpy(float)
