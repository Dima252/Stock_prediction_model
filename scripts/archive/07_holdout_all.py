"""Every model AND every simple rule, on the 2021-2025 holdout, once.

The leaderboard picked the histogram on dev. Before concluding anything we check
whether the logistic - which had the better AUC and the only positive Brier skill
- also fails out of sample, or whether only the histogram was fragile. A
conclusion drawn from one model would not be safe.

Simple rules are included because they are the real competition: if "always trade
the oversold bounce" holds up while the models do not, the models were never the
point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.baseline import baseline_predictions
from src.config import PROCESSED, REPORTS, VALID_END
from src.features import FEATURE_COLS
from src.setups import REVERSION_SETUPS, TREND_SETUPS

SETUP_ORDER = sorted(TREND_SETUPS + REVERSION_SETUPS)
from src.models import ModelSpec, fit_predict
from src.validation import (average_uniqueness, block_bootstrap_ci,
                            calibration_table, evaluate)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

HIST_COLS = ["rsi14", "dist_sma200_atr", "vix_rank_2y", "rv_rank_2y"]


def design(df, cols=None):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].astype(np.float32).copy()
    for s in SETUP_ORDER:
        X[f"setup_{s}"] = (df["setup"] == s).astype(np.int8)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def row(name, r, dates, extra=None):
    lo, hi, nb = block_bootstrap_ci(r, dates, "M") if len(r) > 50 else (np.nan,) * 3
    d = {"strategy": name, "n": len(r), "mean_R": float(r.mean()),
         "ci_lo": lo, "ci_hi": hi, "clears_0": bool(lo > 0) if lo == lo else False}
    if extra:
        d.update(extra)
    return d


def main() -> None:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[~ev.setup.str.startswith("any_liquid")].sort_values("date").reset_index(drop=True)
    dev = ev[ev.date <= VALID_END].reset_index(drop=True)
    te = ev[ev.date > VALID_END].reset_index(drop=True)

    dates = np.sort(dev["date"].unique())
    pos = pd.Series(np.arange(len(dates)), index=dates)
    t0 = pos.reindex(dev["date"].values).to_numpy().astype(int)
    t1 = pos.reindex(dev["exit_date"].values).fillna(len(dates) - 1).to_numpy().astype(int)

    X, Xte = design(dev), None
    Xte = design(te, X.columns)
    y, r = dev["win"].to_numpy(int), dev["r_net"].to_numpy(float)
    yte, rte = te["win"].to_numpy(int), te["r_net"].to_numpy(float)
    dte = te["date"].to_numpy()
    w = average_uniqueness(t0, t1); w /= w.mean()

    print(f"=== dev {len(dev):,} events -> test {len(te):,} events "
          f"({te.date.min().date()} .. {te.date.max().date()}) ===")
    print(f"test base rate {yte.mean():.4f}   test mean R (all events) {rte.mean():+.4f}\n")

    preds = {"histogram": baseline_predictions(dev, te, HIST_COLS, 4, 200)}
    for name, spec in [
        ("logit(C=0.05)", ModelSpec("logit", "logit", {"C": 0.05})),
        ("logit(C=0.01)", ModelSpec("logit", "logit", {"C": 0.01})),
        ("lgbm d3/l7/m400", ModelSpec("lgbm", "lgbm",
                                      {"max_depth": 3, "num_leaves": 7,
                                       "min_child_samples": 400})),
    ]:
        print(f"fitting {name} on full dev ...")
        preds[name] = fit_predict(spec, X, y, w, Xte, t0)

    print("\n=== MODEL METRICS ON HOLDOUT ===")
    mrows = []
    for name, p in preds.items():
        m = evaluate(p, yte, rte)
        mrows.append({"model": name, "auc": m["auc"], "brier_skill": m["brier_skill"],
                      "cal_mae": m["cal_mae"], "r_top_decile": m["r_top_decile"],
                      "r_bot_decile": m["r_bot_decile"], "r_spread": m["r_spread"]})
    md = pd.DataFrame(mrows).sort_values("r_top_decile", ascending=False)
    print(md.round(4).to_string(index=False))
    print("\nr_spread < 0 means the ranking INVERTED out of sample:")
    print("the trades the model liked most did worse than the ones it liked least.\n")

    print("=== STRATEGIES ON HOLDOUT (month-block CIs) ===")
    rows = [row("take EVERY event", rte, dte)]
    for name, p in preds.items():
        m = p >= np.quantile(p, 0.9)
        rows.append(row(f"model {name}: top decile", rte[m], dte[m]))
    for s in sorted(te["setup"].unique()):
        m = (te["setup"] == s).to_numpy()
        rows.append(row(f"rule: always {s}", rte[m], dte[m]))
    sd = pd.DataFrame(rows).sort_values("mean_R", ascending=False)
    print(sd.round(4).to_string(index=False))

    print("\n=== dev -> test decay ===")
    dev_lb = pd.read_csv(REPORTS / "stage6_leaderboard.csv")
    for name in preds:
        d = dev_lb[dev_lb.model == name]
        if len(d):
            print(f"  {name:<18} r_top_decile  dev {d.r_top_decile.iloc[0]:+.4f}"
                  f"  ->  test {md[md.model==name].r_top_decile.iloc[0]:+.4f}")

    md.to_csv(REPORTS / "stage8_holdout_models.csv", index=False)
    sd.to_csv(REPORTS / "stage8_holdout_strategies.csv", index=False)
    print(f"\nwrote {REPORTS/'stage8_holdout_models.csv'}")


if __name__ == "__main__":
    main()
