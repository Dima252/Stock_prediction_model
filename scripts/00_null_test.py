"""Stage 4 exit criterion: the harness must find NO edge in pure noise.

Run this every time the splitter changes. If it reports edge on random numbers,
every downstream result is fiction.

Also contrasts against a deliberately BROKEN (unpurged, random-shuffle) splitter
to show the purging is doing real work rather than being decorative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ModelSpec, run_cv
from src.validation import PurgedSplit, null_test, purged_kfold


def make_runner(splitter, spec):
    def runner(X, y, r, t0, t1):
        splits = splitter(t0, t1)
        return run_cv(spec, X, y, r, t0, t1, splits, use_weights=True)
    return runner


def naive_random_kfold(t0, t1, n_splits=8, seed=0):
    """Deliberately WRONG: random folds, no purge, no embargo."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(t0))
    out = []
    for f in np.array_split(idx, n_splits):
        train = np.setdiff1d(np.arange(len(t0)), f)
        out.append(PurgedSplit(train=train, test=np.sort(f), test_span=(0, 0)))
    return out


def leaky_noise_test(splitter, spec, n=20000, horizon=10, seed=0):
    """Noise features, but labels are made SERIALLY CORRELATED the way real
    overlapping labels are. A purged splitter must still report ~no edge; an
    unpurged one will hallucinate one, because neighbouring samples share an
    outcome path and land on both sides of the split."""
    rng = np.random.default_rng(seed)
    t0 = np.sort(rng.integers(0, n, n))
    t1 = t0 + horizon
    # one latent path; every event inheriting an overlapping slice of it
    path = rng.normal(size=n + horizon + 2)
    y = np.array([(path[a:b + 1].sum() > 0) for a, b in zip(t0, t1)]).astype(int)
    # features carry NOTHING about y - only about t0, which the split leaks
    X = pd.DataFrame({f"noise_{i}": rng.normal(size=n) for i in range(8)})
    X["leaky_time_proxy"] = t0 + rng.normal(0, 5, n)
    r = np.where(y == 1, 2.0, -1.0)
    splits = splitter(t0, t1)
    return run_cv(spec, X, y, r, t0, t1, splits, use_weights=True)


def report(tag: str, res: dict) -> None:
    if res.get("n", 0) == 0:
        print(f"  {tag:<34} (no valid splits)")
        return
    print(f"  {tag:<34} n={res['n']:>6,}  auc={res['auc']:.4f}  "
          f"brier_skill={res['brier_skill']:+.4f}  r_spread={res['r_spread']:+.3f}")


if __name__ == "__main__":
    spec = ModelSpec(name="logit", kind="logit", calibrate=False)
    purged = lambda t0, t1: purged_kfold(t0, t1, n_splits=8, embargo_pct=0.01)

    print("\n=== A. pure noise, independent labels ===")
    print("   (both splitters should report no edge - this only proves the")
    print("    metrics are not broken)")
    report("purged k-fold", null_test(make_runner(purged, spec)))
    report("naive random k-fold", null_test(make_runner(naive_random_kfold, spec)))

    print("\n=== B. pure noise, OVERLAPPING labels + a time-proxy feature ===")
    print("   (this is the real test: purged must stay at ~0.50, naive must")
    print("    hallucinate edge. If naive does NOT, the test itself is too weak)")
    print("   uses a HIGH-CAPACITY model on purpose - a linear model cannot")
    print("   exploit a time proxy, so it would understate the leak")
    hi_cap = ModelSpec(name="rf", kind="rf", calibrate=False)
    r_purged = leaky_noise_test(purged, hi_cap)
    r_naive = leaky_noise_test(naive_random_kfold, hi_cap)
    report("purged k-fold", r_purged)
    report("naive random k-fold", r_naive)

    print("\n=== verdict ===")
    ok = True
    if abs(r_purged["auc"] - 0.5) > 0.03:
        print(f"  FAIL purged splitter leaks: auc={r_purged['auc']:.4f} (want ~0.50)")
        ok = False
    else:
        print(f"  PASS purged splitter is clean: auc={r_purged['auc']:.4f}")
    if r_naive["auc"] - r_purged["auc"] < 0.02:
        print(f"  WARN naive splitter did not hallucinate much edge "
              f"({r_naive['auc']:.4f}); leak test may be too weak")
    else:
        print(f"  PASS purging demonstrably removes leakage "
              f"({r_naive['auc']:.4f} -> {r_purged['auc']:.4f})")
    sys.exit(0 if ok else 1)
