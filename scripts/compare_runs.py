"""Put several per-fold result tables side by side, on the folds they share.

Comparing mean Pearson across runs that finished different numbers of folds is how a change
gets credited with an improvement it did not make: a run that stopped before the hardest fold
looks better than one that did not. Every row here is averaged over the SHARED folds only,
and the per-fold grid above it shows which folds each run actually has.

    python scripts/compare_runs.py \
        "buggy=reports/_retired/old_results.csv" \
        "fixed=reports/perturb_v2_full/v2_full_results.csv"
"""
from __future__ import annotations

import argparse

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

FOLD_COL, METRICS = "fold", ("pearson", "rmse")


def load(spec: str) -> tuple[str, pd.DataFrame]:
    label, _, path = spec.partition("=")
    if not path:
        label, path = path or spec, spec
    p = paths.ROOT / path
    if not p.exists():
        raise SystemExit(f"{p} not found")
    return (label or p.parent.name), pd.read_csv(p).set_index(FOLD_COL).sort_index()


def grid(tables: dict[str, pd.DataFrame], metric: str, fmt: str) -> str:
    """``fmt`` is a complete format spec, e.g. ">+10.3f"."""
    folds = sorted(set().union(*(set(d.index) for d in tables.values())))
    w = max(len(k) for k in tables) + 2
    out = [" " * w + "".join(f"{f'fold {f}':>10}" for f in folds)]
    for k, d in tables.items():
        cells = []
        for f in folds:
            v = d[metric].get(f, np.nan)
            cells.append(f"{'-':>10}" if pd.isna(v) else format(v, fmt))
        out.append(f"{k:<{w}}" + "".join(cells))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/results.csv, or just the path")
    a = ap.parse_args()

    tables = dict(load(s) for s in a.runs)
    shared = sorted(set.intersection(*(set(d.index) for d in tables.values())))

    print("Pearson per fold\n")
    print(grid(tables, "pearson", ">+10.3f"))
    print("\nRMSE per fold\n")
    print(grid(tables, "rmse", ">10.3f"))

    if not shared:
        print("\nno fold is present in every run; no comparable mean")
        return
    print(f"\nMean over the {len(shared)} folds every run has ({shared}):\n")
    w = max(len(k) for k in tables) + 2
    print(f"{'':<{w}}{'pearson':>10}{'rmse':>10}")
    for k, d in tables.items():
        print(f"{k:<{w}}{d.pearson.loc[shared].mean():>+10.3f}{d.rmse.loc[shared].mean():>10.3f}")
    skipped = {k: sorted(set(d.index) - set(shared)) for k, d in tables.items()}
    if any(skipped.values()):
        print("\nfolds left out of the mean: "
              + ", ".join(f"{k} {v}" for k, v in skipped.items() if v))


if __name__ == "__main__":
    main()
