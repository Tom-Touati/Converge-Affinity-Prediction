"""The one results file, and the out-of-fold predictions behind it.

Rule 4: one row per (experiment, fold, seed) appended to ``results/results.csv``. Appending
rather than rewriting means a crashed experiment cannot destroy earlier ones, and the
``git_sha`` column makes a row traceable to the code that produced it.

Predictions are also kept, one file per (experiment, seed), because the error analysis needs
out-of-fold predictions and a metrics table cannot be un-aggregated. The plan asks for error
analysis on the best model's out-of-fold predictions, so this is a prerequisite rather than a
nicety.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from src import paths

RESULTS_DIR = paths.ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
BENCHMARKS_CSV = RESULTS_DIR / "benchmarks.csv"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FAILURES_MD = RESULTS_DIR / "failures.md"
PLOTS_DIR = RESULTS_DIR / "plots"

#: rule 4's column order, so the file is stable and diffable.
COLUMNS = ["exp", "fold", "seed", "n_train", "n_test", "rmse", "pearson", "spearman",
           "acc3", "f1_macro3", "params", "train_minutes", "git_sha"]

#: kept alongside rule 4's columns: mae, and the project's own headline metric so a fusion row
#: can be read against the existing tree ladder without a second run.
EXTRA = ["mae", "per_complex_spearman", "n_complexes_counted", "dataset", "note"]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def append(row: dict, path: Path = RESULTS_CSV) -> None:
    """Append one (exp, fold, seed) row, creating the file with a header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {c: row.get(c, "") for c in COLUMNS + EXTRA}
    frame = pd.DataFrame([full])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def save_predictions(exp: str, seed: int, preds: pd.DataFrame) -> Path:
    """Out-of-fold predictions for one (experiment, seed).

    ``preds`` needs ``row_id, complex, fold, y_true, y_pred``, which is deliberately the same
    schema ``src/train.py`` writes to ``reports/<run>/predictions.csv`` so the project's
    ``src/error_analysis.py`` can read a fusion run unchanged.
    """
    missing = {"row_id", "complex", "fold", "y_true", "y_pred"} - set(preds.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = PREDICTIONS_DIR / f"{exp}_seed{seed}.csv"
    preds.to_csv(out, index=False)
    return out


def record_failure(exp: str, fold, seed, traceback_text: str) -> None:
    """Rule 8: a failure is written down and the run continues."""
    FAILURES_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURES_MD, "a") as f:
        f.write(f"\n## {exp} — fold {fold}, seed {seed}\n\n```\n{traceback_text.strip()}\n```\n")


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS + EXTRA)
    return pd.read_csv(path)
