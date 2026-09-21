"""The three metrics ProtAttBA's Table 1 reports, and their fold assignment.

Kept in one module so that the published predictions and our own predictions are scored by
byte-identical code. The published table turns out to be a per-fold mean with a *population*
standard deviation (ddof=0); ddof=1 moves PCC's spread from 0.050 to 0.053, which rounds the
same but is worth pinning down rather than guessing.

Deliberately not reusing ``src.evaluate``: that harness reports mean per-complex Spearman with
a complex-level bootstrap, which is the right metric for this project but is *not* what
ProtAttBA reports. Comparing a reproduction against a published number requires scoring it the
way the paper scored it. The two are reconciled in the README, not here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
S1131_CSV = HERE / "upstream" / "S1131.csv"
PUBLISHED_PREDICTIONS = HERE / "upstream" / "S1131_published_predictions.csv"
RESULTS = HERE / "results"

#: ``cross_validation/scripts/bash_cross-validation.sh`` sets SEED=3407, and ``trainer.py``
#: feeds that same seed to both ``seed_everything`` and the fold generator. Verified against
#: their shipped per-example predictions in ``verify_published.py``: the fold assignment is an
#: exact match, so the split is fully determined and needs no guessing.
SEED = 3407
N_FOLD = 10
N_ROWS = 1131


def fold_assignment(n_rows: int = N_ROWS, n_fold: int = N_FOLD, seed: int = SEED) -> np.ndarray:
    """Fold id per row, replicating ``utils/data_split.get_K_fold_generator``.

    Their generator is ``KFold(n_splits, shuffle=True, random_state=seed)`` over
    ``np.arange(len(labels))``, so the split is a deterministic function of (n_rows, seed) and
    carries no structural or complex-level grouping at all -- rows from one PDB are spread
    freely across folds. That is the single biggest reason their numbers are not comparable to
    this project's, and it is a property of the benchmark, not a bug in their code.
    """
    folds = np.empty(n_rows, dtype=int)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(kf.split(np.arange(n_rows))):
        folds[test_index] = fold
    return folds


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "pcc": float(pearsonr(y_pred, y_true)[0]),
        "rho": float(spearmanr(y_pred, y_true)[0]),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
    }


def score_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-fold metrics. ``predictions`` needs columns fold, y_true, y_pred."""
    rows = []
    for fold, g in predictions.groupby("fold"):
        row = {"fold": int(fold), "n": len(g)}
        row.update(score(g.y_true.values, g.y_pred.values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and population std over folds, the form Table 1 reports."""
    rows = []
    for metric in ("pcc", "rho", "rmse"):
        rows.append(
            {
                "metric": metric,
                "mean": per_fold[metric].mean(),
                "std": per_fold[metric].std(ddof=0),
                "std_ddof1": per_fold[metric].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def format_summary(per_fold: pd.DataFrame) -> str:
    s = summarise(per_fold).set_index("metric")
    return "  ".join(f"{m.upper()} {s.loc[m, 'mean']:.3f} +/- {s.loc[m, 'std']:.3f}"
                     for m in ("pcc", "rho", "rmse"))


#: ProtAttBA, Bioinformatics 2025, Table 1, S1131 row, ESM2 column.
PUBLISHED = {"pcc": (0.84, 0.05), "rho": (0.75, 0.06), "rmse": (1.31, 0.09)}
