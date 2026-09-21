"""The five metrics the fusion ladder reports, exactly as the plan defines them.

Deliberately separate from ``src/evaluate.py``, which is not touched. The two answer different
questions and mixing them would make the results table unreadable:

* ``src/evaluate.py`` reports **mean per-complex Spearman** with a complex-level bootstrap and
  ``MIN_GROUP=5``. That is this project's headline, chosen because a constant-per-fold
  predictor scores global Spearman -0.36 under grouped CV purely from between-fold label
  shift, so pooled correlations carry a component that has nothing to do with the model.
* the plan asks for **pooled** rmse / pearson / spearman plus a 3-class accuracy and macro-F1,
  which is what makes the numbers comparable to ProtAttBA's own table.

Both are computed for every experiment. ``per_complex_spearman`` here is a convenience copy of
the project's headline so one row of ``results.csv`` can be read against the existing ladder,
and it calls ``src.evaluate`` rather than reimplementing it.

The 3-class labels follow the plan's rule 5 and are **not** the repo's ``CLASS_EDGES``:

| class | plan (rule 5), on \\|ddG\\| | repo ``evaluate.CLASS_EDGES``, on signed ddG |
|---|---|---|
| low / stabilising | \\|ddG\\| < 0.5 | ddG < -0.5 |
| medium / neutral | 0.5 <= \\|ddG\\| <= 2.0 | -0.5 <= ddG <= 0.5 |
| high / destabilising | \\|ddG\\| > 2.0 | ddG > 0.5 |

The plan's bins measure effect *size* and discard direction; the repo's measure direction. They
are not comparable, so both names are kept distinct in the output.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

#: rule 5: |ddG| bins in kcal/mol. Derived from the regression output, never trained directly.
MAGNITUDE_EDGES = (0.5, 2.0)
MAGNITUDE_NAMES = ("low", "medium", "high")


def magnitude_class(y: np.ndarray) -> np.ndarray:
    """0 = low, 1 = medium, 2 = high, by |ddG| against rule 5's edges."""
    return np.digitize(np.abs(np.asarray(y, dtype=float)), MAGNITUDE_EDGES)


def _safe(fn, a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(a, b)[0])


def score(y_true, y_pred, complexes=None) -> dict[str, float]:
    """Every metric one row of ``results.csv`` needs.

    ``complexes`` is optional; when given, the project's headline per-complex Spearman is
    added so a fusion run can be read against the existing tree ladder.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ct, cp = magnitude_class(y_true), magnitude_class(y_pred)

    out = {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "pearson": _safe(stats.pearsonr, y_pred, y_true),
        "spearman": _safe(stats.spearmanr, y_pred, y_true),
        "acc3": float(np.mean(ct == cp)),
        "f1_macro3": float(f1_score(ct, cp, average="macro", labels=[0, 1, 2],
                                    zero_division=0)),
    }
    if complexes is not None:
        # Call the project's own harness rather than reimplementing its headline metric.
        from src import evaluate

        preds = pd.DataFrame({
            "row_id": np.arange(len(y_true)), "complex": np.asarray(complexes),
            "y_true": y_true, "y_pred": y_pred,
        })
        per_cx = evaluate.per_complex(preds)
        out["per_complex_spearman"] = float(per_cx["spearman"].mean()) if len(per_cx) else float("nan")
        out["n_complexes_counted"] = int(len(per_cx))
    return out


def confusion3(y_true, y_pred) -> pd.DataFrame:
    """3x3 confusion over the magnitude bins, for the error-analysis report."""
    ct, cp = magnitude_class(y_true), magnitude_class(y_pred)
    m = pd.crosstab(pd.Series(ct, name="true"), pd.Series(cp, name="pred"),
                    dropna=False).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    m.index = [f"true_{n}" for n in MAGNITUDE_NAMES]
    m.columns = [f"pred_{n}" for n in MAGNITUDE_NAMES]
    return m
