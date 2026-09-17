"""THE evaluation harness. One predictions table in, every reported number out.

Built once, against a dummy predictor, and then frozen. A later score gain must come from the
model, never from a change in how the model is scored.

Headline metric: mean per-complex Spearman. Correlating within a complex and then averaging is
what matters for the real decision ("which mutation hurts binding least, for this target"), and
it is not gamed by the three complexes that hold 29% of the rows. It is also the field's own
convention -- RDE-Network and DiffAffinity both report per-structure correlations, discarding
groups with fewer than ten mutations -- so ``min_group=10`` is the default here for
comparability. The all-complex variant is reported alongside, because dropping small complexes
also drops the hard cases.

Confidence intervals bootstrap over *complexes*, not rows. The score's spread comes almost
entirely from which complexes land in the test fold; a per-row bootstrap would look tighter and
would lie.

Two floors that are not zero, both measured on this data rather than assumed:

* Under grouped CV a *constant-per-fold* predictor scores global Spearman -0.36, because the
  fold a complex sits in is excluded from its own training mean and the folds differ in mean
  ddG (+1.89 to +0.34). Global correlations therefore carry a between-fold component that has
  nothing to do with the model. This is the empirical reason the headline metric is per-complex.
* 86% of rows with |ddG| > 1 are destabilising, so "always predict weaker binding" scores 0.86
  sign accuracy. ``sign_acc_big_base`` reports that floor next to the score.

Ceiling: measurement noise caps per-complex Pearson near 0.91 and global Pearson near 0.96
(SKEMPI's replicate statistics; our own duplicates give a looser 0.97/0.99 because they are
often the same group re-reporting). Quote the conservative number.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

#: class boundaries on ddG (kcal/mol), used for the 3-class framing the brief allows
CLASS_EDGES = (-0.5, 0.5)
CLASS_NAMES = ("stabilising", "neutral", "destabilising")

#: noise ceilings from the EDA, plotted on every chart so a 0.6 is read correctly
CEILING_PER_COMPLEX_R = 0.91
CEILING_GLOBAL_R = 0.96

REQUIRED = ("row_id", "complex", "y_true", "y_pred")


def to_classes(y: np.ndarray) -> np.ndarray:
    return np.digitize(y, CLASS_EDGES)


def _safe_spearman(a, b) -> float:
    """nan rather than a warning when either side is constant -- rung 0 predicts a constant."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(a, b).correlation)


def _safe_pearson(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.pearsonr(a, b)[0])


def per_complex(preds: pd.DataFrame, min_group: int = 10) -> pd.DataFrame:
    """Per-complex correlation table, sorted worst first. The error analysis starts here."""
    rows = []
    for cx, g in preds.groupby("complex"):
        rows.append({
            "complex": cx,
            "n": len(g),
            "ddG_sd": g["y_true"].std(),
            "spearman": _safe_spearman(g["y_true"], g["y_pred"]),
            "pearson": _safe_pearson(g["y_true"], g["y_pred"]),
            "rmse": float(np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))),
        })
    out = pd.DataFrame(rows).sort_values("spearman")
    out["counted"] = out["n"] >= min_group
    return out.reset_index(drop=True)


def metrics(preds: pd.DataFrame, min_group: int = 10) -> dict:
    """Every headline number, from one predictions table."""
    missing = [c for c in REQUIRED if c not in preds.columns]
    if missing:
        raise ValueError(f"predictions table missing columns: {missing}")
    y, p = preds["y_true"].to_numpy(float), preds["y_pred"].to_numpy(float)

    pc = per_complex(preds, min_group)
    kept = pc[pc["counted"]]

    m = {
        "n": len(preds),
        "n_complexes": preds["complex"].nunique(),
        "n_complexes_counted": int(kept.shape[0]),
        # --- headline -------------------------------------------------------------------
        "per_complex_spearman": float(kept["spearman"].mean()),
        "per_complex_pearson": float(kept["pearson"].mean()),
        "per_complex_spearman_all": float(pc["spearman"].mean(skipna=True)),
        # --- comparability with the SKEMPI literature -----------------------------------
        "global_spearman": _safe_spearman(y, p),
        "global_pearson": _safe_pearson(y, p),
        # --- absolute calibration, in physical units ------------------------------------
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
    }

    # --- the classification framing ------------------------------------------------------
    m["macro_f1"] = float(f1_score(to_classes(y), to_classes(p), average="macro",
                                   labels=[0, 1, 2], zero_division=0))

    # --- does it at least get the direction right ----------------------------------------
    # Both of these have a high, non-obvious floor on this data and must always be read against
    # it. 86% of the rows with |ddG| > 1 are destabilising, so "always predict destabilising"
    # already scores 0.86 -- the raw number alone is meaningless.
    binary = (y > 0).astype(int)
    m["auroc"] = float(roc_auc_score(binary, p)) if 0 < binary.sum() < len(binary) else np.nan
    big = np.abs(y) > 1.0
    m["n_big"] = int(big.sum())
    if big.any():
        correct = np.sign(p[big]) == np.sign(y[big])
        pos = y[big] > 0
        m["sign_acc_big"] = float(np.mean(correct))
        m["sign_acc_big_base"] = float(max(pos.mean(), 1 - pos.mean()))
        # balanced across the two directions, so blindness to stabilising mutations shows up
        halves = [correct[pos].mean() if pos.any() else np.nan,
                  correct[~pos].mean() if (~pos).any() else np.nan]
        m["sign_acc_big_balanced"] = float(np.nanmean(halves))
    else:
        m["sign_acc_big"] = m["sign_acc_big_base"] = m["sign_acc_big_balanced"] = np.nan
    return m


def bootstrap(preds: pd.DataFrame, min_group: int = 10, n_boot: int = 1000,
              seed: int = 0, keys=("per_complex_spearman", "global_spearman", "rmse")) -> dict:
    """95% CIs by resampling complexes with replacement."""
    rng = np.random.default_rng(seed)
    groups = {cx: g for cx, g in preds.groupby("complex")}
    names = np.array(list(groups))
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        # relabel so a complex drawn twice counts twice rather than collapsing
        sample = pd.concat(
            [groups[c].assign(complex=f"{c}#{i}") for i, c in enumerate(pick)], ignore_index=True
        )
        try:
            mm = metrics(sample, min_group)
        except Exception:
            continue
        for k in keys:
            draws[k].append(mm[k])
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in draws.items() if v
    }


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str = "per_complex_spearman",
                     min_group: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """Delta between two models on the same resamples. This is how a rung is declared better.

    A rung is kept only if the CI on the delta clears zero.
    """
    merged = a.merge(b[["row_id", "y_pred"]], on="row_id", suffixes=("_a", "_b"), validate="1:1")
    rng = np.random.default_rng(seed)
    groups = {cx: g for cx, g in merged.groupby("complex")}
    names = np.array(list(groups))
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        sample = pd.concat(
            [groups[c].assign(complex=f"{c}#{i}") for i, c in enumerate(pick)], ignore_index=True
        )
        try:
            ma = metrics(sample.rename(columns={"y_pred_a": "y_pred"}), min_group)[metric]
            mb = metrics(sample.rename(columns={"y_pred_b": "y_pred"}), min_group)[metric]
        except Exception:
            continue
        deltas.append(mb - ma)
    d = np.array(deltas, float)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"metric": metric, "delta": float(np.nanmean(d)),
            "ci": (float(lo), float(hi)), "clears_zero": bool(lo > 0 or hi < 0)}


def format_report(m: dict, ci: dict | None = None, title: str = "") -> str:
    ci = ci or {}
    lines = [f"== {title} ==" if title else "== results =="]
    order = [
        ("per_complex_spearman", "per-complex Spearman (n>=10)  HEADLINE"),
        ("per_complex_spearman_all", "per-complex Spearman (all)"),
        ("per_complex_pearson", "per-complex Pearson"),
        ("global_spearman", "global Spearman"),
        ("global_pearson", "global Pearson"),
        ("rmse", "RMSE (kcal/mol)"),
        ("mae", "MAE  (kcal/mol)"),
        ("macro_f1", "3-class macro-F1"),
        ("auroc", "AUROC (sign of ddG)"),
        ("sign_acc_big", "sign acc, |ddG|>1"),
        ("sign_acc_big_base", "  ...majority-class floor"),
        ("sign_acc_big_balanced", "  ...balanced over direction"),
    ]
    for key, label in order:
        v = m.get(key, np.nan)
        s = f"  {label:34s} {v:+.3f}" if np.isfinite(v) else f"  {label:34s}    n/a"
        if key in ci:
            s += f"   95% CI [{ci[key][0]:+.3f}, {ci[key][1]:+.3f}]"
        lines.append(s)
    lines.append(f"  {'complexes (counted / total)':34s} {m['n_complexes_counted']} / {m['n_complexes']}"
                 f"   rows {m['n']}")
    return "\n".join(lines)
