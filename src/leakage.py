"""How much of a published ddG score is the split?

The field's de facto protocol -- RDE-Network, DiffAffinity, DDAffinity, CIR-DDG -- is k-fold
cross-validation grouped by **structure**: no complex appears in two folds. This project groups by
**homology cluster** instead, because thirteen of our 54 complexes are different antibodies
against hen lysozyme and a structure-level split happily spreads them across folds.

That makes our numbers lower by construction, and a reader comparing them to a published table is
comparing two different difficulties. This module measures the difference by running one model
under three groupings that differ in nothing else:

* ``random``    -- rows split at random. Same complex, even same position, on both sides. This is
                   the number a careless pipeline reports.
* ``structure`` -- grouped by complex (``#Pdb``). **The published protocol.**
* ``cluster``   -- grouped by homology cluster. Ours.

Same features, same model, same folds count, same seeds. The only moving part is the grouping, so
the spread between them is the leakage estimate and nothing else.

**What this does and does not license.** It reproduces the published *split methodology*, not the
published *cohort*: RDE-Network and DiffAffinity train on all of SKEMPI, CIR-DDG on 343
antibody-antigen complexes, and we have 54. So the structure-level row here is "our model under
their protocol", which bounds how much of the gap to a published number is split rather than
method. It is not a claim to have reproduced their experiment.

    python -m src.leakage
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from . import evaluate, paths, splits, train
from .model import MODELS

FEATURES = ["chem", "geom", "mpnn"]
REGIMES = ("random", "structure", "cluster")


def _folds(df: pd.DataFrame, regime: str, k: int, seed: int) -> np.ndarray:
    """Assign every row to one of k folds under the requested grouping."""
    out = np.full(len(df), -1)
    if regime == "random":
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        it = splitter.split(df)
    else:
        groups = df["#Pdb"] if regime == "structure" else df["cluster"]
        # GroupKFold is deterministic; the seed only moves the model, which is the honest
        # comparison -- we are not shopping for a favourable grouping.
        it = GroupKFold(n_splits=k).split(df, groups=groups)
    for f, (_, te) in enumerate(it):
        out[te] = f
    assert (out >= 0).all()
    return out


def run_regime(df: pd.DataFrame, X: pd.DataFrame, regime: str, k: int, model: str,
               seeds=(0, 1, 2), single_only: bool = False) -> dict:
    y = df["ddG"].to_numpy(float)
    per_seed = []
    for seed in seeds:
        folds = _folds(df, regime, k, seed)
        preds = np.full(len(df), np.nan)
        for f in range(k):
            te, tr = folds == f, folds != f
            est = MODELS[model](seed).fit(X[tr], y[tr])
            preds[te] = est.predict(X[te])
        out = pd.DataFrame({"row_id": df.row_id, "complex": df["#Pdb"],
                            "y_true": y, "y_pred": preds})
        if single_only:
            out = out[(df["n_mut"] == 1).to_numpy()]
        per_seed.append(evaluate.metrics(out, min_group=10))
    agg = {}
    for key in ("global_pearson", "global_spearman", "per_complex_spearman",
                "per_complex_pearson", "rmse", "mae", "auroc"):
        vals = [m[key] for m in per_seed]
        agg[key] = float(np.nanmean(vals))
        agg[key + "_sd"] = float(np.nanstd(vals))
    agg["n"] = per_seed[0]["n"]
    agg["n_cx_counted"] = per_seed[0]["n_complexes_counted"]
    return agg


def compare(k: int = 3, model: str = "rf", single_only: bool = False,
            seeds=(0, 1, 2), verbose: bool = True) -> pd.DataFrame:
    df = splits.load()
    X = train.build_matrix(df, FEATURES)

    rows = []
    for regime in REGIMES:
        r = run_regime(df, X, regime, k, model, seeds, single_only)
        rows.append({
            "grouping": regime,
            "n": r["n"], "cx_scored": r["n_cx_counted"],
            "global_pearson": round(r["global_pearson"], 3),
            "global_spearman": round(r["global_spearman"], 3),
            "per_cx_spearman": round(r["per_complex_spearman"], 3),
            "rmse": round(r["rmse"], 3),
            "mae": round(r["mae"], 3),
            "auroc": round(r["auroc"], 3),
            "seed_sd": round(r["global_spearman_sd"], 4),
        })
    t = pd.DataFrame(rows)

    if verbose:
        cohort = "single-point rows only" if single_only else "all rows"
        print("=" * 104)
        print(f"LEAKAGE: one model ({model} on {'+'.join(FEATURES)}), three groupings, "
              f"{k}-fold, {len(seeds)} seeds, {cohort}")
        print("=" * 104)
        print(t.to_string(index=False))
        rnd = t[t.grouping == "random"].iloc[0]
        stc = t[t.grouping == "structure"].iloc[0]
        clu = t[t.grouping == "cluster"].iloc[0]
        print()
        print(f"  random -> structure : global Spearman {rnd.global_spearman:+.3f} -> "
              f"{stc.global_spearman:+.3f}   ({100*(1-stc.global_spearman/rnd.global_spearman):.0f}% of the score was same-complex leakage)")
        print(f"  structure -> cluster: global Spearman {stc.global_spearman:+.3f} -> "
              f"{clu.global_spearman:+.3f}   ({100*(1-clu.global_spearman/stc.global_spearman):.0f}% more was homology leakage)")
        print()
        print("  The published protocol is the 'structure' row. Compare that to a published")
        print("  table, not the 'cluster' row -- and remember the cohort differs (54 complexes")
        print("  here against 343 in CIR-DDG and all of SKEMPI in RDE-Network/DiffAffinity).")
    return t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-k", type=int, default=3, help="folds (3 matches the published protocol)")
    p.add_argument("--model", default="rf")
    p.add_argument("--single-only", action="store_true",
                   help="score single-point mutations only, as CIR-DDG does")
    a = p.parse_args()
    t = compare(a.k, a.model, a.single_only)
    out = paths.REPORTS / "leakage"
    out.mkdir(parents=True, exist_ok=True)
    name = f"k{a.k}_{a.model}{'_single' if a.single_only else ''}.csv"
    t.to_csv(out / name, index=False)
    print(f"\nwrote {(out / name).relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
