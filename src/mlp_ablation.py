"""Is the MLP's loss to the random forest a property of the problem, or of one bad configuration?

The neural gate in ARCHITECTURE.md turns on a single comparison: a plain MLP on the 33 scalar
features scored 0.351 against the forest's 0.470, so cross-attention over 48 residues -- which
adds parameters to the losing side -- was not built. That conclusion deserves more than one
configuration behind it.

This runs a small **pre-declared** grid over width, depth and regularisation strength, three
seeds each, and reports every cell. It does not pick a winner: selecting the maximum over twelve
configurations by the outer-fold score would be tuning on the test set, and the resulting number
would not be comparable to any other rung. The question here is narrower and does not need
selection -- does *any* reasonable MLP configuration reach the forest, or does the whole family
sit below it?

    python -m src.mlp_ablation
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from . import evaluate, paths, splits, train

warnings.filterwarnings("ignore")

FEATURES = ["chem", "geom", "mpnn"]
HIDDEN = [(128, 32), (64,), (32,), (16,)]
ALPHA = [1.0, 10.0, 100.0]
SEEDS = (0, 1, 2)


def _fit_cv(X, y, folds, hidden, alpha, seed):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        est = make_pipeline(
            StandardScaler(),
            # early stopping is off on purpose: sklearn carves a RANDOM validation split, which
            # would put near-identical complexes on both sides and leak the homology the outer
            # split exists to prevent.
            MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=2000,
                         early_stopping=False, random_state=seed),
        )
        p[te] = est.fit(X[tr], y[tr]).predict(X[te])
    return p


def run(verbose: bool = True) -> pd.DataFrame:
    d = splits.load()
    X = train.build_matrix(d, FEATURES)
    y = d.ddG.to_numpy(float)
    folds = d.fold.to_numpy()

    def n_params(hidden):
        dims = [X.shape[1], *hidden, 1]
        return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

    rows = []
    for hidden in HIDDEN:
        for alpha in ALPHA:
            scores, rmses = [], []
            for s in SEEDS:
                p = _fit_cv(X, y, folds, hidden, alpha, s)
                m = evaluate.metrics(pd.DataFrame({
                    "row_id": d.row_id, "complex": d["#Pdb"], "y_true": y, "y_pred": p}), 10)
                scores.append(m["per_complex_spearman"])
                rmses.append(m["rmse"])
            rows.append({
                "hidden": str(hidden), "alpha": alpha, "params": n_params(hidden),
                "per_cx_rho": round(float(np.mean(scores)), 3),
                "seed_sd": round(float(np.std(scores)), 4),
                "rmse": round(float(np.mean(rmses)), 3),
            })
            if verbose:
                print(f"  {str(hidden):>10}  alpha={alpha:<6} "
                      f"rho {rows[-1]['per_cx_rho']:+.3f} (sd {rows[-1]['seed_sd']:.3f})",
                      flush=True)

    t = pd.DataFrame(rows).sort_values("per_cx_rho", ascending=False).reset_index(drop=True)
    out = paths.REPORTS / "mlp_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)

    if verbose:
        print()
        print("=" * 78)
        print("MLP ablation -- all cells reported; the maximum is NOT a selectable result")
        print("=" * 78)
        print(t.to_string(index=False))
        print()
        print(f"  best cell          : {t.per_cx_rho.iloc[0]:+.3f}  "
              f"({t.hidden.iloc[0]}, alpha={t.alpha.iloc[0]})")
        print(f"  random forest      : +0.470  (3-seed ensemble, same folds)")
        print(f"  cells beating the forest: {(t.per_cx_rho > 0.470).sum()} of {len(t)}")
        print(f"\nwrote {out.relative_to(paths.ROOT)}")
    return t


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
