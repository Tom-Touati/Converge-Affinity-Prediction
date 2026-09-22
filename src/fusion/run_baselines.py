"""E0 — the baselines every later experiment has to beat, on the frozen 5-fold split.

Four rungs, all tree/linear and therefore runnable on a laptop CPU:

``E0a_rf_handcrafted``      the project's existing best tree feature set (chem, geom, geomrev,
                            mpnn) with a random forest. Re-run here on the frozen 5-fold
                            by-complex split rather than the project's cluster split, so it
                            sits in the same table as everything else.
``E0b_rf_pooled_esm``       sequence only: pooled ESM-2 650M for the wild type and the mutant,
                            plus the cached site scalars.
``E0c_rf_pooled_esm_mpnn``  E0b plus pooled ProteinMPNN representation and its likelihood
                            ratios. **The multimodal baseline.**
``E0e_mean``                predict the training mean. Not in the plan, added because without
                            it the table has no floor, and under grouped CV a constant is not
                            a trivial baseline: the project measured a constant-per-fold
                            predictor at global Spearman -0.36 purely from between-fold label
                            shift.

``src/train.py`` is not touched (rule 9). E0a re-implements only the *invocation*, using the
same ``src.model.MODELS`` registry and the same feature blocks, so the estimator is literally
the project's.

Seeds: the plan's fixed recipe asks for 3 seeds on neural experiments. A random forest is
seeded too, so all three are run and the spread is reported; it is cheap and it shows how much
of any gap is seed noise.

Run: ``python -m src.fusion.run_baselines``            (all four)
     ``python -m src.fusion.run_baselines --exp E0c``  (one)
"""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import pandas as pd

from src import paths
from src import splits
from src.fusion import features, metrics, results, splits_frozen
from src.model import MODELS

SEEDS = (0, 1, 2)

#: (experiment name, feature set, model key, drop wide embedding columns)
EXPERIMENTS: dict[str, tuple[str, str, bool]] = {
    "E0a_rf_handcrafted": ("handcrafted", "rf", False),
    "E0b_rf_pooled_esm": ("pooled_esm", "rf", False),
    "E0c_rf_pooled_esm_mpnn": ("pooled_esm_mpnn", "rf", False),
    "E0e_mean": ("handcrafted", "mean", True),
}


#: Split strategies the ladder can be scored on. ``frozen5`` is the plan's; the other three
#: come from the project's own ``src/splits.py`` and are what its published numbers use.
#: Keeping all four selectable is the point -- the project measured that grouping matters far
#: more than most modelling choices, so a number without its grouping is not a number.
GROUPINGS = ("frozen5", "cluster", "complex", "random")


def dataset_with_folds(grouping: str = "frozen5") -> pd.DataFrame:
    """The 940 rows with a fold column, from whichever split strategy is asked for.

    ``frozen5``
        the plan's ``data/splits/skempi_abag_5fold_by_complex.json``: 5 folds, grouped by PDB.
    ``cluster``
        the project's frozen ``data/folds.csv``: 4 folds grouped by **homology cluster**,
        built with Smith-Waterman over antigen chains. This is the project's headline protocol
        and the hard one -- under it no complex retains a TM > 0.8 training twin, against 42 of
        54 under complex grouping.
    ``complex`` / ``random``
        the project's secondary splits, from ``data/folds_complex.csv`` and a random draw.

    A fold number means nothing without the grouping that produced it, so the grouping is
    recorded on every results row.
    """
    if grouping not in GROUPINGS:
        raise SystemExit(f"grouping must be one of {GROUPINGS}")
    ds = pd.read_parquet(paths.DATASET)
    if grouping == "frozen5":
        ds["fold"] = splits_frozen.fold_of(ds["row_id"])
        ds["cluster"] = ds["#Pdb"]
        return ds
    joined = splits.load(grouping)
    return joined.rename(columns={"fold": "fold"})


def run_one(exp: str, feature_set: str, model_key: str, drop_vectors: bool,
            seeds=SEEDS, grouping: str = "frozen5") -> pd.DataFrame | None:
    """One experiment across every fold and seed. Returns out-of-fold predictions."""
    ds = dataset_with_folds(grouping)
    X = features.build(ds["row_id"], feature_set, drop_vectors=drop_vectors)
    y = ds["ddG"].to_numpy(dtype=float)
    folds = ds["fold"].to_numpy()
    sha = results.git_sha()

    print(f"\n=== {exp} ===")
    print(f"  features {feature_set}: {X.shape[1]} columns over {len(X)} rows, "
          f"model {model_key}, grouping {grouping} ({len(set(folds))} folds)")

    all_preds = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        t0 = time.time()
        n_params = 0
        for k in sorted(set(folds)):
            te = folds == k
            tr = ~te
            est = MODELS[model_key](seed)
            est.fit(X[tr.tolist()] if isinstance(X, list) else X.loc[tr], y[tr])
            oof[te] = est.predict(X.loc[te])
            # "params" for a forest is the total node count, the closest honest analogue.
            if hasattr(est, "estimators_"):
                n_params = int(sum(t.tree_.node_count for t in est.estimators_))
        minutes = (time.time() - t0) / 60

        preds = pd.DataFrame({
            "row_id": ds["row_id"], "complex": ds["#Pdb"], "fold": folds,
            "y_true": y, "y_pred": oof,
        })
        all_preds.append(preds.assign(seed=seed))
        # `exp` already carries the grouping suffix when main() added one, so do not
        # append it twice.
        results.save_predictions(exp, seed, preds)

        for k in sorted(set(folds)):
            m = folds == k
            row = metrics.score(y[m], oof[m], complexes=ds.loc[m, "#Pdb"])
            row.update(exp=exp, fold=int(k), seed=seed, n_train=int((~m).sum()),
                       n_test=int(m.sum()), params=n_params,
                       train_minutes=round(minutes / len(set(folds)), 3),
                       git_sha=sha, dataset="skempi_abag",
                       note=f"{feature_set}/{model_key}/{grouping}")
            results.append(row)

        pooled = metrics.score(y, oof, complexes=ds["#Pdb"])
        print(f"  seed {seed}: pooled pearson {pooled['pearson']:+.3f}  "
              f"spearman {pooled['spearman']:+.3f}  rmse {pooled['rmse']:.3f}  "
              f"acc3 {pooled['acc3']:.3f}  per-complex rho {pooled['per_complex_spearman']:+.3f}"
              f"   [{minutes:.1f} min]")

    return pd.concat(all_preds, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default=None, help="run one experiment instead of all")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--grouping", default="frozen5", choices=GROUPINGS,
                    help="which split strategy to score on")
    cli = ap.parse_args()

    todo = {cli.exp: EXPERIMENTS[cli.exp]} if cli.exp else EXPERIMENTS
    for exp, (fs, model_key, drop) in todo.items():
        try:
            name = exp if cli.grouping == "frozen5" else f"{exp}__{cli.grouping}"
            run_one(name, fs, model_key, drop, seeds=tuple(cli.seeds),
                    grouping=cli.grouping)
        except Exception:                      # rule 8: a failure never stops the ladder
            tb = traceback.format_exc()
            results.record_failure(exp, "all", "all", tb)
            print(f"  FAILED {exp} -- written to {results.FAILURES_MD.name}")
            print(tb.splitlines()[-1])


if __name__ == "__main__":
    main()
