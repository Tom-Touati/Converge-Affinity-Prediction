"""Put our run next to the published table, and next to their per-example predictions.

The aggregate comparison answers "did we land on 0.84". The per-example one answers the more
useful question: *are we predicting the same things they predicted*. Those can come apart. A
reproduction that matches the headline correlation while correlating only 0.5 with their
own per-row output has not reproduced their model, it has reproduced the benchmark's difficulty.

Also reports the two floors that make the headline number readable:

* **predict the training mean** -- what 0.84 has to be compared against.
* **label leakage through duplicate rows.** S1131 is 1131 rows over 112 PDB ids, split by
  row with no grouping, so a test row's complex is essentially always represented in training.
  Worse, the *same* mutation can appear twice with different ddG values. This counts how many
  test rows have an exact (PDB, mutation) twin in their own training split.

Run: ``python compare.py`` (after run_cv.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from metrics import (
    PUBLISHED,
    PUBLISHED_PREDICTIONS,
    RESULTS,
    S1131_CSV,
    fold_assignment,
    score,
    score_by_fold,
    summarise,
)

PROTOCOLS = ("upstream", "honest")


def leakage_report(data: pd.DataFrame) -> None:
    folds = fold_assignment(len(data))
    key = data.PDB.astype(str) + ":" + data.mutation_clean.astype(str)

    same_complex = 0
    exact_twin = 0
    for f in range(folds.max() + 1):
        test = folds == f
        train_pdb = set(data.PDB[~test])
        train_key = set(key[~test])
        same_complex += int(data.PDB[test].isin(train_pdb).sum())
        exact_twin += int(key[test].isin(train_key).sum())

    n = len(data)
    print(f"    test rows whose PDB also appears in training      "
          f"{same_complex:5d} / {n}  ({same_complex / n:.1%})")
    print(f"    test rows with an exact (PDB, mutation) twin in   "
          f"{exact_twin:5d} / {n}  ({exact_twin / n:.1%})")
    print(f"      training  -- i.e. the same mutation, measured twice, split across folds")


def main() -> None:
    data = pd.read_csv(S1131_CSV)
    pub = pd.read_csv(PUBLISHED_PREDICTIONS).rename(
        columns={"true_label": "y_true", "prediction": "y_pred", "test_indices": "row"}
    )
    pub_fold = score_by_fold(pub)

    print("=" * 78)
    print("S1131, ESM2, 10-fold CV")
    print("=" * 78)

    rows = [("published Table 1", *[f"{PUBLISHED[m][0]:.2f} +/- {PUBLISHED[m][1]:.2f}"
                                    for m in ("pcc", "rho", "rmse")], "")]
    s = summarise(pub_fold).set_index("metric")
    rows.append(("their shipped predictions",
                 *[f"{s.loc[m, 'mean']:.3f} +/- {s.loc[m, 'std']:.3f}"
                   for m in ("pcc", "rho", "rmse")], "n=1131"))

    ours = {}
    for proto in PROTOCOLS:
        path = RESULTS / f"{proto}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ours[proto] = df
        pf = score_by_fold(df)
        s = summarise(pf).set_index("metric")
        rows.append((f"our run, {proto} protocol",
                     *[f"{s.loc[m, 'mean']:.3f} +/- {s.loc[m, 'std']:.3f}"
                       for m in ("pcc", "rho", "rmse")],
                     f"n={len(df)}, {df.fold.nunique()} folds"))

    # floor: predict the training mean of each fold
    folds = fold_assignment(len(data))
    y = data.ddG.values
    floor = np.empty(len(data))
    for f in range(folds.max() + 1):
        floor[folds == f] = y[folds != f].mean()
    ff = score_by_fold(pd.DataFrame({"fold": folds, "y_true": y, "y_pred": floor}))
    s = summarise(ff).set_index("metric")
    rows.append(("floor: training mean",
                 *[f"{s.loc[m, 'mean']:.3f} +/- {s.loc[m, 'std']:.3f}"
                   for m in ("pcc", "rho", "rmse")], "constant per fold"))

    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'':<{w}}{'PCC':<18}{'Spearman':<18}{'RMSE':<18}note")
    for r in rows:
        print(f"{r[0]:<{w}}{r[1]:<18}{r[2]:<18}{r[3]:<18}{r[4]}")

    # ---- per-example agreement ---------------------------------------------------------
    if ours:
        print("\n" + "-" * 78)
        print("per-example agreement with their shipped predictions (all 1131 rows)")
        print("-" * 78)
        for proto, df in ours.items():
            j = df.merge(pub[["row", "y_pred"]], on="row", suffixes=("", "_pub"))
            r = pearsonr(j.y_pred, j.y_pred_pub)[0]
            rho = spearmanr(j.y_pred, j.y_pred_pub)[0]
            mad = float(np.abs(j.y_pred - j.y_pred_pub).mean())
            print(f"    {proto:<10s} n={len(j):5d}   PCC {r:.4f}   rho {rho:.4f}   "
                  f"mean |ours - theirs| {mad:.3f} kcal/mol")

    print("\n" + "-" * 78)
    print("what the row-wise split gives the model for free")
    print("-" * 78)
    leakage_report(data)

    print("\n" + "-" * 78)
    print("dataset shape")
    print("-" * 78)
    print(f"    rows {len(data)}, distinct PDB ids {data.PDB.nunique()}, "
          f"ddG mean {data.ddG.mean():.2f} std {data.ddG.std():.2f} kcal/mol")
    sizes = data.PDB.value_counts()
    print(f"    rows per PDB: median {int(sizes.median())}, max {sizes.max()} "
          f"({sizes.idxmax()}), top 3 hold {sizes.head(3).sum() / len(data):.0%} of rows")


if __name__ == "__main__":
    main()
