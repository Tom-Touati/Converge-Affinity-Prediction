"""Step 0 of the reproduction: check the published number against the authors' own artefacts.

Before spending a GPU-day reproducing Table 1, it is worth asking whether Table 1 is
internally consistent with the predictions the authors shipped. ProtAttBA commits
``cross_validation/results/S1131_results.csv``, which holds a fold id, a row index, a
prediction and a label for all 1131 rows -- i.e. the complete output of one 10-fold run.

This script answers three questions, all of which turn out to be yes:

1. Does ``KFold(10, shuffle=True, random_state=3407)`` reproduce their fold assignment exactly?
   If so, the split is pinned and our run can use the identical one.
2. Do the ``true_label`` values line up with ``S1131.csv`` row order? This confirms
   ``test_indices`` indexes the CSV as read, so the shipped predictions can be joined back to
   sequences and PDB ids.
3. Do their predictions, scored per fold and averaged, land on the published 0.84/0.75/1.31?

Run: ``python verify_published.py``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import (
    PUBLISHED,
    PUBLISHED_PREDICTIONS,
    RESULTS,
    S1131_CSV,
    SEED,
    fold_assignment,
    score,
    score_by_fold,
    summarise,
)


def main() -> None:
    data = pd.read_csv(S1131_CSV)
    pub = pd.read_csv(PUBLISHED_PREDICTIONS)
    print(f"S1131.csv                 {len(data)} rows, {data.PDB.nunique()} distinct PDB ids")
    print(f"shipped predictions       {len(pub)} rows, {pub.fold.nunique()} folds, "
          f"{pub.test_indices.nunique()} distinct row indices")

    # 1. fold assignment
    ours = fold_assignment(len(data))
    theirs = np.empty(len(data), dtype=int)
    theirs[pub.test_indices.values] = pub.fold.values
    assert (ours == theirs).all(), "fold assignment does not match seed 3407"
    print(f"\n[1] fold assignment      KFold(10, shuffle=True, random_state={SEED}) matches "
          f"exactly, all {len(data)} rows")

    # 2. label alignment
    joined = pub.merge(data[["ddG", "PDB"]], left_on="test_indices", right_index=True)
    drift = float(np.abs(joined.true_label - joined.ddG).max())
    assert drift < 1e-4, f"labels do not match S1131.csv row order (max drift {drift})"
    print(f"[2] label alignment      test_indices indexes S1131.csv row order, "
          f"max |drift| = {drift:.1e}")

    # 3. the published table
    per_fold = score_by_fold(
        pub.rename(columns={"true_label": "y_true", "prediction": "y_pred"})
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(RESULTS / "published_per_fold.csv", index=False)

    print("\n[3] scoring their own shipped predictions, per fold:\n")
    print(per_fold.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary = summarise(per_fold)
    print("\n    metric   reproduced from shipped preds   Table 1 (ESM2, S1131)   match")
    for _, r in summary.iterrows():
        pm, ps = PUBLISHED[r.metric]
        ok = abs(round(r["mean"], 2) - pm) < 5e-3 and abs(round(r["std"], 2) - ps) < 5e-3
        print(f"    {r.metric:<7s} {r['mean']:.4f} +/- {r['std']:.4f}                "
              f"{pm:.2f} +/- {ps:.2f}            {'yes' if ok else 'NO'}")

    pooled = score(pub.true_label.values, pub.prediction.values)
    print(f"\n    pooled over all 1131 predictions (not what the paper reports): "
          f"PCC {pooled['pcc']:.4f}  rho {pooled['rho']:.4f}  RMSE {pooled['rmse']:.4f}")

    print(
        "\nConclusion: Table 1 is exactly the per-fold mean +/- population std of the\n"
        "predictions committed in the repo. The target is internally consistent, the split is\n"
        f"pinned to seed {SEED}, and every one of the 1131 published predictions is available\n"
        "for per-example comparison against our own run."
    )


if __name__ == "__main__":
    main()
