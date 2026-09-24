"""How many test rows are easy, medium or hard — by structural distance to the training fold.

"Performance" on this dataset is an average over test rows that differ enormously in how much
help the training folds give them. A complex with a TM > 0.8 twin in training is being asked a
near-retrieval question; one with nothing above 0.5 is being asked to generalise. Reporting a
single number over both conceals which of the two the model can actually do.

Difficulty is the **maximum TM-score from a test complex to any complex in its training
folds**, computed per fold, so it reflects the split the model was actually trained under:

    easy    max TM >= 0.8    a near-identical training twin exists
    medium  0.5 <= TM < 0.8  same fold, different structure
    hard    max TM  < 0.5    no structural relative in training

`data/tm_tiers.csv` already carries this for the cluster split, where by construction nothing
is easy. This computes it for `frozen5`, the by-complex split the headline numbers use.

    python scripts/difficulty_tiers.py [--run ...]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

EASY, MEDIUM = 0.80, 0.50


def tiers_for_split(folds: pd.Series, tm: pd.DataFrame) -> pd.DataFrame:
    """folds: complex -> fold id. Returns per-complex max TM to its own training folds."""
    out = []
    for cx, f in folds.items():
        if cx not in tm.index:
            continue
        train = [c for c, g in folds.items() if g != f and c in tm.columns and c != cx]
        best = float(tm.loc[cx, train].max()) if train else np.nan
        out.append({"complex": cx, "fold": f, "max_tm_to_train": best,
                    "tier": "easy" if best >= EASY else
                            "medium" if best >= MEDIUM else "hard"})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="E0a_rf_handcrafted_seed0")
    ap.add_argument("--net", default="cat128_reg2_l1")   # the submitted model
    a, _ = ap.parse_known_args()

    tm = pd.read_csv("data/tm_matrix.csv", index_col=0)
    ref = pd.read_csv(paths.REPORTS / "perturb_v3_base" / "predictions.csv")
    if "fold" not in ref.columns:
        raise SystemExit("predictions file has no fold column")
    folds = ref.drop_duplicates("complex").set_index("complex").fold

    t = tiers_for_split(folds, tm)
    d = ref.merge(t[["complex", "max_tm_to_train", "tier"]], on="complex", how="left")

    order = ["easy", "medium", "hard"]
    print("Difficulty = max TM-score from a test complex to any complex in its TRAINING folds")
    print(f"easy >= {EASY} | medium >= {MEDIUM} | hard < {MEDIUM}   (frozen5, by complex)\n")

    cx = (t.groupby("tier").size().reindex(order).fillna(0).astype(int)
          .rename("complexes").to_frame())
    cx["rows"] = d.groupby("tier").size().reindex(order).fillna(0).astype(int)
    cx["rows %"] = (100 * cx["rows"] / cx["rows"].sum()).round(1)
    cx["median max TM"] = t.groupby("tier").max_tm_to_train.median().reindex(order).round(3)
    print(cx.to_string())

    # per-tier performance, for the forest and the network, on the same rows
    preds = {}
    p = paths.REPORTS / a.run / "predictions.csv"
    if p.exists():
        preds["forest"] = pd.read_csv(p).set_index("row_id").y_pred
    o = pathlib.Path(f"results/oof/{a.net}.csv")
    if o.exists():
        preds[a.net] = pd.read_csv(o).groupby("row_id").ddg_pred.mean()

    print("\nper-complex r within each tier (complexes with >= 5 rows):\n")
    hdr = f"{'tier':<9}{'complexes':>11}{'rows':>7}" + "".join(f"{k:>17}" for k in preds)
    print(hdr); print("-" * len(hdr))
    D = d.set_index("row_id")
    for tier in order:
        idx = D.index[D.tier == tier]
        cells = []
        for k, pr in preds.items():
            ii = idx.intersection(pr.index)
            rs = []
            for _, g in pr.loc[ii].groupby(D.loc[ii, "complex"]):
                tt = D.loc[g.index, "y_true"]
                if len(g) < 5 or tt.std() == 0 or g.std() == 0:
                    continue
                rs.append(float(np.corrcoef(g, tt)[0, 1]))
            cells.append(f"{np.mean(rs):+.3f} ({len(rs)})" if rs else "   -")
        print(f"{tier:<9}{int((t.tier == tier).sum()):>11}{len(idx):>7}"
              + "".join(f"{c:>17}" for c in cells))
    print("\n(n) is the number of complexes that clear the >=5-row threshold in that tier.")

    # the same question for the cluster split, which is why it is the harder protocol
    ct = pathlib.Path("data/tm_tiers.csv")
    if ct.exists():
        c = pd.read_csv(ct)
        print("\nFor contrast, the CLUSTER split (data/tm_tiers.csv):")
        print(c.groupby("tier").size().reindex(order).fillna(0).astype(int).to_string())
        print("By construction the cluster split withholds structural relatives, so a complex")
        print("with an easy twin cannot exist -- which is exactly what makes it the harder test.")


if __name__ == "__main__":
    main()
