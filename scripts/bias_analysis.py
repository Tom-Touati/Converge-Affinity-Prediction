"""Where the model's errors concentrate: class imbalance, complex size, mutation type.

The headline metrics average over a dataset that is badly unbalanced in three separate ways,
and each one hides a different failure:

  * the LABEL is 13/34/53 stabilising/neutral/destabilising, so a model can score well while
    never predicting the minority class;
  * the COMPLEXES are 2 to 87 rows, and three of them hold 24% of the data, so a pooled
    statistic is partly a statement about those three;
  * the MUTATIONS are dominated by alanine scanning, so "performance on a mutation" mostly
    means performance on X->A.

Each section below asks whether error depends on the imbalance, not merely whether the
imbalance exists. Mutations are parsed from the row id (``3SE8_HL_G|QH65A``), so this needs
nothing but a predictions file.

    python scripts/bias_analysis.py [--run E0a_rf_handcrafted_seed0]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths                 # noqa: E402
from src.evaluate import to_classes   # noqa: E402

MUT = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")
NAMES = ["stabilising", "neutral", "destabilising"]


def parse(row_id: str) -> dict:
    """`3SE8_HL_G|QH65A` -> wild-type Q, mutant A, 1 mutation."""
    muts = row_id.split("|", 1)[1].split(",") if "|" in row_id else []
    ok = [MUT.match(m) for m in muts]
    ok = [m for m in ok if m]
    return {"k": len(muts),
            "wt": ok[0].group(1) if ok else "?",
            "mt": ok[0].group(4) if ok else "?"}


def block(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def grouped(d: pd.DataFrame, by: str, order=None, min_n: int = 12) -> pd.DataFrame:
    """Signed error, |error| and within-group correlation for each level of `by`."""
    out = []
    for k, g in d.groupby(by):
        if len(g) < min_n:
            continue
        r = (float(np.corrcoef(g.y_pred, g.y_true)[0, 1])
             if g.y_true.std() > 0 and g.y_pred.std() > 0 else np.nan)
        out.append({by: k, "n": len(g), "true": g.y_true.mean(),
                    "pred": g.y_pred.mean(), "bias": (g.y_pred - g.y_true).mean(),
                    "mae": (g.y_pred - g.y_true).abs().mean(), "r": r})
    t = pd.DataFrame(out)
    if order is not None and len(t):
        t[by] = pd.Categorical(t[by], order)
        t = t.sort_values(by)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="E0a_rf_handcrafted_seed0")
    a, _ = ap.parse_known_args()

    d = pd.read_csv(paths.REPORTS / a.run / "predictions.csv")
    d = d.dropna(subset=["y_pred"]).copy()
    meta = pd.DataFrame([parse(r) for r in d.row_id])
    d = pd.concat([d.reset_index(drop=True), meta], axis=1)
    d["cls"] = to_classes(d.y_true.values)
    d["err"] = d.y_pred - d.y_true
    n_cx = d["complex"].nunique()
    print(f"{a.run}: {len(d)} rows, {n_cx} complexes\n"
          f"overall bias {d.err.mean():+.3f}, MAE {d.err.abs().mean():.3f}")

    # ---- 1. label imbalance -------------------------------------------------------
    block("1. LABEL IMBALANCE -- error by true class")
    t = grouped(d.assign(cls_name=[NAMES[c] for c in d.cls]), "cls_name", NAMES)
    print(t.round(3).to_string(index=False))
    lo = d[d.cls == 0]; hi = d[d.cls == 2]
    print(f"\n   stabilising rows are pulled UP by {lo.err.mean():+.3f} kcal/mol on average,")
    print(f"   destabilising rows pulled DOWN by {hi.err.mean():+.3f}. That is regression to")
    print(f"   the mean, and it is why thresholding at the LABEL edges starves the minority")
    print(f"   class: the predictions simply do not reach far enough out.")
    span_t = d.y_true.max() - d.y_true.min()
    span_p = d.y_pred.max() - d.y_pred.min()
    print(f"   predicted range {span_p:.2f} vs true range {span_t:.2f} "
          f"({span_p/span_t:.0%} of it); sd {d.y_pred.std():.3f} vs {d.y_true.std():.3f}")

    # ---- 2. complex imbalance -----------------------------------------------------
    block("2. COMPLEX IMBALANCE -- does performance depend on how many rows a complex has?")
    size = d.groupby("complex").size().rename("n")
    print(f"   rows per complex: median {int(size.median())}, min {size.min()}, "
          f"max {size.max()}; top-3 hold {size.nlargest(3).sum()/len(d):.0%} of all rows")
    per = []
    for cxn, g in d.groupby("complex"):
        if len(g) < 5 or g.y_true.std() == 0 or g.y_pred.std() == 0:
            continue
        per.append({"complex": cxn, "n": len(g),
                    "r": float(np.corrcoef(g.y_pred, g.y_true)[0, 1]),
                    "mae": g.err.abs().mean(), "spread": g.y_true.std()})
    per = pd.DataFrame(per)
    print(f"   {len(per)} complexes have the >=5 rows needed to score a correlation; "
          f"{n_cx - len(per)} do not and are invisible to the headline metric")
    q = pd.qcut(per.n, 3, labels=["small", "medium", "large"])
    print("\n" + per.assign(size=q).groupby("size", observed=True)
          .agg(complexes=("complex", "size"), rows=("n", "sum"),
               mean_r=("r", "mean"), mean_mae=("mae", "mean"),
               label_spread=("spread", "mean")).round(3).to_string())
    c1 = per.n.corr(per.r, method="spearman")
    c2 = per.spread.corr(per.r, method="spearman")
    print(f"\n   corr(rows, per-complex r)        = {c1:+.3f}")
    print(f"   corr(label spread, per-complex r) = {c2:+.3f}")
    print("   If the second is much larger, the metric is not measuring sample size but")
    print("   how much variation a complex has to rank in the first place.")
    worst = per.nsmallest(5, "r")[["complex", "n", "r", "spread"]]
    print("\n   five worst complexes:\n" + worst.round(3).to_string(index=False))

    # ---- 3. mutation imbalance ----------------------------------------------------
    block("3. MUTATION IMBALANCE -- alanine scanning dominates")
    d["to_ala"] = d.mt == "A"
    d["from_gly"] = d.wt == "G"
    print(f"   X->A is {d.to_ala.mean():.0%} of all rows")
    print(grouped(d.assign(kind=np.where(d.to_ala, "X->A", "other")), "kind")
          .round(3).to_string(index=False))
    block("   by wild-type residue (>=12 rows)")
    print(grouped(d, "wt").sort_values("n", ascending=False).round(3).to_string(index=False))
    block("   by mutant residue (>=12 rows)")
    print(grouped(d, "mt").sort_values("n", ascending=False).round(3).to_string(index=False))
    block("   single vs multi-point")
    print(grouped(d.assign(points=np.where(d.k > 1, "multi", "single")), "points")
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
