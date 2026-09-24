"""The submission's results table: every model, one truth, one aggregation convention.

Two numbers are reported per model and they answer different questions.

``ensemble``  averages the seeds' predictions and scores once. This is what you would ship
              if you trained k models and averaged them, and it is what ``publish`` writes.
``per seed``  scores each seed separately and reports the mean and the spread. This is what
              ONE training run gives you, and the spread is the only honest way to read a
              difference between two models on this dataset.

Mixing them is how a model appears to gain 0.05 by being written up differently -- the same
run reads +0.300 as a three-seed ensemble and +0.249 as the mean of its seeds. Every row
here is computed both ways from the same predictions.

Classification is reported alongside, because the ranking metrics hide a real defect: the
forest wins per-complex correlation while finding 7 of 126 stabilising mutations. Balanced
accuracy is reported rather than accuracy, since the classes are 13/34/53 and predicting the
majority everywhere scores 53%.

    python scripts/final_table.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths            # noqa: E402
from src.evaluate import to_classes  # noqa: E402

REFERENCE = "perturb_v3_base"
MIN_ROWS = 5

#: label, and where the per-seed predictions come from
FOREST = {"E0a_rf_handcrafted (chem+geom+MPNN)": "results/predictions/E0a_rf_handcrafted_seed{}.csv"}
NETS = ["l1_gated", "cat128_reg2_l1", "gf_reg2", "st64_nopca_grouped", "area_concat",
        "st64_noattn", "st64_xattn_rev_nopca", "st64_film_struct"]


def load_truth():
    ref = pd.read_csv(paths.REPORTS / REFERENCE / "predictions.csv").set_index("row_id")
    ref = ref.sort_index()
    return ref.y_true, ref["complex"]


def per_complex(pred: pd.Series, truth: pd.Series, cx: pd.Series) -> tuple[float, int]:
    idx = pred.index.intersection(truth.index)
    p, t, c = pred.loc[idx], truth.loc[idx], cx.loc[idx]
    rs, neg = [], 0
    for _, g in p.groupby(c):
        tt = t.loc[g.index]
        if len(g) < MIN_ROWS or tt.std() == 0 or g.std() == 0:
            continue
        r = float(np.corrcoef(g, tt)[0, 1])
        rs.append(r)
        neg += r < 0
    return (float(np.mean(rs)) if rs else float("nan")), neg


def balanced(pred: pd.Series, truth: pd.Series, calibrate: bool = False) -> tuple[float, list]:
    idx = pred.index.intersection(truth.index)
    p, t = pred.loc[idx].values, to_classes(truth.loc[idx].values)
    if calibrate:
        # cut the score at its OWN quantiles so the predicted class shares match the true
        # ones. Monotone, so the ranking above is untouched.
        q = np.quantile(p, np.cumsum([float((t == k).mean()) for k in range(2)]))
        pc = np.digitize(p, q)
    else:
        pc = to_classes(p)
    rec = [float(((pc == k) & (t == k)).sum()) / max(float((t == k).sum()), 1)
           for k in range(3)]
    return float(np.mean(rec)), rec


def seeds_of(name: str) -> list[pd.Series]:
    """Per-seed prediction series for one model, from whichever store holds it."""
    out = []
    if name in FOREST:
        for s in range(3):
            f = pathlib.Path(FOREST[name].format(s))
            if f.exists():
                out.append(pd.read_csv(f).set_index("row_id").y_pred)
        return out
    f = pathlib.Path(f"results/oof/{name}.csv")
    if not f.exists():
        return out
    d = pd.read_csv(f)
    for s in sorted(d.seed.unique()):
        g = d[d.seed == s]
        if g.fold.nunique() >= 5:            # only complete seeds are comparable
            out.append(g.set_index("row_id").ddg_pred)
    return out


def main() -> None:
    truth, cx = load_truth()
    print(f"scored on {len(truth)} rows over {cx.nunique()} complexes; "
          f"per-complex needs >= {MIN_ROWS} rows\n")
    hdr = (f"{'model':<34}{'ens':>7}{'per seed':>10}{'spread':>8}{'n':>3}"
           f"{'neg':>5}{'bal':>7}{'bal-cal':>9}{'stab':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name in list(FOREST) + NETS:
        ss = seeds_of(name)
        if not ss:
            continue
        ens = pd.concat(ss, axis=1).mean(axis=1)
        e_r, neg = per_complex(ens, truth, cx)
        singles = [per_complex(s, truth, cx)[0] for s in ss]
        bal, rec = balanced(ens, truth)
        bal_c, _ = balanced(ens, truth, calibrate=True)
        rows.append((name, e_r, float(np.mean(singles)),
                     (max(singles) - min(singles)) if len(singles) > 1 else float("nan"),
                     len(ss), neg, bal, bal_c, rec[0]))
    for n, e, m, sp, k, neg, bal, balc, stab in sorted(rows, key=lambda r: -r[1]):
        print(f"{n:<34}{e:>+7.3f}{m:>+10.3f}{sp:>8.3f}{k:>3}{neg:>5}"
              f"{bal:>7.3f}{balc:>9.3f}{stab:>7.2f}")

    # the floor, which several of these metrics reward more than any model
    ref = pd.read_csv(paths.REPORTS / REFERENCE / "predictions.csv").set_index("row_id")
    cmn = ref.groupby("complex").y_true.transform("mean")
    f_r, _ = per_complex(cmn, truth, cx)
    f_bal, f_rec = balanced(cmn, truth)
    print("-" * len(hdr))
    print(f"{'[complex mean only] -- the floor':<34}{f_r:>+7.3f}{'':>10}{'':>8}{'':>3}{'':>5}"
          f"{f_bal:>7.3f}{'':>9}{f_rec[0]:>7.2f}")
    print("\nens      = seeds averaged, then scored once (what you would ship)")
    print("per seed = each seed scored separately, then averaged (what one run gives)")
    print("spread   = max - min over seeds; differences smaller than this are not real")
    print("neg      = complexes with a NEGATIVE within-complex correlation")
    print("bal      = balanced 3-class accuracy at the label edges (-0.5, +0.5)")
    print("bal-cal  = the same after quantile calibration, which cannot change the ranking")
    print("stab     = recall on STABILISING mutations, the class design actually cares about")


if __name__ == "__main__":
    main()
