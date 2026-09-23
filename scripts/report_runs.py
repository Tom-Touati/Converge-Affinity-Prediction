"""One table of the metrics a run is actually judged on, for any set of runs.

Reporting per-complex Pearson alone has repeatedly been too thin. Pooled Pearson disagrees
with it (predicting each complex's mean scores +0.672 pooled and 0.000 per complex, better
than any model here on the first and useless on the second), RMSE is dominated by the same
between-complex structure, and neither says whether a model gets the DIRECTION of a
mutation right -- which is the question a designer asks.

Every run is scored against ONE truth, taken from a reference run's predictions file, so
models trained on differently clipped labels stay comparable. That caught a real problem
earlier: the forest was being compared at RMSE 1.532 against nets at 1.5-1.6 while being
scored on unclipped labels; on the common truth it is 1.335.

    python scripts/report_runs.py perturb_st64_chem perturb_st64_xattn_rev
    python scripts/report_runs.py --all-st            # every sitetok run plus the baselines
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

#: the run whose y_true every model is scored against
REFERENCE = "perturb_v3_base"
BASELINES = ["E0a_rf_handcrafted_seed0", "perturb_mlp_delta_chem_reg"]
MIN_ROWS = 5          # a complex needs this many to carry a rank


def metrics(d: pd.DataFrame) -> dict:
    e = d.y_pred - d.y_true
    per_r, per_s, neg = [], [], 0
    for _, g in d.groupby("complex"):
        if len(g) < MIN_ROWS or g.y_true.std() == 0 or g.y_pred.std() == 0:
            continue
        r = float(np.corrcoef(g.y_pred, g.y_true)[0, 1])
        per_r.append(r)
        per_s.append(float(g.y_pred.corr(g.y_true, method="spearman")))
        neg += r < 0

    # Direction, on the mutations big enough for the sign to be meaningful. A 0.1 kcal/mol
    # error of sign is noise; a 1.5 one is a wrong answer.
    big = d[d.y_true.abs() > 0.5]
    sign = float((np.sign(big.y_pred) == np.sign(big.y_true)).mean()) if len(big) else np.nan
    # the same, balanced, so a dataset that is 85% destabilising cannot score well by
    # always saying "destabilising"
    bal = np.nan
    if len(big):
        pos, negm = big[big.y_true > 0], big[big.y_true < 0]
        if len(pos) and len(negm):
            bal = 0.5 * (float((big.loc[pos.index].y_pred > 0).mean())
                         + float((big.loc[negm.index].y_pred < 0).mean()))

    # Concordance WITHIN a complex: of all pairs of mutations on the same complex, how
    # often is the ordering right. This is the design question stated as a metric.
    conc, npairs = [], 0
    for _, g in d.groupby("complex"):
        if len(g) < 2:
            continue
        t, p = g.y_true.values, g.y_pred.values
        dt = t[:, None] - t[None, :]
        dp = p[:, None] - p[None, :]
        iu = np.triu_indices(len(g), 1)
        keep = np.abs(dt[iu]) > 0.5           # pairs too close to call are excluded
        if keep.sum():
            conc.append(float((np.sign(dt[iu][keep]) == np.sign(dp[iu][keep])).mean()))
            npairs += int(keep.sum())

    return {
        "n": len(d), "n_cx": len(per_r),
        "per_cx_r": np.mean(per_r) if per_r else np.nan,
        "per_cx_rho": np.mean(per_s) if per_s else np.nan,
        "cx_neg": neg,
        "pooled_r": float(np.corrcoef(d.y_pred, d.y_true)[0, 1]),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "bias": float(e.mean()),
        "rmse_deb": float(np.sqrt(((e - e.mean()) ** 2).mean())),
        "sign": sign, "sign_bal": bal,
        "conc": np.mean(conc) if conc else np.nan, "pairs": npairs,
    }


def load(run: str, truth: pd.Series) -> pd.DataFrame | None:
    p = paths.REPORTS / run / "predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).dropna(subset=["y_pred"]).set_index("row_id").sort_index()
    d = d.loc[d.index.intersection(truth.index)].copy()
    d["y_true"] = truth.loc[d.index]
    return d if len(d) > 2 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all-st", action="store_true", help="every sitetok run, plus baselines")
    a = ap.parse_args()

    ref = pd.read_csv(paths.REPORTS / REFERENCE / "predictions.csv").set_index("row_id")
    truth = ref.sort_index().y_true

    runs = list(a.runs)
    if a.all_st or not runs:
        runs = sorted(p.name for p in paths.REPORTS.iterdir()
                      if p.is_dir() and ("st64" in p.name or "st_" in p.name))
    runs = BASELINES + [r for r in runs if r not in BASELINES]

    rows = []
    for r in runs:
        d = load(r, truth)
        if d is None:
            continue
        rows.append({"run": r.replace("perturb_", ""), **metrics(d)})
    if not rows:
        raise SystemExit("nothing to report")

    t = pd.DataFrame(rows).sort_values("per_cx_r", ascending=False)
    # the floor: predicting each complex's own mean, which is pure complex identity
    cm = ref.sort_index().assign(y_pred=lambda x: x.groupby("complex").y_true.transform("mean"))
    base = metrics(cm.reset_index())

    hdr = (f"{'run':<26}{'per-cx r':>9}{'per-cx rho':>11}{'neg':>5}{'pooled':>8}"
           f"{'rmse':>7}{'bias':>7}{'deb':>6}{'sign':>6}{'bal':>6}{'conc':>6}")
    print(f"\nscored on {len(truth)} rows, one common truth; "
          f"per-complex needs >= {MIN_ROWS} rows\n")
    print(hdr)
    print("-" * len(hdr))
    for r in t.itertuples():
        print(f"{r.run:<26}{r.per_cx_r:>+9.3f}{r.per_cx_rho:>+11.3f}{r.cx_neg:>5}"
              f"{r.pooled_r:>+8.3f}{r.rmse:>7.3f}{r.bias:>+7.3f}{r.rmse_deb:>6.3f}"
              f"{r.sign:>6.2f}{r.sign_bal:>6.2f}{r.conc:>6.2f}")
    print("-" * len(hdr))
    print(f"{'[complex mean only]':<26}{0.0:>+9.3f}{0.0:>+11.3f}{'-':>5}"
          f"{base['pooled_r']:>+8.3f}{base['rmse']:>7.3f}{base['bias']:>+7.3f}"
          f"{base['rmse_deb']:>6.3f}{base['sign']:>6.2f}{base['sign_bal']:>6.2f}"
          f"{base['conc']:>6.2f}")
    print("\nneg = complexes with a NEGATIVE within-complex correlation")
    print("sign/bal = direction correct on |ddG| > 0.5, raw and class-balanced")
    print("conc = within-complex pairwise ordering correct, pairs > 0.5 apart "
          f"({t.pairs.iloc[0]:,} pairs)")


if __name__ == "__main__":
    main()
