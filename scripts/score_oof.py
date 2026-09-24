"""Score out-of-fold predictions the way the metric section of the report does.

Pooled Pearson is not reported here, and that is the point: predicting each complex's own
mean ddG -- a model that cannot see which mutation it is looking at -- scores +0.672 pooled
and +0.000 per complex. Any number that a complex-identity lookup can win is not measuring
what we are building.

What is reported:

``per-cx``   mean within-complex Pearson over complexes with >= 5 rows, averaged over seeds.
             The honest headline. ``+/-`` beside it is the spread across seeds, which on this
             dataset reaches 0.117 and is larger than most architectural effects.
``ens``      the same, on the mean of the seeds' predictions. Ensembling three seeds is free
             at inference and is how the model would actually be shipped.
``neg``      complexes whose within-complex correlation is NEGATIVE in the ensemble. A model
             can carry a good mean while being actively wrong on a handful of complexes, and
             that is a different failure from being uniformly mediocre.
``wc s|d``   within-complex AUC, destabilising over stabilising, averaged over complexes.
             Prevalence-free, threshold-free and identity-free: the complex-mean baseline
             scores 0.500 here by construction where pooled AUC hands it 0.931.
``multiMAE`` MAE on multi-point rows only. They are 21% of the data and the forest's weakest
             region, so it is tracked separately rather than being averaged away.

Runs are skipped unless every (fold, seed) cell asked for is present -- a partial table
scored as though it were complete gives a per-complex number that is nonsense.

    python scripts/score_oof.py twobranch_cg_rev twobranch_cg ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ROWS = 5        # a complex needs this many mutations before a correlation means anything
EDGES = (-0.5, 0.5)


def per_complex(d: pd.DataFrame, cx: pd.Series) -> tuple[float, int]:
    r, neg = [], 0
    for _, g in d.assign(cx=cx.reindex(d.row_id).to_numpy()).groupby("cx"):
        if len(g) < MIN_ROWS or g.ddg_true.std() == 0 or g.ddg_pred.std() == 0:
            continue
        v = float(np.corrcoef(g.ddg_true, g.ddg_pred)[0, 1])
        r.append(v)
        neg += v < 0
    return (float(np.mean(r)) if r else np.nan), neg


def wc_auc(d: pd.DataFrame, cx: pd.Series) -> float:
    """AUC(destabilising over stabilising) inside each complex, then averaged."""
    c = np.digitize(d.ddg_true.to_numpy(), EDGES)
    g = pd.DataFrame({"s": d.ddg_pred.to_numpy(), "c": c,
                      "cx": cx.reindex(d.row_id).to_numpy()})
    out = []
    for _, h in g.groupby("cx"):
        lo, hi = h.s[h.c == 0].to_numpy(), h.s[h.c == 2].to_numpy()
        if not len(lo) or not len(hi):
            continue
        out.append(float((hi[:, None] > lo[None, :]).mean()
                         + 0.5 * (hi[:, None] == lo[None, :]).mean()))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    rows = pd.read_parquet(
        Path("../baseline-protattba-repro-82952e/.perturb_local/perturb_rows.parquet"))
    cx = rows.set_index(rows.row_id.astype(str)).complex_key
    nmut = rows.set_index(rows.row_id.astype(str)).row_id.str.split("|").str[1].str.count(",") + 1

    names = sys.argv[1:] or [p.stem for p in sorted(Path("results/oof").glob("*.csv"))]
    out = []
    for n in names:
        f = Path("results/oof") / f"{n}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if "seed" not in d or d.fold.nunique() < 5:
            continue
        seeds = sorted(d.seed.unique())
        cells = d.groupby(["fold", "seed"]).size()
        if len(cells) < 5 * len(seeds):
            continue                      # partial: scoring it would be misleading
        per = [per_complex(d[d.seed == s], cx)[0] for s in seeds]
        ens = d.groupby("row_id", as_index=False).agg(
            ddg_true=("ddg_true", "first"), ddg_pred=("ddg_pred", "mean"))
        e, neg = per_complex(ens, cx)
        m = nmut.reindex(ens.row_id).to_numpy() > 1
        out.append({
            "run": n, "seeds": len(seeds),
            "per-cx": float(np.mean(per)), "+/-": float(np.ptp(per)),
            "ens": e, "neg": neg, "wc s|d": wc_auc(ens, cx),
            "multiMAE": float(np.abs(ens.ddg_true - ens.ddg_pred)[m].mean()),
        })
    t = pd.DataFrame(out).sort_values("ens", ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
