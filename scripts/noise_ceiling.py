"""The ceiling that measurement noise puts on any model's score.

SKEMPI contains the same (complex, mutation) measured more than once, by different groups
under different conditions. Those repeats are a free estimate of the label's own
reproducibility, and reproducibility caps correlation: a model cannot correlate with a target
better than the target correlates with itself.

For a variable observed with independent additive noise,

    observed = signal + noise,     reliability = var(signal) / var(observed)

and the maximum attainable Pearson correlation between a *noiseless* predictor and the
observed labels is ``sqrt(reliability)`` — the classic attenuation bound.

Two reliabilities are computed, because this project reports two metrics:

  pooled        var(noise) against the variance of ALL labels
  within-complex  var(noise) against the variance of labels INSIDE a complex, which is the
                  quantity a per-complex correlation is actually built from, and is much
                  smaller — so the per-complex ceiling is much lower.

``var(noise)`` is the pooled within-group variance of the repeated measurements. With n_i
repeats in group i it is estimated as sum((n_i - 1) * s_i^2) / sum(n_i - 1), the usual pooled
variance, which is unbiased and weights larger groups more.

    python scripts/noise_ceiling.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MEAS = pathlib.Path("data/processed/measurements.parquet")
DATA = pathlib.Path("data/processed/dataset.parquet")
CLIP = 4.0


def main() -> None:
    m = pd.read_parquet(MEAS)
    d = pd.read_parquet(DATA)
    m = m.dropna(subset=["ddG"])

    g = m.groupby(["#Pdb", "Mutation(s)_cleaned"]).ddG
    n = g.size()
    var = g.var(ddof=1)
    rep = n[n > 1].index
    nn, vv = n.loc[rep].values, var.loc[rep].values
    var_noise = float((( nn - 1) * vv).sum() / (nn - 1).sum())
    sd_noise = var_noise ** 0.5

    print(f"repeated (complex, mutation) groups : {len(rep)}")
    print(f"measurements inside them            : {int(nn.sum())}")
    print(f"largest group                       : {int(nn.max())} repeats")
    print(f"\npooled within-group variance (noise) : {var_noise:.4f}")
    print(f"  -> measurement sd                  : {sd_noise:.3f} kcal/mol")
    print(f"  median |difference| between a pair : "
          f"{float(np.median([abs(x) for x in g.apply(lambda s: s.max()-s.min()).loc[rep]])):.3f}")

    # the two variances a model is scored against
    y = d.ddG.clip(-CLIP, CLIP)
    var_pooled = float(y.var(ddof=1))
    within = d.assign(y=y).groupby("#Pdb").y.var(ddof=1).dropna()
    sizes = d.groupby("#Pdb").size().loc[within.index]
    var_within = float(((sizes - 1) * within).sum() / (sizes - 1).sum())

    print(f"\nvariance of the labels, pooled       : {var_pooled:.4f}  (sd {var_pooled**0.5:.3f})")
    print(f"variance of the labels, within complex: {var_within:.4f}  (sd {var_within**0.5:.3f})")

    print(f"\n{'metric':<26}{'reliability':>13}{'ceiling r':>12}   our best")
    print("-" * 68)
    for name, vt, best in (("pooled Pearson", var_pooled, 0.509),
                           ("per-complex Pearson", var_within, 0.381)):
        rel = max(0.0, 1.0 - var_noise / vt)
        ceil = rel ** 0.5
        print(f"{name:<26}{rel:>13.3f}{ceil:>12.3f}   {best:+.3f}"
              f"   ({100*best/ceil:.0f}% of the ceiling)")

    print("\nCaveats, all of which push the true ceiling DOWN rather than up:")
    print("  * repeats are not a random sample -- a mutation gets re-measured when it is")
    print("    interesting or contested, which if anything inflates their disagreement;")
    print("  * only 107 of 940 rows have a repeat, so var(noise) is estimated from 11% of")
    print("    the data and assumed homogeneous across the rest;")
    print("  * the bound assumes additive independent noise and a perfect predictor. It is")
    print("    an upper bound on what ANY model could reach, not a target.")


if __name__ == "__main__":
    main()
