"""Antibody-side vs antigen-side performance split, for a given model's OOF predictions.

Matches ERROR_ANALYSIS.md Part I §2's original methodology (which was scoped to
`cat128_reg2_l1` and is not being carried over silently to the current leader): pooled Spearman
between ensemble prediction and true label, restricted to rows whose `mut_side` is "antibody"
or "antigen", with a cluster (by-complex) bootstrap CI and a `top_cx` diagnostic -- the largest
fraction of that slice's rows coming from any single complex, so a correlation dominated by one
complex reads as a warning rather than a clean result.

    python scripts/mut_side_split.py --net struct_film_chem_kendall
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

N_BOOT = 2000
SEED = 0


def cluster_bootstrap_ci(y_pred, y_true, complex_id, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    complexes = complex_id.unique()
    by_cx = {c: np.flatnonzero(complex_id.values == c) for c in complexes}
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(complexes, size=len(complexes), replace=True)
        idx = np.concatenate([by_cx[c] for c in draw])
        if len(set(y_true[idx])) < 2 or len(set(y_pred[idx])) < 2:
            continue
        boots.append(spearmanr(y_pred[idx], y_true[idx])[0])
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    a = ap.parse_args()

    oof = pd.read_csv(f"results/oof/{a.net}.csv")
    ens = oof.groupby("row_id", as_index=False).agg(
        ddg_true=("ddg_true", "first"), ddg_pred=("ddg_pred", "mean"))

    data = pd.read_parquet(paths.DATASET)[["row_id", "#Pdb", "mut_side"]]
    d = ens.merge(data, on="row_id", validate="one_to_one")

    print(f"model: {a.net}  ({len(d)} rows total)\n")
    hdr = f"{'side':<10}{'n':>6}{'complexes':>11}{'top_cx':>9}{'rho':>9}{'95% CI':>20}"
    print(hdr); print("-" * len(hdr))
    for side in ("antibody", "antigen"):
        sub = d[d.mut_side == side]
        n = len(sub)
        n_cx = sub["#Pdb"].nunique()
        top_cx = float(sub["#Pdb"].value_counts().iloc[0] / n)
        rho = spearmanr(sub.ddg_pred, sub.ddg_true)[0]
        lo, hi = cluster_bootstrap_ci(sub.ddg_pred.to_numpy(), sub.ddg_true.to_numpy(),
                                      sub["#Pdb"])
        print(f"{side:<10}{n:>6}{n_cx:>11}{top_cx:>9.2f}{rho:>+9.3f}"
             f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}")


if __name__ == "__main__":
    main()
