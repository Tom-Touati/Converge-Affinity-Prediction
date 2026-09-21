"""Structural-similarity difficulty tiers, the way the SKEMPI literature defines them.

A published categorisation of SKEMPI targets sorts each PDB by how much structurally similar
training data it has: **easy** when 50 or more training measurements come from complexes with
TM-score > 0.8 to it, **medium** for 1-50, **hard** for none. That is a sharper question than
"how many mutations does this complex have", because it asks what the model could have learned
*transferably*, and it is the axis on which a per-complex Spearman is most easily flattered.

Two decisions worth stating, since both change the numbers:

* **Antigen chains are aligned, not whole complexes.** Antibody frameworks are structurally
  near-identical across unrelated targets -- it is why the homology clustering in splits.py
  uses a 90% identity threshold on the antibody side and 30% on the antigen side. Aligning
  whole complexes would score almost every pair above 0.8 on framework alone and collapse the
  tiers into one. For the antibody-vs-antibody complex (1DVF) the declared side is used.
* **Tiers are computed per fold.** "Training set" means the other three folds under the frozen
  split in data/folds.csv, so a complex can be easy in one fold and hard in another. Computing
  it once over all 997 rows would count a complex's own neighbours as training data for itself.

TM-score is computed with TM-align via `tmtools`, normalised by the shorter chain, on CA
coordinates.

    python -m src.tmscore          # writes data/tm_tiers.csv and prints the breakdown
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths, splits
from .structures import BACKBONE, load_structures

CA = BACKBONE.index("CA")
TM_THRESHOLD = 0.8
EASY_MIN = 50


def antigen_ca(struct, chain_group: str):
    """CA coordinates and one-letter sequence for a chain group, NaN residues dropped."""
    xyz, seq = [], []
    for c in chain_group:
        ch = struct.chains.get(c)
        if ch is None:
            continue
        bb = np.asarray(ch.backbone)[:, CA, :]
        ok = np.isfinite(bb).all(axis=1)
        xyz.append(bb[ok])
        seq.append("".join(a for a, keep in zip(ch.seq, ok) if keep))
    if not xyz:
        return None, ""
    return np.concatenate(xyz).astype(np.float64), "".join(seq)


def tm_matrix(verbose: bool = True) -> pd.DataFrame:
    """Symmetric TM-score matrix over the complexes, normalised by the shorter chain."""
    from tmtools import tm_align

    d = splits.load()
    cx = d.drop_duplicates("#Pdb").set_index("#Pdb")
    structures = load_structures(cx["pdb"].unique())

    coords = {}
    for k, r in cx.iterrows():
        group = r["ag_chains"] or r["side2"]      # 1DVF: antibody vs antibody
        xyz, seq = antigen_ca(structures[r["pdb"]], group)
        if xyz is not None and len(xyz) >= 5:
            coords[k] = (xyz, seq)
    keys = sorted(coords)
    if verbose:
        print(f"aligning antigen chains of {len(keys)} complexes "
              f"({len(keys) * (len(keys) - 1) // 2} pairs)", flush=True)

    M = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
    for i, a in enumerate(keys):
        xa, sa = coords[a]
        for b in keys[i + 1:]:
            xb, sb = coords[b]
            try:
                r = tm_align(xa, xb, sa, sb)
                # normalised by the shorter chain: the conservative choice, since normalising
                # by the longer one would let a small domain hide inside a large antigen.
                v = float(max(r.tm_norm_chain1, r.tm_norm_chain2))
            except Exception:
                v = 0.0
            M.loc[a, b] = M.loc[b, a] = v
    return M


MATRIX_PATH = paths.DATA / "tm_matrix.csv"


def cached_matrix(verbose: bool = True) -> pd.DataFrame:
    """The pairwise matrix, computed once. 1,431 alignments is ~10 minutes."""
    if MATRIX_PATH.exists():
        return pd.read_csv(MATRIX_PATH, index_col=0)
    M = tm_matrix(verbose=verbose)
    paths.DATA.mkdir(parents=True, exist_ok=True)
    M.to_csv(MATRIX_PATH)
    return M


def tiers(M: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Per (fold, complex): training rows above the TM threshold, and the resulting tier."""
    d = splits.load()
    M = cached_matrix(verbose=verbose) if M is None else M
    rows_per_cx = d.groupby("#Pdb").size()

    out = []
    for f in sorted(d.fold.unique()):
        train_cx = d.loc[d.fold != f, "#Pdb"].unique()
        for k in d.loc[d.fold == f, "#Pdb"].unique():
            if k not in M.index:
                out.append({"fold": f, "complex": k, "n_similar_train": 0, "tier": "hard"})
                continue
            sim = [c for c in train_cx if c in M.index and M.loc[k, c] > TM_THRESHOLD]
            n = int(rows_per_cx.reindex(sim).fillna(0).sum())
            tier = "hard" if n == 0 else ("easy" if n >= EASY_MIN else "medium")
            out.append({"fold": f, "complex": k, "n_similar_train": n, "tier": tier})
    t = pd.DataFrame(out)
    if verbose:
        print(t.groupby("tier").agg(complexes=("complex", "nunique"),
                                    rows=("complex", "size")).to_string())
    return t


def main():
    t = tiers()
    paths.DATA.mkdir(parents=True, exist_ok=True)
    t.to_csv(paths.DATA / "tm_tiers.csv", index=False)
    print(f"wrote {paths.DATA / 'tm_tiers.csv'}")


if __name__ == "__main__":
    main()
