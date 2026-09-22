"""Precompute one crop per row and ship it, so the trainer needs no PDB files.

The crop is a function of the wild-type backbone alone, so it is the same every epoch and for
both branches. Computing it once here means the VM never parses a structure, never needs
``data/PDBs``, and cannot accidentally derive a different crop for MUT than for ITW.

Saved as a single ``.npz`` keyed by row id. Per row:

    ab_idx, ag_idx        positions kept, as indices into the concatenated side sequence
    ab_chain, ag_chain    chain type per kept token (heavy / light / antigen)
    ab_res,  ag_res       ORIGINAL per-chain residue index, so RoPE is not renumbered by the crop
    ab_site, ag_site      1.0 at a mutated residue
    dist                  the cropped (n_ab, n_ag) Ca-Ca submatrix
    {ab,ag}_{wt,mt}_aa    the substitution at each flagged site of that side, in crop order

Run: ``python -m src.perturb.build_crops [--r-iface 12] [--r-site 10]``
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from src import paths
from src.perturb import data as D
from src.perturb.crop import (
    R_IFACE_DEFAULT,
    R_SITE_DEFAULT,
    build_crop,
    chain_layout,
)

OUT = paths.FEATURES / "perturb_crops.npz"


def build(r_iface: float = R_IFACE_DEFAULT, r_site: float = R_SITE_DEFAULT,
          verbose: bool = True, out: Path | None = None,
          prefer_chain_id: bool = True) -> str:
    from src.structures import load_structures

    rows = D.load_rows()
    meta = pd.read_parquet(paths.DATASET).set_index("row_id")
    structures = load_structures({r.pdb for r in rows})

    store: dict[str, np.ndarray] = {}
    sizes, disconnected = [], 0

    for r in rows:
        st = structures[r.pdb]
        ab = meta.loc[r.row_id, "ab_chains"] or meta.loc[r.row_id, "side1"]
        ag = meta.loc[r.row_id, "ag_chains"] or meta.loc[r.row_id, "side2"]
        lens = {c: len(st.chains[c].seq) for c in st.chains}
        t_ab, i_ab, _ = chain_layout(ab, lens, "ab", prefer_chain_id=prefer_chain_id)
        t_ag, i_ag, _ = chain_layout(ag, lens, "ag")

        dist = np.load(D.DIST_DIR / f"{r.complex_key}.npy")
        c = build_crop(dist, r.sites_ab, r.sites_ag, len(t_ab), len(t_ag),
                       t_ab, t_ag, i_ab, i_ag, r_iface=r_iface, r_site=r_site)

        d = np.full((len(t_ab), len(t_ag)), 99.0, np.float32)
        d[: dist.shape[0], : dist.shape[1]] = dist[: len(t_ab), : len(t_ag)]
        sub = d[np.ix_(c.ab_idx, c.ag_idx)] if c.ab_idx.size and c.ag_idx.size else \
            np.zeros((len(c.ab_idx), len(c.ag_idx)), np.float32)

        k = r.row_id
        store[f"{k}|ab_idx"] = c.ab_idx.astype(np.int32)
        store[f"{k}|ag_idx"] = c.ag_idx.astype(np.int32)
        store[f"{k}|ab_chain"] = c.ab_chain_type.astype(np.int8)
        store[f"{k}|ag_chain"] = c.ag_chain_type.astype(np.int8)
        store[f"{k}|ab_res"] = c.ab_res_index.astype(np.int32)
        store[f"{k}|ag_res"] = c.ag_res_index.astype(np.int32)
        store[f"{k}|ab_site"] = c.ab_is_site.astype(np.float32)
        store[f"{k}|ag_site"] = c.ag_is_site.astype(np.float32)
        store[f"{k}|dist"] = sub.astype(np.float16)

        # Substitution letters in CROP order: the j-th flagged token of a side gets the
        # letters of the mutation that landed there. np.flatnonzero returns the crop's sites
        # by ascending position, which is not the order the mutation string lists them, so
        # the position has to be looked up rather than counted. Indexing the row's global
        # wt_aa by a count within one side -- what the trainers used to do -- reads another
        # side's residue on the 82 rows (8.7%) whose orders disagree.
        for side, idx, flags, sites, wt_s, mt_s in (
                ("ab", c.ab_idx, c.ab_is_site, r.sites_ab, r.wt_ab, r.mt_ab),
                ("ag", c.ag_idx, c.ag_is_site, r.sites_ag, r.wt_ag, r.mt_ag)):
            at = {p: i for i, p in enumerate(sites)}
            order = [at[int(idx[j])] for j in np.flatnonzero(flags)]
            store[f"{k}|{side}_wt_aa"] = np.array([wt_s[o] for o in order], dtype="U1")
            store[f"{k}|{side}_mt_aa"] = np.array([mt_s[o] for o in order], dtype="U1")
        sizes.append(c.size)
        disconnected += c.disconnected

    dest = Path(out).resolve() if out else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **store)

    s = np.array(sizes)
    if verbose:
        mb = dest.stat().st_size / 1e6
        shown = dest.relative_to(paths.ROOT) if dest.is_relative_to(paths.ROOT) else dest
        print(f"wrote {shown} ({mb:.1f} MB) for {len(rows)} rows")
        print(f"  crop size: median {np.median(s):.0f}, p10 {np.percentile(s,10):.0f}, "
              f"p90 {np.percentile(s,90):.0f}, max {s.max()}")
        print(f"  inside the expected 40-120 band: {100*((s>=40)&(s<=120)).mean():.0f}%")
        print(f"  disconnected (site does not touch the interface): {disconnected} "
              f"({100*disconnected/len(s):.0f}%)")
        print(f"  radii: r_iface {r_iface} A, r_site {r_site} A")
        print(f"  heavy/light from the chain id where present: {prefer_chain_id}")
    return str(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r-iface", type=float, default=R_IFACE_DEFAULT)
    ap.add_argument("--r-site", type=float, default=R_SITE_DEFAULT)
    ap.add_argument("--out", default=None, help="write here instead of the default path")
    ap.add_argument("--length-only", action="store_true",
                    help="assign heavy/light by length alone, ignoring an H/L chain id")
    a = ap.parse_args()
    build(a.r_iface, a.r_site, out=a.out, prefer_chain_id=not a.length_only)


if __name__ == "__main__":
    main()
