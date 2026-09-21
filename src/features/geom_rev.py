"""Geometry that reverses: a site half that is symmetric and a residue half that swaps.

The problem this solves. ``geometry.py`` computes every feature from the wild-type residue's
heavy atoms -- SASA over the full side chain, contacts over the full side chain, and even the
SASA normalisation keyed on the wild-type identity. All of it is therefore side-chain dependent,
and under reverse-mutation augmentation none of it can flip, because the mutant structure does
not exist. A reversed row ends up carrying identical geometry with an opposite label.

That is not a small issue: ``geom:n_contacts`` is the single strongest feature in the project
(|rho| 0.449), and augmentation makes every one of its values map to both +ddG and -ddG. Measured
cost: per-complex Spearman 0.487 -> 0.416.

**The decomposition.** ``n_contacts`` is really measuring two things multiplied together -- how
crowded this position is, and how much side chain is sitting in it. The first is a property of
the *site* and is identical whichever residue occupies it. The second is a property of the
*residue* and swaps on reversal. Separating them gives a block where every column has a defined
behaviour under reversal:

* **Site columns** (``bb_*``) are computed from backbone N, CA, C, O plus a *virtual* C-beta
  derived from the backbone. No side-chain atoms are read at all, so these are genuinely
  identical for any residue at that position. They are unchanged by reversal -- exactly, not as
  an approximation.
* **Residue columns** (``wt_*`` / ``mut_*``) depend only on amino-acid identity and exchange
  values on reversal.
* **Interaction columns** are products of the two. They reverse correctly *because* the site half
  is symmetric and the identity half swaps.

What we give up: the true solvent accessibility of the actual wild-type side chain, replaced by
residue volume scaled by how buried the site is. That is a worse description of the forward
direction. It is a *reversible* one, which the forward-only version is not, and the experiment is
whether the trade pays.

The virtual C-beta uses the standard construction from N, CA and C, the same one ProteinMPNN and
AlphaFold use, so glycine positions get a C-beta too and the block has no identity-dependent
gaps.

    python -m src.features.geom_rev
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .. import paths
from ..structures import load_structures, parse_mutations
from .chem import HYDROPATHY, VOLUME
from .geometry import MAX_ASA, PROBE, VDW, VDW_DEFAULT, shrake_rupley

CB_CONTACT = 8.0      # partner atoms within this of the virtual C-beta
NEIGHBOUR = 10.0      # same-side CA density


def virtual_cb(bb: np.ndarray) -> np.ndarray:
    """C-beta position implied by backbone N, CA, C. Defined even for glycine."""
    n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca


def _backbone_atoms(structure, chain_group: str):
    """Backbone N, CA, C, O plus virtual C-beta for every residue of a chain group."""
    coords, radii, owner, keys, cb = [], [], [], [], []
    for ch_id in chain_group:
        ch = structure.chains.get(ch_id)
        if ch is None:
            continue
        cbs = virtual_cb(ch.backbone)
        for i, k in enumerate(ch.keys):
            bb = ch.backbone[i]
            atoms = [a for a in bb if np.isfinite(a).all()]
            if not atoms:
                continue
            ridx = len(keys)
            pts = np.vstack(atoms + [cbs[i]]) if np.isfinite(cbs[i]).all() else np.vstack(atoms)
            coords.append(pts)
            radii.append([VDW["N"], VDW["C"], VDW["C"], VDW["O"]][:len(atoms)]
                         + ([VDW["C"]] if len(pts) > len(atoms) else []))
            owner.append(np.full(len(pts), ridx))
            keys.append((ch_id, k))
            cb.append(cbs[i] if np.isfinite(cbs[i]).all() else bb[1])
    if not keys:
        return None
    return (np.vstack(coords), np.concatenate([np.asarray(r, float) for r in radii]),
            np.concatenate(owner), keys, np.vstack(cb))


def _site_geometry(structure, ab_chains: str, ag_chains: str) -> dict:
    """Per-residue site descriptors that do not read a single side-chain atom."""
    out = {}
    for side, partner in ((ab_chains, ag_chains), (ag_chains, ab_chains)):
        me = _backbone_atoms(structure, side)
        them = _backbone_atoms(structure, partner)
        if me is None:
            continue
        coords, radii, owner, keys, cb = me
        n_res = len(keys)

        sasa_unbound = shrake_rupley(coords, radii, owner, n_res)
        if them is None:
            sasa_bound, pcoords = sasa_unbound, None
        else:
            pc, pr, po, pkeys, _ = them
            all_c = np.vstack([coords, pc])
            all_r = np.concatenate([radii, pr])
            all_o = np.concatenate([owner, po + n_res])
            sasa_bound = shrake_rupley(all_c, all_r, all_o, n_res + len(pkeys))[:n_res]
            pcoords = pc

        for i, k in enumerate(keys):
            if pcoords is not None:
                d = np.linalg.norm(pcoords - cb[i], axis=1)
                n_contacts = float((d <= CB_CONTACT).sum())
                min_dist = float(d.min())
            else:
                n_contacts, min_dist = 0.0, 99.0
            near = float((np.linalg.norm(cb - cb[i], axis=1) <= NEIGHBOUR).sum() - 1)
            # normalise by the backbone+CB reference, not by a residue-specific max ASA
            out[k] = {
                "bb_sasa_bound": sasa_bound[i] / 110.0,
                "bb_sasa_unbound": sasa_unbound[i] / 110.0,
                "bb_d_sasa": (sasa_unbound[i] - sasa_bound[i]) / 110.0,
                "bb_n_contacts": n_contacts,
                "bb_min_dist": min_dist,
                "bb_n_neighbors": near,
            }
    return out


SITE = ("bb_sasa_bound", "bb_sasa_unbound", "bb_d_sasa",
        "bb_n_contacts", "bb_min_dist", "bb_n_neighbors")


def build(df: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    paths.ensure_dirs()
    if df is None:
        df = pd.read_parquet(paths.DATASET)
    structures = load_structures(df["pdb"].unique())

    t0 = time.perf_counter()
    geo = {}
    cx = df.drop_duplicates("#Pdb")
    for n, (key, pdb, abc, agc) in enumerate(
        zip(cx["#Pdb"], cx["pdb"], cx["ab_chains"], cx["ag_chains"])
    ):
        geo[key] = _site_geometry(structures[pdb], abc or key.split("_")[1],
                                  agc or key.split("_")[2])
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(cx)} complexes  {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for row_id, key, mutations in zip(df["row_id"], df["#Pdb"], df["mutations"]):
        acc = {k: 0.0 for k in SITE}
        acc["bb_min_dist"] = 99.0
        wt_v = mut_v = wt_h = mut_h = wt_a = mut_a = 0.0
        wt_bur = mut_bur = wt_con = mut_con = 0.0
        for m in parse_mutations(mutations):
            g = geo[key].get((m.chain, m.key))
            if g is None:
                continue
            for k in SITE:
                if k == "bb_min_dist":
                    acc[k] = min(acc[k], g[k])
                else:
                    acc[k] += g[k]
            wt_v += VOLUME[m.wt]; mut_v += VOLUME[m.mut]
            wt_h += HYDROPATHY[m.wt]; mut_h += HYDROPATHY[m.mut]
            wt_a += MAX_ASA.get(m.wt, 200.0); mut_a += MAX_ASA.get(m.mut, 200.0)
            # interaction: how much side chain the partner is burying, per identity
            wt_bur += VOLUME[m.wt] * g["bb_d_sasa"]
            mut_bur += VOLUME[m.mut] * g["bb_d_sasa"]
            wt_con += VOLUME[m.wt] / (g["bb_min_dist"] + 1.0)
            mut_con += VOLUME[m.mut] / (g["bb_min_dist"] + 1.0)
        rows.append({
            "row_id": row_id, **acc,
            "wt_vol": wt_v, "mut_vol": mut_v,
            "wt_hyd": wt_h, "mut_hyd": mut_h,
            "wt_maxasa": wt_a, "mut_maxasa": mut_a,
            "wt_buried_vol": wt_bur, "mut_buried_vol": mut_bur,
            "wt_contact_est": wt_con, "mut_contact_est": mut_con,
        })

    block = pd.DataFrame(rows).set_index("row_id").astype(np.float32)
    out = paths.FEATURES / "geomrev.parquet"
    block.to_parquet(out)
    if verbose:
        print(f"wrote {out.relative_to(paths.ROOT)}  {block.shape}  "
              f"in {time.perf_counter()-t0:.1f}s")
        print(f"  site columns (symmetric under reversal): {len(SITE)}")
        print(f"  residue/interaction columns (swap on reversal): {block.shape[1]-len(SITE)}")
    return block


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()


if __name__ == "__main__":
    main()
