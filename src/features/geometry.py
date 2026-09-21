"""Structure modality, part 1: interface geometry. No learned model, no GPU, no downloads.

This block is not a warm-up for the inverse-folding features -- it is a serious baseline. The
2025 AbAgym benchmark found that across six published ΔΔG methods, the best held-out antibody
result was Spearman 0.28, and that **relative solvent accessibility alone was competitive with
all of them**. Our own EDA says the same thing from the other direction: SKEMPI's interface
annotation sorts mean ΔΔG monotonically, +1.55 core through +0.09 surface.

Features per mutated position, summed over positions for multi-point mutations:

* ``rsasa_bound`` / ``rsasa_unbound`` -- relative solvent accessibility of the residue in the
  complex, and in its own chain group alone.
* ``d_rsasa`` -- the difference. This is burial *by the partner*, and it is the cleanest
  available definition of "is this residue at the interface", derived rather than taken from
  SKEMPI's own annotation so the two can be cross-checked.
* ``n_contacts`` -- heavy atoms within 5 A of any partner-side heavy atom.
* ``min_dist_partner`` -- closest approach to the partner side.
* ``n_neighbors`` -- residues within 8 A on the same side, i.e. local packing density.

SASA is Shrake-Rupley, implemented here rather than taken from a library so it runs on exactly
the coordinates the EDA verified, with no second parser to disagree with the first.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .. import paths
from ..structures import load_structures, parse_mutations

PROBE = 1.4          # water radius, angstroms
N_SPHERE = 96        # Shrake-Rupley sampling points; 96 is the usual accuracy/speed compromise
CONTACT_CUTOFF = 5.0
NEIGHBOUR_CUTOFF = 8.0

# Bondi van der Waals radii
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90, "P": 1.80}
VDW_DEFAULT = 1.70

# Theoretical maximum ASA per residue (Tien et al. 2013), for normalising to relative SASA
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "Q": 225, "E": 223, "G": 104,
    "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155,
    "T": 172, "W": 285, "Y": 263, "V": 174,
}


def _sphere(n: int = N_SPHERE) -> np.ndarray:
    """Golden-spiral points on the unit sphere -- deterministic, unlike random sampling."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


SPHERE = _sphere()


def shrake_rupley(coords: np.ndarray, radii: np.ndarray, owner: np.ndarray,
                  n_res: int) -> np.ndarray:
    """Per-residue solvent accessible surface area, in square angstroms.

    ``owner[i]`` gives the residue index of atom ``i``; the result has length ``n_res``.
    """
    from scipy.spatial import cKDTree

    if len(coords) == 0:
        return np.zeros(n_res)
    r = radii + PROBE
    tree = cKDTree(coords)
    out = np.zeros(n_res)
    max_r = float(r.max())

    for i in range(len(coords)):
        # neighbours that could possibly occlude atom i
        cand = tree.query_ball_point(coords[i], r[i] + max_r)
        cand = [j for j in cand if j != i]
        pts = coords[i] + r[i] * SPHERE
        if cand:
            d2 = ((pts[:, None, :] - coords[cand][None, :, :]) ** 2).sum(-1)
            accessible = (d2 >= (r[cand] ** 2)[None, :]).all(axis=1)
        else:
            accessible = np.ones(len(pts), bool)
        out[owner[i]] += 4.0 * np.pi * r[i] ** 2 * accessible.mean()
    return out


def _atoms_of(structure, chain_group: str):
    """Flatten a chain group into (coords, radii, owner, residue keys)."""
    coords, radii, owner, keys = [], [], [], []
    for c in chain_group:
        ch = structure.chains.get(c)
        if ch is None:
            continue
        for i, k in enumerate(ch.keys):
            xyz, el = ch.heavy[i], ch.heavy_elem[i]
            if len(xyz) == 0:
                continue
            ridx = len(keys)
            coords.append(xyz)
            radii.append([VDW.get(str(e).upper(), VDW_DEFAULT) for e in el])
            owner.append(np.full(len(xyz), ridx))
            keys.append((c, k))
    if not coords:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int), []
    return (np.vstack(coords).astype(np.float64),
            np.concatenate(radii), np.concatenate(owner).astype(int), keys)


def _complex_geometry(structure, ab_chains: str, ag_chains: str) -> dict:
    """rSASA bound/unbound, contacts and neighbour counts for every residue in the complex."""
    from scipy.spatial import cKDTree

    if not ab_chains or not ag_chains:            # 1DVF: antibody vs antibody
        ab_chains = ab_chains or structure.chains and "".join(sorted(structure.chains))[:1]
    sides = {"ab": ab_chains, "ag": ag_chains}

    both = (ab_chains or "") + (ag_chains or "")
    bc, br, bo, bkeys = _atoms_of(structure, both)
    bound = shrake_rupley(bc, br, bo, len(bkeys))
    bound_of = {k: bound[i] for i, k in enumerate(bkeys)}

    unbound_of, side_of = {}, {}
    atom_sets = {}
    for name, group in sides.items():
        if not group:
            continue
        c, r, o, keys = _atoms_of(structure, group)
        sasa = shrake_rupley(c, r, o, len(keys))
        for i, k in enumerate(keys):
            unbound_of[k] = sasa[i]
            side_of[k] = name
        atom_sets[name] = (c, o, keys)

    out = {}
    for name, (c, o, keys) in atom_sets.items():
        other = "ag" if name == "ab" else "ab"
        if other not in atom_sets:
            continue
        oc, _, _ = atom_sets[other]
        ptree = cKDTree(oc)
        stree = cKDTree(c)
        contacts = np.zeros(len(keys))
        mind = np.full(len(keys), np.inf)
        for i in range(len(c)):
            near = ptree.query_ball_point(c[i], CONTACT_CUTOFF)
            contacts[o[i]] += len(near)
            d, _ = ptree.query(c[i])
            mind[o[i]] = min(mind[o[i]], d)
        # residue-level neighbour density on the same side, by CA-ish centroid
        cent = np.zeros((len(keys), 3))
        cnt = np.zeros(len(keys))
        np.add.at(cent, o, c)
        np.add.at(cnt, o, 1)
        cent /= np.maximum(cnt, 1)[:, None]
        ctree = cKDTree(cent)
        nbr = np.array([len(ctree.query_ball_point(p, NEIGHBOUR_CUTOFF)) - 1 for p in cent])

        for i, k in enumerate(keys):
            out[k] = {
                "sasa_bound": bound_of.get(k, np.nan),
                "sasa_unbound": unbound_of.get(k, np.nan),
                "n_contacts": contacts[i],
                "min_dist_partner": mind[i] if np.isfinite(mind[i]) else 99.0,
                "n_neighbors": nbr[i],
            }
    return out


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
        fallback_ab = abc or key.split("_")[1]
        fallback_ag = agc or key.split("_")[2]
        geo[key] = _complex_geometry(structures[pdb], fallback_ab, fallback_ag)
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(cx)} complexes  {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for row_id, key, pdb, mutations in zip(df["row_id"], df["#Pdb"], df["pdb"], df["mutations"]):
        muts = parse_mutations(mutations)
        acc = {"rsasa_bound": 0.0, "rsasa_unbound": 0.0, "d_rsasa": 0.0,
               "n_contacts": 0.0, "n_neighbors": 0.0}
        mind = 99.0
        for m in muts:
            g = geo[key].get((m.chain, m.key))
            if g is None:
                continue
            norm = MAX_ASA.get(m.wt, 200.0)
            rb = g["sasa_bound"] / norm
            ru = g["sasa_unbound"] / norm
            acc["rsasa_bound"] += rb
            acc["rsasa_unbound"] += ru
            acc["d_rsasa"] += ru - rb
            acc["n_contacts"] += g["n_contacts"]
            acc["n_neighbors"] += g["n_neighbors"]
            mind = min(mind, g["min_dist_partner"])
        rows.append({"row_id": row_id, **acc, "min_dist_partner": mind,
                     "is_interface": float(acc["d_rsasa"] > 0.05)})

    block = pd.DataFrame(rows).set_index("row_id").astype(np.float32)
    out = paths.FEATURES / "geom.parquet"
    block.to_parquet(out)
    if verbose:
        print(f"wrote {out.relative_to(paths.ROOT)}  {block.shape}  "
              f"in {time.perf_counter()-t0:.1f}s")
    return block


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()


if __name__ == "__main__":
    main()
