"""The crop: which residues a branch actually sees, computed once from the wild-type structure.

Section 1's token set is not the whole complex. It is

    interface residues            Ca-Ca < r_iface (12 A) to any residue of the partner side
  U mutation-site neighbourhood   Ca-Ca < r_site  (10 A) to any mutated residue, both sides
  U the mutated residues +/- 2 in sequence

Two properties matter more than the radii:

* **It is identical for ITW and MUT.** The crop comes from the wild-type backbone, which both
  branches share, so the two branches cannot differ in what they look at. Required test (c)
  asserts this rather than trusting it.
* **A non-interface mutation yields two disconnected regions, and that is intended.** The site
  neighbourhood and the interface set need not touch. Whether the site can still reach the
  interface through the distance-biased attention is a question the error analysis asks, not
  something the crop should paper over.

Each token carries its ``chain_type`` (heavy / light / antigen), its ``chain_id``, and its
**original per-chain residue index**, not its position in the concatenated array. RoPE uses
the original index, so cropping does not silently renumber the protein.

Heavy and light are told apart by length and order within the antibody group: the longer
chain is the heavy chain. That is a heuristic and is logged as one -- a Kabat/Chothia
annotation would be better and is not available here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: chain_type codes, used as indices into the 3 x 64 chain embedding
HEAVY, LIGHT, ANTIGEN = 0, 1, 2
CHAIN_TYPES = ("heavy", "light", "antigen")

R_IFACE_DEFAULT = 12.0
R_SITE_DEFAULT = 10.0
SEQ_WINDOW = 2


@dataclass
class Crop:
    """Indices into the concatenated side arrays, plus per-token metadata."""

    ab_idx: np.ndarray        # positions kept on the antibody side
    ag_idx: np.ndarray        # positions kept on the antigen side
    ab_chain_type: np.ndarray
    ag_chain_type: np.ndarray
    ab_res_index: np.ndarray  # original per-chain residue index
    ag_res_index: np.ndarray
    ab_is_site: np.ndarray
    ag_is_site: np.ndarray
    disconnected: bool        # site neighbourhood does not touch the interface set

    @property
    def size(self) -> int:
        return len(self.ab_idx) + len(self.ag_idx)


def chain_layout(chain_ids: str, lengths: dict[str, int], side: str) -> tuple:
    """(chain_type per position, original residue index per position) for one side.

    ``lengths`` maps chain id to length, in the order the sequences were concatenated.
    """
    types, res_index, chain_of = [], [], []
    if side == "ab":
        present = [c for c in chain_ids if c in lengths]
        # the longer chain is taken to be heavy; a heuristic, logged as one
        heavy = max(present, key=lambda c: lengths[c]) if present else None
        for c in present:
            n = lengths[c]
            types.extend([HEAVY if c == heavy else LIGHT] * n)
            res_index.extend(range(n))
            chain_of.extend([c] * n)
    else:
        for c in (c for c in chain_ids if c in lengths):
            n = lengths[c]
            types.extend([ANTIGEN] * n)
            res_index.extend(range(n))
            chain_of.extend([c] * n)
    return (np.asarray(types, np.int64), np.asarray(res_index, np.int64), chain_of)


def build_crop(dist: np.ndarray, sites_ab, sites_ag, n_ab: int, n_ag: int,
               ab_types: np.ndarray, ag_types: np.ndarray,
               ab_res: np.ndarray, ag_res: np.ndarray,
               r_iface: float = R_IFACE_DEFAULT, r_site: float = R_SITE_DEFAULT) -> Crop:
    """``dist`` is the (n_ab, n_ag) Ca-Ca matrix of the wild-type complex."""
    d = dist[:n_ab, :n_ag] if dist.size else np.full((n_ab, n_ag), 99.0, np.float32)
    if d.shape != (n_ab, n_ag):
        pad = np.full((n_ab, n_ag), 99.0, np.float32)
        pad[: d.shape[0], : d.shape[1]] = d
        d = pad

    iface_ab = d.min(axis=1) < r_iface
    iface_ag = d.min(axis=0) < r_iface

    site_ab = np.zeros(n_ab, bool)
    site_ag = np.zeros(n_ag, bool)
    for p in sites_ab:
        if 0 <= p < n_ab:
            site_ab[p] = True
    for p in sites_ag:
        if 0 <= p < n_ag:
            site_ag[p] = True

    # neighbourhood of the mutated residues, on BOTH sides
    near_ab = np.zeros(n_ab, bool)
    near_ag = np.zeros(n_ag, bool)
    if site_ab.any():
        # same side: sequence window only, since there is no intra-side distance matrix here
        for p in np.flatnonzero(site_ab):
            near_ab[max(0, p - SEQ_WINDOW): p + SEQ_WINDOW + 1] = True
        near_ag |= (d[site_ab].min(axis=0) < r_site)
    if site_ag.any():
        for p in np.flatnonzero(site_ag):
            near_ag[max(0, p - SEQ_WINDOW): p + SEQ_WINDOW + 1] = True
        near_ab |= (d[:, site_ag].min(axis=1) < r_site)

    keep_ab = iface_ab | near_ab | site_ab
    keep_ag = iface_ag | near_ag | site_ag
    # never return an empty side: attention needs at least one key
    if not keep_ab.any():
        keep_ab[: min(n_ab, 1)] = True
    if not keep_ag.any():
        keep_ag[: min(n_ag, 1)] = True

    site_touches_iface = bool((site_ab & iface_ab).any() or (site_ag & iface_ag).any()
                              or (near_ab & iface_ab).any() or (near_ag & iface_ag).any())

    ab_idx = np.flatnonzero(keep_ab)
    ag_idx = np.flatnonzero(keep_ag)
    return Crop(
        ab_idx=ab_idx, ag_idx=ag_idx,
        ab_chain_type=ab_types[ab_idx], ag_chain_type=ag_types[ag_idx],
        ab_res_index=ab_res[ab_idx], ag_res_index=ag_res[ag_idx],
        ab_is_site=site_ab[ab_idx].astype(np.float32),
        ag_is_site=site_ag[ag_idx].astype(np.float32),
        disconnected=not site_touches_iface,
    )
