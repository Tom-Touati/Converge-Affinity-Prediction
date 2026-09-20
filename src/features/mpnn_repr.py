"""The structure encoder's *representation*, not just its output score.

The gap this closes. ``proteinmpnn.py`` reads ProteinMPNN's 20-way output softmax and keeps five
scalars. But the model is a geometric graph network: three encoder layers over a 48-neighbour
graph whose edges carry RBF-expanded distances between N, CA, C, O and a virtual C-beta plus
relative orientations. That machinery produces a **128-dimensional representation per residue**,
``h_V``, and the log-probabilities are what is left after squeezing it through a linear layer and
a softmax over 20 letters.

So the multimodal comparison we had been running was not fair to structure: 480 dimensions of
pretrained sequence encoder against 5 numbers of pretrained structure encoder. Measured
consequence -- fusing the two scored 0.230 against ProteinMPNN's own 0.243, i.e. fusion appeared
to hurt. This module supplies the structure side's actual representation so the comparison is
between like and like.

``unconditional_probs`` computes ``h_V`` and discards it, so the encoder half is replicated here
rather than patched into the vendored repo. The arithmetic is copied from that method exactly,
stopping before the decoder.

**Three blocks, and the third is the one that should matter.**

* ``hc*`` -- ``h_V`` at the mutated position, conditioned on the whole complex.
* ``ha*`` -- the same with the partner chain deleted.
* ``hd*`` -- their difference: the part of the site's representation that the partner's presence
  creates. This is the representation-level analogue of ``llr_delta``, which is the only
  ProteinMPNN scalar that predicts binding (global rho -0.266, against -0.033 and +0.099 for the
  two absolute scores). If the contrast is where binding lives in the output layer, it should be
  where binding lives in the representation too.

**What this is and is not.** ``h_V`` comes from backbone geometry alone -- no sequence enters the
encoder -- so it describes the *site*, identically for whichever residue occupies it. It cannot
distinguish two substitutions at the same position; 256 of 696 single-point rows sit at a site
that is also mutated to some other amino acid, and this block is constant across them. That is
not a defect to hide but the correct division of labour: the structure encoder says what kind of
position this is, and the sequence encoder says what the substitution does to it. Fusing them is
combining a site descriptor with a substitution descriptor, which is why fusion is worth trying
at all.

    python -m src.features.mpnn_repr --device auto
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..structures import load_structures, parse_mutations
from ..util import resolve_device
from .proteinmpnn import DEFAULT_WEIGHTS, MPNN_DIR, _featurize, _load_model


@torch.no_grad()
def encoder_h_V(model, structure, chain_group: str, device: str):
    """Per-residue geometric representation (L, 128), plus this group's index offsets.

    Replicates the encoder half of ``ProteinMPNN.unconditional_probs`` and stops before the
    decoder, which is where h_V would otherwise be consumed and thrown away.
    """
    sys.path.insert(0, str(MPNN_DIR))
    from protein_mpnn_utils import cat_neighbors_nodes, gather_nodes  # noqa: F401

    X, mask, residue_idx, chain_enc, offsets = _featurize(structure, chain_group, device)
    E, E_idx = model.features(X, mask, residue_idx, chain_enc)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    return h_V[0].float().cpu().numpy(), offsets


def build(device: str = "auto", weights: str = DEFAULT_WEIGHTS,
          verbose: bool = True) -> pd.DataFrame:
    paths.ensure_dirs()
    device = resolve_device(device)
    df = pd.read_parquet(paths.DATASET)
    structures = load_structures(df["pdb"].unique())
    model, _ = _load_model(device, weights)
    t0 = time.perf_counter()

    cache = {}
    cx = df.drop_duplicates("#Pdb")
    for n, (key, pdb, abc, agc) in enumerate(
        zip(cx["#Pdb"], cx["pdb"], cx["ab_chains"], cx["ag_chains"])
    ):
        st = structures[pdb]
        ab = abc or key.split("_")[1]
        ag = agc or key.split("_")[2]
        entry = {"complex": encoder_h_V(model, st, ab + ag, device),
                 "ab": encoder_h_V(model, st, ab, device),
                 "ag": encoder_h_V(model, st, ag, device)}
        entry["sides"] = {c: "ab" for c in ab}
        entry["sides"].update({c: "ag" for c in ag})
        cache[key] = entry
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(cx)} complexes  {time.perf_counter()-t0:.0f}s", flush=True)

    dim = cache[cx['#Pdb'].iloc[0]]["complex"][0].shape[1]
    rows = []
    for row_id, key, pdb, mutations in zip(df["row_id"], df["#Pdb"], df["pdb"], df["mutations"]):
        st = structures[pdb]
        e = cache[key]
        hc = np.zeros(dim, np.float32)
        ha = np.zeros(dim, np.float32)
        for m in parse_mutations(mutations):
            pos = st.chains[m.chain].index[m.key]
            H_c, off_c = e["complex"]
            hc += H_c[off_c[m.chain] + pos]
            side = e["sides"][m.chain]
            H_a, off_a = e[side]
            ha += H_a[off_a[m.chain] + pos]
        rows.append((row_id, hc, ha))

    idx = [r[0] for r in rows]
    blocks = []
    for j, prefix in enumerate(("hc", "ha")):
        blocks.append(pd.DataFrame(np.vstack([r[j + 1] for r in rows]), index=idx,
                                   columns=[f"{prefix}{i}" for i in range(dim)]))
    # the partner-attributable representation: what the interface does to this site
    delta = blocks[0].to_numpy() - blocks[1].to_numpy()
    blocks.append(pd.DataFrame(delta, index=idx, columns=[f"hd{i}" for i in range(dim)]))

    block = pd.concat(blocks, axis=1).astype(np.float32)
    block.index.name = "row_id"
    out = paths.FEATURES / "mpnnrep.parquet"
    block.to_parquet(out)
    if verbose:
        print(f"\nwrote {out.relative_to(paths.ROOT)}  {block.shape}  "
              f"in {time.perf_counter()-t0:.1f}s")
        print(f"  {dim} dims each for complex-conditioned, chain-alone, and their difference")
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    build(a.device)


if __name__ == "__main__":
    main()
