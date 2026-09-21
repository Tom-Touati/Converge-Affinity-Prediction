"""Structure modality, part 2: ProteinMPNN inverse-folding log-odds.

Why ProteinMPNN rather than ESM-IF1 as the primary structure encoder: on the closest published
setup to ours -- SKEMPI antibody-antigen interface single-point mutations -- CIR-DDG measures
zero-shot ProteinMPNN at Spearman 0.172 against ESM-IF's 0.119, and ProteinMPNN is pure PyTorch
with no torch-geometric dependency tree. See BACKBONE_COMPARISON.md.

The score is ``unconditional_probs``: p(amino acid at position i | backbone geometry), with no
sequence input at all. Three consequences, all of them good for us:

* It is deterministic -- no decoding order, no sampling, and ``augment_eps`` is forced to 0.
* Wild type and mutant share one forward pass, because the backbone is identical and only the
  identity read off it changes. LLR = log p(mut) - log p(wt) at the mutated position.
* It is a pure structure signal, uncontaminated by the sequence model we already have.

**The control is the point.** de Kanter & Greiff showed ESM-IF1 predicts a *control* epitope's
affinity change as well as the real one (r 0.64 vs 0.59), i.e. inverse-folding likelihood tracks
protein *quality* rather than *interaction*. Their run was single-chain, so multichain
conditioning is the untested variable. This module therefore scores every mutation twice:

* ``llr_complex`` -- conditioned on the whole antibody-antigen complex.
* ``llr_alone``   -- conditioned on the mutated residue's own side only, partner deleted.
* ``llr_delta``   -- the difference, i.e. the part of the score that the partner's presence
  creates. This is the only column that can contain binding-specific information.

If ``llr_complex`` predicts ddG no better than ``llr_alone``, our structure modality is a
foldability predictor wearing a binding hat, and ``llr_delta`` is where we find out.

    python -m src.features.proteinmpnn --device auto
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

MPNN_DIR = paths.ROOT / "third_party" / "ProteinMPNN"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}
DEFAULT_WEIGHTS = "v_48_020.pt"


def _load_model(device: str, weights: str = DEFAULT_WEIGHTS):
    if not MPNN_DIR.exists():
        raise FileNotFoundError(
            f"{MPNN_DIR} not found. Clone it with:\n"
            f"  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git "
            f"third_party/ProteinMPNN"
        )
    sys.path.insert(0, str(MPNN_DIR))
    from protein_mpnn_utils import ProteinMPNN

    ckpt = torch.load(MPNN_DIR / "vanilla_model_weights" / weights,
                      map_location=device, weights_only=False)
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0,                    # training-time coordinate noise off: determinism
        k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device), ckpt


def _featurize(structure, chain_group: str, device: str):
    """Build ProteinMPNN's tensors straight from our verified Structure.

    Deliberately not using the repo's own ``parse_PDB``: a second parser could disagree with the
    one the EDA validated, and every residue index here has to line up with ``chain.index``.
    Conventions copied from ``tied_featurize``: residue_idx carries a 100-offset per chain so the
    relative positional encoding cannot bridge chains, and chain_encoding is 1-based.
    """
    xs, res_idx, chain_enc, offsets = [], [], [], {}
    cursor = 0
    for c_i, cid in enumerate(chain_group):
        ch = structure.chains.get(cid)
        if ch is None:
            continue
        n = len(ch.seq)
        xs.append(ch.backbone)                              # (n, 4, 3) N, CA, C, O
        res_idx.append(100 * c_i + np.arange(n))
        chain_enc.append(np.full(n, c_i + 1))
        offsets[cid] = cursor
        cursor += n

    X = np.concatenate(xs, 0)[None]                         # (1, L, 4, 3)
    finite = np.isfinite(X).all(axis=(2, 3))                # (1, L)
    X = np.nan_to_num(X, nan=0.0)
    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    return (t(X, torch.float32), t(finite.astype(np.float32), torch.float32),
            t(np.concatenate(res_idx)[None], torch.long),
            t(np.concatenate(chain_enc)[None], torch.long), offsets)


@torch.no_grad()
def _log_probs(model, structure, chain_group: str, device: str):
    """Per-position log p(aa | backbone) for one chain group, plus that group's index offsets."""
    X, mask, residue_idx, chain_enc, offsets = _featurize(structure, chain_group, device)
    lp = model.unconditional_probs(X, mask, residue_idx, chain_enc)
    return lp[0].float().cpu().numpy(), offsets                 # (L, 21)


def extract(device: str = "auto", weights: str = DEFAULT_WEIGHTS,
            verbose: bool = True) -> pd.DataFrame:
    paths.ensure_dirs()
    device = resolve_device(device)
    df = pd.read_parquet(paths.DATASET)
    structures = load_structures(df["pdb"].unique())
    model, ckpt = _load_model(device, weights)
    if verbose:
        print(f"ProteinMPNN {weights}: k_neighbors={ckpt['num_edges']} "
              f"noise_level={ckpt['noise_level']} device={device}", flush=True)

    cx = df.drop_duplicates("#Pdb")
    cache = {}
    t0 = time.perf_counter()
    for n, (key, pdb, abc, agc) in enumerate(
        zip(cx["#Pdb"], cx["pdb"], cx["ab_chains"], cx["ag_chains"])
    ):
        st = structures[pdb]
        ab = abc or key.split("_")[1]
        ag = agc or key.split("_")[2]
        entry = {"complex": _log_probs(model, st, ab + ag, device)}
        # each side scored with the partner deleted -- the control
        entry["ab"] = _log_probs(model, st, ab, device)
        entry["ag"] = _log_probs(model, st, ag, device)
        entry["sides"] = {c: "ab" for c in ab}
        entry["sides"].update({c: "ag" for c in ag})
        cache[key] = entry
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(cx)} complexes  {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for row_id, key, pdb, mutations in zip(df["row_id"], df["#Pdb"], df["pdb"], df["mutations"]):
        st = structures[pdb]
        e = cache[key]
        llr_c = llr_a = 0.0
        p_wt_c = p_mut_c = 0.0
        for m in parse_mutations(mutations):
            pos_in_chain = st.chains[m.chain].index[m.key]
            iw, im = AA_IDX[m.wt], AA_IDX[m.mut]

            lp_c, off_c = e["complex"]
            i_c = off_c[m.chain] + pos_in_chain
            llr_c += float(lp_c[i_c, im] - lp_c[i_c, iw])
            p_wt_c += float(lp_c[i_c, iw])
            p_mut_c += float(lp_c[i_c, im])

            side = e["sides"][m.chain]
            lp_a, off_a = e[side]
            i_a = off_a[m.chain] + pos_in_chain
            llr_a += float(lp_a[i_a, im] - lp_a[i_a, iw])

        rows.append({
            "row_id": row_id,
            "llr_complex": llr_c,
            "llr_alone": llr_a,
            "llr_delta": llr_c - llr_a,      # the only binding-specific column
            "logp_wt_complex": p_wt_c,
            "logp_mut_complex": p_mut_c,
        })

    block = pd.DataFrame(rows).set_index("row_id").astype(np.float32)
    out = paths.FEATURES / "mpnn.parquet"
    block.to_parquet(out)
    if verbose:
        print(f"wrote {out.relative_to(paths.ROOT)}  {block.shape}  "
              f"in {time.perf_counter()-t0:.1f}s")
    return block


def build(df: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    return extract(verbose=verbose)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    a = p.parse_args()
    extract(a.device, a.weights)


if __name__ == "__main__":
    main()
