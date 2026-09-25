"""Cache ESM-IF1's per-residue encoder output, as an alternative to ProteinMPNN.

This project picked ProteinMPNN over ESM-IF1 originally on the strength of one published
comparison (CIR-DDG, docs/BACKBONE_COMPARISON.md): ESM-IF scores Spearman 0.119 against
ProteinMPNN's 0.172 on the closest published cohort to ours. This builds the cache to test
that directly, on our own data and our own downstream head, rather than trusting the
transfer from a different benchmark.

Mirrors ``src/fusion/mpnn_per_residue.py`` exactly in structure: same complex iteration, same
``sides_for``/chain-resolution fallback (imported, not reimplemented, so the two encoders
describe the IDENTICAL set of residues per side), same ``.npz`` layout with ``h_ab``/``h_ag``
-- a drop-in alternative to ``mpnn_per_residue/`` for whichever loader reads it.

**Runs in an isolated venv, deliberately.** ``esm.inverse_folding`` needs torch_geometric,
torch_scatter and a biotite old enough to still export ``filter_backbone`` (removed in
current biotite, which forced numpy<2 and, in turn, scipy<1.12 to avoid an ABI break). None
of that may touch the venv the live training fleet is using.

**Side-alone, not complex-conditioned**, matching what the trainer actually reads: the
downstream ``Cache.mpnn()`` uses ``h_ab``/``h_ag`` -- ProteinMPNN's *side-alone* field, not
``h_complex`` -- so this computes only the equivalent side-alone field and does not pay for
the complex-conditioned one at all.

Run (on the box, with the isolated venv):
    /home/ubuntu/venv_esmif/bin/python _esmif_ec2.py --full-root /home/ubuntu/full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def build_esmif_encoder(model, alphabet):
    """Returns a function (structure, chain_group) -> (h_V (L, 512), offsets)."""
    import torch
    from esm.inverse_folding.util import CoordBatchConverter

    device = next(model.parameters()).device
    converter = CoordBatchConverter(alphabet)
    PAD = 10  # ESM-IF1's own convention (esm.inverse_folding.multichain_util), reused so the
              # model sees the chain-break pattern it was trained on

    @torch.no_grad()
    def run(structure, chain_group: str):
        # Two coordinate systems on purpose: `padded_offset`/`cursor` locate a chain in the
        # PADDED array the model actually sees (it needs the 10-residue NaN gap to read the
        # chain break); `offsets` locates it in the CLEAN, padding-free array this function
        # returns, which is what every downstream consumer (site indices, crop indices)
        # expects -- ProteinMPNN's own cache has no physical gap between chains either.
        coords_list, spans, cursor = [], [], 0
        pad = np.full((PAD, 3, 3), np.nan, dtype=np.float32)
        first = True
        for cid in chain_group:
            ch = structure.chains.get(cid)
            if ch is None:
                continue
            if not first:
                coords_list.append(pad)
                cursor += PAD
            first = False
            # backbone is (n, 4, 3) N, CA, C, O -- ESM-IF1 wants only N, CA, C
            coords_list.append(ch.backbone[:, :3, :].astype(np.float32))
            spans.append((cid, cursor, len(ch.seq)))     # position in the PADDED array
            cursor += len(ch.seq)
        if not coords_list:
            return np.zeros((0, 512), np.float32), {}
        all_coords = np.concatenate(coords_list, axis=0)

        batch = [(all_coords, None, None)]
        coords_t, confidence, _, _, padding_mask = converter(batch)
        # CoordBatchConverter builds its tensors on CPU regardless of the model's device
        coords_t, confidence, padding_mask = (
            coords_t.to(device), confidence.to(device), padding_mask.to(device))
        enc = model.encoder.forward(coords_t, padding_mask, confidence,
                                    return_all_hiddens=False)
        # (L+2, 1, D) with bos/eos -- strip them, matching esm's own get_encoder_output
        h_padded = enc["encoder_out"][0][1:-1, 0].float().cpu().numpy()
        if h_padded.shape[0] != cursor:
            raise RuntimeError(f"encoder returned {h_padded.shape[0]} residues, expected "
                               f"{cursor} (chain_group {chain_group!r}, WITH padding)")

        # drop the NaN-gap rows, rebuild offsets into the now-contiguous, padding-free array
        clean_chunks, offsets, clean_cursor = [], {}, 0
        for cid, start, n in spans:
            clean_chunks.append(h_padded[start:start + n])
            offsets[cid] = clean_cursor
            clean_cursor += n
        h = np.concatenate(clean_chunks, axis=0) if clean_chunks else h_padded[:0]
        return h, offsets
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", default="/home/ubuntu/full",
                    help="a checkout with src/ and data/PDBs/ -- reused for structure "
                         "parsing only; chain assignment comes from complex_key itself")
    ap.add_argument("--rows", default="/home/ubuntu/perturb/perturb_rows.parquet",
                    help="perturb_rows.parquet -- complex_key already encodes the resolved "
                         "<pdb>_<ab_chains>_<ag_chains>, including the 1DVF tie-break, so "
                         "nothing here needs to re-derive it")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent.parent
                                        / "esmif_per_residue"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, a.full_root)
    import pandas as pd
    from src.structures import load_structures

    import torch
    import esm

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(a.rows)
    cx = rows.drop_duplicates("complex_key")
    todo = [k for k in cx["complex_key"] if a.force or not (out_dir / f"{k}.npz").exists()]
    print(f"{len(cx)} complexes, {len(todo)} to compute", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    encoder_h_V = build_esmif_encoder(model, alphabet)

    structures = load_structures(cx["pdb"].unique())
    t0 = time.perf_counter()
    written = 0
    for key, pdb in zip(cx["complex_key"], cx["pdb"]):
        if key not in todo:
            continue
        st = structures[pdb]
        _, ab, ag = key.split("_")

        h_ab, off_ab = encoder_h_V(st, ab)
        h_ag, off_ag = encoder_h_V(st, ag)

        expected_ab = sum(len(st.chains[c].seq) for c in ab if c in st.chains)
        expected_ag = sum(len(st.chains[c].seq) for c in ag if c in st.chains)
        if h_ab.shape[0] != expected_ab or h_ag.shape[0] != expected_ag:
            raise RuntimeError(f"{key}: field length mismatch, ab {h_ab.shape[0]} vs "
                               f"{expected_ab}, ag {h_ag.shape[0]} vs {expected_ag}")

        np.savez_compressed(
            out_dir / f"{key}.npz",
            h_ab=h_ab.astype(np.float16), h_ag=h_ag.astype(np.float16),
            meta=np.array(json.dumps({
                "complex_key": key, "pdb": pdb, "chains_ab": ab, "chains_ag": ag,
                "off_ab": off_ab, "off_ag": off_ag, "dim": int(h_ab.shape[1]),
                "encoder": "esm_if1_gvp4_t16_142M_UR50",
            })),
        )
        written += 1
        if written % 10 == 0:
            print(f"  {written}/{len(todo)}  {time.perf_counter() - t0:.0f}s", flush=True)

    total = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    print(f"wrote {written} complexes to {out_dir} ({total / 1e6:.1f} MB) "
          f"in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
