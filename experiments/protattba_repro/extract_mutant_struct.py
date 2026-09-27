"""Per-ROW ProteinMPNN structural embedding, from the FoldX mutant structure.

mpnn_per_residue.py's cache (src/fusion/mpnn_per_residue.py) is one entry per COMPLEX, from the
wild-type structure only -- correct for the forward direction (h_V comes from backbone geometry
alone, identical for wild-type and mutant, per that module's own docstring). The reverse-mutation
augmentation (DS.__getitem__ in _perturb_v2_colab.py, the `swap` branch) needs different geometry:
in that orientation the true mutant plays the role of the reference structure, and reusing the
wild-type complex's structure for it is a per-complex constant blind to which mutation occurred --
exactly the gap this closes.

Self-contained (no `src` package import) so it can run directly on a training box: `parse_pdb`,
`Structure`/`Chain`, `_featurize` and `encoder_h_V` below are copied verbatim from
src/structures.py and src/features/{proteinmpnn,mpnn_repr}.py -- same encoder, same featurization,
same offset convention -- just pointed at data_mutants/<row_safe_name>.pdb per row instead of the
wild-type complex PDB, so the two embedding spaces are directly comparable.

    python extract_mutant_struct.py --row-chains row_chains.csv --mutant-dir data_mutants \
        --mpnn-dir stabddg_test/third_party/ProteinMPNN --out mpnn_per_residue_mutant
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
BACKBONE = ("N", "CA", "C", "O")


@dataclass
class Chain:
    id: str
    seq: str = ""
    keys: list = field(default_factory=list)
    index: dict = field(default_factory=dict)
    backbone: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))


@dataclass
class Structure:
    pdb: str
    chains: dict


def parse_pdb(path: Path) -> Structure:
    """Verbatim copy of src.structures.parse_pdb, minus heavy-atom bookkeeping this doesn't need."""
    raw, order, names = {}, {}, {}
    with open(path) as fh:
        for ln in fh:
            rec = ln[:6]
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            resname = ln[17:20].strip()
            if rec == "HETATM" and resname != "MSE":
                continue
            aa = THREE2ONE.get(resname)
            if aa is None:
                continue
            if ln[16] not in (" ", "A"):
                continue
            element, atom = ln[76:78].strip().upper(), ln[12:16].strip()
            if element == "H" or (not element and atom.lstrip("0123456789").startswith("H")):
                continue
            ch, key = ln[21], (int(ln[22:26]), ln[26].strip())
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], np.float32)
            res = raw.setdefault(ch, {})
            if key not in res:
                res[key] = {}
                order.setdefault(ch, []).append(key)
                names.setdefault(ch, {})[key] = aa
            if atom not in res[key]:
                res[key][atom] = xyz

    chains = {}
    for ch, keys in order.items():
        c = Chain(id=ch, keys=keys)
        c.seq = "".join(names[ch][k] for k in keys)
        c.index = {k: i for i, k in enumerate(keys)}
        bb = np.full((len(keys), len(BACKBONE), 3), np.nan, np.float32)
        for i, k in enumerate(keys):
            atoms = raw[ch][k]
            for j, name in enumerate(BACKBONE):
                if name in atoms:
                    bb[i, j] = atoms[name]
        c.backbone = bb
        chains[ch] = c
    return Structure(pdb=path.stem, chains=chains)


def _featurize(structure: Structure, chain_group: str, device: str):
    """Verbatim copy of src.features.proteinmpnn._featurize."""
    xs, res_idx, chain_enc, offsets = [], [], [], {}
    cursor = 0
    for c_i, cid in enumerate(chain_group):
        ch = structure.chains.get(cid)
        if ch is None:
            continue
        n = len(ch.seq)
        xs.append(ch.backbone)
        res_idx.append(100 * c_i + np.arange(n))
        chain_enc.append(np.full(n, c_i + 1))
        offsets[cid] = cursor
        cursor += n

    X = np.concatenate(xs, 0)[None]
    finite = np.isfinite(X).all(axis=(2, 3))
    X = np.nan_to_num(X, nan=0.0)
    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    return (t(X, torch.float32), t(finite.astype(np.float32), torch.float32),
            t(np.concatenate(res_idx)[None], torch.long),
            t(np.concatenate(chain_enc)[None], torch.long), offsets)


@torch.no_grad()
def encoder_h_V(model, structure: Structure, chain_group: str, device: str):
    """Verbatim copy of src.features.mpnn_repr.encoder_h_V."""
    from protein_mpnn_utils import gather_nodes  # noqa: F401  (imported for parity/availability)

    X, mask, residue_idx, chain_enc, offsets = _featurize(structure, chain_group, device)
    E, E_idx = model.features(X, mask, residue_idx, chain_enc)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    return h_V[0].float().cpu().numpy(), offsets


def load_model(mpnn_dir: Path, device: str, weights: str = "v_48_020.pt"):
    sys.path.insert(0, str(mpnn_dir))
    from protein_mpnn_utils import ProteinMPNN

    path = mpnn_dir / "vanilla_model_weights" / weights
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def row_safe_name(row_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", row_id)


def sides_for(key: str, ab_chains: str, ag_chains: str) -> tuple[str, str]:
    return (ab_chains or key.split("_")[1]), (ag_chains or key.split("_")[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-chains", required=True, help="csv: row_id,#Pdb,ab_chains,ag_chains")
    ap.add_argument("--mutant-dir", required=True, type=Path)
    ap.add_argument("--mpnn-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    # ab_chains/ag_chains are "" (not NaN) for 1DVF_AB_CD's 38 rows -- the antibody-vs-antibody
    # tie complex sides_for() resolves via the "#Pdb" key itself -- but pandas round-trips ""
    # through CSV as NaN, so it must be restored before sides_for()'s falsy-string fallback works.
    df = pd.read_csv(a.row_chains).rename(columns={"#Pdb": "pdb_key"})
    df["ab_chains"] = df["ab_chains"].fillna("")
    df["ag_chains"] = df["ag_chains"].fillna("")
    a.out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.mpnn_dir, a.device)

    t0 = time.perf_counter()
    written, skipped = 0, []
    for n, r in enumerate(df.itertuples()):
        dest = a.out / f"{row_safe_name(r.row_id)}.npz"
        if dest.exists() and not a.force:
            continue
        pdb_path = a.mutant_dir / f"{row_safe_name(r.row_id)}.pdb"
        if not pdb_path.exists():
            skipped.append(r.row_id)
            continue
        st = parse_pdb(pdb_path)
        ab, ag = sides_for(r.pdb_key, r.ab_chains, r.ag_chains)
        h_ab, _ = encoder_h_V(model, st, ab, a.device)
        h_ag, _ = encoder_h_V(model, st, ag, a.device)
        np.savez_compressed(dest, h_ab=h_ab.astype(np.float16), h_ag=h_ag.astype(np.float16))
        written += 1
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(df)}  written={written}  {time.perf_counter()-t0:.0f}s",
                  flush=True)

    print(f"done: wrote {written}, skipped {len(skipped)}, in {time.perf_counter()-t0:.0f}s")
    if skipped:
        print("skipped (no mutant pdb):", skipped[:10])


if __name__ == "__main__":
    main()
