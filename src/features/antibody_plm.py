"""Sequence modality, second attempt: an antibody-specific language model, with a local window.

Two changes from ``esm2.py``, each motivated by a measurement rather than a hunch.

**A different kind of model.** ESM-2 failed here three times -- log-likelihood ratio alone scored
-0.066, the position-wise embedding difference +0.074, and adding its scalars to the best feature
set cost -0.032. The diagnosis is in the feature itself: mean LLR by interface location runs
INT -1.165, SUR -0.922, SUP -0.575, COR -0.449, RIM -0.350, which is *backwards* for binding.
ESM-2 measures evolutionary conservation, and antibody binding residues sit in CDR loops that
evolution deliberately does not conserve. A different *general* protein model would share that
bias. AntiBERTy is trained on antibody repertoires (OAS), where CDR variation is the signal
rather than the noise, so "surprising" means something different: unusual *for an antibody*, not
unusual for a protein. BACKBONE_COMPARISON.md carries the one positive antibody-specific
datapoint we have -- CurrAb at +0.074 Spearman over its ESM-2 base, against AntiFold at -0.097 on
the structure side.

**A window rather than a single position.** ESM-2 features took the embedding difference at
exactly the mutated residue, plus a whole-chain mean that dilutes one changed residue among ~215.
Nothing in between. But a transformer's attention means mutating a position also shifts its
neighbours' representations, and *how far that shift reaches* is informative: a substitution that
disrupts a tight local motif perturbs its neighbourhood more than a conservative swap on a
flexible loop. This module keeps the difference across +/- ``WINDOW`` residues and summarises the
decay.

**Scope.** Antibody models read antibody variable domains. Mutations on the antigen side -- 332
of 997 rows -- have no value here, and are emitted as 0 with ``ab_applicable = 0`` so a tree can
tell "absent" from "measured zero". This is deliberate rather than a limitation to apologise for:
the error analysis found the antibody side is exactly where the model is blind (per-complex
Spearman 0.15 against 0.62 on the antigen side), so a feature that only speaks there targets the
measured weakness.

A sequence-window is *not* a spatial neighbourhood. Residues +/-8 along the chain are rarely the
residues touching the mutation in 3D; that is the geometry block's job. This feature describes
local chain context -- helix, turn, loop -- and should be read as a stability/context signal
rather than a binding one.

    python -m src.features.antibody_plm --device auto
"""
from __future__ import annotations

import argparse
import re
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .. import paths, splits
from ..structures import apply_mutations, load_structures, parse_mutations
from ..util import resolve_device

WINDOW = 8          # +/- residues around the mutation
MAX_V = 140         # fallback truncation when no FR4 motif is found

# Framework-4 motifs, which sit at the end of the variable domain. Heavy chains end WGxG;
# light chains end FGxG. Together these cover both chain types without needing ANARCI.
FR4_HEAVY = re.compile(r"WG[QKRAESTG]G")
FR4_LIGHT = re.compile(r"FG[QGSTEAP]G")


def variable_domain(seq: str) -> int:
    """Length of the variable domain: the FR4 motif plus the ~11 residues that follow it."""
    best = None
    for pat in (FR4_HEAVY, FR4_LIGHT):
        for m in pat.finditer(seq[:MAX_V + 20]):
            end = min(len(seq), m.end() + 11)
            if best is None or end < best:
                best = end
    return best if best is not None else min(len(seq), MAX_V)


class AntiBERTy:
    def __init__(self, device: str = "auto"):
        warnings.filterwarnings("ignore")
        from antiberty import AntiBERTyRunner

        self.r = AntiBERTyRunner()
        self.device = resolve_device(device)
        self.r.model.to(self.device).eval()
        self.dim = self.r.model.config.hidden_size
        self._cache = {}

    def idx(self, aa: str) -> int:
        return self.r.tokenizer.token_to_id[aa]

    @torch.no_grad()
    def forward(self, seq: str):
        """Per-residue representations (L, dim) and per-position log-probabilities (L, vocab).

        Two calls rather than one: ``_forward`` returns prediction logits but not hidden states,
        while ``embed`` returns hidden states but not logits. At 26M parameters on ~125-residue
        variable domains the second pass costs almost nothing.
        """
        rep = self.r.embed([seq])[0][1:len(seq) + 1].float().cpu()
        outputs, _, _ = self.r._forward([seq])
        logp = torch.log_softmax(outputs.prediction_logits[0], -1)[1:len(seq) + 1].float().cpu()
        return rep, logp

    def wt(self, key, seq: str):
        if key not in self._cache:
            self._cache[key] = self.forward(seq)
        return self._cache[key]


def build(df: pd.DataFrame, device: str = "auto", window: int = WINDOW,
          verbose: bool = True) -> pd.DataFrame:
    enc = AntiBERTy(device)
    structures = load_structures(df["pdb"].unique())
    t0 = time.perf_counter()

    ab_chains = {r["complex_key"]: set(r["ab_chains"] or "")
                 for _, r in df.drop_duplicates("complex_key").iterrows()}
    # 1DVF is antibody bound to antibody: both sides are immunoglobulin, so both are in scope.
    for _, r in df.drop_duplicates("complex_key").iterrows():
        if r["role_basis"] == "tie":
            ab_chains[r["complex_key"]] = set(r["side1"] + r["side2"])

    rows, vectors = [], []
    n_cov = n_skip_pos = 0
    for n, r in enumerate(df.itertuples()):
        muts = [m for m in parse_mutations(r.mutations)
                if m.chain in ab_chains[r.complex_key]]
        applicable = False
        llr_wt = llr_mask = at_norm = win_norm = 0.0
        prof = np.zeros(window, np.float32)
        wide = np.zeros(enc.dim, np.float32)

        if muts:
            mutant_seqs = apply_mutations(structures[r.pdb], parse_mutations(r.mutations))
            used = 0
            for m in muts:
                chain = structures[r.pdb].chains[m.chain]
                pos = chain.index[m.key]
                vlen = variable_domain(chain.seq)
                if pos >= vlen:          # mutation sits in the constant domain
                    n_skip_pos += 1
                    continue
                wt_seq = chain.seq[:vlen]
                mt_seq = mutant_seqs[m.chain][:vlen]

                wrep, wlogp = enc.wt((r.pdb, m.chain, vlen), wt_seq)
                mrep, _ = enc.forward(mt_seq)

                llr_wt += float(wlogp[pos, enc.idx(m.mut)] - wlogp[pos, enc.idx(m.wt)])
                _, mlogp = enc.forward(wt_seq[:pos] + "_" + wt_seq[pos + 1:])
                llr_mask += float(mlogp[pos, enc.idx(m.mut)] - mlogp[pos, enc.idx(m.wt)])

                d = (mrep - wrep).numpy()                      # (L, dim) full perturbation field
                lo, hi = max(0, pos - window), min(len(wt_seq), pos + window + 1)
                at_norm += float(np.linalg.norm(d[pos]))
                win_norm += float(np.linalg.norm(d[lo:hi], axis=1).mean())
                wide += d[lo:hi].mean(0)
                for k in range(1, window + 1):
                    vals = [np.linalg.norm(d[pos + s]) for s in (-k, k)
                            if 0 <= pos + s < len(wt_seq)]
                    if vals:
                        prof[k - 1] += float(np.mean(vals))
                used += 1
            if used:
                applicable = True
                n_cov += 1
                prof /= used

        rows.append({
            "row_id": r.row_id,
            "ab_applicable": float(applicable),
            "ab_llr_wt": llr_wt,
            "ab_llr_masked": llr_mask,
            "ab_d_at_pos": at_norm,
            "ab_d_window": win_norm,
            # how far the perturbation reaches relative to its size at the mutated residue
            "ab_reach": float(win_norm / at_norm) if at_norm > 0 else 0.0,
            **{f"ab_prof{k+1}": float(prof[k]) for k in range(window)},
        })
        vectors.append(wide)

        if verbose and (n + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {n+1:4d}/{len(df)}  {el:6.1f}s  eta {el/(n+1)*(len(df)-n-1):6.1f}s", flush=True)

    scalars = pd.DataFrame(rows).set_index("row_id").astype(np.float32)
    vec = pd.DataFrame(np.vstack(vectors), index=scalars.index,
                       columns=[f"d{i}" for i in range(enc.dim)]).astype(np.float32)
    block = pd.concat([scalars, vec], axis=1)
    out = paths.FEATURES / "abplm.parquet"
    block.to_parquet(out)

    if verbose:
        print(f"\nrows with an antibody-side mutation inside the variable domain: "
              f"{n_cov}/{len(df)} ({100*n_cov/len(df):.0f}%)")
        print(f"mutated positions skipped as constant-domain: {n_skip_pos}")
        print(f"wrote {out.relative_to(paths.ROOT)}  {block.shape}  in {time.perf_counter()-t0:.0f}s")
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=int, default=WINDOW)
    a = p.parse_args()
    df = splits.load()
    df = df.rename(columns={"#Pdb": "complex_key"})
    build(df, a.device, a.window)


if __name__ == "__main__":
    main()
