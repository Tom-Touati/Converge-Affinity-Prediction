"""Sequence modality via any HuggingFace masked protein language model, with a local window.

Written to serve two models that differ in exactly one thing, so a difference between them is
attributable:

* ``brineylab/150M_full_checkpoint-500000`` -- **CurrAb**, ESM-2 150M fine-tuned on antibody
  repertoires by curriculum. AbBiBench measures it at +0.074 Spearman over its base, the one
  positive antibody-specific datapoint in BACKBONE_COMPARISON.md.
* ``facebook/esm2_t30_150M_UR50D`` -- the **control**: same architecture (30 layers, hidden 640),
  same parameter count, no antibody fine-tuning.

Run both and the delta isolates the fine-tuning. That is why this module exists rather than
reusing ``antibody_plm.py``: AntiBERTy changes architecture, size, tokenizer and training corpus
all at once, so a result from it cannot be attributed to anything in particular.

**Two features, both absent from ``esm2.py``.**

*The surprise indicator* is the same quantity ESM-2 supplies -- log p(mutant) - log p(wild type)
at the mutated position -- but read from a model whose notion of "surprising" is different. ESM-2
scores conservation, and antibody CDR loops are hypervariable by design, which is why its LLR
ordered *backwards* by interface location (INT -1.165 through RIM -0.350). A model fine-tuned on
antibody repertoires should not share that inversion.

*The window.* ``esm2.py`` keeps the embedding difference at exactly the mutated residue, plus a
whole-chain mean that dilutes one changed residue among hundreds. Nothing in between. But
attention means mutating a position also shifts its neighbours' representations, and how far the
shift reaches is informative in its own right: a substitution that disrupts a tight local motif
perturbs its neighbourhood more than a conservative swap on a flexible loop. This module keeps
the difference across +/- ``WINDOW`` residues and summarises the decay as a profile.

Unlike ``antibody_plm.py`` this covers **all** rows, antigen side included, because an ESM-2
derivative still reads ordinary protein sequence. Whether the antibody fine-tuning helps or hurts
on antigen chains is then something the error analysis can slice rather than something we assume.

A sequence window is not a spatial neighbourhood: residues +/-8 along the chain are rarely the
ones contacting the mutation in 3D. That is the geometry block's job. Read this as local chain
context -- helix, turn, loop -- rather than as a binding signal.

    python -m src.features.hf_plm --model brineylab/150M_full_checkpoint-500000 \
        --tokenizer brineylab/CurrAb --tag currab150M
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .. import paths, splits
from ..structures import apply_mutations, load_structures, parse_mutations
from ..util import resolve_device

WINDOW = 8


class HFEncoder:
    def __init__(self, model_id: str, tokenizer_id: str | None = None, device: str = "auto"):
        warnings.filterwarnings("ignore")
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = resolve_device(device)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.hidden_size
        # ESM position embeddings reserve two slots for BOS/EOS and one for padding.
        self.max_len = self.model.config.max_position_embeddings - 4
        self._cache = {}

    def idx(self, aa: str) -> int:
        return self.tok.convert_tokens_to_ids(aa)

    def window_seq(self, seq: str, focus: int) -> tuple:
        """Truncate a long chain around the mutated position. Returns (subseq, new index)."""
        if len(seq) <= self.max_len:
            return seq, focus
        half = self.max_len // 2
        start = max(0, min(focus - half, len(seq) - self.max_len))
        return seq[start:start + self.max_len], focus - start

    @torch.no_grad()
    def forward(self, seq: str, mask_at: int | None = None):
        """Per-residue hidden states (L, dim) and log-probabilities (L, vocab)."""
        chars = list(seq)
        if mask_at is not None:
            chars[mask_at] = self.tok.mask_token     # "<mask>" -- one token, not six characters
        enc = self.tok("".join(chars), return_tensors="pt", add_special_tokens=True)
        assert enc["input_ids"].shape[1] == len(seq) + 2, (
            f"tokenised length {enc['input_ids'].shape[1]} != {len(seq)} + 2; the mask token "
            f"did not collapse to a single position and every index below would be shifted"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        n = len(seq)
        rep = out.hidden_states[-1][0, 1:n + 1].float().cpu()
        logp = torch.log_softmax(out.logits[0], -1)[1:n + 1].float().cpu()
        return rep, logp

    def wt(self, key, seq: str):
        if key not in self._cache:
            if len(self._cache) > 200:          # bounded: chains are reused within a complex
                self._cache.clear()
            self._cache[key] = self.forward(seq)
        return self._cache[key]


def build(model_id: str, tokenizer_id: str | None, tag: str, device: str = "auto",
          window: int = WINDOW, verbose: bool = True) -> pd.DataFrame:
    enc = HFEncoder(model_id, tokenizer_id, device)
    df = splits.load()
    structures = load_structures(df["pdb"].unique())
    t0 = time.perf_counter()
    if verbose:
        print(f"{model_id}: hidden={enc.dim} max_len={enc.max_len} device={enc.device}", flush=True)

    rows, vectors = [], []
    for n, r in enumerate(df.itertuples()):
        muts = parse_mutations(r.mutations)
        mutant_seqs = apply_mutations(structures[r.pdb], muts)

        llr_wt = llr_mask = at_norm = win_norm = 0.0
        prof = np.zeros(window, np.float32)
        wide = np.zeros(enc.dim, np.float32)
        # Keep the two halves, not only their difference. Subtracting two sequences that differ
        # in one character mostly encodes *which substitution it is* -- measured at 82% of the
        # difference vector's variance -- and discards what kind of position it happened at.
        wt_vec = np.zeros(enc.dim, np.float32)
        mt_vec = np.zeros(enc.dim, np.float32)

        for m in muts:
            chain = structures[r.pdb].chains[m.chain]
            pos = chain.index[m.key]
            wt_seq, wpos = enc.window_seq(chain.seq, pos)
            mt_seq, _ = enc.window_seq(mutant_seqs[m.chain], pos)

            wrep, wlogp = enc.wt((r.pdb, m.chain, len(wt_seq), wpos), wt_seq)
            mrep, _ = enc.forward(mt_seq)
            _, mlogp = enc.forward(wt_seq, mask_at=wpos)

            llr_wt += float(wlogp[wpos, enc.idx(m.mut)] - wlogp[wpos, enc.idx(m.wt)])
            llr_mask += float(mlogp[wpos, enc.idx(m.mut)] - mlogp[wpos, enc.idx(m.wt)])

            d = (mrep - wrep).numpy()                     # (L, dim) perturbation field
            lo, hi = max(0, wpos - window), min(len(wt_seq), wpos + window + 1)
            at_norm += float(np.linalg.norm(d[wpos]))
            win_norm += float(np.linalg.norm(d[lo:hi], axis=1).mean())
            wide += d[lo:hi].mean(0)
            wt_vec += wrep[wpos].numpy()
            mt_vec += mrep[wpos].numpy()
            for k in range(1, window + 1):
                vals = [np.linalg.norm(d[wpos + s]) for s in (-k, k)
                        if 0 <= wpos + s < len(wt_seq)]
                if vals:
                    prof[k - 1] += float(np.mean(vals))

        prof /= max(len(muts), 1)
        rows.append({
            "row_id": r.row_id,
            "llr_wt": llr_wt,
            "llr_masked": llr_mask,
            "d_at_pos": at_norm,
            "d_window": win_norm,
            "reach": float(win_norm / at_norm) if at_norm > 0 else 0.0,
            **{f"prof{k+1}": float(prof[k]) for k in range(window)},
        })
        vectors.append((wide, wt_vec, mt_vec))

        if verbose and (n + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {n+1:4d}/{len(df)}  {el:6.1f}s  eta {el/(n+1)*(len(df)-n-1):6.1f}s", flush=True)

    scalars = pd.DataFrame(rows).set_index("row_id").astype(np.float32)
    parts = [scalars]
    for j, prefix in enumerate(("d", "wt_d", "mt_d")):
        parts.append(pd.DataFrame(
            np.vstack([v[j] for v in vectors]), index=scalars.index,
            columns=[f"{prefix}{i}" for i in range(enc.dim)]).astype(np.float32))
    block = pd.concat(parts, axis=1)
    out = paths.FEATURES / f"{tag}.parquet"
    block.to_parquet(out)
    if verbose:
        print(f"\nwrote {out.relative_to(paths.ROOT)}  {block.shape}  "
              f"in {time.perf_counter()-t0:.0f}s")
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--tag", required=True, help="cache name, e.g. currab150M")
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=int, default=WINDOW)
    a = p.parse_args()
    build(a.model, a.tokenizer, a.tag, a.device, a.window)


if __name__ == "__main__":
    main()
