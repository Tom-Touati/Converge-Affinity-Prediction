"""Sequence modality: frozen ESM-2 features, cached per row_id.

Two families of feature, because they answer different questions:

* **Log-likelihood ratios.** log p(mutant residue) - log p(wild-type residue) at the mutated
  position. This is the standard zero-shot variant score, and it is a single number the head
  cannot overfit. Both variants are computed: *wt-marginal* reads the logits from one unmasked
  pass over the wild-type chain and is nearly free (one pass per chain), while *masked-marginal*
  masks the position first, which is the better-calibrated scorer and costs one pass per unique
  position.
* **Embedding difference at the mutated position.** The residue-level representation of the
  mutant sequence minus that of the wild type, at the position that changed. Whole-sequence
  pooling alone washes out a single-residue change, which is why the difference is taken
  position-wise and the pooled version is kept only as a scalar norm.

Multi-point mutations sum their per-position contributions. The SKEMPI double-mutant cycles say
this is only half true (345 additive against 421 context-dependent), so the additivity probe
tests it rather than assuming it.

Only the chain carrying the mutation is embedded. ESM-2 is a single-chain model: feeding it a
concatenated complex would invent a covalent link that does not exist. Cross-chain context is
the structure modality's job.

    python -m src.features.esm2 --model esm2_t12_35M_UR50D --device auto
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..structures import apply_mutations, load_structures, parse_mutations
from ..util import resolve_device

MODELS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}
MAX_LEN = 1022  # ESM-2 positional budget, minus BOS/EOS


def _window(seq: str, focus: int) -> tuple:
    """Truncate a long chain around the mutated position; returns (subseq, new index)."""
    if len(seq) <= MAX_LEN:
        return seq, focus
    half = MAX_LEN // 2
    start = max(0, min(focus - half, len(seq) - MAX_LEN))
    return seq[start:start + MAX_LEN], focus - start


class Encoder:
    def __init__(self, model_name: str, device: str = "auto"):
        import esm as esm_lib

        self.name = model_name
        self.layer = MODELS[model_name]
        self.device = resolve_device(device)
        self.model, self.alphabet = getattr(esm_lib.pretrained, model_name)()
        self.model = self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)          # frozen: 997 rows cannot fine-tune a PLM
        self.bc = self.alphabet.get_batch_converter()
        self.dim = self.model.embed_dim
        self._cache = {}

    @torch.no_grad()
    def forward(self, seq: str, mask_at: int = -1):
        """Return (per-residue representations [L, D], per-residue log-probs [L, V])."""
        _, _, toks = self.bc([("x", seq)])
        if mask_at >= 0:
            toks[0, mask_at + 1] = self.alphabet.mask_idx   # +1 skips BOS
        toks = toks.to(self.device)
        out = self.model(toks, repr_layers=[self.layer], return_contacts=False)
        rep = out["representations"][self.layer][0, 1:len(seq) + 1].float().cpu()
        logp = torch.log_softmax(out["logits"][0, 1:len(seq) + 1].float(), dim=-1).cpu()
        return rep, logp

    def wt_chain(self, key: tuple, seq: str):
        """Wild-type pass, memoised: one forward per chain rather than one per mutation."""
        if key not in self._cache:
            self._cache[key] = self.forward(seq)
        return self._cache[key]

    def idx(self, aa: str) -> int:
        return self.alphabet.get_idx(aa)


def extract(model_name: str, device: str = "auto", masked: bool = True,
            verbose: bool = True) -> pd.DataFrame:
    paths.ensure_dirs()
    df = pd.read_parquet(paths.DATASET)
    structures = load_structures(df["pdb"].unique())
    enc = Encoder(model_name, device)
    if verbose:
        print(f"{model_name}: dim={enc.dim} layer={enc.layer} device={enc.device}  "
              f"rows={len(df)}", flush=True)

    t0 = time.perf_counter()
    rows, vectors = [], []
    for n, (row_id, pdb, mutations) in enumerate(
        zip(df["row_id"], df["pdb"], df["mutations"])
    ):
        st = structures[pdb]
        muts = parse_mutations(mutations)
        mutant_seqs = apply_mutations(st, muts)

        d_sum = np.zeros(enc.dim, np.float32)
        llr_wt = llr_mask = cos_sum = 0.0
        pooled = np.zeros(enc.dim, np.float32)

        for m in muts:
            chain = st.chains[m.chain]
            pos = chain.index[m.key]
            wt_seq, wpos = _window(chain.seq, pos)
            mt_seq, _ = _window(mutant_seqs[m.chain], pos)

            wt_rep, wt_logp = enc.wt_chain((pdb, m.chain, wt_seq), wt_seq)
            mt_rep, _ = enc.forward(mt_seq)

            d = (mt_rep[wpos] - wt_rep[wpos]).numpy()
            d_sum += d
            pooled += (mt_rep.mean(0) - wt_rep.mean(0)).numpy()
            cos_sum += float(
                torch.nn.functional.cosine_similarity(mt_rep[wpos], wt_rep[wpos], dim=0)
            )
            llr_wt += float(wt_logp[wpos, enc.idx(m.mut)] - wt_logp[wpos, enc.idx(m.wt)])

            if masked:
                _, mlogp = enc.forward(wt_seq, mask_at=wpos)
                llr_mask += float(mlogp[wpos, enc.idx(m.mut)] - mlogp[wpos, enc.idx(m.wt)])

        rows.append({
            "row_id": row_id,
            "llr_wt": llr_wt,
            "llr_masked": llr_mask if masked else np.nan,
            "diff_norm": float(np.linalg.norm(d_sum)),
            "diff_cos": cos_sum / len(muts),
            "pooled_norm": float(np.linalg.norm(pooled)),
        })
        vectors.append(d_sum)

        if verbose and (n + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {n+1:4d}/{len(df)}  {el:6.1f}s  "
                  f"eta {el/(n+1)*(len(df)-n-1):6.1f}s", flush=True)

    scalars = pd.DataFrame(rows).set_index("row_id")
    vec = pd.DataFrame(np.vstack(vectors), index=scalars.index,
                       columns=[f"d{i}" for i in range(enc.dim)]).astype(np.float32)
    block = pd.concat([scalars.astype(np.float32), vec], axis=1)

    tag = model_name.replace("esm2_", "").replace("_UR50D", "")
    out = paths.FEATURES / f"esm2_{tag}.parquet"
    block.to_parquet(out)
    if verbose:
        el = time.perf_counter() - t0
        print(f"wrote {out.relative_to(paths.ROOT)}  {block.shape}  in {el:.1f}s "
              f"({el/len(df):.2f}s/row)")
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="esm2_t12_35M_UR50D", choices=sorted(MODELS))
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--no-masked", action="store_true",
                   help="skip masked-marginal LLR (halves the forward passes)")
    a = p.parse_args()
    extract(a.model, a.device, masked=not a.no_masked)


if __name__ == "__main__":
    main()
