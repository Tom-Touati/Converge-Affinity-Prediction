"""Per-residue ESM-2 650M tokens for our antibody-antigen rows. The missing prerequisite.

The mission assumes these are cached. They are not: every block under ``data/features/`` is
pooled or scalar, one row per mutation, and the only per-residue ESM cache in the repo is for
S1131 -- a different dataset with no antibody-antigen complexes in it.

Four sequences per row (antibody wild type and mutant, antigen wild type and mutant) come from
``experiments/protattba_repro/cache/project_sequences.parquet``, which was built from the PDB
chains with the project's own parser and validated: 940 rows, 884 distinct sequences, 335,261
residues, and every row's mutation changes at least one side.

Deliberate choices:

* **Keyed on token ids, not on sequence strings.** A consumer only has the tokenised input, so
  keying on that makes the lookup independent of any bookkeeping agreeing with ours, and a miss
  raises instead of silently re-encoding.
* **fp16 on disk.** 335k residues at 1280 dimensions is 858 MB in fp32 and half that in fp16.
  The values are order 1 and everything downstream reduces them through PCA-256, so half
  precision costs nothing that survives the projection.
* **No truncation.** 34 sequences exceed ESM2's nominal 1022 positions; rotary embeddings
  extrapolate and were measured working to 1492 tokens, so windowing would lose real residues
  for no reason.

Run: ``python -m src.perturb.esm_tokens [--device auto|cpu|cuda]``
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from src import paths

SEQ_TABLE = (paths.ROOT / "experiments" / "protattba_repro" / "cache"
             / "project_sequences.parquet")
ESM_DIR = (paths.ROOT / "experiments" / "protattba_repro" / "model" / "esm2_650m")

CACHE = paths.FEATURES / "esm_per_residue"
EMB = CACHE / "esm2_650m_tokens.npy"
INDEX = CACHE / "esm2_650m_index.pkl"

SEQ_COLUMNS = ("ab_wt", "ag_wt", "ab_mt", "ag_mt")


def distinct_sequences(frame: pd.DataFrame) -> list[str]:
    """Longest first, so the worst-case batch is the first one and OOM happens immediately."""
    seqs: set[str] = set()
    for col in SEQ_COLUMNS:
        seqs.update(s for s in frame[col].astype(str) if s and s.lower() != "nan")
    return sorted(seqs, key=lambda s: (-len(s), s))


def build(device: str = "auto", batch_tokens: int = 8192, verbose: bool = True) -> int:
    import torch
    from transformers import AutoTokenizer, EsmModel

    from src.util import resolve_device

    if not SEQ_TABLE.exists():
        raise SystemExit(
            f"{SEQ_TABLE} not found -- build it first with "
            "experiments/protattba_repro/build_project_sequences.py"
        )
    frame = pd.read_parquet(SEQ_TABLE)
    seqs = distinct_sequences(frame)
    device = resolve_device(device)
    if verbose:
        print(f"{len(frame)} rows, {len(seqs)} distinct sequences, "
              f"{sum(len(s) for s in seqs)} residues, device {device}")

    tok = AutoTokenizer.from_pretrained(str(ESM_DIR))
    model = EsmModel.from_pretrained(str(ESM_DIR)).to(device).eval()
    hidden = model.config.hidden_size
    assert hidden == 1280, f"expected ESM2-650M (1280), got {hidden}"

    ids = [tok(s, return_tensors="np")["input_ids"][0].astype(np.int32) for s in seqs]
    lengths = np.array([len(t) for t in ids])
    total = int(lengths.sum())
    if verbose:
        print(f"{total} tokens including <cls>/<eos>; cache is "
              f"{total * hidden * 2 / 1e9:.2f} GB fp16")

    CACHE.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(EMB, mode="w+", dtype=np.float16,
                                      shape=(total, hidden))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    pad = tok.pad_token_id
    t0, done, i, nb = time.time(), 0, 0, 0
    while i < len(seqs):
        size = max(1, batch_tokens // int(lengths[i]))
        j = min(i + size, len(seqs))
        chunk = ids[i:j]
        width = max(len(c) for c in chunk)
        x = np.full((len(chunk), width), pad, dtype=np.int64)
        m = np.zeros((len(chunk), width), dtype=np.int64)
        for r, c in enumerate(chunk):
            x[r, : len(c)] = c
            m[r, : len(c)] = 1
        with torch.no_grad():
            out = model(input_ids=torch.from_numpy(x).to(device),
                        attention_mask=torch.from_numpy(m).to(device)
                        ).last_hidden_state.half().cpu().numpy()
        for r, c in enumerate(chunk):
            store[offsets[i + r]: offsets[i + r] + len(c)] = out[r, : len(c)]
        done += int(lengths[i:j].sum())
        i, nb = j, nb + 1
        if verbose and (nb % 10 == 0 or i == len(seqs)):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"  {i:5d}/{len(seqs)} seqs  {done:7d}/{total} tokens  "
                  f"{el/60:5.1f} min, {rate:6.0f} tok/s, "
                  f"{(total-done)/max(rate,1e-9)/60:5.1f} min left", flush=True)

    store.flush()
    index = {ids[k].tobytes(): (int(offsets[k]), int(lengths[k])) for k in range(len(seqs))}
    assert len(index) == len(seqs), "two distinct sequences tokenised identically"
    with open(INDEX, "wb") as f:
        pickle.dump({"index": index, "hidden": hidden, "total_tokens": total,
                     "checkpoint": "facebook/esm2_t33_650M_UR50D"}, f)
    if verbose:
        print(f"wrote {EMB.name} and {INDEX.name} in {(time.time()-t0)/60:.1f} min")
    return total


def load():
    """(memmapped fp16 store, {token_id_bytes: (offset, length)}, hidden)."""
    if not INDEX.exists():
        raise FileNotFoundError(f"{INDEX} not found -- run: python -m src.perturb.esm_tokens")
    with open(INDEX, "rb") as f:
        meta = pickle.load(f)
    return np.load(EMB, mmap_mode="r"), meta["index"], meta["hidden"]


def tokens_for(sequence: str, tokenizer, store, index) -> np.ndarray:
    """(L+2, 1280) fp32 for one sequence, including <cls>/<eos> as ProtAttBA's head sees them."""
    key = tokenizer(sequence, return_tensors="np")["input_ids"][0].astype(np.int32).tobytes()
    hit = index.get(key)
    if hit is None:
        raise KeyError(f"no cached tokens for a {len(sequence)}-residue sequence")
    off, n = hit
    return np.asarray(store[off: off + n], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    args = ap.parse_args()
    build(device=args.device, batch_tokens=args.batch_tokens)


if __name__ == "__main__":
    main()
