"""Cache ESM2-650M per-residue embeddings for every distinct sequence in S1131.

This is the whole reason the reproduction fits on a laptop. S1131's four sequence columns
(``a``, ``b``, ``a_mut``, ``b_mut``) hold 4524 cells but only **1021 distinct sequences**,
because 112 PDB ids share wild-type chains and a single-point mutation leaves the partner chain
untouched -- ``b_mut == b`` for 894 of the 1131 rows. So the frozen encoder has 127k residues of
real work to do, not 711k, and it only has to do them once instead of once per epoch per fold.

Their loop recomputes all of it on every step: 711k residues x 4 encoder calls x ~40 epochs x
10 folds. Caching turns roughly 1700 CPU-hours into roughly 30 minutes.

Output is a single fp32 memmap plus an index keyed by *token ids*, not by sequence string. That
matters: at training time ``CachedEsmEncoder`` only sees the ``input_ids`` their collator
produced, so keying on the token ids makes the lookup independent of any bookkeeping of ours
agreeing with theirs. A key miss is a hard error, never a silent re-encode.

fp32 rather than fp16 throughout: their run is fp32 (``--precision`` is passed but the Trainer
line that consumes it is commented out upstream), the cache is only 660 MB, and rounding the
encoder output would put a floor under how closely we can match their per-example predictions.

Run: ``python extract_embeddings.py [--device cpu|cuda|auto] [--batch-tokens 8192]``
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

from metrics import S1131_CSV
from paths_local import CACHE, EMBEDDINGS, EMBEDDINGS_INDEX, ESM2_DIR

SEQ_COLUMNS = ("a", "b", "a_mut", "b_mut")


def distinct_sequences(csv_path=S1131_CSV) -> list[str]:
    """Every distinct sequence across the four columns, longest first.

    Longest-first so the first batch is the worst case: if it fits, the run will not die of
    memory 40 minutes in.
    """
    df = pd.read_csv(csv_path)
    seqs = set()
    for col in SEQ_COLUMNS:
        seqs.update(df[col].astype(str).tolist())
    return sorted(seqs, key=lambda s: (-len(s), s))


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        # ESM2-650M is 2.6 GB in fp32 and this box has ~1.2 GB of VRAM free, so extraction runs
        # on the CPU even though head training does not. Kept as a flag so a real GPU just works.
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free > 4e9:
                return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-tokens", type=int, default=8192,
                    help="pad-inclusive token budget per forward pass")
    args = ap.parse_args()

    device = pick_device(args.device)
    seqs = distinct_sequences()
    residues = sum(len(s) for s in seqs)
    print(f"{len(seqs)} distinct sequences, {residues} residues "
          f"(vs {len(pd.read_csv(S1131_CSV)) * 4} sequence cells in the csv)")
    print(f"device {device}")

    tok = AutoTokenizer.from_pretrained(str(ESM2_DIR))
    model = EsmModel.from_pretrained(str(ESM2_DIR)).to(device).eval()
    hidden = model.config.hidden_size
    assert hidden == 1280, (
        f"hidden size {hidden} != 1280; wrong ESM2 checkpoint for HIDDEN_SIZE=1280"
    )

    # Tokenise once up front so the exact output length is known and the memmap can be sized
    # rather than grown. +2 per sequence for ESM2's <cls>/<eos>, which their code feeds to the
    # head along with the residues (the attention mask covers them), so we cache them too.
    token_ids = [tok(s, return_tensors="np")["input_ids"][0].astype(np.int32) for s in seqs]
    lengths = np.array([len(t) for t in token_ids])
    total = int(lengths.sum())
    print(f"{total} tokens including <cls>/<eos>; cache will be "
          f"{total * hidden * 4 / 1e9:.2f} GB fp32")

    CACHE.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(
        EMBEDDINGS, mode="w+", dtype=np.float32, shape=(total, hidden)
    )
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    pad_id = tok.pad_token_id
    t0 = time.time()
    done_tokens = 0
    i = 0
    n_batches = 0
    while i < len(seqs):
        # length-sorted input, so a batch is near-rectangular and padding waste is tiny
        longest = lengths[i]
        size = max(1, args.batch_tokens // int(longest))
        j = min(i + size, len(seqs))
        chunk = token_ids[i:j]
        width = max(len(c) for c in chunk)

        ids = np.full((len(chunk), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(chunk), width), dtype=np.int64)
        for r, c in enumerate(chunk):
            ids[r, : len(c)] = c
            mask[r, : len(c)] = 1

        with torch.no_grad():
            out = model(
                input_ids=torch.from_numpy(ids).to(device),
                attention_mask=torch.from_numpy(mask).to(device),
            ).last_hidden_state.float().cpu().numpy()

        for r, c in enumerate(chunk):
            store[offsets[i + r] : offsets[i + r] + len(c)] = out[r, : len(c)]

        done_tokens += int(lengths[i:j].sum())
        i = j
        n_batches += 1
        if n_batches % 10 == 0 or i == len(seqs):
            el = time.time() - t0
            rate = done_tokens / el
            print(f"  {i:5d}/{len(seqs)} seqs  {done_tokens:7d}/{total} tokens  "
                  f"{el / 60:5.1f} min elapsed, {rate:6.0f} tok/s, "
                  f"{(total - done_tokens) / rate / 60:5.1f} min left", flush=True)

    store.flush()
    index = {
        token_ids[k].tobytes(): (int(offsets[k]), int(lengths[k])) for k in range(len(seqs))
    }
    assert len(index) == len(seqs), "two distinct sequences tokenised identically"
    with open(EMBEDDINGS_INDEX, "wb") as f:
        pickle.dump({"index": index, "hidden": hidden, "total_tokens": total}, f)

    print(f"\nwrote {EMBEDDINGS.name} ({total * hidden * 4 / 1e9:.2f} GB) and "
          f"{EMBEDDINGS_INDEX.name} ({len(index)} keys) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
