"""Justify lifting the frozen ESM2 encoder out of ProtAttBA's training loop.

ProtAttBA runs ``EsmModel`` inside ``SeqBindModel.forward``, four times per example, on every
step of every epoch. With ``--freeze_backbone`` that recomputes a constant. On a 4-core laptop
CPU the encoder alone costs roughly 4 hours per epoch over S1131's 711k residues, against
~1700 hours for the full 10-fold run, so running their file literally as written is not a
bounded debugging problem -- it is arithmetically out of reach on this hardware. Caching the
encoder output once costs about 45 minutes and leaves their head code untouched.

That substitution is only faithful if the encoder output is a pure function of one sequence.
Three things have to hold, and this script checks all three against the actual downloaded
checkpoint rather than assuming them:

1. **No stochasticity in train mode.** ``SeqBindModel`` holds the encoder as a submodule, so
   ``model.train()`` puts it in train mode too. If ESM2's dropout probabilities were non-zero
   the cached (eval-mode) embedding would differ from what their loop sees. ESM2 ships
   ``hidden_dropout_prob = attention_probs_dropout_prob = 0.0``, which we verify, and then
   confirm empirically that two train-mode passes agree bit-for-bit.

2. **No cross-example coupling.** ESM2 normalises with LayerNorm only -- no BatchNorm -- so one
   row's embedding cannot depend on its batch-mates. Checked by scanning the module tree.

3. **Padding invariance.** Their collator pads to the longest sequence in the batch, so the same
   sequence appears with different padding widths across epochs. Real-token outputs must be
   unaffected. ESM2 uses rotary position embeddings plus an additive attention mask, so this is
   expected to hold; we measure it, because "expected to hold" is how reproductions go wrong.

Run: ``python check_encoder_assumptions.py``
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from metrics import S1131_CSV
from paths_local import ESM2_DIR


def main() -> None:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(ESM2_DIR))
    model = EsmModel.from_pretrained(str(ESM2_DIR))
    cfg = model.config

    print(f"checkpoint                {ESM2_DIR.name}")
    print(f"num_hidden_layers         {cfg.num_hidden_layers}")
    print(f"hidden_size               {cfg.hidden_size}   "
          f"(bash_cross-validation.sh sets HIDDEN_SIZE=1280)")
    print(f"position_embedding_type   {cfg.position_embedding_type}")
    print(f"parameters                {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    # ---- 1. dropout is off -------------------------------------------------------------
    print("\n[1] stochasticity in train mode")
    print(f"    hidden_dropout_prob            {cfg.hidden_dropout_prob}")
    print(f"    attention_probs_dropout_prob   {cfg.attention_probs_dropout_prob}")
    assert cfg.hidden_dropout_prob == 0.0, "ESM2 hidden dropout is non-zero"
    assert cfg.attention_probs_dropout_prob == 0.0, "ESM2 attention dropout is non-zero"

    import pandas as pd

    # Spread of lengths on purpose: the first rows of column `a` are all one PDB, so taking
    # them in order pads 191 -> 193 and barely tests anything. These four span 4.7x in length,
    # so the shortest is padded to nearly five times its own width.
    pool = sorted(set(pd.read_csv(S1131_CSV).a.astype(str)), key=len)
    seqs = [pool[0], pool[len(pool) // 3], pool[2 * len(pool) // 3], pool[-1]]
    enc = tok(seqs[:1], return_tensors="pt")

    model.train()
    with torch.no_grad():
        a = model(**enc).last_hidden_state
        b = model(**enc).last_hidden_state
    print(f"    two train-mode passes, max |diff|   {float((a - b).abs().max()):.3e}")
    assert torch.equal(a, b), "train-mode encoder is stochastic"

    model.eval()
    with torch.no_grad():
        c = model(**enc).last_hidden_state
    print(f"    train mode vs eval mode, max |diff| {float((a - c).abs().max()):.3e}")
    assert torch.equal(a, c), "train mode and eval mode disagree"

    # ---- 2. no batch-coupled normalisation ---------------------------------------------
    batchnorms = [n for n, m in model.named_modules()
                  if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                    torch.nn.SyncBatchNorm))]
    print(f"\n[2] cross-example coupling")
    print(f"    BatchNorm modules in the encoder    {len(batchnorms)}")
    assert not batchnorms, f"encoder contains batch-coupled normalisation: {batchnorms}"

    # ---- 3. padding invariance ---------------------------------------------------------
    print("\n[3] padding invariance (their collator pads to the batch maximum)")
    padded = tok(seqs, padding=True, return_tensors="pt")
    with torch.no_grad():
        batch_out = model(**padded).last_hidden_state

    worst = 0.0
    for i, s in enumerate(seqs):
        single = tok([s], return_tensors="pt")
        with torch.no_grad():
            solo = model(**single).last_hidden_state[0]
        n = int(padded["attention_mask"][i].sum())
        diff = float((batch_out[i, :n] - solo).abs().max())
        worst = max(worst, diff)
        print(f"    seq {i} len {len(s):4d} padded to {padded['input_ids'].shape[1]:4d}   "
              f"max |diff| on real tokens {diff:.3e}")

    # fp32 attention over a 4x wider key set reorders float additions, so this is float noise
    # rather than an exact match. 1e-4 is far below the scale of the embeddings themselves.
    scale = float(batch_out.abs().mean())
    print(f"    mean |embedding| {scale:.3f}, so worst relative drift is {worst / scale:.1e}")
    assert worst < 1e-3, f"padding changes real-token embeddings by {worst}"

    print(
        "\nConclusion: with --freeze_backbone the encoder output is a deterministic,\n"
        "padding-invariant function of a single sequence. Caching it once and running\n"
        "ProtAttBA's own head code on the cache is equivalent to their loop, up to float\n"
        "reassociation noise of order 1e-4 in the embeddings."
    )


if __name__ == "__main__":
    main()
