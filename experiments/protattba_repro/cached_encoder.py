"""A drop-in stand-in for ``EsmModel`` that serves cached embeddings.

``SeqBindModel.__init__`` does ``self.encoder = EsmModel.from_pretrained(args.model_locate)``
and its ``forward`` does ``self.encoder(input_ids=..., attention_mask=...).last_hidden_state``.
Matching those two signatures is enough to run their model file with the encoder lifted out of
the loop, with **no edit to any upstream source file** -- ``run_cv.py`` rebinds the name
``EsmModel`` inside their already-imported ``model`` module and their code is otherwise
untouched.

Two design points worth stating, because they are what make this exact rather than approximate:

* **Lookup is keyed on token ids, not on sequence strings.** The encoder only ever sees the
  ``input_ids`` their collator produced. Keying on those means the cache cannot be silently
  misaligned by our own sequence bookkeeping disagreeing with theirs, and a miss raises.

* **Padding rows are filled with zeros, which is provably what their model already sees.**
  Every consumer of the encoder output is an ``AttnTransform``, whose softmax is
  ``masked_fill_(~mask, -inf)`` and therefore exactly 0 at pad positions; ``attn * x`` then
  zeroes those rows outright whatever the encoder put there. So the real encoder's pad-position
  activations cannot reach the head, and omitting them changes nothing.

  Note the separate, *non*-cosmetic consequence of their masking: inside
  ``MutilHeadSelfAttn.self_attn`` the key mask is applied as ``masked_fill(mask == 0, 1e-10)``
  rather than ``-inf``, so pad keys keep a real share of the softmax mass. Their prediction for
  an example therefore depends on how wide its batch happened to be padded. That is upstream
  behaviour, not something the cache introduces, and it is why ``run_cv.py`` reproduces their
  batch size and iteration order exactly rather than picking convenient ones.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from paths_local import EMBEDDINGS, EMBEDDINGS_INDEX


@dataclass
class EncoderOutput:
    """Just enough of ``BaseModelOutputWithPooling`` for their two attribute accesses."""

    last_hidden_state: torch.Tensor


#: Set PROTATTBA_CACHE_DEVICE=cpu to force the memmap path. Default is to move the whole cache
#: onto the training device when there is room, because the memmap path is the bottleneck:
#: every step reads four [B, L, 1280] blocks out of a 0.66 GB file and copies them over PCIe,
#: which is ~55 MB of host-to-device traffic per step and dwarfs the head's own arithmetic.
#: Resident is a pure throughput change -- same values, same order, same results.
CACHE_DEVICE_ENV = "PROTATTBA_CACHE_DEVICE"


class CachedEsmEncoder(nn.Module):
    """Serves ``extract_embeddings.py``'s cache. Holds no parameters, so it trains nothing."""

    def __init__(self, store, index: dict[bytes, tuple[int, int]], hidden: int):
        super().__init__()
        # Deliberately not a buffer or a parameter: it must stay out of state_dict so their
        # ModelCheckpoint keeps writing head-only checkpoints, and out of .to(device) so
        # Lightning does not try to move 0.66 GB on every hook.
        self._store = store
        self._index = index
        self.hidden = hidden
        self._resident = torch.is_tensor(store)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> "CachedEsmEncoder":
        """Signature-compatible with ``EsmModel.from_pretrained``; the path is ignored.

        The path is ignored on purpose: the cache was built by ``extract_embeddings.py`` from
        the checkpoint at ``paths_local.ESM2_DIR``, and that provenance is asserted there
        (hidden size 1280) rather than re-derived here.
        """
        store = np.load(EMBEDDINGS, mmap_mode="r")
        with open(EMBEDDINGS_INDEX, "rb") as f:
            meta = pickle.load(f)
        if store.shape != (meta["total_tokens"], meta["hidden"]):
            raise RuntimeError(
                f"embedding cache shape {store.shape} disagrees with its index "
                f"{(meta['total_tokens'], meta['hidden'])}; re-run extract_embeddings.py"
            )

        want = os.environ.get(CACHE_DEVICE_ENV, "auto")
        nbytes = store.shape[0] * store.shape[1] * 4
        if want == "auto":
            want = "cuda" if torch.cuda.is_available() else "cpu"
        if want.startswith("cuda") and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            # Leave room for the head, its AdamW state and the activations.
            if free > nbytes + 3e9:
                store = torch.from_numpy(np.ascontiguousarray(store)).to(want)
                print(f"[cache] {nbytes / 1e9:.2f} GB of embeddings resident on {want}")
            else:
                print(f"[cache] only {free / 1e9:.2f} GB free on {want}; keeping the memmap "
                      f"on the host (this is much slower per step)")
        return cls(store, meta["index"], meta["hidden"])

    def _locate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Per-row (offset, length) from the token ids. Small CPU hop, ~100 kB per batch."""
        ids = input_ids.detach().to("cpu", torch.int32).numpy()
        lengths = attention_mask.detach().to("cpu").sum(dim=1).numpy().astype(int)
        spans = []
        for row in range(ids.shape[0]):
            n = int(lengths[row])
            hit = self._index.get(ids[row, :n].tobytes())
            if hit is None:
                raise KeyError(
                    f"embedding cache miss for a {n}-token sequence in batch row {row}. "
                    "The cache was built from the dataset's four sequence columns; a miss "
                    "means a sequence reached the model that extraction never saw."
                )
            offset, cached_len = hit
            if cached_len != n:
                raise RuntimeError(f"cached length {cached_len} != masked length {n}")
            spans.append((offset, n))
        return spans

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput:
        spans = self._locate(input_ids, attention_mask)
        batch, width = input_ids.shape

        if self._resident:
            out = torch.zeros(batch, width, self.hidden, dtype=torch.float32,
                              device=self._store.device)
            for row, (offset, n) in enumerate(spans):
                out[row, :n] = self._store[offset:offset + n]
            return EncoderOutput(out.to(input_ids.device))

        host = np.zeros((batch, width, self.hidden), dtype=np.float32)
        for row, (offset, n) in enumerate(spans):
            host[row, :n] = self._store[offset:offset + n]
        return EncoderOutput(
            torch.from_numpy(host).to(device=input_ids.device, dtype=torch.float32)
        )
