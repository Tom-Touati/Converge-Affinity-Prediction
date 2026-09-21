"""Download the frozen encoder ProtAttBA uses, into the layout their scripts expect.

The upstream repo ships ``model/readme.txt`` reading "you can download the pretrain model
weights from hugging face and put in here" and never names the checkpoint. Neither does the
paper. It is pinned by two independent things in their own config:

* ``bash_cross-validation.sh`` sets ``MODEL_LOCATE="./model/esm2_650m"``.
* the same script sets ``HIDDEN_SIZE=1280``, and ``SeqBindModel`` wires that straight into every
  head dimension. Of the ESM2 family only the 650M checkpoint has hidden size 1280 -- 35M is
  480, 150M is 640, 3B is 2560 -- so a different size would not even build.

Hence ``facebook/esm2_t33_650M_UR50D``. ``check_encoder_assumptions.py`` asserts the
downloaded config's hidden size is 1280, so a wrong checkpoint fails loudly rather than
quietly changing the result.

Run: ``python fetch_esm2.py``   (~2.6 GB, skipped if already present)
"""
from __future__ import annotations

import time

from huggingface_hub import snapshot_download

from paths_local import ESM2_DIR, ESM2_HF_ID

#: pytorch_model.bin is the same weights again; fetching only safetensors halves the download.
ALLOW = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json",
         "model.safetensors"]


def main() -> None:
    if (ESM2_DIR / "config.json").exists() and (ESM2_DIR / "model.safetensors").exists():
        print(f"already present: {ESM2_DIR}")
        return
    ESM2_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ESM2_HF_ID} -> {ESM2_DIR}")
    t0 = time.time()
    snapshot_download(
        repo_id=ESM2_HF_ID,
        local_dir=str(ESM2_DIR),
        allow_patterns=ALLOW,
        local_dir_use_symlinks=False,
    )
    total = sum(f.stat().st_size for f in ESM2_DIR.rglob("*") if f.is_file())
    print(f"done in {time.time() - t0:.0f}s, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
