#!/usr/bin/env bash
# Build the ESM-2 650M token cache ON the instance rather than shipping it.
#
# The cache is 823 MB. Uploading it from a laptop ran at about 0.5 MB/s -- roughly half an
# hour -- while a T4 regenerates it in about four minutes at 2,000 tokens/s over the 448,772
# tokens this dataset needs. The weights come from HuggingFace at datacentre speed, so the
# whole thing is faster than the transfer it replaces by a wide margin.
#
# It is also the more honest artefact: the cache is derived data, and deriving it here means
# the instance holds exactly what this code produces rather than whatever happened to be on
# the laptop.
set -uo pipefail
export PERTURB_ROOT=${PERTURB_ROOT:-/home/ubuntu/perturb}
V=/home/ubuntu/venv/bin
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "torch: $($V/python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"

if [ ! -f "$PERTURB_ROOT/model/esm2_650m/config.json" ]; then
  say "downloading ESM-2 650M weights"
  $V/python -m pip install -q huggingface_hub >/dev/null 2>&1
  $V/python - <<'PY'
import os
from huggingface_hub import snapshot_download
root = os.environ.get("PERTURB_ROOT", "/home/ubuntu/perturb")
# allow_patterns keeps this to what EsmModel actually loads: the TensorFlow and ONNX copies
# in that repo are several more gigabytes and are never read.
p = snapshot_download("facebook/esm2_t33_650M_UR50D",
                      local_dir=f"{root}/model/esm2_650m",
                      allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"])
print("weights at", p)
PY
else
  say "weights already present"
fi

# Check the TOKEN STORE, not the index. The index is 1.3 MB and was shipped with the
# small files, so guarding on it reported "already built" for a cache whose 823 MB of
# actual embeddings were not there.
if [ -s "$PERTURB_ROOT/esm2_650m_tokens.npy" ]; then
  say "token cache already built"
else
  say "extracting per-residue embeddings"
  cd "$PERTURB_ROOT" && $V/python _esm_colab.py
fi

say "cache: $(du -sh "$PERTURB_ROOT/esm2_650m_tokens.npy" 2>/dev/null | cut -f1 || echo missing)"
say "ESM DONE"
