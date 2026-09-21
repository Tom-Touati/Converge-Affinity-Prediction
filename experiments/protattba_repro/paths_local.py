"""Paths for the ProtAttBA reproduction, and the upstream checkout it reads code from.

Mirrors the role of ``src/paths.py`` for this experiment, so no script depends on the working
directory it was launched from. Deliberately separate from ``src/paths.py``: nothing under
``src/`` should learn about an external baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Upstream checkout of https://github.com/code4luck/ProtAttBA, pinned at 9adbf98.
#: Gitignored; ``setup_upstream.sh`` clones it. Their head code is imported from here
#: unmodified rather than copied, so there is no chance of a silent edit.
UPSTREAM = HERE / "ProtAttBA"
UPSTREAM_CV = UPSTREAM / "cross_validation"
UPSTREAM_SRC = UPSTREAM_CV / "src_s1131"

#: facebook/esm2_t33_650M_UR50D. Their scripts expect it at ``./model/esm2_650m`` and the repo
#: ships only a readme saying to download it; the checkpoint size is pinned by HIDDEN_SIZE=1280
#: in bash_cross-validation.sh, which is 650M's hidden dim and no other ESM2 size's.
ESM2_DIR = HERE / "model" / "esm2_650m"
ESM2_HF_ID = "facebook/esm2_t33_650M_UR50D"

CACHE = HERE / "cache"
EMBEDDINGS = CACHE / "esm2_650m_embeddings.npy"
EMBEDDINGS_INDEX = CACHE / "esm2_650m_index.pkl"

RESULTS = HERE / "results"


def add_upstream_to_path() -> None:
    """Make ProtAttBA's own modules importable exactly as their bash script arranges.

    ``bash_cross-validation.sh`` exports ``PYTHONPATH=./`` from ``cross_validation/`` and runs
    ``python ./src_s1131/trainer.py``, which puts both that directory and the script's own
    directory on the path -- hence ``from model_module.rope_attn import ...`` alongside
    ``from model import SeqBindModel``. We replicate that, so their imports resolve untouched.
    """
    if not UPSTREAM.exists():
        raise FileNotFoundError(
            f"{UPSTREAM} not found. Run setup_upstream.sh first (see README)."
        )
    for p in (UPSTREAM_CV, UPSTREAM_SRC):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
