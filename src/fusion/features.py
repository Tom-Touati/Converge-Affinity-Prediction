"""Feature matrices for the E0 baselines, assembled from the caches that already exist.

Reads ``data/features/*.parquet`` and never writes them (plan rule 9). Every block is indexed
by ``row_id``, which is ``#Pdb|mutations`` and therefore content-based, so the 997-row caches
built before the censored-affinity drop still serve the current 940 rows.

One deviation from the plan, recorded here and in ``docs/decisions.md``:

    The plan's E0b asks for "mean-pooled ESM per chain, WT and mutant, concatenated, plus the
    ESM delta at the mutation site". The cache stores a **whole-sequence** pooled vector per
    branch (``wt_d0..1279``, ``mt_d0..1279``) and a pooled difference (``d0..1279``), not a
    per-chain pooling, because ``src/features/esm2.py`` deliberately embeds only the chain
    carrying the mutation -- ESM-2 is a single-chain model and concatenating a complex would
    invent a covalent bond. Re-extracting per-chain pooled vectors would mean re-running the
    650M encoder over the dataset, which is a Colab job, not a local one. E0b therefore uses
    the pooled WT and MT vectors plus the site scalars that are cached, and the per-chain
    variant is left to the per-residue extraction that E1-E5 need anyway.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import paths

#: the wide embedding columns, which some views drop
_VEC = re.compile(r"^(wt_d|mt_d|d|hc)\d+$")


def _block(name: str) -> pd.DataFrame:
    path = paths.FEATURES / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Cached blocks: "
            f"{sorted(p.stem for p in paths.FEATURES.glob('*.parquet'))}"
        )
    return pd.read_parquet(path)


def _take(block: pd.DataFrame, row_ids: pd.Series, prefix: str) -> pd.DataFrame:
    missing = set(row_ids) - set(block.index)
    if missing:
        raise RuntimeError(
            f"block '{prefix}' is missing {len(missing)} row_ids, e.g. {sorted(missing)[:3]}"
        )
    out = block.loc[row_ids].copy()
    out.columns = [f"{prefix}__{c}" for c in out.columns]
    return out.reset_index(drop=True)


#: The three E0 feature sets. E0c is the "multimodal baseline everything else must beat".
FEATURE_SETS: dict[str, list[str]] = {
    # E0a: the project's existing best tree feature set, unchanged.
    "handcrafted": ["chem", "geom", "geomrev", "mpnn"],
    # E0b: sequence only -- pooled ESM-2 650M, WT and mutant, plus the site scalars.
    "pooled_esm": ["esm650M_pair"],
    # E0c: E0b plus pooled ProteinMPNN representation and its likelihood ratios.
    "pooled_esm_mpnn": ["esm650M_pair", "mpnnrep", "mpnn"],
}


def build(row_ids: pd.Series, feature_set: str, drop_vectors: bool = False) -> pd.DataFrame:
    """Assemble one feature set for the given rows, in a stable column order.

    ``drop_vectors`` keeps only the summary scalars, which is the view a tree head wants when
    the alternative is 3,840 embedding columns against 752 training rows.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}; have {sorted(FEATURE_SETS)}")
    parts = [_take(_block(name), row_ids, name) for name in FEATURE_SETS[feature_set]]
    X = pd.concat(parts, axis=1)
    if drop_vectors:
        X = X[[c for c in X.columns if not _VEC.match(c.split("__", 1)[1])]]
    # A constant column carries no information and upsets some estimators; drop and report.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    return X.astype(np.float32).fillna(0.0)
