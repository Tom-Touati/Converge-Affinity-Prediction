"""Weighted sampling over the union of datasets, with our own data drawn twice as often.

When a model trains on SKEMPI ∪ AB645 ∪ AB1101, the auxiliary sets outnumber our 940 rows
almost two to one. Left uniform, most gradient steps would come from data we are not evaluated
on, and AB645/AB1101 overlap our complexes heavily — 19 of 25 and 20 of 28 of their complexes
are among our 53 — so that is not a harmless imbalance but a slow route to fitting the test
complexes' neighbourhoods.

The rule here is the one Tom set: **a row from our dataset is twice as likely to be drawn as a
row from any auxiliary dataset.** That is a per-row odds ratio, not a share of the batch, and
the resulting share depends on how many rows each pool has. ``describe`` prints both so the
configured ratio and its consequence are never confused.

Leakage exclusion happens before weighting, not after: for a given fold, any auxiliary row
whose PDB id is in that fold's test set is dropped outright (weight zero is not enough, since
a zero-weight row can still be reported as "training data"). The per-fold usable counts are
precomputed in ``data/splits/skempi_abag_5fold_by_complex.json``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Tom's default: our rows are twice as likely to be drawn as any auxiliary row.
OURS = "skempi_abag"
DEFAULT_WEIGHT = 2.0


def row_weights(datasets: pd.Series, ours: str = OURS,
                weight: float = DEFAULT_WEIGHT) -> np.ndarray:
    """Per-row sampling weight: ``weight`` for our dataset, 1.0 for everything else."""
    return np.where(datasets.to_numpy() == ours, float(weight), 1.0)


def exclude_leaked(frame: pd.DataFrame, test_complexes: set[str],
                   pdb_col: str = "pdb") -> pd.DataFrame:
    """Drop auxiliary rows whose complex is in this fold's test set.

    Applied before weighting. A zero weight would keep the row in the training table, where it
    can still be counted, logged, or used by anything that does not consult the sampler.
    """
    keep = ~frame[pdb_col].astype(str).str.upper().isin({c.upper() for c in test_complexes})
    return frame.loc[keep].copy()


def make_sampler(frame: pd.DataFrame, dataset_col: str = "dataset",
                 ours: str = OURS, weight: float = DEFAULT_WEIGHT, seed: int = 0):
    """A torch ``WeightedRandomSampler`` over ``frame``, one epoch = one pass worth of draws.

    Returns None when torch is absent, so the module stays importable for the tree ladder.
    """
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ImportError:
        return None
    w = torch.as_tensor(row_weights(frame[dataset_col], ours, weight), dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(w, num_samples=len(frame), replacement=True,
                                 generator=generator)


def describe(frame: pd.DataFrame, dataset_col: str = "dataset", ours: str = OURS,
             weight: float = DEFAULT_WEIGHT) -> pd.DataFrame:
    """Rows, configured weight, and the expected share of draws, per dataset.

    The last column is the one that matters and the one a ratio alone does not give you: at a
    2x per-row weight our share of drawn examples is set by our share of rows as well.
    """
    counts = frame[dataset_col].value_counts()
    w = {d: (weight if d == ours else 1.0) for d in counts.index}
    mass = {d: counts[d] * w[d] for d in counts.index}
    total = sum(mass.values())
    return pd.DataFrame({
        "rows": counts,
        "row_weight": pd.Series(w),
        "share_of_rows": (counts / counts.sum()).round(3),
        "share_of_draws": pd.Series({d: mass[d] / total for d in counts.index}).round(3),
    }).sort_values("share_of_draws", ascending=False)
