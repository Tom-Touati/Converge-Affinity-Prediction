"""PCA down to 128 dimensions for the ESM blocks, fit on the training fold only.

The pooled ESM features are 1280 numbers per branch against ~750 training rows, and the E0
ladder measured what that costs: pooled ESM plus pooled ProteinMPNN reaches pooled Pearson
0.188 on the homology-cluster split against 0.340 for 49 handcrafted columns. Reducing to 128
tests whether that gap is dilution or representation.

**The fit is per fold, on training rows only.** Fitting the projection on all rows before
splitting is a textbook leak: the components would be chosen using the variance structure of
the test complexes, and on a grouped split that is exactly the homology the split exists to
withhold. `fit_transform_folds` therefore takes the fold assignment and refits inside each
fold, which is why it returns a per-fold matrix rather than a single transformed table.

The scaler matters as much as the rotation. ESM dimensions differ in scale by more than an
order of magnitude, so PCA without standardisation is dominated by a handful of high-variance
channels; `StandardScaler` is fit on the same training rows and applied the same way.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#: Columns that are genuinely embedding dimensions, as opposed to derived scalars. Matches the
#: three blocks features.py assembles: wild-type pooled, mutant pooled, and their difference.
EMBEDDING_COL = re.compile(r"__(wt_d|mt_d|d)\d+$")

DEFAULT_COMPONENTS = 128


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(embedding columns, everything else). The scalars are passed through untouched."""
    emb = [c for c in X.columns if EMBEDDING_COL.search(c)]
    rest = [c for c in X.columns if c not in set(emb)]
    return emb, rest


def fit_transform_folds(X: pd.DataFrame, folds: np.ndarray,
                        n_components: int = DEFAULT_COMPONENTS,
                        seed: int = 0, verbose: bool = False) -> dict[int, tuple]:
    """Per fold, fit on that fold's training rows and return (X_train, X_test) matrices.

    Returns ``{fold: (train_frame, test_frame)}``. Scalars are concatenated back on, so a
    downstream estimator sees the reduced embeddings *plus* the handcrafted columns it would
    otherwise have had.
    """
    emb_cols, rest_cols = split_columns(X)
    out: dict[int, tuple] = {}

    for k in sorted(set(folds)):
        te = folds == k
        tr = ~te
        if not emb_cols:
            out[k] = (X.loc[tr], X.loc[te])
            continue

        n_comp = min(n_components, len(emb_cols), int(tr.sum()) - 1)
        scaler = StandardScaler()
        pca = PCA(n_components=n_comp, random_state=seed)

        train_emb = scaler.fit_transform(X.loc[tr, emb_cols].to_numpy(dtype=np.float64))
        train_z = pca.fit_transform(train_emb)
        test_z = pca.transform(scaler.transform(X.loc[te, emb_cols].to_numpy(dtype=np.float64)))

        names = [f"pc{i}" for i in range(n_comp)]
        tr_frame = pd.DataFrame(train_z, columns=names, index=X.index[tr])
        te_frame = pd.DataFrame(test_z, columns=names, index=X.index[te])
        if rest_cols:
            tr_frame = pd.concat([X.loc[tr, rest_cols], tr_frame], axis=1)
            te_frame = pd.concat([X.loc[te, rest_cols], te_frame], axis=1)
        out[k] = (tr_frame.astype(np.float32), te_frame.astype(np.float32))

        if verbose:
            ev = float(pca.explained_variance_ratio_.sum())
            print(f"    fold {k}: {len(emb_cols)} embedding cols -> {n_comp} components, "
                  f"{ev:.1%} of variance, fit on {int(tr.sum())} training rows")
    return out
