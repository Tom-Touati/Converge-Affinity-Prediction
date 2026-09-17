"""Heads. Every encoder stays frozen; only these small models train.

997 rows cannot fine-tune a protein language model without overfitting it, so the ladder holds
the encoders fixed and varies only what sits on top. With this sample size a regularised linear
or tree head on good features is hard to beat, and the write-up says so rather than reaching for
a transformer.

No hyperparameter search beyond the small fixed grid below, chosen in advance. With ~250 test
rows per fold, a search would mostly fit the validation folds.
"""
from __future__ import annotations

import numpy as np


class MeanPredictor:
    """Rung 0: predict the training mean. The floor -- any model below this is broken."""

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


def make_gbt(seed: int = 0):
    """Gradient-boosted trees, regularised hard for the sample size.

    Early stopping is deliberately OFF. sklearn's built-in early stopping carves out a *random*
    validation split, which would put near-identical complexes on both sides of it and leak
    exactly the homology the outer split exists to prevent. A fixed, conservative iteration
    budget with strong regularisation is the honest alternative.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=128,
        early_stopping=False,
        random_state=seed,
    )


def make_ridge(seed: int = 0):
    """Linear head with an internal alpha sweep; the natural baseline for embedding features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 25)),
    )


def make_mlp(seed: int = 0):
    """Two-layer MLP with dropout-equivalent regularisation, for the fusion rung."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 32), alpha=1.0, max_iter=2000,
                     early_stopping=False, random_state=seed),
    )


MODELS = {
    "mean": lambda seed=0: MeanPredictor(),
    "gbt": make_gbt,
    "ridge": make_ridge,
    "mlp": make_mlp,
}
