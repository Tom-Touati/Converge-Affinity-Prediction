"""Heads. Every encoder stays frozen; only these small models train.

997 rows cannot fine-tune a protein language model without overfitting it, so the ladder holds
the encoders fixed and varies only what sits on top. With this sample size a regularised linear
or tree head on good features is hard to beat, and the write-up says so rather than reaching for
a transformer.

No hyperparameter search beyond the small fixed grid below, chosen in advance. With ~250 test
rows per fold, a search would mostly fit the validation folds.
"""
from __future__ import annotations

import re

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


def make_rf(seed: int = 0):
    """Random forest. Not a formality -- it is a diagnostic for label noise.

    Boosting and bagging fail in opposite directions. Gradient boosting fits residuals
    sequentially, which reduces bias and lets it chase label noise; a random forest averages
    decorrelated trees, which reduces variance and is markedly more robust to noisy targets.
    With a measurement floor near 0.5 kcal/mol on this data, the GBT-vs-RF gap is informative in
    itself: if the forest matches or beats the boosted trees, we are limited by label noise
    rather than by model capacity, and reaching for a bigger model is the wrong response.

    Fixed configuration, chosen in advance, no search.

    ``max_depth`` is None, so the constraint that actually limits tree depth is
    ``min_samples_leaf``. It was 5 for the first run (reported at per-complex rho 0.475) and is
    now 3: a deliberate single step, not a swept grid, because selecting a value by the outer
    fold scores would be tuning on the test set. Smaller leaves average fewer training rows per
    prediction, which is the direct mechanism behind the prediction compression measured in
    ERROR_ANALYSIS.md -- so this is a test of that mechanism as much as a hyperparameter change.
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
    )


#: wide embedding columns, as emitted by the PLM extractors and prefixed by train.build_matrix
_LATENT = re.compile(r":d\d+$")
_WT = re.compile(r":wt_d\d+$")
_MT = re.compile(r":mt_d\d+$")


def _latent_cols(X):
    return [c for c in X.columns if _LATENT.search(c)]


def _wt_cols(X):
    return [c for c in X.columns if _WT.search(c)]


def _mt_cols(X):
    return [c for c in X.columns if _MT.search(c)]


def _scalar_cols(X):
    return [c for c in X.columns
            if not (_LATENT.search(c) or _WT.search(c) or _MT.search(c))]


def make_rf20(seed: int = 0):
    """Random forest considering 20% of features per split instead of sqrt(p).

    The control for make_pca_rf: it isolates the max_features change from the PCA features, so a
    gain cannot be credited to the wrong one.
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=500, max_features=0.2, min_samples_leaf=3,
        max_depth=None, n_jobs=-1, random_state=seed,
    )


def make_pca_rf(seed: int = 0, n_components: int = 32):
    """Compress the PLM latent to 32 components in-fold, keep the scalars, then a forest.

    Why this shape. A raw 480-dimensional embedding difference concatenated onto 33 scalar
    features drops the headline metric from 0.487 to 0.383 -- not because the latent is empty
    (a forest on it alone scores 0.221) but because 513 columns of which 33 are informative
    means each split sees almost nothing useful. Compressing the latent first restores the
    balance: 33 scalars beside 32 components.

    The PCA sits inside a Pipeline, so it is refitted on the training rows of every fold.
    Fitting it once over all 997 rows would choose components using test-fold feature structure
    -- no label leakage, but still an inflated score a reviewer would rightly flag.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    reduce_latent = make_pipeline(StandardScaler(), PCA(n_components=n_components,
                                                        random_state=seed))
    return Pipeline([
        ("split", ColumnTransformer(
            [("latent", reduce_latent, _latent_cols), ("scalars", "passthrough", _scalar_cols)],
            remainder="drop")),
        ("rf", RandomForestRegressor(
            n_estimators=500, max_features=0.2, min_samples_leaf=3,
            max_depth=None, n_jobs=-1, random_state=seed)),
    ])


def make_pca_pair_rf(seed: int = 0, n_components: int = 16):
    """Compress the wild-type and mutant embeddings to 16 components EACH, then concatenate.

    The alternative to subtraction. Taking mutant minus wild type at the same position destroys
    the site: 82% of that difference vector's variance is explained by the substitution type
    alone (66% on types with >=10 examples), so it is a 480-dimensional restatement of "Y>A",
    which ``chem`` already supplies in 21 interpretable numbers. Keeping the halves lets the head
    see both *what the position is* and *what it became*, and form its own comparison rather than
    being handed a subtraction.

    The two blocks are reduced separately, not jointly: a joint PCA over the concatenation would
    mix the two and could spend components on their shared variance, which is exactly the part
    subtraction already proved uninformative.

    Both PCAs sit inside the Pipeline, so they refit on the training rows of every fold.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _reduce():
        return make_pipeline(StandardScaler(),
                             PCA(n_components=n_components, random_state=seed))

    return Pipeline([
        ("split", ColumnTransformer(
            [("wt", _reduce(), _wt_cols),
             ("mt", _reduce(), _mt_cols),
             ("scalars", "passthrough", _scalar_cols)],
            remainder="drop")),
        ("rf", RandomForestRegressor(
            n_estimators=500, max_features=0.2, min_samples_leaf=3,
            max_depth=None, n_jobs=-1, random_state=seed)),
    ])


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
    "rf": make_rf,
    "rf20": make_rf20,
    "pca_rf": make_pca_rf,
    "pca_pair_rf": make_pca_pair_rf,
    "ridge": make_ridge,
    "mlp": make_mlp,
}
