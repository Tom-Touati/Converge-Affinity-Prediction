"""Reverse-mutation augmentation: swap wild type and mutant, flip the label's sign.

Binding free energy is a state function, so ddG(A->B) = -ddG(B->A) exactly. Every row therefore
implies a second measurement we never recorded, and the implied ones are concentrated exactly
where the data is thin: 544 rows are destabilising beyond +0.5 against only 130 clearly
stabilising, and the error analysis found the model ranks stabilising mutations at rho -0.16,
worse than chance. Reversing the destabilising rows manufactures 544 synthetic stabilising
examples against the 130 real ones.

PLAN.md deferred this behind a trigger -- "antisymmetry probe fails, or the stabilising slice
stays at chance" -- and the trigger has fired.

**What actually reverses, and what only pretends to.**

* ``chem`` reverses *exactly*. Every column is a function of (wild type, mutant), so the honest
  implementation is to reverse the mutation string ("YH33A" -> "AH33Y") and recompute the block
  from scratch rather than hand-flipping signs and hoping the bookkeeping is right.
* ``mpnn`` reverses *exactly*. ProteinMPNN's unconditional log-probabilities come from the
  backbone alone and give the full 20-way distribution at each position, so the reverse reads two
  different entries of the same vector: the log-ratios negate, and the wild-type/mutant
  log-probabilities swap.
* PLM difference vectors negate and the wild-type/mutant halves swap. Log-likelihood ratios
  negate, which is exact for the masked-marginal and an approximation for the wt-marginal, since
  the latter is read from the wild-type sequence's logits and the true reverse would read the
  mutant's.
* Norm-like features (``d_at_pos``, ``reach``, ``prof*``, ``diff_norm``) are magnitudes and are
  genuinely unchanged by reversal.
* **``geom`` does not reverse at all.** Every geometric feature is computed on the wild-type
  backbone, and the mutant structure does not exist. A reversed row therefore carries *identical*
  geometry to its source with an opposite label.

That last point is the whole risk, and it should be stated rather than buried. ``geom:n_contacts``
is the single strongest feature we have (|rho| 0.449). After augmentation every value of it maps
to both +ddG and -ddG, so it can no longer predict sign on its own -- only magnitude. For this to
help, the model has to learn magnitude from the symmetric features and sign from the antisymmetric
ones, which is a multiplicative interaction to infer from ~750 rows. It may well fail. That is
what the experiment is for.

**Leakage.** Augmented rows are added to *training folds only* and never scored. A reversed row
is the same measurement with a flipped sign, so allowing one into a test fold whose source sat in
training would be as severe a leak as duplicating the row outright.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .structures import parse_mutations

#: columns whose sign flips under reversal
_NEGATE = re.compile(
    r":(llr_complex|llr_alone|llr_delta|llr_wt|llr_masked|d_charge|d\d+)$"
)
#: (wild-type column, mutant column) pairs that exchange values
_SWAPS = (
    (re.compile(r":logp_wt_complex$"), ":logp_mut_complex"),
    (re.compile(r":wt_d(\d+)$"), r":mt_d\1"),
    # geom_rev residue and interaction columns. The negative lookahead keeps the wide
    # embedding columns (wt_d0, wt_d1, ...) on the rule above, which maps them to mt_*.
    (re.compile(r":wt_(?!d\d)(\w+)$"), r":mut_\1"),
)


def reverse_mutations(spec: str) -> str:
    """'YH33A' -> 'AH33Y'; multi-point reverses every token."""
    out = []
    for m in parse_mutations(spec):
        out.append(f"{m.mut}{m.chain}{m.resnum}{m.icode}{m.wt}")
    return ",".join(out)


def _swap_map(columns) -> dict:
    """Map every column onto the column it exchanges values with, where one exists."""
    cols = set(columns)
    mapping = {}
    for pat, repl in _SWAPS:
        for c in columns:
            if pat.search(c):
                partner = pat.sub(repl, c)
                if partner in cols:
                    mapping[c] = partner
                    mapping[partner] = c
    return mapping


def reverse_features(X: pd.DataFrame, df: pd.DataFrame, blocks) -> pd.DataFrame:
    """Build the feature matrix for the reversed version of every row in X."""
    R = X.copy()

    # chem is recomputed from the reversed mutation strings -- exact, no sign bookkeeping
    if any(c.startswith("chem:") for c in X.columns):
        from .features.chem import build as chem_build

        rev = df.assign(mutations=[reverse_mutations(s) for s in df["mutations"]])
        rev = rev.assign(row_id=rev["#Pdb"] + "|" + rev["mutations"])
        block = chem_build(rev).add_prefix("chem:")
        for c in block.columns:
            if c in R.columns:
                R[c] = block[c].to_numpy()

    for c in X.columns:
        if c.startswith("chem:"):
            continue
        if _NEGATE.search(c):
            R[c] = -X[c].to_numpy()

    for a, b in _swap_map(X.columns).items():
        R[a] = X[b].to_numpy()

    return R


def augment_training_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, df_tr: pd.DataFrame,
                          blocks) -> tuple:
    """Return (X, y) with each training row's reverse appended. Test folds never see this."""
    R = reverse_features(X_tr, df_tr, blocks)
    return (pd.concat([X_tr, R], ignore_index=True),
            np.concatenate([y_tr, -y_tr]))


def antisymmetry_probe(model, X: pd.DataFrame, df: pd.DataFrame, blocks) -> dict:
    """Does the fitted model already satisfy f(reverse) = -f(forward)?

    A model that respects the physics needs no augmentation. One that does not is either missing
    the constraint or relying on features that cannot express it.
    """
    from scipy import stats

    fwd = np.asarray(model.predict(X), float)
    rev = np.asarray(model.predict(reverse_features(X, df, blocks)), float)
    return {
        "mean_fwd_plus_rev": float(np.mean(fwd + rev)),   # 0 if perfectly antisymmetric
        "rmse_from_antisymmetry": float(np.sqrt(np.mean((fwd + rev) ** 2))),
        "corr_fwd_vs_negrev": float(stats.pearsonr(fwd, -rev)[0]),
    }
