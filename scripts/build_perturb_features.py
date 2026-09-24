"""Assemble the tabular block the fusion head sees, and its mirror image.

Two things happen here that used to be missing.

**The model scores go in.** The 49-column forest gets ProteinMPNN's inverse-folding
log-odds and ESM-2's masked-language log-odds as ordinary columns; the net only ever
got ProteinMPNN's. Those two scalars are the published zero-shot predictors of this
exact quantity -- they are what a reviewer means by "did you at least compare to the
likelihood" -- and handing them to the head costs six numbers. ESM's come from the
650M model, the same checkpoint whose per-residue embeddings the attention consumes,
so the scalar and the tokens cannot disagree about which model was asked.

**The reverse augmentation gets a chem table to match.** Half of every training batch
is a mutation run backwards: the sequence embeddings swap, the BLOSUM rows swap, the
label changes sign. The chem vector did not, so ``mpnn__llr_delta`` -- the strongest
single correlate of ddG in the whole block at -0.231 -- was shown its forward value
against a flipped label on half the rows it appeared in. A column that predicts both
+y and -y from one value is not a weak column, it is a column the head is being
trained to zero out.

So each row is written twice, forward and reversed, and the reversal is per-column:

``NEGATE``
    differences of the form mut - wt: the five physicochemical deltas and every
    log-odds ratio. Running the mutation backwards negates them exactly.
``SWAP``
    pairs that exchange roles, ``(n_to_pro, n_from_pro)`` and friends, plus
    ``(logp_wt_complex, logp_mut_complex)`` and the two residue identities.
    ``n_to_ala`` had no partner, so ``n_from_ala`` is computed here to give it one --
    without it, reversing an alanine scan silently loses the alanine.
``KEEP``
    counts, magnitudes and symmetric scores: ``n_mut``, the ``abs_d_*`` family,
    BLOSUM (a symmetric matrix), the three ESM distance scalars (a norm and a cosine
    do not care which end you start from), and ``n_charge_reversal``.

The seven geometry columns are also kept. They describe the wild-type structure, and
running the mutation backwards would want the mutant's -- which does not exist here.
That is the same approximation the structure branch already makes, since the
ProteinMPNN tokens are wild-type-only and are not swapped either; it is recorded
rather than hidden.

Standardisation is deliberately NOT done here. The reversed table has to be scaled by
the forward table's mean and standard deviation, or the two orientations land on
different scales and the negation stops meaning anything; the trainer owns that, and
it owns the per-cluster variant too.

    python scripts/build_perturb_features.py --src <worktree-with-caches>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# mut - wt differences: running the mutation backwards negates them
NEGATE = (
    "chem__d_hydropathy", "chem__d_volume", "chem__d_charge",
    "chem__d_polarity", "chem__d_mw",
    "mpnn__llr_complex", "mpnn__llr_alone", "mpnn__llr_delta",
    "esm__llr_wt", "esm__llr_masked",
)
# columns that exchange values with each other
SWAP = (
    ("chem__n_to_pro", "chem__n_from_pro"),
    ("chem__n_to_gly", "chem__n_from_gly"),
    ("chem__n_to_ala", "chem__n_from_ala"),
    ("chem__n_large_to_small", "chem__n_small_to_large"),
    ("chem__wt_aa", "chem__mut_aa"),
    ("mpnn__logp_wt_complex", "mpnn__logp_mut_complex"),
)
ESM_COLS = ("llr_wt", "llr_masked", "d_at_pos", "d_window", "reach")
MUT = re.compile(r"^([A-Z])([A-Za-z0-9]+?)([A-Z])$")


def n_from_ala(row_id: str) -> float:
    """Count wild-type alanines, the partner ``n_to_ala`` never had.

    Parsed from the row id's mutation list (``1AHW_AB_C|KC138A,DC139A``) rather than
    from a column, because no column records the wild-type letter for multi-point rows
    -- ``chem__wt_aa`` is -1 whenever there is more than one mutation.
    """
    if "|" not in row_id:
        return 0.0
    return float(sum(bool(m) and m.group(1) == "A"
                     for m in (MUT.match(s) for s in row_id.split("|", 1)[1].split(","))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="worktree holding data/features and .perturb_local")
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--esm", default="esm650M_pair.parquet",
                    help="table carrying the sequence model's scalar scores")
    a = ap.parse_args()

    feat = a.src / "data" / "features"
    rows = pd.read_parquet(a.src / ".perturb_local" / "perturb_rows.parquet")
    ids = rows.row_id.astype(str)

    chem = pd.read_parquet(feat / "chem.parquet").add_prefix("chem__")
    mpnn = pd.read_parquet(feat / "mpnn.parquet").add_prefix("mpnn__")
    geom = pd.read_parquet(feat / "geom.parquet")
    esm = pd.read_parquet(feat / a.esm)[list(ESM_COLS)].add_prefix("esm__")

    fwd = pd.concat([t.reindex(ids) for t in (chem, mpnn, esm, geom)], axis=1)
    fwd.insert(fwd.columns.get_loc("chem__n_to_pro"), "chem__n_from_ala",
               pd.Series([n_from_ala(r) for r in ids], index=ids, dtype=np.float32))
    fwd.index.name = "row_id"

    miss = fwd.isna().sum()
    if miss.any():
        print("filling NaN with the column mean:")
        for c, n in miss[miss > 0].items():
            print(f"  {c:<28} {n:4d} rows")
        fwd = fwd.fillna(fwd.mean(numeric_only=True))

    rev = fwd.copy()
    for c in NEGATE:
        if c in rev:
            rev[c] = -rev[c]
    for x, y in SWAP:
        if x in rev and y in rev:
            rev[x], rev[y] = fwd[y].to_numpy(), fwd[x].to_numpy()

    known = set(NEGATE) | {c for p in SWAP for c in p}
    kept = [c for c in fwd.columns if c not in known]
    print(f"\n{len(fwd)} rows x {fwd.shape[1]} columns")
    print(f"  negated {len([c for c in NEGATE if c in fwd])}, "
          f"swapped {len([p for p in SWAP if p[0] in fwd])} pairs, kept {len(kept)}")
    print("  kept unchanged: " + ", ".join(kept))

    # A column that is neither negated nor swapped and yet is strongly antisymmetric
    # would be a classification error, so say which of the kept columns actually moved.
    moved = [c for c in kept if not np.allclose(fwd[c].astype(float),
                                                rev[c].astype(float), equal_nan=True)]
    print(f"  kept columns that changed anyway: {moved or 'none'}")

    a.out.mkdir(parents=True, exist_ok=True)
    for name, t in (("chem_perturb_v2", fwd), ("chem_perturb_v2_rev", rev)):
        t.reset_index().to_parquet(a.out / f"{name}.parquet", index=False)
        print(f"  wrote {a.out / name}.parquet")

    y = rows.set_index(ids).ddg
    print("\ncorrelation with ddG, forward orientation:")
    for c in ("mpnn__llr_complex", "mpnn__llr_alone", "mpnn__llr_delta",
              "esm__llr_wt", "esm__llr_masked", "esm__d_at_pos", "esm__d_window"):
        if c in fwd:
            print(f"  {c:<24} {np.corrcoef(fwd[c].astype(float), y)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
