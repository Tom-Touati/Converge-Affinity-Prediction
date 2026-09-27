"""Merge the StaB-ddG columns onto the existing 39-column chem block, for the ablation
_perturb_v2_colab.py already knows how to run (CHEM_TABLE + chem_dim -- see below).

The 39-column table (``chem_perturb_v2.parquet``) is left completely untouched; this writes a
SEPARATE ``chem_perturb_v2_stab.parquet`` (46 columns) so the existing leader recipe's own table
never changes underneath it.

**The reverse-mutation table is derived algebraically, not by re-running the model.**
build_perturb_features.py established the pattern this follows: reversing a mutation means
scoring the wild-type residue where you used to score the mutant and vice versa, so any column
whose two "sides" are wt/mut swaps them, and any column that IS mut-minus-wt (in whatever
sense) negates. StaB-ddG's six f(...) terms are three such SWAP pairs by construction --
f_wt_complex/f_mut_complex simply relabel which sequence is "reference" -- and stab_ddg itself
is a mut-wt double-difference, so it negates. Unlike the existing table's geometry columns
(wild-type-only, kept as a documented approximation on reversal), this reversal is EXACT: no
approximation, because f(s) never assumed which sequence was "wild-type" in the first place.

    python scripts/build_stab_ddg_table.py [--stab data/features/stab_ddg.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = Path("data/features")
# (wt column, mut column): swapped under reversal, exactly like chem_perturb_v2's own
# (logp_wt_complex, logp_mut_complex) pair.
SWAP_PAIRS = (
    ("stab__f_wt_complex", "stab__f_mut_complex"),
    ("stab__f_wt_ab_alone", "stab__f_mut_ab_alone"),
    ("stab__f_wt_ag_alone", "stab__f_mut_ag_alone"),
)
NEGATE = ("stab__stab_ddg",)


def reverse(block: pd.DataFrame) -> pd.DataFrame:
    rev = block.copy()
    for wt_col, mut_col in SWAP_PAIRS:
        rev[wt_col], rev[mut_col] = block[mut_col], block[wt_col]
    for col in NEGATE:
        rev[col] = -block[col]
    return rev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab", type=Path, default=FEATURES / "stab_ddg.parquet")
    ap.add_argument("--chem", type=Path, default=FEATURES / "chem_perturb_v2.parquet")
    ap.add_argument("--out", type=Path, default=FEATURES / "chem_perturb_v2_stab.parquet")
    a = ap.parse_args()

    stab = pd.read_parquet(a.stab)
    units = stab.pop("stab__unit").unique()
    assert len(units) == 1, f"expected one unit across the whole extraction, got {units}"
    unit = units[0]
    stab_cols = [c for c in stab.columns if c != "row_id"]
    stab[stab_cols] = stab[stab_cols].astype(np.float32)

    chem = pd.read_parquet(a.chem)
    chem_rev = pd.read_parquet(str(a.chem).replace(".parquet", "_rev.parquet"))

    merged = chem.merge(stab, on="row_id", how="inner", validate="one_to_one")
    assert len(merged) == len(chem), (
        f"row_id mismatch: chem has {len(chem)} rows, merged has {len(merged)} -- "
        f"stab_ddg.parquet must cover exactly the same rows as {a.chem.name}"
    )
    stab_rev = reverse(stab)
    merged_rev = chem_rev.merge(stab_rev, on="row_id", how="inner", validate="one_to_one")
    assert len(merged_rev) == len(chem_rev)

    merged.to_parquet(a.out, index=False)
    rev_out = Path(str(a.out).replace(".parquet", "_rev.parquet"))
    merged_rev.to_parquet(rev_out, index=False)

    print(f"wrote {a.out}  {merged.shape}  (chem_dim={merged.shape[1] - 1})")
    print(f"wrote {rev_out}  {merged_rev.shape}")
    print(f"stab__* columns are in units: {unit}")
    print(f"\nrun the ablation with:")
    print(f"  ON:  CHEM_TABLE=chem_perturb_v2_stab  --overrides '{{\"chem_dim\":"
          f"{merged.shape[1] - 1},...}}'")
    print(f"  OFF: CHEM_TABLE=chem_perturb_v2       --overrides '{{\"chem_dim\":39,...}}'  "
          f"(the existing leader recipe, unchanged)")


if __name__ == "__main__":
    main()
