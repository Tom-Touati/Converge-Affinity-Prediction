"""Build the modelling table from raw SKEMPI 2.0.

Produces two artefacts:

* ``measurements.parquet`` -- one row per surviving *measurement*, pre-deduplication. Kept
  because the spread within a repeated (complex, mutation) group is our only subset-specific
  estimate of label noise, and therefore of the correlation ceiling.
* ``dataset.parquet`` -- one row per unique (complex, mutation) pair, which is the honest
  modelling unit. This is 997 rows, not the 1,211 the raw subset advertises.

Every dropped row is counted and printed. Run with ``--verbose`` to see the audit table.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from . import paths
from .structures import (
    apply_mutations,
    assign_roles,
    load_structures,
    parse_mutations,
    verify,
)

R_GAS = 0.0019872041  # kcal / (mol K)

# Curator notes that mark a non-native reference state: the "wild type" is itself an
# affinity-matured Fab or another mutant from the same paper, so the measurement is really
# mutant-to-mutant. The SKEMPI paper flags these; we keep them but report with and without.
NONNATIVE_RE = re.compile(
    r"affinity[- ]matured|taken as wild[- ]?type|wild[- ]?type is .*(?:matured|mutant)", re.I
)
NEAR_IDENTICAL_RE = re.compile(r"very similar|nearly identical|almost identical", re.I)


def _first(s: pd.Series):
    """Most common non-null value in a group, for categorical columns."""
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan


def build(verbose: bool = True, keep_censored: bool = False) -> pd.DataFrame:
    paths.ensure_dirs()
    raw = pd.read_csv(paths.RAW_CSV, sep=";", low_memory=False)
    ab = raw[raw["Hold_out_type"].fillna("").str.contains("AB/AG")].copy()

    audit = [("AB/AG rows in SKEMPI 2.0", len(ab), ab["#Pdb"].nunique())]

    # --- label ------------------------------------------------------------------------------
    # Temperature is free text: "298", "298 (assumed)", "277". Parse the number, default 298 K,
    # and carry the assumption forward as a flag -- 48% of this subset is assumed, which makes
    # the ddG scale approximate for those rows.
    tnum = ab["Temperature"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    ab["temp_assumed"] = ab["Temperature"].astype(str).str.contains("assumed", case=False)
    ab["T"] = tnum.fillna(298.0)

    for side in ("mut", "wt"):
        ab[f"{side}_is_bound"] = (
            ab[f"Affinity_{side} (M)"].astype(str).str.contains(r"[<>]", regex=True)
        )

    ab["ddG"] = R_GAS * ab["T"] * np.log(ab["Affinity_mut_parsed"] / ab["Affinity_wt_parsed"])
    ab = ab[np.isfinite(ab["ddG"])].copy()
    audit.append(("after dropping unparsable affinities", len(ab), ab["#Pdb"].nunique()))

    # Censored affinities are dropped. SKEMPI writes a detection limit as ">1e-6" and parses it
    # into a bare number, so the row says "Kd is at least this" and the pipeline reads "Kd is
    # exactly this". The resulting ddG is a lower bound presented as a measurement, and it is
    # biased in one direction: censored rows average +2.09 kcal/mol against +0.97 for the rest,
    # because a censored affinity means binding too weak to measure.
    #
    # Dropping is the conservative reading, not the right one. The correct treatment is
    # censored regression (a one-sided loss that penalises predicting below the bound but not
    # above it) or a ranking constraint, either of which would use the row as the real evidence
    # it is. Until that exists, training on a bound as though it were a point estimate teaches
    # the model a number nobody measured. See README, "Open defects".
    censored = ab["mut_is_bound"] | ab["wt_is_bound"]
    if keep_censored:
        audit.append((f"KEEPING {int(censored.sum())} censored rows (--keep-censored)",
                      len(ab), ab["#Pdb"].nunique()))
    else:
        ab = ab[~censored].copy()
        audit.append((f"after dropping {int(censored.sum())} censored affinities",
                      len(ab), ab["#Pdb"].nunique()))

    # --- identifiers ------------------------------------------------------------------------
    ab["pdb"] = ab["#Pdb"].str.split("_").str[0]
    ab["side1"] = ab["#Pdb"].str.split("_").str[1]
    ab["side2"] = ab["#Pdb"].str.split("_").str[2]
    ab["mutations"] = ab["Mutation(s)_cleaned"].str.strip()
    ab["n_mut"] = ab["mutations"].str.count(",") + 1
    ab["row_id"] = ab["#Pdb"] + "|" + ab["mutations"]

    notes = ab["Notes"].fillna("")
    ab["nonnative_ref"] = notes.str.contains(NONNATIVE_RE)
    ab["near_identical_note"] = notes.str.contains(NEAR_IDENTICAL_RE)

    # --- structure: verify, and assign chain roles ------------------------------------------
    structures = load_structures(ab["pdb"].unique())

    problems, sides, is_ala = [], [], []
    for _, r in ab.iterrows():
        st = structures[r["pdb"]]
        muts = parse_mutations(r["mutations"])
        problems.append(verify(st, muts))
        is_ala.append(all(m.mut == "A" for m in muts))
        sides.append({m.chain for m in muts})
    ab["struct_problems"] = ["; ".join(p) for p in problems]
    ab["is_alanine"] = is_ala

    n_bad = int((ab["struct_problems"] != "").sum())
    n_pos = int(sum(len(parse_mutations(m)) for m in ab["mutations"]))
    if n_bad:
        raise AssertionError(
            f"{n_bad} rows fail structure verification; the EDA proved this should be zero.\n"
            + ab.loc[ab["struct_problems"] != "", ["row_id", "struct_problems"]].head(10).to_string()
        )
    if verbose:
        print(f"structure check: {n_pos:,} mutated positions, 0 mismatches")

    roles = {}
    for cx, g in ab.groupby("#Pdb"):
        r = g.iloc[0]
        abc, agc, basis = assign_roles(structures[r["pdb"]], r["side1"], r["side2"])
        roles[cx] = (abc, agc, basis)
    ab["ab_chains"] = ab["#Pdb"].map(lambda k: roles[k][0])
    ab["ag_chains"] = ab["#Pdb"].map(lambda k: roles[k][1])
    ab["role_basis"] = ab["#Pdb"].map(lambda k: roles[k][2])

    def side_of(r, touched):
        if r["role_basis"] == "tie":
            return "ab_vs_ab"          # 1DVF_AB_CD: anti-idiotype, no antigen exists
        on_ab = bool(touched & set(r["ab_chains"]))
        on_ag = bool(touched & set(r["ag_chains"]))
        return "both" if on_ab and on_ag else ("antibody" if on_ab else "antigen")

    ab["mut_side"] = [side_of(r, t) for (_, r), t in zip(ab.iterrows(), sides)]

    # --- assert the mutant sequences can actually be built -----------------------------------
    for _, r in ab.iterrows():
        apply_mutations(structures[r["pdb"]], parse_mutations(r["mutations"]))

    meas = ab.reset_index(drop=True)
    meas.to_parquet(paths.MEASUREMENTS, index=False)

    # --- deduplicate to the modelling unit ---------------------------------------------------
    # Repeat measurements of the same (complex, mutation) are aggregated by median. Their spread
    # is retained as ddG_sd / ddG_ptp: this is the subset's own label-noise estimate.
    keep_first = [
        "#Pdb", "pdb", "side1", "side2", "mutations", "n_mut", "ab_chains", "ag_chains",
        "role_basis", "mut_side", "is_alanine", "Protein 1", "Protein 2",
    ]
    agg = {c: (c, "first") for c in keep_first}
    agg.update(
        ddG=("ddG", "median"),
        ddG_n=("ddG", "size"),
        ddG_sd=("ddG", "std"),
        ddG_ptp=("ddG", lambda s: s.max() - s.min()),
        T=("T", "median"),
        temp_assumed=("temp_assumed", "any"),
        mut_is_bound=("mut_is_bound", "any"),
        wt_is_bound=("wt_is_bound", "any"),
        nonnative_ref=("nonnative_ref", "any"),
        near_identical_note=("near_identical_note", "any"),
        location=("iMutation_Location(s)", _first),
        method=("Method", _first),
        reference=("Reference", _first),
    )
    ds = meas.groupby("row_id", as_index=False).agg(**agg)
    ds = ds.sort_values("row_id").reset_index(drop=True)
    audit.append(("unique (complex, mutation) pairs", len(ds), ds["#Pdb"].nunique()))

    ds.to_parquet(paths.DATASET, index=False)

    if verbose:
        tbl = pd.DataFrame(audit, columns=["step", "rows", "complexes"])
        tbl["delta"] = tbl["rows"].diff().fillna(0).astype(int)
        print()
        print(tbl.to_string(index=False))
        print()
        print(f"wrote {paths.MEASUREMENTS.relative_to(paths.ROOT)}  ({len(meas):,} measurements)")
        print(f"wrote {paths.DATASET.relative_to(paths.ROOT)}  ({len(ds):,} modelling rows)")
        print()
        print("mutation side:")
        print(ds["mut_side"].value_counts().to_string())
        print()
        print(f"repeated pairs: {(ds.ddG_n > 1).sum()}  covering {ds.loc[ds.ddG_n > 1, 'ddG_n'].sum()} measurements")
        print(f"ddG: mean {ds.ddG.mean():+.2f}  sd {ds.ddG.std():.2f}  "
              f"range [{ds.ddG.min():+.2f}, {ds.ddG.max():+.2f}]")
    return ds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--keep-censored", action="store_true",
                   help="keep affinities recorded as a detection limit (\">1e-6\") and treat "
                        "them as exact. This is how every result before 2026-09-21 was "
                        "produced; it is wrong and the flag exists only to reproduce them.")
    a = p.parse_args()
    build(verbose=not a.quiet, keep_censored=a.keep_censored)


if __name__ == "__main__":
    main()
