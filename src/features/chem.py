"""Rung 1: substitution chemistry only. No protein context, no structure, no learned encoder.

This is deliberately the dumbest defensible model, and it is not a strawman. The 2025 AbAgym
benchmark found that across six published methods the best held-out antibody result was Spearman
0.28, and that relative solvent accessibility alone was competitive with all of them. Whatever
the encoders do later has to beat *this*, on the frozen split, by a paired-bootstrap delta whose
CI clears zero.

Deliberately excluded: SKEMPI's ``iMutation_Location(s)`` annotation. It is a structural
property, and putting it here would contaminate the rung-1 versus rung-3 comparison that tells
us whether structure adds anything. It is computed as its own block for later rungs and slices.

Multi-point mutations sum their per-position deltas. The SKEMPI paper's double-mutant cycles say
this is only half true -- 345 additive against 421 context-dependent -- so the approximation is
tested by the additivity probe rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..structures import parse_mutations

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}

# Kyte-Doolittle hydropathy
HYDROPATHY = dict(zip(AAS, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                            1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
# residue volume, cubic angstroms (Zamyatnin)
VOLUME = dict(zip(AAS, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
                        162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6]))
# formal charge at pH 7
CHARGE = dict(zip(AAS, [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
# Grantham polarity
POLARITY = dict(zip(AAS, [8.1, 5.5, 13.0, 12.3, 5.2, 9.0, 10.4, 5.2, 11.3, 4.9,
                          5.7, 11.6, 8.0, 10.5, 10.5, 9.2, 8.6, 5.9, 5.4, 6.2]))
# molecular weight
MW = dict(zip(AAS, [71.1, 103.1, 115.1, 129.1, 147.2, 57.1, 137.1, 113.2, 128.2, 113.2,
                    131.2, 114.1, 97.1, 128.1, 156.2, 87.1, 101.1, 99.1, 186.2, 163.2]))


def _blosum():
    from Bio.Align import substitution_matrices
    return substitution_matrices.load("BLOSUM62")


def build(df: pd.DataFrame) -> pd.DataFrame:
    """One row of chemistry features per dataset row, indexed by row_id."""
    bl = _blosum()
    out = []
    for row_id, mutations in zip(df["row_id"], df["mutations"]):
        muts = parse_mutations(mutations)
        wt = [m.wt for m in muts]
        mu = [m.mut for m in muts]

        d_hyd = [HYDROPATHY[b] - HYDROPATHY[a] for a, b in zip(wt, mu)]
        d_vol = [VOLUME[b] - VOLUME[a] for a, b in zip(wt, mu)]
        d_chg = [CHARGE[b] - CHARGE[a] for a, b in zip(wt, mu)]
        d_pol = [POLARITY[b] - POLARITY[a] for a, b in zip(wt, mu)]
        d_mw = [MW[b] - MW[a] for a, b in zip(wt, mu)]
        blos = [float(bl[a, b]) for a, b in zip(wt, mu)]

        out.append({
            "row_id": row_id,
            "n_mut": len(muts),
            # substitution likelihood: the single strongest cheap signal
            "blosum_sum": sum(blos),
            "blosum_min": min(blos),
            # physicochemical deltas, summed over positions
            "d_hydropathy": sum(d_hyd),
            "d_volume": sum(d_vol),
            "d_charge": sum(d_chg),
            "d_polarity": sum(d_pol),
            "d_mw": sum(d_mw),
            # magnitude matters independently of direction for packing effects
            "abs_d_hydropathy": sum(abs(x) for x in d_hyd),
            "abs_d_volume": sum(abs(x) for x in d_vol),
            "abs_d_charge": sum(abs(x) for x in d_chg),
            # the special residues, which break backbone geometry rather than side-chain contacts
            "n_to_ala": sum(b == "A" for b in mu),
            "n_to_pro": sum(b == "P" for b in mu),
            "n_from_pro": sum(a == "P" for a in wt),
            "n_to_gly": sum(b == "G" for b in mu),
            "n_from_gly": sum(a == "G" for a in wt),
            "n_charge_reversal": sum(
                CHARGE[a] * CHARGE[b] < 0 for a, b in zip(wt, mu)
            ),
            "n_large_to_small": sum(VOLUME[a] - VOLUME[b] > 50 for a, b in zip(wt, mu)),
            "n_small_to_large": sum(VOLUME[b] - VOLUME[a] > 50 for a, b in zip(wt, mu)),
            # identity of the substitution, for single-point rows only
            "wt_aa": AA_IDX[wt[0]] if len(muts) == 1 else -1,
            "mut_aa": AA_IDX[mu[0]] if len(muts) == 1 else -1,
        })
    return pd.DataFrame(out).set_index("row_id").astype(np.float32)


CATEGORICAL = ("wt_aa", "mut_aa")
