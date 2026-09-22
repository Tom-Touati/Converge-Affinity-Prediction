"""Normalise ProtAttBA's three benchmarks into one table, with their id defects repaired.

AB645, AB1101 and S1131 ship in two different shapes. S1131 carries the two partner sides
already concatenated (``a``/``b``); the AB sets carry four chains separately
(``antibody_light_seq``, ``antibody_heavy_seq``, ``antigen_a_seq``, ``antigen_b_seq``) and
their loader concatenates light+heavy and antigen_a+antigen_b. This produces the same four
columns for all three, matching what ``utils/common.py`` feeds their model.

Two defects in their released CSVs are repaired here rather than worked around:

* **S1131's PDB id ``1E96`` is stored as ``1.00E+96``** in 2 rows -- Excel read the id as
  scientific notation. The structure ``1E96.pdb`` is shipped, so the rows are recoverable; left
  alone they would silently drop from any structure-based feature.
* **``HM_1KTZ`` and four siblings are homology models, not a chain suffix.** Splitting the id
  on ``_`` and taking the first field yields ``HM`` for 87 AB645 rows. They are legitimate
  distinct structures with their own ``HM_*.pdb`` files and keep their full name.

Run: ``python -m src.fusion.benchmark_data``  (prints the inventory it will extract from)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import paths

CHECKOUT = paths.ROOT / "experiments" / "protattba_repro" / "ProtAttBA"
CSV = {
    "AB645": CHECKOUT / "cross_validation" / "data" / "csv" / "AB645.csv",
    "AB1101": CHECKOUT / "cross_validation" / "data" / "csv" / "AB1101.csv",
    "S1131": paths.ROOT / "experiments" / "protattba_repro" / "upstream" / "S1131.csv",
}
PDB_DIRS = [
    CHECKOUT / "source_data" / "SKEMPI" / "S1131" / "PDB",
    CHECKOUT / "source_data" / "AB-bind" / "PDB",
    paths.PDB_DIR,
]

#: Excel turned "1E96" into "1.00E+96". The pattern is <digit>.00E+<digits>; the original id is
#: the leading digit, then "E", then the exponent.
_SCIENTIFIC = re.compile(r"^(\d)\.0+E\+(\d+)$", re.IGNORECASE)


def normalise_pdb_id(raw: str) -> str:
    """Their PDB column to a usable structure id."""
    s = str(raw).strip().upper()
    m = _SCIENTIFIC.match(s)
    if m:
        return f"{m.group(1)}E{m.group(2)}"        # 1.00E+96 -> 1E96
    if s.startswith("HM_"):
        return s                                    # homology model, a structure in its own right
    return s.split("_")[0]                          # 1DQJ_HL_Y -> 1DQJ


def cat_seq(a, b) -> str:
    """Their ``utils.common.cat_seq``: a missing chain contributes nothing."""
    a = "" if (a is None or (isinstance(a, float)) or str(a).lower() == "nan") else str(a)
    b = "" if (b is None or (isinstance(b, float)) or str(b).lower() == "nan") else str(b)
    return a + b


def load(name: str) -> pd.DataFrame:
    """One benchmark as (dataset, row_id, pdb, partners, mutation, ab_wt, ag_wt, ab_mt, ag_mt, ddG)."""
    path = CSV[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (the upstream checkout is gitignored)")
    d = pd.read_csv(path)

    if name == "S1131":
        ab_wt, ag_wt = d["a"].astype(str), d["b"].astype(str)
        ab_mt, ag_mt = d["a_mut"].astype(str), d["b_mut"].astype(str)
        mutation = d["mutation"]
    else:
        ab_wt = [cat_seq(x, y) for x, y in zip(d["antibody_light_seq"], d["antibody_heavy_seq"])]
        ab_mt = [cat_seq(x, y) for x, y in
                 zip(d["antibody_light_seq_mut"], d["antibody_heavy_seq_mut"])]
        ag_wt = [cat_seq(x, y) for x, y in zip(d["antigen_a_seq"], d["antigen_b_seq"])]
        ag_mt = [cat_seq(x, y) for x, y in zip(d["antigen_a_seq_mut"], d["antigen_b_seq_mut"])]
        mutation = d["Mutation"]

    out = pd.DataFrame({
        "dataset": name,
        "pdb": [normalise_pdb_id(p) for p in d["PDB"]],
        "pdb_raw": d["PDB"].astype(str),
        "partners": d["Partners"].astype(str) if "Partners" in d else "",
        "mutation": mutation.astype(str),
        "ab_wt": list(ab_wt), "ag_wt": list(ag_wt),
        "ab_mt": list(ab_mt), "ag_mt": list(ag_mt),
        "ddG": d["ddG"].astype(float),
    })
    out["row_id"] = out["dataset"] + "|" + out["pdb"] + "|" + out["mutation"]
    return out


def load_all() -> pd.DataFrame:
    return pd.concat([load(n) for n in CSV], ignore_index=True)


def find_structure(pdb_id: str) -> Path | None:
    for d in PDB_DIRS:
        p = d / f"{pdb_id}.pdb"
        if p.exists():
            return p
    return None


def distinct_sequences(frame: pd.DataFrame) -> list[str]:
    """Every distinct sequence across the four columns, longest first."""
    seqs = set()
    for col in ("ab_wt", "ag_wt", "ab_mt", "ag_mt"):
        seqs.update(s for s in frame[col].astype(str) if s and s.lower() != "nan")
    return sorted(seqs, key=lambda s: (-len(s), s))


def main() -> None:
    all_rows = load_all()
    print(f"{len(all_rows)} rows across {all_rows.dataset.nunique()} benchmarks\n")
    for name, g in all_rows.groupby("dataset"):
        seqs = distinct_sequences(g)
        missing = sorted({p for p in g.pdb.unique() if find_structure(p) is None})
        repaired = g[g.pdb != g.pdb_raw.str.upper().str.split("_").str[0]]
        print(f"{name}")
        print(f"  rows {len(g):5d}   pdb ids {g.pdb.nunique():3d}   "
              f"structures found {g.pdb.nunique() - len(missing):3d}"
              + (f"   MISSING {missing}" if missing else ""))
        print(f"  distinct sequences {len(seqs):5d}   residues {sum(len(s) for s in seqs):7d}"
              f"   longest {len(seqs[0]) if seqs else 0}")
        if len(repaired):
            print(f"  ids repaired: {sorted(set(zip(repaired.pdb_raw, repaired.pdb)))[:5]}"
                  f"  ({len(repaired)} rows)")

    seqs = distinct_sequences(all_rows)
    print(f"\nunion across all three: {len(seqs)} distinct sequences, "
          f"{sum(len(s) for s in seqs)} residues, longest {len(seqs[0])}")
    over = sum(1 for s in seqs if len(s) > 1022)
    print(f"  longer than ESM2's 1022 positions: {over} "
          "(rotary extrapolates; measured fine to 1492 tokens)")


if __name__ == "__main__":
    main()
