"""Freeze the evaluation protocol for the fusion ladder, and account for cross-dataset leakage.

Rule 1 of the overnight plan: build the folds once, before any modelling, and never
regenerate them. This writes ``data/splits/skempi_abag_5fold_by_complex.json`` — 5 folds over
the 940-row antibody–antigen dataset, grouped by PDB id, seed 0 — and refuses to overwrite an
existing file unless asked, so no model run can re-split the data underneath itself.

The repo's own ``data/folds.csv`` (4 folds, grouped by *homology cluster*) is left alone. The
two are not interchangeable and the difference is not cosmetic: grouping by complex lets a
test complex keep a structurally near-identical training twin, and the project measured that
42 of 54 complexes gain a TM > 0.8 training neighbour under complex grouping against 0 of 54
under cluster grouping. Numbers on this 5-fold split are therefore *looser* than the project's
headline and must not be quoted beside them. Both are reported.

Rule 2, leakage from auxiliary datasets, is handled here too, because the exclusion has to be
derived from the same frozen fold assignment:

* every (pdb, mutation) pair is canonicalised and duplicates across datasets are counted;
* for each fold, any auxiliary row whose pdb id is in that fold's test set is excluded from
  that fold's training pool.

The counts are written into the json so a later run can be audited without recomputing them.

Run: ``python -m src.fusion.splits_frozen``  (add ``--force`` to deliberately rebuild)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import paths

SPLIT_DIR = paths.DATA / "splits"
SKEMPI_SPLIT = SPLIT_DIR / "skempi_abag_5fold_by_complex.json"

N_FOLDS = 5
SEED = 0

#: The ProtAttBA benchmark csvs, as auxiliary training pools. Paths are relative to the repo
#: root; the checkout is gitignored, so a missing file is reported rather than fatal.
AUX = {
    "AB645": "experiments/protattba_repro/ProtAttBA/cross_validation/data/csv/AB645.csv",
    "AB1101": "experiments/protattba_repro/ProtAttBA/cross_validation/data/csv/AB1101.csv",
    "S1131": "experiments/protattba_repro/upstream/S1131.csv",
}


#: SKEMPI's cleaned form, which this repo stores in `mutations`: wt, chain, position, mutant.
#: e.g. "DC167A" -> D at chain C position 167 becomes A. Comma-separated for multi-point.
_SKEMPI_RE = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")

#: ProtAttBA's form, in the `Mutation` column of AB645/AB1101/S1131: chain, wt, position,
#: mutant. e.g. "C:D101A".
_PROTATTBA_RE = re.compile(r"^([A-Za-z0-9]):([A-Z])(-?\d+[A-Za-z]?)([A-Z])$")

#: AB1101 concatenates multi-point mutations with no separator ("D:L483TD:V486PD:H487A").
#: A bare findall over that is ambiguous, because a position may carry a PDB insertion code
#: ("100A") that is indistinguishable from the mutant residue letter: on "C:K138AC:D139A" it
#: reads the position as "138A" and the mutant as "C". Splitting on the field boundary first
#: -- immediately before each "<chain>:" -- removes the ambiguity.
_FIELD_BOUNDARY = re.compile(r"(?=[A-Za-z0-9]:)")


def canonical_mutations(text: str) -> str:
    """Order- and format-insensitive key for a mutation list.

    Both conventions in play here describe the same thing in different orders, and the first
    version of this function simply lowercased and sorted the raw strings. That made the
    cross-dataset duplicate count **vacuously zero**: our "DC101A" and AB645's "C:D101A" are
    the same mutation of 1DQJ and never compared equal. Rule 2 exists to catch leakage, so a
    key that cannot match is worse than no key.

    Both forms normalise to ``chain:wt:pos:mut`` tuples, sorted, so a multi-point mutation is
    order-insensitive too.

    Positions are taken as written. Note that S1131 also ships a ``mutation_clean`` column
    whose positions differ from its own ``mutation`` column (``A:C171A`` against ``CA182A``),
    i.e. sequence index against PDB numbering; only the PDB-numbered form is used here, which
    is the convention this repo's ``mutations`` column also follows.
    """
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""

    # Commas separate fields in both conventions; AB1101 additionally uses no separator, so
    # split on commas and then on the "<chain>:" boundary.
    chunks: list[str] = []
    for comma_part in s.split(","):
        comma_part = comma_part.strip()
        if not comma_part:
            continue
        if ":" in comma_part:
            chunks.extend(c for c in _FIELD_BOUNDARY.split(comma_part) if c)
        else:
            chunks.append(comma_part)

    parts = []
    for chunk in chunks:
        chunk = chunk.strip()
        m = _PROTATTBA_RE.match(chunk)
        if m:
            ch, wt, pos, mut = m.groups()
            parts.append(f"{ch.upper()}:{wt}:{pos}:{mut}")
            continue
        m = _SKEMPI_RE.match(chunk.upper())
        if m:
            wt, ch, pos, mut = m.groups()
            parts.append(f"{ch}:{wt}:{pos}:{mut}")
            continue
        parts.append(f"UNPARSED[{chunk.upper()}]")   # visible, never silently dropped
    return ",".join(sorted(parts))


def aux_frame(name: str, path: Path) -> pd.DataFrame | None:
    """Normalise one auxiliary csv to (dataset, pdb, mutations, ddG)."""
    if not path.exists():
        return None
    d = pd.read_csv(path)
    mut_col = "Mutation" if "Mutation" in d.columns else "mutation"
    out = pd.DataFrame({
        "dataset": name,
        "pdb": d["PDB"].astype(str).str.upper().str.split("_").str[0],
        "mutations": d[mut_col].map(canonical_mutations),
        "ddG": d["ddG"].astype(float),
    })
    return out


def build(force: bool = False) -> dict:
    if SKEMPI_SPLIT.exists() and not force:
        raise SystemExit(
            f"{SKEMPI_SPLIT} already exists. It is a frozen artefact -- rule 1 of the plan "
            f"says never regenerate it. Pass --force only if you mean to."
        )

    ds = pd.read_parquet(paths.DATASET)
    ds["pdb_u"] = ds["pdb"].astype(str).str.upper()
    ds["mut_key"] = ds["mutations"].map(canonical_mutations)

    # GroupKFold is deterministic and needs no seed, but the plan asks for seed 0, so the
    # complex order is shuffled with it first. That makes the seed meaningful and recorded.
    rng = np.random.default_rng(SEED)
    complexes = np.array(sorted(ds["pdb_u"].unique()))
    rng.shuffle(complexes)
    order = {c: i for i, c in enumerate(complexes)}
    ds = ds.assign(_order=ds["pdb_u"].map(order)).sort_values("_order", kind="stable")

    folds = np.empty(len(ds), dtype=int)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (_, te) in enumerate(gkf.split(ds, groups=ds["pdb_u"])):
        folds[te] = k
    ds["fold"] = folds

    # no complex may straddle two folds
    straddle = ds.groupby("pdb_u")["fold"].nunique()
    assert (straddle == 1).all(), f"complexes in >1 fold: {straddle[straddle > 1].index.tolist()}"

    payload: dict = {
        "name": "skempi_abag_5fold_by_complex",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "grouped_by": "pdb id",
        "n_rows": int(len(ds)),
        "n_complexes": int(ds["pdb_u"].nunique()),
        "note": (
            "Grouped by complex, which is LOOSER than the project's frozen data/folds.csv "
            "(grouped by homology cluster). Not comparable to the project's headline numbers."
        ),
        "folds": {},
        "row_fold": {},
        "auxiliary": {},
    }
    for k in range(N_FOLDS):
        m = ds["fold"] == k
        payload["folds"][str(k)] = {
            "test_complexes": sorted(ds.loc[m, "pdb_u"].unique().tolist()),
            "n_test_rows": int(m.sum()),
            "n_train_rows": int((~m).sum()),
        }
    payload["row_fold"] = dict(zip(ds["row_id"], ds["fold"].astype(int)))

    # ---- rule 2: dedup and per-fold leakage exclusion -------------------------------------
    own = set(zip(ds["pdb_u"], ds["mut_key"]))
    for name, rel in AUX.items():
        frame = aux_frame(name, Path(rel))
        if frame is None:
            payload["auxiliary"][name] = {"status": "csv not present (checkout is gitignored)"}
            continue
        exact_dup = frame.apply(lambda r: (r.pdb, r.mutations) in own, axis=1)
        shared_pdb = frame["pdb"].isin(ds["pdb_u"])
        per_fold = {}
        for k in range(N_FOLDS):
            test_cx = set(payload["folds"][str(k)]["test_complexes"])
            blocked = frame["pdb"].isin(test_cx)
            per_fold[str(k)] = {
                "excluded_rows": int(blocked.sum()),
                "usable_rows": int((~blocked & ~exact_dup).sum()),
            }
        payload["auxiliary"][name] = {
            "status": "ok",
            "rows": int(len(frame)),
            "complexes": int(frame["pdb"].nunique()),
            "complexes_shared_with_skempi": int(frame.loc[shared_pdb, "pdb"].nunique()),
            "rows_in_shared_complexes": int(shared_pdb.sum()),
            "exact_pdb_mutation_duplicates_of_skempi": int(exact_dup.sum()),
            "per_fold": per_fold,
        }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    SKEMPI_SPLIT.write_text(json.dumps(payload, indent=2))
    return payload


def load() -> dict:
    if not SKEMPI_SPLIT.exists():
        raise SystemExit(f"{SKEMPI_SPLIT} not built -- run: python -m src.fusion.splits_frozen")
    return json.loads(SKEMPI_SPLIT.read_text())


def fold_of(row_ids: pd.Series) -> pd.Series:
    """Fold id per row_id, from the frozen file. Raises if a row is not in it."""
    table = load()["row_fold"]
    out = row_ids.map(table)
    if out.isna().any():
        missing = row_ids[out.isna()].tolist()[:5]
        raise RuntimeError(f"{int(out.isna().sum())} rows not in the frozen split, e.g. {missing}")
    return out.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild a split that already exists")
    p = build(force=ap.parse_args().force)

    print(f"wrote {SKEMPI_SPLIT.relative_to(paths.ROOT)}")
    print(f"  {p['n_rows']} rows, {p['n_complexes']} complexes, {p['n_folds']} folds, "
          f"seed {p['seed']}, grouped by {p['grouped_by']}")
    for k, f in p["folds"].items():
        print(f"  fold {k}: {f['n_test_rows']:4d} test rows over "
              f"{len(f['test_complexes']):2d} complexes, {f['n_train_rows']:4d} train rows")
    print("\nauxiliary datasets (rule 2 leakage accounting):")
    for name, a in p["auxiliary"].items():
        if a.get("status") != "ok":
            print(f"  {name:8s} {a['status']}")
            continue
        print(f"  {name:8s} {a['rows']:5d} rows, {a['complexes']:3d} complexes; "
              f"{a['complexes_shared_with_skempi']:3d} shared with skempi "
              f"({a['rows_in_shared_complexes']:5d} rows), "
              f"{a['exact_pdb_mutation_duplicates_of_skempi']:4d} exact (pdb,mutation) duplicates")
        usable = [a["per_fold"][str(k)]["usable_rows"] for k in range(p["n_folds"])]
        print(f"           usable per fold after exclusion: {usable}")


if __name__ == "__main__":
    main()
