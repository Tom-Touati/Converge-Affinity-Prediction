"""Canonical paths, and the one place that knows the repo layout.

Every script resolves its inputs and outputs through this module, so behaviour does not depend
on the working directory a command happens to be launched from.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW_CSV = DATA / "skempi_v2.csv"
PDB_TGZ = DATA / "SKEMPI2_PDBs.tgz"
PDB_DIR = DATA / "PDBs"

PROCESSED = DATA / "processed"
DATASET = PROCESSED / "dataset.parquet"      # one row per (complex, mutation), the modelling table
MEASUREMENTS = PROCESSED / "measurements.parquet"  # pre-dedup rows, kept for the noise estimate

FOLDS = DATA / "folds.csv"                   # frozen split, committed to git
FEATURES = DATA / "features"                 # one file per (encoder, variant), keyed by row_id

REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"


def ensure_dirs() -> None:
    for d in (PROCESSED, FEATURES, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


def ensure_pdbs() -> Path:
    """Unpack the structure archive once; return the directory holding the .pdb files."""
    if PDB_DIR.exists() and any(PDB_DIR.glob("*.pdb")):
        return PDB_DIR
    if not PDB_TGZ.exists():
        raise FileNotFoundError(
            f"{PDB_TGZ} not found. See README 'Getting the data' for the download link."
        )
    with tarfile.open(PDB_TGZ) as t:
        t.extractall(DATA)
    if not any(PDB_DIR.glob("*.pdb")):
        raise RuntimeError(f"unpacked {PDB_TGZ.name} but {PDB_DIR} holds no .pdb files")
    return PDB_DIR
