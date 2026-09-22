"""Write the perturbation runs into the dashboard's sweep schema.

The dashboard's Configurations table expects the column names the rank-fusion sweep used
(noise, fdrop, drop, wd, d, train_rho, gap, headline metrics). A row without them renders as
"undefined" in every cell, which looks like a broken run rather than a different one. This
fills what applies and leaves the rest blank, reading the per-fold results the poller pulls
off the VM.

Run: ``python scripts/perturb_sweep_row.py``   (safe to re-run; it rewrites only its own rows)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "reports" / "rank_fusion_sweep.csv"
COLS = ["name", "noise", "fdrop", "drop", "wd", "d", "train_rho", "gap",
        "per_cx_rho", "global_rho", "rmse", "minutes", "note", "resid", "min_grp",
        "dataset", "model"]


def rows_for(exp_dir: Path) -> dict | None:
    res = next(exp_dir.glob("*_results.csv"), None)
    if res is None:
        return None
    d = pd.read_csv(res)
    if d.empty:
        return None
    hist = exp_dir / "history.csv"
    train_rho = ""
    if hist.exists():
        h = pd.read_csv(hist)
        v = pd.to_numeric(h.get("val_rho"), errors="coerce")
        train_rho = round(float(v.max()), 3) if v.notna().any() else ""
    return {
        "name": exp_dir.name,
        "noise": "", "fdrop": "", "drop": 0.2, "wd": 0.01, "d": 256,
        "train_rho": train_rho, "gap": "",
        "per_cx_rho": "", "global_rho": round(float(d.pearson.mean()), 3),
        "rmse": round(float(d.rmse.mean()), 3),
        "minutes": round(float(d.train_minutes.sum()), 1),
        "note": f"{len(d)} of 15 (fold,seed) done",
        "resid": False, "min_grp": 5, "dataset": "skempi_abag", "model": "perturb",
    }


def main() -> None:
    existing = {}
    if SWEEP.exists():
        for r in pd.read_csv(SWEEP).to_dict("records"):
            existing[r.get("name")] = r
    n = 0
    for d in sorted((ROOT / "reports").glob("perturb_*")):
        if not d.is_dir():
            continue
        row = rows_for(d)
        if row:
            existing[row["name"]] = row
            n += 1
    frame = pd.DataFrame(list(existing.values()))
    for c in COLS:
        if c not in frame.columns:
            frame[c] = ""
    frame[COLS].to_csv(SWEEP, index=False)
    print(f"updated {n} perturbation row(s) in {SWEEP.relative_to(ROOT)}")
    print(frame[frame.model == "perturb"][["name", "global_rho", "rmse", "train_rho", "note"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
