"""Publish a perturbation run into ``reports/<name>/`` so the dashboard can analyse it.

The perturbation trainers write an out-of-fold table of their own shape
(``row_id, ddg_true, ddg_pred, fold, seed``) into ``results/oof/``. Every consumer of a run
-- ``available_runs`` for the dropdown, ``analyse`` for the error tab, ``catalog`` for the
Runs table -- reads ``reports/<name>/predictions.csv`` and its three companions instead, and
a directory without them is invisible to all three. That is why the perturbation runs never
appeared in the run selector: they existed on disk but in the wrong shape.

Run: ``python -m src.perturb.publish v2_full [--name perturb_v2_full]``
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import evaluate, paths
from src import splits

OOF_DIR = paths.ROOT / "results" / "oof"


def publish(exp: str, name: str | None = None, oof: str | None = None,
            model: str = "perturb_v2", grouping: str = "cluster",
            dataset: str = "skempi_abag", note: str = "") -> str:
    src = paths.ROOT / oof if oof else OOF_DIR / f"{exp}.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found -- nothing to publish for {exp}")

    d = pd.read_csv(src).rename(columns={"ddg_true": "y_true", "ddg_pred": "y_pred"})
    d = d.dropna(subset=["y_pred"])
    if "seed" in d.columns and d.seed.nunique() > 1:
        # Several seeds over the same rows: average them, which is what the run's headline
        # number means. Leaving them stacked would count every row once per seed.
        d = (d.groupby(["row_id", "fold"], as_index=False)
               .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))

    d["complex"] = d.row_id.str.rsplit("|", n=1).str[0]
    try:
        meta = splits.load(grouping)[["row_id", "cluster"]]
        d = d.merge(meta, on="row_id", how="left")
        d["cluster"] = d.cluster.fillna(d.complex)
    except SystemExit:
        d["cluster"] = d["complex"]              # split not built; degrade visibly

    ordered = d[["row_id", "complex", "cluster", "fold", "y_true", "y_pred"]]

    run = name or f"perturb_{exp}"
    out = paths.REPORTS / run
    out.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"metrics": evaluate.metrics(ordered), "ci": None}, indent=2))
    evaluate.per_complex(ordered).to_csv(out / "per_complex.csv", index=False)
    (out / "run.json").write_text(json.dumps({
        "name": run, "source": "src.perturb", "exp": exp, "seed": 0,
        "dataset": dataset, "model": model, "features": [],
        "grouping_for_cluster_labels": grouping, "residual": False,
        "min_group": evaluate.MIN_GROUP,
        "note": note or f"perturbation run {exp}; out-of-fold predictions from "
                        f"{src.relative_to(paths.ROOT)}",
    }, indent=2))
    m = evaluate.metrics(ordered)
    print(f"published reports/{run}/  ({len(ordered)} rows, "
          f"{ordered.complex.nunique()} complexes)")
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()
                           if isinstance(v, (int, float))))
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp", help="experiment name, e.g. v2_full")
    ap.add_argument("--name", default=None, help="report directory name")
    ap.add_argument("--oof", default=None, help="OOF csv, relative to the repo root")
    ap.add_argument("--model", default="perturb_v2")
    a = ap.parse_args()
    publish(a.exp, a.name, a.oof, a.model)


if __name__ == "__main__":
    main()
