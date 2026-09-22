"""Publish a fusion run into the project's own report format, so main's tooling reads it.

``src/error_analysis.py`` and ``scripts/dashboard/`` both consume
``reports/<run>/predictions.csv`` with columns ``row_id, complex, cluster, fold, y_true,
y_pred``, plus ``metrics.json`` for the dashboard's run catalogue. The fusion ladder writes its
own out-of-fold predictions to ``results/predictions/<exp>_seed<n>.csv`` instead, because the
plan asks for one results file with its own schema.

This bridges the two rather than duplicating either. After exporting, both of these work on a
fusion run with no change to them:

    python -m src.error_analysis --run E0c_rf_pooled_esm_mpnn_seed0
    python scripts/dashboard/serve.py        # the run appears in the Runs and Error tabs

Two details that make the bridge correct rather than approximate:

* **``cluster`` comes from the project's split, not from the fusion split.** A homology cluster
  is a property of the complex, computed once by ``src/splits.py`` with Smith-Waterman over
  antigen chains; it is independent of how folds were later drawn. So a run scored on the
  5-fold by-complex split still gets its true cluster labels, and the dashboard's per-cluster
  panel is meaningful.
* **``fold`` comes from the run.** ``error_analysis.load`` drops ``cluster``/``fold`` from the
  dataset side and keeps the prediction file's, which is what we want: the fold column has to
  describe the split the model was actually scored on.

``metrics.json`` is written by the project's own ``src/evaluate.metrics``, not recomputed here,
so the headline number in the dashboard is the project's definition of it.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import evaluate, paths, splits
from src.fusion import results as R


def export(exp: str, seed: int, name: str | None = None, n_boot: int = 0,
           grouping: str = "cluster") -> str:
    """Write one fusion (experiment, seed) into ``reports/<name>/``. Returns the run name."""
    src_path = R.PREDICTIONS_DIR / f"{exp}_seed{seed}.csv"
    if not src_path.exists():
        raise SystemExit(f"{src_path} not found -- run the experiment first")
    preds = pd.read_csv(src_path).dropna(subset=["y_pred"])

    # cluster labels from the project's frozen homology clustering, keyed by row_id
    try:
        meta = splits.load(grouping)[["row_id", "cluster"]]
        preds = preds.merge(meta, on="row_id", how="left")
    except SystemExit:
        preds["cluster"] = preds["complex"]      # split not built; degrade visibly

    ordered = preds[["row_id", "complex", "cluster", "fold", "y_true", "y_pred"]]

    run = name or f"{exp}_seed{seed}"
    out = paths.REPORTS / run
    out.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(out / "predictions.csv", index=False)

    m = evaluate.metrics(ordered)
    ci = evaluate.bootstrap(ordered, n_boot=n_boot) if n_boot else None
    (out / "metrics.json").write_text(json.dumps({"metrics": m, "ci": ci}, indent=2))

    per_cx = evaluate.per_complex(ordered)
    per_cx.to_csv(out / "per_complex.csv", index=False)

    (out / "run.json").write_text(json.dumps({
        "name": run, "source": "src.fusion", "exp": exp, "seed": seed,
        "model": exp, "features": [], "grouping_for_cluster_labels": grouping,
        "residual": None, "min_group": evaluate.MIN_GROUP,
        "note": "exported from results/predictions/ by src.fusion.export; "
                "fold column is the fusion split, cluster labels are the project's",
        "git_sha": R.git_sha(),
    }, indent=2))
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", required=True, help="experiment name(s)")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--name", default=None, help="report dir name (single exp/seed only)")
    ap.add_argument("--n-boot", type=int, default=0,
                    help="bootstrap iterations for metrics.json (0 = skip, it is slow)")
    cli = ap.parse_args()

    if cli.name and (len(cli.exp) > 1 or len(cli.seed) > 1):
        raise SystemExit("--name only makes sense for a single --exp and --seed")

    for exp in cli.exp:
        for seed in cli.seed:
            try:
                run = export(exp, seed, cli.name, cli.n_boot)
            except SystemExit as e:
                print(f"  skipped {exp} seed {seed}: {e}")
                continue
            m = json.loads((paths.REPORTS / run / "metrics.json").read_text())["metrics"]
            print(f"  {run:44s} per-complex rho {m['per_complex_spearman']:+.3f}  "
                  f"global rho {m['global_spearman']:+.3f}  rmse {m['rmse']:.3f}  "
                  f"n={m['n']} cx={m['n_complexes']}")
    print(f"\nreports/ now holds {sum(1 for p in paths.REPORTS.iterdir() if p.is_dir())} runs")
    print("main's tooling now reads these directly:")
    print("  python -m src.error_analysis --run <name>")
    print("  python scripts/dashboard/serve.py     (Runs and Error analysis tabs)")


if __name__ == "__main__":
    main()
