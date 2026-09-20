"""Run one rung of the ladder over the frozen split and write its predictions.

One command, one experiment, one directory under ``reports/``. The outer test fold is never
touched for any decision -- no early stopping on it, no feature selection, no threshold fitting.

    python -m src.train --model gbt --features chem --name rung1_chem_gbt

Feature blocks are looked up by name and concatenated, so adding a modality is a change to the
``--features`` list and nothing else. Every encoder writes a cached block keyed by row_id; the
head just reads them.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate, paths, splits
from .model import MODELS


def _feature_block(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Resolve a feature block by name, preferring a cached file over recomputation."""
    cache = paths.FEATURES / f"{name}.parquet"
    if cache.exists():
        block = pd.read_parquet(cache)
        missing = set(df["row_id"]) - set(block.index)
        if missing:
            raise RuntimeError(
                f"cached block '{name}' is missing {len(missing)} row_ids; delete "
                f"{cache} and re-extract"
            )
        return block.loc[df["row_id"]]

    if name == "chem":
        from .features.chem import build
    else:
        raise KeyError(
            f"unknown feature block '{name}'. Cached blocks found: "
            f"{sorted(p.stem for p in paths.FEATURES.glob('*.parquet'))}"
        )
    block = build(df)
    paths.FEATURES.mkdir(parents=True, exist_ok=True)
    block.to_parquet(cache)
    return block.loc[df["row_id"]]


def build_matrix(df: pd.DataFrame, names: list) -> pd.DataFrame:
    blocks = []
    for n in names:
        b = _feature_block(n, df).add_prefix(f"{n}:")
        blocks.append(b.reset_index(drop=True))
    X = pd.concat(blocks, axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "uncommitted"


def run(model: str, features: list, name: str, seed: int = 0, min_group: int = 10,
        n_boot: int = 1000, verbose: bool = True, center: bool = False) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = splits.load()
    X = build_matrix(df, features)
    y = df["ddG"].to_numpy(float)
    complexes = df["#Pdb"].to_numpy()

    preds = np.full(len(df), np.nan)
    for f in sorted(df["fold"].unique()):
        te = (df["fold"] == f).to_numpy()
        tr = ~te
        est = MODELS[model](seed)

        if center:
            # Train on the within-complex deviation instead of the raw label.
            #
            # 38% of this dataset's label variance is between complexes -- an offset set by the
            # wild type's reference affinity, the assay and the temperature, none of which the
            # mutation's features can predict for an unseen complex. Squared error spends that
            # share of its gradient on it, while the headline metric (per-complex Spearman)
            # ignores it entirely, being rank-based within each complex.
            #
            # Means come from TRAINING rows only. Test complexes never appear in training under
            # grouped CV, so no test-complex mean exists to leak -- which is the point.
            tr_mean = pd.Series(y[tr]).groupby(complexes[tr]).mean()
            est.fit(X[tr], y[tr] - pd.Series(complexes[tr]).map(tr_mean).to_numpy())
            # Add the global training mean back so predictions stay on a physical scale. The
            # model still makes no attempt at per-complex offsets, so its RMSE is not comparable
            # to an uncentered run's -- only the rank metrics are.
            preds[te] = est.predict(X[te]) + y[tr].mean()
        else:
            est.fit(X[tr], y[tr])
            preds[te] = est.predict(X[te])
    assert np.isfinite(preds).all(), "some rows never landed in a test fold"

    out = pd.DataFrame({
        "row_id": df["row_id"], "complex": df["#Pdb"], "cluster": df["cluster"],
        "fold": df["fold"], "y_true": y, "y_pred": preds,
    })

    m = evaluate.metrics(out, min_group)
    ci = evaluate.bootstrap(out, min_group, n_boot=n_boot, seed=seed) if n_boot else {}
    pc = evaluate.per_complex(out, min_group)

    d = paths.REPORTS / name
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "predictions.csv", index=False)
    pc.to_csv(d / "per_complex.csv", index=False)
    (d / "metrics.json").write_text(json.dumps({"metrics": m, "ci": ci}, indent=2, default=float))
    (d / "run.json").write_text(json.dumps({
        "name": name, "model": model, "features": features, "seed": seed,
        "min_group": min_group, "n_boot": n_boot, "center": center,
        "n_folds": int(df["fold"].nunique()),
        "n_rows": len(df), "n_features": X.shape[1], "git_sha": _git_sha(),
        "python": platform.python_version(), "machine": platform.processor(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }, indent=2))

    if verbose:
        print(evaluate.format_report(m, ci, title=f"{name}  [{model} on {'+'.join(features)}]"))
        print(f"\n  per-fold headline:")
        for f in sorted(out["fold"].unique()):
            sub = out[out["fold"] == f]
            fm = evaluate.metrics(sub, min_group)
            print(f"    fold {f}:  per-complex rho {fm['per_complex_spearman']:+.3f}"
                  f"   rmse {fm['rmse']:.3f}   n={fm['n']:4d}"
                  f"   complexes {fm['n_complexes_counted']}/{fm['n_complexes']}")
        print(f"\n  wrote {d.relative_to(paths.ROOT)}/  in {time.perf_counter() - t0:.1f}s")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gbt", choices=sorted(MODELS))
    p.add_argument("--features", default="chem", help="comma-separated feature blocks")
    p.add_argument("--name", default=None, help="report directory name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-group", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--center", action="store_true",
                   help="train on the within-complex deviation instead of the raw ddG")
    a = p.parse_args()
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    run(a.model, feats, a.name or f"{a.model}_{'+'.join(feats)}", a.seed, a.min_group,
        a.n_boot, center=a.center)


if __name__ == "__main__":
    main()
