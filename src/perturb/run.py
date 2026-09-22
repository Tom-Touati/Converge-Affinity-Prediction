"""Single entry point: one experiment from its config, over folds and seeds.

    python -m src.perturb.run --config configs/full.yaml
    python -m src.perturb.run --config configs/no_structure.yaml --fold 0 --seed 0

Every number in ``results/summary.md`` must be reproducible from a config plus the frozen
split, so the config carries the whole model configuration and nothing is passed on the command
line except which fold and seed to run.

Writes one row per (experiment, fold, seed) to ``results/results.csv`` and out-of-fold
predictions to ``results/oof/<exp>.csv``. The error analysis reads only the latter.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src import paths
from src.fusion import results as R
from src.perturb import data as D
from src.perturb.model import PerturbConfig
from src.perturb.train import MAX_EPOCHS, pearson, train_fold

OOF_DIR = R.RESULTS_DIR / "oof"
SEEDS = (0, 1, 2)


def metrics(y_true, y_pred, complexes=None) -> dict:
    """Section 2's metric list, plus per-complex Pearson over complexes with >= 5 mutations."""
    from scipy import stats
    from sklearn.metrics import f1_score

    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    edges = (0.5, 2.0)
    ct = np.digitize(np.abs(y_true), edges)
    cp = np.digitize(np.abs(y_pred), edges)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())

    out = {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "pearson": pearson(y_pred, y_true),
        "spearman": float(stats.spearmanr(y_pred, y_true)[0]) if len(y_true) > 2 else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "acc3": float(np.mean(ct == cp)),
        "f1_macro3": float(f1_score(ct, cp, average="macro", labels=[0, 1, 2], zero_division=0)),
    }
    if complexes is not None:
        per = []
        for c in pd.unique(complexes):
            m = np.asarray(complexes) == c
            if m.sum() >= 5:
                r = pearson(y_pred[m], y_true[m])
                if not np.isnan(r):
                    per.append(r)
        out["per_complex_pearson"] = float(np.mean(per)) if per else float("nan")
    return out


def load_config(path: Path) -> tuple[str, PerturbConfig, dict]:
    raw = yaml.safe_load(path.read_text())
    exp = raw.pop("exp")
    extra = {k: raw.pop(k) for k in list(raw) if k in ("max_epochs", "seeds", "note")}
    return exp, PerturbConfig(**raw), extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = ap.parse_args()

    exp, cfg, extra = load_config(cli.config)
    seeds = (cli.seed,) if cli.seed is not None else tuple(extra.get("seeds", SEEDS))
    max_epochs = int(extra.get("max_epochs", MAX_EPOCHS))

    rows = D.load_rows()
    folds = [cli.fold] if cli.fold is not None else sorted({r.fold for r in rows})
    meta = pd.read_parquet(paths.DATASET)[["row_id", "#Pdb", "mutations"]]
    meta = meta.rename(columns={"#Pdb": "complex_key"}).set_index("row_id")
    sha = R.git_sha()
    print(f"=== {exp} === folds {folds} seeds {seeds} device {cli.device}")
    print(f"    {cfg}")

    collected = []
    for fold in folds:
        for seed in seeds:
            t0 = time.time()
            try:
                ids, yt, yp, epochs, mins, n_params, _ = train_fold(
                    rows, fold, cfg, seed, cli.device, max_epochs=max_epochs,
                    history_name=exp)
            except Exception:
                tb = traceback.format_exc()
                R.record_failure(exp, fold, seed, tb)
                print(f"  FAILED fold {fold} seed {seed}: {tb.splitlines()[-1]}")
                continue

            cx = meta.loc[ids, "complex_key"].to_numpy()
            m = metrics(yt, yp, cx)
            m.update(exp=exp, split="frozen5", fold=fold, seed=seed,
                     n_train=len([r for r in rows if r.fold != fold]), n_test=len(ids),
                     params=n_params, train_minutes=round(mins, 2), git_sha=sha,
                     dataset="skempi_abag", note=extra.get("note", ""))
            R.append(m)
            collected.append(pd.DataFrame({
                "row_id": ids, "pdb": [i.split("_")[0] for i in ids],
                "mutation": meta.loc[ids, "mutations"].to_numpy(),
                "ddg_true": yt, "ddg_pred": yp, "fold": fold, "seed": seed,
            }))
            print(f"  fold {fold} seed {seed}: pearson {m['pearson']:+.3f} "
                  f"spearman {m['spearman']:+.3f} rmse {m['rmse']:.3f} "
                  f"per-cx {m.get('per_complex_pearson', float('nan')):+.3f} "
                  f"| {epochs} epochs, {mins:.1f} min", flush=True)

    if collected:
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        out = OOF_DIR / f"{exp}.csv"
        frame = pd.concat(collected, ignore_index=True)
        if out.exists():                    # keep folds/seeds from earlier partial runs
            frame = pd.concat([pd.read_csv(out), frame], ignore_index=True)
            frame = frame.drop_duplicates(["row_id", "fold", "seed"], keep="last")
        frame.to_csv(out, index=False)
        pooled = metrics(frame.ddg_true, frame.ddg_pred)
        print(f"\n  pooled over {len(frame)} predictions: pearson {pooled['pearson']:+.3f} "
              f"spearman {pooled['spearman']:+.3f} rmse {pooled['rmse']:.3f}")
        print(f"  wrote {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
