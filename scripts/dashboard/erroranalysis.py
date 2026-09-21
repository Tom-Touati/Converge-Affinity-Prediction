"""Error-analysis slices for the dashboard, computed from any run's predictions.csv.

Every slice here answers "where is this model weak", not "how good is it", so each one
reports its own n alongside the metric: a per-complex Spearman over three complexes is not
comparable to one over twenty, and the difference is usually what makes a slice look good.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
MIN_GROUP = 5           # lowered from 10: the old rule dropped 27 of 54 complexes, and the
                        # dropped ones carry the highest error. See evaluate.MIN_GROUP.


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return None
    v = float(np.corrcoef(ra, rb)[0, 1])
    return None if not np.isfinite(v) else v


def _by(g: pd.DataFrame, key: str, min_group: int):
    v = [_spearman(x.y_true, x.y_pred) for _, x in g.groupby(key) if len(x) >= min_group]
    return [x for x in v if x is not None]


def _block(g: pd.DataFrame) -> dict:
    """Micro (pooled) and macro together, at both thresholds and both grouping units.

    Four numbers rather than one because they disagree, and the disagreement is the finding:
    micro is carried by between-complex spread, macro at 10 is computed on the easy half of
    the data, and a per-complex average weights a ten-complex cluster ten times over a
    singleton.
    """
    err = (g.y_pred - g.y_true).to_numpy()
    per5 = _by(g, "complex", 5)
    per10 = _by(g, "complex", 10)
    out = {
        "n": int(len(g)),
        "n_complexes": int(g.complex.nunique()),
        "n_counted": len(per5),
        "n_counted10": len(per10),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "sd_true": float(g.y_true.std()),
        "micro_rho": _spearman(g.y_true, g.y_pred),
        "macro_rho": float(np.mean(per5)) if per5 else None,
        "macro_rho10": float(np.mean(per10)) if per10 else None,
    }
    if "cluster" in g.columns:
        cl = _by(g, "cluster", 5)
        out["n_clusters"] = int(g.cluster.nunique())
        out["n_clusters_counted"] = len(cl)
        out["cluster_rho"] = float(np.mean(cl)) if cl else None
    return out


def load(run: str) -> pd.DataFrame | None:
    p = ROOT / "reports" / run / "predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d.dropna(subset=["y_pred"])


def analyse(run: str) -> dict:
    d = load(run)
    if d is None or d.empty:
        return {"error": f"no predictions for {run}"}
    d = d.copy()
    d["err"] = d.y_pred - d.y_true
    out: dict = {"run": run, "overall": _block(d)}

    # ---- direction of effect. The dataset is heavily destabilising, so a model can score
    # well overall while being useless on the stabilising mutations that matter for design.
    def direction(v):
        return "stabilising (<-0.5)" if v < -0.5 else (
            "destabilising (>+0.5)" if v > 0.5 else "neutral (|ddG|<=0.5)")
    d["direction"] = d.y_true.map(direction)
    out["direction"] = {k: _block(g) for k, g in d.groupby("direction")}

    # ---- support: how much data the complex itself contributed
    n_per = d.groupby("complex").size()
    d["cx_n"] = d.complex.map(n_per)
    bins = [0, 5, 10, 20, 50, 10 ** 6]
    labels = ["1-5", "6-10", "11-20", "21-50", "50+"]
    d["support"] = pd.cut(d.cx_n, bins=bins, labels=labels, right=True)
    out["support"] = {str(k): _block(g) for k, g in d.groupby("support", observed=True)}

    # ---- per-complex scatter: size against error, the picture behind the slice tables
    pts = []
    for cx, g in d.groupby("complex"):
        pts.append({
            "complex": cx, "n": int(len(g)),
            "rmse": float(np.sqrt(np.mean(g.err ** 2))),
            "bias": float(g.err.mean()),
            "sd_true": float(g.y_true.std()),
            "rho": _spearman(g.y_true, g.y_pred) if len(g) >= MIN_GROUP else None,
            "fold": int(g.fold.iloc[0]) if "fold" in g else -1,
        })
    out["per_complex"] = sorted(pts, key=lambda r: -r["n"])

    if "cluster" in d.columns:
        cpts = []
        for cl, g in d.groupby("cluster"):
            cpts.append({
                "cluster": cl, "n": int(len(g)), "n_complexes": int(g.complex.nunique()),
                "rmse": float(np.sqrt(np.mean(g.err ** 2))),
                "sd_true": float(g.y_true.std()),
                "rho": _spearman(g.y_true, g.y_pred) if len(g) >= MIN_GROUP else None,
            })
        out["per_cluster"] = sorted(cpts, key=lambda r: -r["n"])

    # ---- structural-similarity tiers (TM-score > 0.8 against the fold's training set)
    tp = ROOT / "data" / "tm_tiers.csv"
    if tp.exists() and "fold" in d:
        t = pd.read_csv(tp)
        m = d.merge(t, on=["fold", "complex"], how="left")
        if m.tier.notna().any():
            out["tier"] = {str(k): _block(g) for k, g in m.dropna(subset=["tier"]).groupby("tier")}
            out["tier_counts"] = (t.drop_duplicates(["fold", "complex"])
                                  .groupby("tier").size().to_dict())
    else:
        out["tier_note"] = "run `python -m src.tmscore` to build data/tm_tiers.csv"

    # ---- other thin slices worth naming rather than averaging over
    extra = {}
    if "cluster" in d:
        cl_n = d.groupby("cluster").size()
        thin = cl_n[cl_n < 30].index
        if len(thin):
            extra["clusters with <30 rows"] = _block(d[d.cluster.isin(thin)])
            extra["clusters with >=30 rows"] = _block(d[~d.cluster.isin(thin)])
    big = d[d.y_true.abs() > 2]
    if len(big) >= 10:
        extra["|ddG| > 2 (large effects)"] = _block(big)
    small = d[d.y_true.abs() <= 0.5]
    if len(small) >= 10:
        extra["|ddG| <= 0.5 (within noise)"] = _block(small)
    out["extra"] = extra
    return out


def available_runs() -> list:
    r = ROOT / "reports"
    return sorted(p.name for p in r.iterdir()
                  if p.is_dir() and (p / "predictions.csv").exists()) if r.exists() else []
