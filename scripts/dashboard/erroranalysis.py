"""Error-analysis slices for the dashboard, computed from any run's predictions.csv.

Every slice here answers "where is this model weak", not "how good is it", so each one
reports its own n alongside the metric: a per-complex Spearman over three complexes is not
comparable to one over twenty, and the difference is usually what makes a slice look good.
"""
from __future__ import annotations

import pathlib

import json

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
    sd = float(g.y_true.std())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    out = {
        "n": int(len(g)),
        "n_complexes": int(g.complex.nunique()),
        "n_counted": len(per5),
        "n_counted10": len(per10),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        # RMSE against the only baseline that needs no model: predict this slice's own mean.
        # Correlation says whether the ordering is right and stays silent about magnitude, so
        # a slice can show a healthy rho while every prediction is off by 2 kcal/mol. Below 1
        # the model beats the slice mean; at or above 1 it does not, whatever rho says.
        "rmse_over_sd": (rmse / sd) if sd and sd == sd and sd > 0 else None,
        "bias": float(np.mean(err)),
        "sd_true": sd,
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
    """Predictions joined to the mutation metadata the slices need.

    predictions.csv carries only row_id/complex/cluster/fold/y_true/y_pred, so mut_side and
    the mutation string have to come back from the dataset. The join is on row_id and is
    strictly optional -- if the dataset is missing, the structural slices still work.
    """
    p = ROOT / "reports" / run / "predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).dropna(subset=["y_pred"])
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from src import splits
        meta = splits.load()[["row_id", "mut_side", "mutations", "location"]]
        d = d.merge(meta, on="row_id", how="left")
        d["n_mut"] = d.mutations.fillna("").str.count(",") + 1
        d["multiplicity"] = np.where(d.n_mut > 1, "multi-point", "single-point")
    except Exception:
        pass
    return d


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

    # ---- what kind of mutation. Both of these looked decisive earlier in this project and
    # both collapsed once micro was replaced by macro (antibody-side +0.152 -> -0.048,
    # multi-point +0.222 -> -0.006), so they are reported at both aggregations here.
    if "multiplicity" in d.columns:
        out["multiplicity"] = {str(k): _block(g) for k, g in d.groupby("multiplicity")}
        by_n = d[d.n_mut <= 4].copy()
        by_n["k"] = by_n.n_mut.map(lambda v: f"{int(v)}-point")
        out["n_mut"] = {str(k): _block(g) for k, g in by_n.groupby("k")}
    if "mut_side" in d.columns and d.mut_side.notna().any():
        out["mut_side"] = {str(k): _block(g) for k, g in d.dropna(subset=["mut_side"]).groupby("mut_side")}

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
        # The per-fold tier is constant here (every complex is `hard`, because the homology
        # clustering keeps each structural twin in its own fold), so it is reported for
        # completeness but discriminates nothing. The intrinsic tier is the one that varies:
        # it asks whether the antigen fold has ANY relative in SKEMPI-AB, split aside, and
        # separates the 8 complexes that no split could ever make easy.
        if "tier" in m and m.tier.notna().any():
            out["tier"] = {str(k): _block(g) for k, g in m.dropna(subset=["tier"]).groupby("tier")}
            out["tier_counts"] = (t.drop_duplicates(["fold", "complex"])
                                  .groupby("tier").size().to_dict())
        if "intrinsic_tier" in m and m.intrinsic_tier.notna().any():
            mi = m.dropna(subset=["intrinsic_tier"])
            out["intrinsic"] = {str(k): _block(g) for k, g in mi.groupby("intrinsic_tier")}
            out["intrinsic_counts"] = (t.drop_duplicates("complex")
                                       .groupby("intrinsic_tier").size().to_dict())
            hard = mi[mi.intrinsic_tier == "hard"]
            out["intrinsic_hard_complexes"] = sorted(hard.complex.unique().tolist())
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


#: Runs with no dataset recorded are this project's own antibody-antigen SKEMPI subset --
#: every run that existed before the ProtAttBA benchmarks were added is ours.
OUR_DATASET = "skempi_abag"


def run_dataset(d) -> str:
    """Which dataset a run was scored on, from its run.json."""
    try:
        return json.loads((d / "run.json").read_text()).get("dataset") or OUR_DATASET
    except Exception:
        return OUR_DATASET


def available_runs() -> list:
    """[{run, dataset, when, age}], newest first, so the UI can find the run that just landed.

    A bare list of names cannot express which dataset a run belongs to, and the whole point
    of the tick box is that a benchmark number must never be mistaken for one of ours.

    Sorted by mtime, not by name. A hundred and forty runs have accumulated; alphabetical
    order buries whatever finished a minute ago somewhere in the middle and puts a benchmark
    from another dataset at the top. ``age`` splits the list so the UI can group the handful
    that are current apart from the archive.
    """
    import time

    r = ROOT / "reports"
    if not r.exists():
        return []
    out = []
    for p in r.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        f = p / "predictions.csv"
        if not f.exists():
            continue
        m = f.stat().st_mtime
        out.append({"run": p.name, "dataset": run_dataset(p), "mtime": m,
                    "when": time.strftime("%d %b %H:%M", time.localtime(m))})
    out.sort(key=lambda x: -x["mtime"])
    for i, x in enumerate(out):
        x["age"] = "recent" if i < 12 else "earlier"
    return out


_CATALOG: dict = {}


def catalog(live: set | None = None) -> list:
    """One row per run: what it is, when it landed, and its headline numbers.

    Ninety-odd runs have accumulated and a dropdown of them sorted alphabetically is useless
    for finding the one that just finished. Rows are newest first and carry the metrics from
    metrics.json rather than recomputing, so the listing stays cheap; the parse is memoised on
    (path, mtime) because the poller hits this endpoint on a timer.

    `live` marks the configurations belonging to the sweep currently running on the VM, which
    the poller knows from the sweep CSV it is already pulling.
    """
    live = live or set()
    out = []
    rdir = ROOT / "reports"
    if not rdir.exists():
        return out
    for d in rdir.iterdir():
        pred = d / "predictions.csv"
        if not d.is_dir() or not pred.exists():
            continue
        mtime = pred.stat().st_mtime
        cached = _CATALOG.get(d.name)
        if cached and cached["_mtime"] == mtime:
            out.append(cached)
            continue
        row = {"_mtime": mtime, "run": d.name,
               "when": __import__("datetime").datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M"),
               "model": None, "rows": None, "macro5": None, "macro10": None,
               "global": None, "rmse": None, "conc": None, "residual": None,
               "min_group": None}
        try:
            m = json.loads((d / "metrics.json").read_text())["metrics"]
            # min_group travels with the number. Runs written before the threshold moved from
            # 10 to 5 stored per_complex_spearman AT 10, so their macro5 column is really a
            # macro10 and the two read identically. Showing the threshold makes that visible
            # instead of silently comparing two different metrics down one column.
            row["min_group"] = m.get("min_group")
            row.update(rows=m.get("n"), macro5=m.get("per_complex_spearman"),
                       macro10=m.get("per_complex_spearman_min10"),
                       global_=m.get("global_spearman"), rmse=m.get("rmse"),
                       conc=m.get("concordance_micro_m05"))
        except Exception:
            pass
        try:
            r = json.loads((d / "run.json").read_text())
            row["model"] = r.get("model")
            cfg = r.get("config") or {}
            row["residual"] = cfg.get("residual")
        except Exception:
            pass
        _CATALOG[d.name] = row
        out.append(row)
    for r in out:
        r["live"] = r["run"] in live
    return sorted(out, key=lambda r: -r["_mtime"])
