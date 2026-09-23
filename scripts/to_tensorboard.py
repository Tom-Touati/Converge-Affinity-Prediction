"""Mirror every run in ``reports/`` into TensorBoard event files.

The bespoke dashboard grew two tabs, a hundred and forty runs in one dropdown and an error
view of its own design. TensorBoard already does the three things that were actually wanted:
overlay training curves across folds and runs, sort runs by a metric, and keep the whole
history addressable. This is a one-way mirror -- it reads the CSVs the trainers already
write and never changes them, so nothing depends on it and it can be re-run at will.

Layout, chosen so TensorBoard's own run filter does the work:

    runs/tb/<run>/             one point per fold on test/*, the HParams row, the figures
    runs/tb/<run>__fold<k>/    the epoch curves for that fold

Both are top-level directories rather than one nested inside the other, because TensorBoard
derives a run's name from its path with ``os.sep``: nested, the folds come out named
``perturb_v2_full\fold0`` on Windows, and a backslash in a name the filter box treats as a
regex is unusable. Flat, typing ``perturb_v2_full`` selects the summary and all five folds,
``perturb_v2_full$`` just the summary, and ``fold0`` overlays fold 0 across every run.

    python scripts/to_tensorboard.py                 # the live comparison (see DEFAULT)
    python scripts/to_tensorboard.py arch_ ab_       # add older families by substring
    python scripts/to_tensorboard.py --all           # all 131, spaghetti, use the filter box
    tensorboard --logdir runs/tb
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src import paths  # noqa: E402

TB = paths.ROOT / "runs" / "tb"

#: TensorBoard plots every run in the log directory checked by default, so mirroring all 131
#: reproduces the problem the custom dashboard had: everything on screen at once and nothing
#: legible. The default is the live comparison instead -- the perturbation models against the
#: forest baselines on our own antibody-antigen subset -- and older families are added by
#: name. Nothing is deleted from reports/ either way; this only decides what gets mirrored.
DEFAULT = ("perturb_", "v3_", "E0")

#: history.csv exists in two schemas -- the perturbation trainers' and the older fusion
#: sweeps'. Both carry these under one name or another; anything missing is skipped rather
#: than faked, so a curve that is absent is absent rather than flat at zero.
CURVES = {
    "train/loss": ("loss", "train_loss_eval"),
    "val/pearson": ("val_rho",),
    # what early stopping now maximises, and the quantity the model is judged on
    "val/per_complex_rho": ("val_cx_rho",),
    "val/rmse": ("val_rmse",),
    "train/pearson": ("train_rho",),
    "train/loss_gap": ("loss_gap",),
    # Gradient health, measured before clipping. The per-group norms are the point: a
    # difference-of-branches head can starve the path the edit travels without anything
    # in the loss curve moving. On the first smoke run branch.gamma -- the FiLM the delta
    # is injected through -- carried gradients ~80x smaller than the attention.
    "grad/norm": ("grad_norm",),
    "grad/clipped_frac": ("grad_clipped",),
    "grad/dead_frac": ("grad_dead_frac",),
    "grad/head": ("g_head",),
    "grad/attn": ("g_attn",),
    "grad/fuse": ("g_fuse",),
    "grad/film_gamma": ("g_film",),
    "grad/film_beta": ("g_film_b",),
    "grad/reduce_seq": ("g_red_seq",),
    "grad/reduce_struct": ("g_red_str",),
    "grad/blosum": ("g_blosum",),
    "grad/chain": ("g_chain",),
    "grad/other": ("g_other",),
}


def pick(frame: pd.DataFrame, names: tuple) -> pd.Series | None:
    for n in names:
        if n in frame.columns:
            s = pd.to_numeric(frame[n], errors="coerce")
            if s.notna().any():
                return s
    return None


def read_csv(p: pathlib.Path) -> pd.DataFrame | None:
    """None for a file that is missing, empty or truncated.

    A poller writes these while a run is in flight and a killed transfer can leave a
    zero-byte file behind, so an unreadable CSV is an ordinary state here, not a bug. It
    should skip that one run rather than abort the whole mirror.
    """
    try:
        return pd.read_csv(p)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
        return None


def results_csv(d: pathlib.Path) -> pd.DataFrame | None:
    """The per-fold table, whatever the trainer happened to call it."""
    for p in sorted(d.glob("*_results.csv")):
        return read_csv(p)
    return None


def fresh(d: pathlib.Path) -> pathlib.Path:
    """Empty a run directory before writing it.

    EventFileWriter APPENDS, and TensorBoard reads every event file in a directory. Without
    this, re-running the mirror draws each scalar once per run: run/epochs came back as 52
    points for 5 folds, the same five values repeating, and a curve that looks nothing like
    the number it reports.
    """
    if d.exists():
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_curves(run: str, hist: pd.DataFrame) -> int:
    from tb_writer import Writer as SummaryWriter

    n = 0
    folds = sorted(hist.fold.dropna().unique()) if "fold" in hist.columns else [0]
    for f in folds:
        g = hist[hist.fold == f] if "fold" in hist.columns else hist
        if "epoch" in g.columns:
            g = g.sort_values("epoch")
        if g.empty:
            continue
        step = (pd.to_numeric(g.epoch, errors="coerce").fillna(0).astype(int).tolist()
                if "epoch" in g.columns else list(range(len(g))))
        w = SummaryWriter(str(fresh(TB / f"{run}__fold{int(f)}")))
        for tag, names in CURVES.items():
            s = pick(g, names)
            if s is None:
                continue
            for e, v in zip(step, s):
                if pd.notna(v):
                    w.add_scalar(tag, float(v), e)
            n += 1
        if "seconds" in g.columns:
            for e, v in zip(step, pd.to_numeric(g.seconds, errors="coerce")):
                if pd.notna(v):
                    w.add_scalar("time/seconds", float(v), e)
        w.close()
    return n


def figures(run: str, preds: pd.DataFrame, w) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Predicted against true, coloured by fold, with y=x and the fitted offset. A systematic
    # offset is invisible in a correlation and unmissable here.
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    for f, g in preds.groupby("fold"):
        ax.scatter(g.y_true, g.y_pred, s=8, alpha=0.55, label=f"fold {int(f)}")
    lo = float(min(preds.y_true.min(), preds.y_pred.min())) - 0.3
    hi = float(max(preds.y_true.max(), preds.y_pred.max())) + 0.3
    ax.plot([lo, hi], [lo, hi], lw=1, color="#888888", zorder=0)
    b = float((preds.y_pred - preds.y_true).mean())
    ax.plot([lo, hi], [lo + b, hi + b], lw=1, ls="--", color="#d95926", zorder=0,
            label=f"bias {b:+.2f}")
    ax.set_xlabel("true ddG (kcal/mol)")
    ax.set_ylabel("predicted")
    ax.set_title(f"{run}  ({len(preds)} held-out rows)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    w.add_figure("predictions/scatter", fig, 0)
    plt.close(fig)

    # Per-complex Pearson: the quantity a pooled fold correlation hides, since most of a
    # pooled r can come from differences BETWEEN complexes rather than ranking within one.
    rows = [(c, len(g), float(np.corrcoef(g.y_pred, g.y_true)[0, 1]))
            for c, g in preds.groupby("complex") if len(g) >= 5 and g.y_true.std() > 0]
    if not rows:
        return
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.22 * len(rows))), dpi=110)
    ax.barh([f"{c}  (n={n})" for c, n, _ in rows], [r for _, _, r in rows],
            color=["#d95926" if r < 0 else "#3987e5" for _, _, r in rows], height=0.7)
    ax.axvline(0, lw=0.8, color="#444444")
    ax.set_xlabel("Pearson r within complex")
    ax.tick_params(labelsize=6)
    ax.set_title(f"{run}  ({len(rows)} complexes with 5 or more rows)", fontsize=9)
    fig.tight_layout()
    w.add_figure("predictions/per_complex", fig, 0)
    plt.close(fig)


def mirror(run: str, d: pathlib.Path) -> dict:
    from tb_writer import Writer as SummaryWriter

    meta = {}
    rj = d / "run.json"
    if rj.exists():
        try:
            meta = json.loads(rj.read_text())
        except json.JSONDecodeError:
            meta = {}

    n_curves = 0
    hist = d / "history.csv"
    if hist.exists():
        try:
            h = read_csv(hist)
            if h is not None:
                n_curves = write_curves(run, h)
        except Exception as e:                  # a malformed history must not stop the rest
            print(f"  {run}: history skipped ({type(e).__name__}: {e})")

    w = SummaryWriter(str(fresh(TB / run)))
    summary = {"run": run, "curves": n_curves}

    res = results_csv(d)
    if res is not None and "fold" in res.columns:
        for _, r in res.sort_values("fold").iterrows():
            f = int(r.fold)
            for tag, col in (("test/pearson", "pearson"), ("test/rmse", "rmse"),
                             ("test/mae", "mae"), ("run/epochs", "epochs"),
                             ("run/train_minutes", "train_minutes")):
                if col in res.columns and pd.notna(r[col]):
                    w.add_scalar(tag, float(r[col]), f)
        if "pearson" in res.columns:
            summary["mean_pearson"] = float(res.pearson.mean())
        if "rmse" in res.columns:
            summary["mean_rmse"] = float(res.rmse.mean())
        summary["folds"] = len(res)
        try:
            w.add_text("per_fold", res.to_markdown(index=False), 0)
        except Exception:
            w.add_text("per_fold", res.to_string(index=False), 0)

    preds = d / "predictions.csv"
    if preds.exists():
        p = read_csv(preds)
        p = p.dropna(subset=["y_pred"]) if p is not None else None
        if p is not None and len(p) > 2:
            e = p.y_pred - p.y_true
            pooled = {
                "oof/pearson": float(np.corrcoef(p.y_pred, p.y_true)[0, 1]),
                "oof/spearman": float(p.y_pred.corr(p.y_true, method="spearman")),
                "oof/rmse": float(np.sqrt((e ** 2).mean())),
                "oof/bias": float(e.mean()),
                # The offset removed. Equal to the label sd means the model is no better
                # than predicting that run's own mean, whatever its correlation says.
                "oof/rmse_debiased": float(np.sqrt(((e - e.mean()) ** 2).mean())),
                "oof/label_sd": float(p.y_true.std()),
            }
            per_cx = [float(np.corrcoef(g.y_pred, g.y_true)[0, 1])
                      for _, g in p.groupby("complex")
                      if len(g) >= 5 and g.y_true.std() > 0]
            if per_cx:
                pooled["oof/per_complex_pearson"] = float(np.mean(per_cx))
            for k, v in pooled.items():
                w.add_scalar(k, v, 0)
            summary.update({k.split("/")[-1]: v for k, v in pooled.items()})
            summary["n_rows"] = len(p)
            try:
                figures(run, p, w)
            except Exception as ex:
                print(f"  {run}: figures skipped ({type(ex).__name__}: {ex})")

    # The HParams tab is the run table: sortable, filterable, every run in one grid.
    hp = {"model": str(meta.get("model") or "?"),
          "dataset": str(meta.get("dataset") or "?"),
          "source": str(meta.get("source") or "?"),
          "grouping": str(meta.get("grouping_for_cluster_labels") or "?"),
          "residual": str(meta.get("residual")),
          "seed": int(meta.get("seed") or 0)}
    if res is not None and "params" in res.columns and res.params.notna().any():
        hp["params"] = int(res.params.iloc[0])
    if res is not None and "split" in res.columns:
        hp["split"] = str(res.split.iloc[0])
    # add_hparams also writes each metric as a scalar in this run, so passing all of them
    # doubles every oof/* tag under an hp/* name and those cards sort to the top of the tag
    # list. Only the four a run is actually judged on go in.
    metrics = {}
    for k in ("per_complex_pearson", "mean_pearson", "rmse", "bias"):
        v = summary.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not np.isnan(float(v)):
            metrics[f"hp/{k}"] = float(v)
    if metrics:
        w.add_hparams(hp, metrics, run_name=".")
    w.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match", nargs="*",
                    help=f"only runs whose name contains one of these (default: {DEFAULT})")
    ap.add_argument("--all", action="store_true", help="mirror every run in reports/")
    ap.add_argument("--clean", action="store_true", help="delete runs/tb first")
    a = ap.parse_args()

    if a.clean and TB.exists():
        shutil.rmtree(TB)
    TB.mkdir(parents=True, exist_ok=True)

    dirs = [d for d in sorted(paths.REPORTS.iterdir())
            if d.is_dir() and not d.name.startswith("_")
            and ((d / "predictions.csv").exists() or (d / "history.csv").exists())]
    n_all = len(dirs)
    match = tuple(a.match) or (() if a.all else DEFAULT)
    if match:
        dirs = [d for d in dirs if any(m in d.name for m in match)]
    if not dirs:
        raise SystemExit(f"no run name contains any of {match}")

    out = [mirror(d.name, d) for d in dirs]
    s = pd.DataFrame(out)
    cols = [c for c in ("run", "folds", "n_rows", "mean_pearson", "per_complex_pearson",
                        "bias", "rmse", "rmse_debiased", "label_sd", "curves")
            if c in s.columns]
    s = s[cols]
    if "mean_pearson" in cols:
        s = s.sort_values("mean_pearson", ascending=False)
    print(f"\nmirrored {len(out)} of {n_all} runs into {TB.relative_to(paths.ROOT)}"
          + (f"  (matching {', '.join(match)})" if match else "  (all)") + "\n")
    print(s.to_string(index=False))
    print(f"\n  tensorboard --logdir {TB.relative_to(paths.ROOT)}")
    if match:
        print(f"  {n_all - len(out)} runs were left out; add them by name, or --all")


if __name__ == "__main__":
    main()
