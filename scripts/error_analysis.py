"""Error analysis for the fusion ladder, on a model's out-of-fold predictions.

Writes ``results/error_analysis.md`` and plots under ``results/plots/``. Covers the sections
the overnight plan asks for, in its order.

Separate from ``src/error_analysis.py``, which is not touched. That module answers the
project's own question with complex-level bootstraps and its own slice definitions; this one
answers the plan's, on the plan's frozen 5-fold split, and adds the plan's specific plots. The
two agree on the one principle that matters here: **a slice number is read against that
slice's own spread, never on its own.** ``rmse_over_sd`` is reported for every slice, and at
1.0 the model adds nothing within that group no matter how its RMSE compares to other slices.

Two guards against the imbalances that can manufacture a finding on this data:

* ``n_cx`` and ``top_cx_share`` next to every slice. 53 complexes, median 10 rows, and the
  three largest hold a quarter of the data, so a slice can be one complex wearing a label. At
  ``top_cx_share`` 0.6 that is what it is.
* macro (per complex, then averaged) alongside micro (rows pooled). Under grouped CV they
  disagree and the disagreement is informative rather than noise.

Run: ``python scripts/error_analysis.py --exp E0a_rf_handcrafted [--seed 0]``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import paths  # noqa: E402
from src.fusion import metrics as FM  # noqa: E402
from src.fusion import results as R  # noqa: E402

REPORT = R.RESULTS_DIR / "error_analysis.md"

#: SKEMPI's interface codes. COR/SUP/RIM are at the interface, INT/SUR are not.
INTERFACE_CODES = {"COR", "SUP", "RIM"}


def load(exp: str, seed: int) -> pd.DataFrame:
    """Out-of-fold predictions joined to every column the slices need."""
    path = R.PREDICTIONS_DIR / f"{exp}_seed{seed}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the experiment first.")
    preds = pd.read_csv(path)
    ds = pd.read_parquet(paths.DATASET)
    chem = pd.read_parquet(paths.FEATURES / "chem.parquet")

    m = preds.merge(ds.drop(columns=[c for c in ("fold",) if c in ds.columns]),
                    on="row_id", validate="1:1")
    m = m.merge(chem[["abs_d_charge", "n_to_ala"]], left_on="row_id", right_index=True,
                how="left")

    m["resid"] = m.y_pred - m.y_true            # positive = over-predicts ddG
    m["abs_resid"] = m.resid.abs()
    m["abs_true"] = m.y_true.abs()
    m["is_single"] = m.n_mut == 1
    m["to_alanine"] = m.is_alanine
    m["charge_change"] = m.abs_d_charge.fillna(0) > 0
    # A multi-point row lists one code per position; call it interface if any position is.
    m["interface"] = m.location.fillna("").apply(
        lambda s: any(c in INTERFACE_CODES for c in str(s).split(",")))
    m["magnitude_bin"] = pd.cut(m.abs_true, [-0.01, 0.5, 1.0, 2.0, 1e9],
                                labels=["<=0.5", "0.5-1", "1-2", ">2"])
    return m


def _slice_row(g: pd.DataFrame) -> dict:
    sd = float(g.y_true.std(ddof=0))
    rmse = float(np.sqrt(np.mean(g.resid ** 2)))
    per_cx = [FM.score(h.y_true, h.y_pred)["spearman"]
              for _, h in g.groupby("complex") if len(h) >= 5]
    per_cx = [v for v in per_cx if not np.isnan(v)]
    top = g.complex.value_counts()
    return {
        "n": len(g),
        "n_cx": int(g.complex.nunique()),
        "top_cx_share": round(float(top.iloc[0] / len(g)), 2) if len(top) else np.nan,
        "sd_true": round(sd, 3),
        "rmse": round(rmse, 3),
        "rmse_over_sd": round(rmse / sd, 3) if sd > 0 else np.nan,
        "mae": round(float(g.abs_resid.mean()), 3),
        "bias": round(float(g.resid.mean()), 3),
        "micro_rho": round(FM.score(g.y_true, g.y_pred)["spearman"], 3),
        "macro_rho": round(float(np.mean(per_cx)), 3) if per_cx else np.nan,
    }


def slice_table(m: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = {}
    for key, g in m.groupby(by, dropna=False, observed=True):
        rows[str(key)] = _slice_row(g)
    return pd.DataFrame(rows).T


def md(df: pd.DataFrame, index_name: str) -> str:
    out = df.reset_index().rename(columns={"index": index_name})
    head = "| " + " | ".join(out.columns) + " |"
    rule = "|" + "|".join(["---"] * len(out.columns)) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in out.itertuples(index=False)]
    return "\n".join([head, rule] + body)


# ----------------------------------------------------------------------------------- plots
def plot_scatter(m: pd.DataFrame, exp: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k, g in m.groupby("fold"):
        ax.scatter(g.y_true, g.y_pred, s=14, alpha=0.6, label=f"fold {k}")
    lo, hi = float(min(m.y_true.min(), m.y_pred.min())), float(max(m.y_true.max(), m.y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    s = FM.score(m.y_true, m.y_pred)
    ax.set(xlabel="true ddG (kcal/mol)", ylabel="predicted ddG (kcal/mol)",
           title=f"{exp}\npooled r={s['pearson']:.3f}  rho={s['spearman']:.3f}  "
                 f"rmse={s['rmse']:.3f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    out = R.PLOTS_DIR / f"{exp}_scatter.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_residual_vs_magnitude(m: pd.DataFrame, exp: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.scatter(m.abs_true, m.resid, s=12, alpha=0.45)
    ax.axhline(0, color="k", lw=1)
    # binned mean residual: regression to the mean shows up as a downward slope
    bins = np.array([0, 0.5, 1, 2, 3, 5, 100.0])
    idx = np.digitize(m.abs_true, bins) - 1
    centres, means = [], []
    for b in range(len(bins) - 1):
        g = m[idx == b]
        if len(g) >= 8:
            centres.append(g.abs_true.mean()); means.append(g.resid.mean())
    ax.plot(centres, means, "o-", color="crimson", label="binned mean residual")
    ax.set(xlabel="|true ddG| (kcal/mol)", ylabel="residual (pred - true)",
           title=f"{exp}: do we regress to the mean on large effects?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    out = R.PLOTS_DIR / f"{exp}_residual_vs_magnitude.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_per_complex(per_cx: pd.DataFrame, exp: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(per_cx))
    ax.bar(x, per_cx.spearman, color=["crimson" if v < 0 else "steelblue"
                                      for v in per_cx.spearman])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(per_cx.index, per_cx.n)],
                       rotation=90, fontsize=5.5)
    ax.axhline(0, color="k", lw=1)
    ax.set(ylabel="Spearman within complex", title=f"{exp}: per-complex Spearman (n>=5)")
    ax.grid(alpha=0.25, axis="y")
    out = R.PLOTS_DIR / f"{exp}_per_complex.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="E0a_rf_handcrafted")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    m = load(cli.exp, cli.seed)
    R.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    overall = FM.score(m.y_true, m.y_pred, complexes=m.complex)

    per_cx = pd.DataFrame({
        c: {"n": len(g), "spearman": FM.score(g.y_true, g.y_pred)["spearman"],
            "rmse": float(np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)))}
        for c, g in m.groupby("complex") if len(g) >= 5
    }).T.sort_values("spearman")
    per_cx["n"] = per_cx["n"].astype(int)

    p1 = plot_scatter(m, cli.exp)
    p2 = plot_residual_vs_magnitude(m, cli.exp)
    p3 = plot_per_complex(per_cx, cli.exp)

    lines = [
        f"# Error analysis — {cli.exp} (seed {cli.seed})", "",
        "Out-of-fold predictions on the frozen 5-fold by-complex split",
        "(`data/splits/skempi_abag_5fold_by_complex.json`).", "",
        "Read every slice against its own `sd_true`: **`rmse_over_sd` >= 1 means the model adds",
        "nothing within that slice**, whatever its RMSE looks like next to other slices. And",
        "check `top_cx_share` before believing a slice — at 0.6 the slice result is one",
        "complex's result wearing a category label.", "",
        "## Overall", "",
        f"| n | complexes | pearson | spearman | rmse | mae | acc3 | f1_macro3 | per-complex rho |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {len(m)} | {m.complex.nunique()} | {overall['pearson']:.3f} | "
        f"{overall['spearman']:.3f} | {overall['rmse']:.3f} | {overall['mae']:.3f} | "
        f"{overall['acc3']:.3f} | {overall['f1_macro3']:.3f} | "
        f"{overall['per_complex_spearman']:.3f} |", "",
        f"![scatter]({p1.relative_to(R.RESULTS_DIR).as_posix()})", "",
        "## Regression to the mean", "",
        f"![residual]({p2.relative_to(R.RESULTS_DIR).as_posix()})", "",
        md(slice_table(m, "magnitude_bin"), "|ddG| bin"), "",
        "A negative bias that grows with |ddG| is the signature of shrinking large effects",
        "toward the mean.", "",
        "## Per complex", "",
        f"![per complex]({p3.relative_to(R.RESULTS_DIR).as_posix()})", "",
        f"{len(per_cx)} of {m.complex.nunique()} complexes have n >= 5. The five worst:", "",
        md(per_cx.head(5).round(3), "complex"), "",
        "The five best:", "",
        md(per_cx.tail(5).iloc[::-1].round(3), "complex"), "",
        "## By mutation category", "",
        "### to-alanine vs other", "", md(slice_table(m, "to_alanine"), "to_alanine"), "",
        "### charge change vs none", "", md(slice_table(m, "charge_change"), "charge_change"), "",
        "### interface vs non-interface (SKEMPI COR/SUP/RIM)", "",
        md(slice_table(m, "interface"), "interface"), "",
        "### single vs multi-point", "", md(slice_table(m, "is_single"), "single_point"), "",
        "## By chain", "",
        "`ab_vs_ab` is 1DVF, the anti-idiotope antibody-antibody pair, where the",
        "antibody/antigen labels are conventions rather than facts.", "",
        md(slice_table(m, "mut_side"), "mutated_side"), "",
        "## 3-class calibration (rule 5 bins on |ddG|)", "",
        md(FM.confusion3(m.y_true, m.y_pred), "actual"), "",
        "## Attention sanity check", "",
        "*Not applicable: requires E3 or later (distance-biased cross-attention). No",
        "attention-based experiment has produced out-of-fold predictions yet.*", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"  plots: {p1.name}, {p2.name}, {p3.name}")
    print(f"  worst complexes: {', '.join(per_cx.head(5).index)}")


if __name__ == "__main__":
    main()
