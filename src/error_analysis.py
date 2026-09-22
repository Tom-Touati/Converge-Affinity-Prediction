"""Error analysis, built so that imbalance cannot fake a finding.

Three kinds of imbalance run through this dataset, and each one can manufacture a conclusion that
is not there:

* **Complex imbalance.** 54 complexes, median 10 rows, max 87; the top three hold 23% of the
  data and 27 complexes fall below the 10-row rule the headline metric uses. A slice that looks
  bad may simply *be* one bad complex.
* **Label imbalance.** 70.8% of rows are destabilising by sign, only 130 are clearly stabilising.
  Any error metric computed on a slice has to be read against that slice's own spread, or
  "high RMSE" just means "high variance".
* **Category imbalance.** Interface locations run COR 295 to INT 34; assay methods run SPR 475 to
  single digits. A three-decimal number on 34 rows is not a measurement.

Every slice therefore reports, alongside its error numbers:

* ``n`` and ``n_cx`` -- rows *and* how many complexes they came from.
* ``top_cx`` -- the share of the slice contributed by its single largest complex. At 0.6 the
  slice result is that complex's result wearing a category label.
* ``sd_true`` next to ``rmse`` -- if they are equal the model has no skill on the slice,
  regardless of how the RMSE compares to other slices.
* Confidence intervals bootstrapped **over complexes within the slice**, never over rows. Row
  bootstrapping on a slice dominated by one complex reports a tight interval around an artefact.

Both aggregations are reported for every slice, because under imbalance they disagree and the
disagreement is informative:

* **micro** -- pool all rows in the slice. Answers "what is the error on a random measurement".
* **macro** -- compute per complex, then average over complexes. Answers "what is the error on a
  random target", and matches the headline metric.

    python -m src.error_analysis --run rungN0_chem_geom_mpnn_rf
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluate, paths, splits

MIN_CX_FOR_CI = 3      # fewer contributing complexes than this and a CI is not meaningful
MIN_ROWS_SLICE = 15    # below this a slice is listed but its numbers are not interpreted


# ----------------------------------------------------------------------------- loading
def load(run: str) -> pd.DataFrame:
    """Predictions joined to every column the slices need."""
    p = pd.read_csv(paths.REPORTS / run / "predictions.csv")
    d = splits.load()
    m = p.merge(d.drop(columns=[c for c in ("cluster", "fold") if c in d.columns]),
                on="row_id", validate="1:1")
    m["resid"] = m["y_pred"] - m["y_true"]          # positive = model over-predicts ddG
    m["abs_resid"] = m["resid"].abs()
    m["is_single"] = m["n_mut"] == 1
    m["direction"] = np.select(
        [m.y_true < -0.5, m.y_true > 0.5], ["stabilising", "destabilising"], "neutral")
    m["magnitude"] = pd.cut(m.y_true.abs(), [-0.01, 0.5, 1.0, 2.0, 100],
                            labels=["|ddG|<=0.5", "0.5-1", "1-2", ">2"])
    # Location is one code per mutated position, comma-joined for multi-point rows. Slicing it
    # only makes sense on single-point rows; 117 apparent "levels" are just combinations.
    m["location_single"] = np.where(m.is_single, m.location, np.nan)
    return m


# ----------------------------------------------------------------------------- statistics
def _stats(g: pd.DataFrame) -> dict:
    """Micro (row-pooled) statistics for one group."""
    y, p = g.y_true.to_numpy(float), g.y_pred.to_numpy(float)
    out = {
        "n": len(g),
        "n_cx": g.complex.nunique(),
        "top_cx": g.complex.value_counts().iloc[0] / len(g),
        "sd_true": y.std(),
        "bias": float(np.mean(p - y)),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
    }
    out["rho"] = evaluate._safe_spearman(y, p)
    # skill against predicting this slice's own mean: 1 means perfect, 0 means no better than
    # the slice mean, negative means worse. This is what stops high-variance slices from
    # automatically looking bad.
    out["skill"] = 1 - out["rmse"] ** 2 / y.var() if y.var() > 0 else np.nan
    return out


def _macro(g: pd.DataFrame, min_rows: int = 5) -> dict:
    """Per-complex statistics averaged over complexes -- each target counts once."""
    per = []
    for _, sub in g.groupby("complex"):
        if len(sub) < min_rows:
            continue
        y, p = sub.y_true.to_numpy(float), sub.y_pred.to_numpy(float)
        per.append({
            "bias": np.mean(p - y),
            "rmse": np.sqrt(np.mean((p - y) ** 2)),
            "rho": evaluate._safe_spearman(y, p),
        })
    if not per:
        return {"macro_bias": np.nan, "macro_rmse": np.nan, "macro_rho": np.nan, "macro_cx": 0}
    df = pd.DataFrame(per)
    return {"macro_bias": df.bias.mean(), "macro_rmse": df.rmse.mean(),
            "macro_rho": df.rho.mean(skipna=True), "macro_cx": len(df)}


def _ci(g: pd.DataFrame, key: str, n_boot: int = 400, seed: int = 0) -> tuple:
    """Bootstrap a micro statistic by resampling COMPLEXES within the slice."""
    cx = g.complex.unique()
    if len(cx) < MIN_CX_FOR_CI:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    groups = {c: sub for c, sub in g.groupby("complex")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cx, size=len(cx), replace=True)
        s = pd.concat([groups[c] for c in pick], ignore_index=True)
        try:
            vals.append(_stats(s)[key])
        except Exception:
            continue
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return (np.nan, np.nan) if v.size < 20 else tuple(np.percentile(v, [2.5, 97.5]))


def slice_table(m: pd.DataFrame, by: str, n_boot: int = 400, dropna: bool = True) -> pd.DataFrame:
    """One row per level of ``by``, with both aggregations and complex-bootstrapped CIs."""
    g = m.dropna(subset=[by]) if dropna else m
    rows = []
    for level, sub in g.groupby(by, observed=True):
        r = {by: level}
        r.update(_stats(sub))
        r.update(_macro(sub))
        lo, hi = _ci(sub, "bias", n_boot)
        r["bias_lo"], r["bias_hi"] = lo, hi
        lo, hi = _ci(sub, "rho", n_boot)
        r["rho_lo"], r["rho_hi"] = lo, hi
        # a finding on this slice is suspect if one complex dominates it or it is tiny
        r["trust"] = (
            "one-complex" if r["top_cx"] > 0.6 else
            "thin" if r["n"] < MIN_ROWS_SLICE or r["n_cx"] < MIN_CX_FOR_CI else
            "ok"
        )
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def fmt(t: pd.DataFrame, by: str) -> str:
    """Compact printable view of a slice table."""
    d = t.copy()
    d["bias_ci"] = [f"[{a:+.2f},{b:+.2f}]" if np.isfinite(a) else "  --  "
                    for a, b in zip(d.bias_lo, d.bias_hi)]
    d["rho_ci"] = [f"[{a:+.2f},{b:+.2f}]" if np.isfinite(a) else "  --  "
                   for a, b in zip(d.rho_lo, d.rho_hi)]
    cols = [by, "n", "n_cx", "top_cx", "sd_true", "bias", "bias_ci", "rmse", "skill",
            "rho", "rho_ci", "macro_rho", "trust"]
    d = d[cols].round({"top_cx": 2, "sd_true": 2, "bias": 2, "rmse": 2,
                       "skill": 2, "rho": 2, "macro_rho": 2})
    return d.to_string(index=False)


# ----------------------------------------------------------------------------- probes
def alanine_probe(m: pd.DataFrame, n_boot: int = 400) -> pd.DataFrame:
    """Does performance live entirely in the alanine scan?

    53% of single-point mutations substitute alanine, and they average +1.17 against +0.87 for
    the rest. A model can score respectably by learning "alanine scan means destabilising"
    without learning any binding physics.
    """
    s = m[m.is_single].copy()
    s["group"] = np.where(s.is_alanine, "X->Ala", "other substitutions")
    return slice_table(s, "group", n_boot)


def additivity_probe(m: pd.DataFrame) -> pd.DataFrame:
    """Double-mutant cycles: is the model's error on doubles explained by non-additivity?

    For every 2-point mutation whose both constituent singles are measured in the same complex,
    compare observed epistasis (true double minus the sum of trues) against predicted epistasis.
    If the model tracks epistasis at all, the two correlate.
    """
    singles = {(r.complex, r.mutations): (r.y_true, r.y_pred)
               for r in m[m.n_mut == 1].itertuples()}
    rows = []
    for r in m[m.n_mut == 2].itertuples():
        parts = [p.strip() for p in r.mutations.split(",")]
        keys = [(r.complex, p) for p in parts]
        if not all(k in singles for k in keys):
            continue
        ts = sum(singles[k][0] for k in keys)
        ps = sum(singles[k][1] for k in keys)
        rows.append({
            "row_id": r.row_id, "complex": r.complex,
            "true_double": r.y_true, "sum_true_singles": ts, "obs_epistasis": r.y_true - ts,
            "pred_double": r.y_pred, "sum_pred_singles": ps, "pred_epistasis": r.y_pred - ps,
        })
    return pd.DataFrame(rows)


def worst_cases(m: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k largest absolute residuals, with the context needed to read them by hand."""
    cols = ["row_id", "complex", "mutations", "y_true", "y_pred", "resid",
            "location", "mut_side", "method", "n_mut", "ddG_n", "ddG_sd",
            "nonnative_ref", "mut_is_bound", "wt_is_bound"]
    w = m.nlargest(k, "abs_resid")[[c for c in cols if c in m.columns]]
    n = m.groupby("complex").size()
    w = w.assign(cx_rows=w.complex.map(n))
    return w.reset_index(drop=True)


# ----------------------------------------------------------------------------- report
SLICES = [
    ("location_single", "interface location (single-point rows only)"),
    ("direction", "direction of the true effect"),
    ("magnitude", "magnitude of the true effect"),
    ("mut_side", "which side of the interface was mutated"),
    ("is_single", "single-point vs multi-point"),
    ("method", "assay method"),
    ("fold", "cross-validation fold"),
    ("is_alanine", "alanine substitution"),
    ("nonnative_ref", "non-native reference state (wild type is itself engineered)"),
]


def report(run: str, n_boot: int = 400, verbose: bool = True) -> dict:
    m = load(run)
    out_dir = paths.REPORTS / run / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}

    if verbose:
        print("=" * 110)
        print(f"ERROR ANALYSIS  --  {run}   ({len(m)} rows, {m.complex.nunique()} complexes)")
        print("=" * 110)
        print("bias = mean(predicted - true); positive means the model over-predicts ddG.")
        print("skill = 1 - MSE/Var within the slice; 0 means no better than that slice's own mean.")
        print("trust: 'one-complex' = >60% of the slice is a single complex; 'thin' = too few rows"
              " or complexes.")
        print()

    # ---- per complex, worst first
    per_cx = []
    for cx, g in m.groupby("complex"):
        r = {"complex": cx}
        r.update(_stats(g))
        per_cx.append(r)
    per_cx = pd.DataFrame(per_cx).sort_values("rho").reset_index(drop=True)
    tables["per_complex"] = per_cx
    per_cx.to_csv(out_dir / "per_complex.csv", index=False)

    if verbose:
        print("-" * 110)
        print("PER COMPLEX, worst first (only complexes with >=10 rows carry the headline metric)")
        print("-" * 110)
        show = per_cx[per_cx.n >= 10][["complex", "n", "sd_true", "bias", "rmse", "skill", "rho"]]
        print(show.round(2).head(12).to_string(index=False))
        print(f"  ... {len(per_cx)} complexes total, {(per_cx.n < 10).sum()} below the 10-row rule")

    # ---- the slices
    for col, label in SLICES:
        if col not in m.columns:
            continue
        t = slice_table(m, col, n_boot)
        tables[col] = t
        t.to_csv(out_dir / f"slice_{col}.csv", index=False)
        if verbose:
            print()
            print("-" * 110)
            print(f"BY {label.upper()}")
            print("-" * 110)
            print(fmt(t, col))

    # ---- alanine probe
    ala = alanine_probe(m, n_boot)
    tables["alanine"] = ala
    ala.to_csv(out_dir / "probe_alanine.csv", index=False)
    if verbose:
        print()
        print("-" * 110)
        print("PROBE: does performance live in the alanine scan?")
        print("-" * 110)
        print(fmt(ala, "group"))

    # ---- additivity probe
    add = additivity_probe(m)
    tables["additivity"] = add
    add.to_csv(out_dir / "probe_additivity.csv", index=False)
    if verbose:
        print()
        print("-" * 110)
        print("PROBE: double-mutant cycles -- does the model track epistasis?")
        print("-" * 110)
        if len(add) >= 5:
            r = stats.spearmanr(add.obs_epistasis, add.pred_epistasis)
            print(f"  usable double-mutant cycles: {len(add)} across {add.complex.nunique()} complexes")
            print(f"  observed epistasis:  mean {add.obs_epistasis.mean():+.2f}  "
                  f"sd {add.obs_epistasis.std():.2f}  range "
                  f"[{add.obs_epistasis.min():+.2f}, {add.obs_epistasis.max():+.2f}]")
            print(f"  predicted epistasis: mean {add.pred_epistasis.mean():+.2f}  "
                  f"sd {add.pred_epistasis.std():.2f}")
            print(f"  correlation observed vs predicted: Spearman {r.correlation:+.3f} "
                  f"(p={r.pvalue:.3f})")
            frac = (add.obs_epistasis.abs() > 1.0).mean()
            print(f"  cycles with |epistasis| > 1 kcal/mol: {frac:.0%} -- additivity is "
                  f"{'often violated' if frac > 0.2 else 'mostly a fair approximation'}")
        else:
            print(f"  only {len(add)} usable cycles; not interpretable")

    # ---- worst cases
    worst = worst_cases(m, 10)
    tables["worst"] = worst
    worst.to_csv(out_dir / "worst_cases.csv", index=False)
    if verbose:
        print()
        print("-" * 110)
        print("TEN WORST RESIDUALS (for hand inspection against the structures)")
        print("-" * 110)
        cols = ["row_id", "y_true", "y_pred", "resid", "location", "mut_side", "method",
                "cx_rows", "ddG_n", "nonnative_ref"]
        print(worst[[c for c in cols if c in worst.columns]].round(2).to_string(index=False))
        print()
        print(f"wrote {out_dir.relative_to(paths.ROOT)}/")
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="rungN0_chem_geom_mpnn_rf")
    p.add_argument("--n-boot", type=int, default=400)
    a = p.parse_args()
    report(a.run, a.n_boot)
    figures(a.run)



# ----------------------------------------------------------------------------- figures
def figures(run: str = "rungN0_chem_geom_mpnn_rf") -> str:
    """Four diagnostics, chosen so that each one answers a question the tables raise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import DESTAB, GRID, INK, INK_2, INK_MUTED, NEUTRAL, STAB, SURFACE
    from .analysis import _style as _base_style

    BLUE = STAB  # categorical slot 1; same hex as the cool diverging pole

    def _style(ax, title=""):
        _base_style(ax)
        if title:
            ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        return ax

    m = load(run)
    out = paths.REPORTS / run / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.4), facecolor=SURFACE)

    # (a) predicted vs true -- the shape of the failure
    ax = _style(axes[0, 0], "a. Predicted vs true: the model compresses towards the mean")
    colors = {"stabilising": STAB, "neutral": NEUTRAL, "destabilising": DESTAB}
    for lab, c in colors.items():
        s = m[m.direction == lab]
        ax.scatter(s.y_true, s.y_pred, s=13, c=c, alpha=0.65, linewidth=0, zorder=3, label=lab)
    lim = [m.y_true.min() - 0.5, m.y_true.max() + 0.5]
    ax.plot(lim, lim, color=INK, linewidth=1.2, zorder=4)
    ax.axhline(m.y_true.mean(), color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    ax.text(lim[0] + 0.2, m.y_true.mean() + 0.18, "training mean", fontsize=8.5, color=INK_2)
    ax.set_xlabel("true ddG (kcal/mol)"); ax.set_ylabel("predicted ddG")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # (b) bias by magnitude band -- shrinkage, quantified, with complex-bootstrapped CIs
    ax = _style(axes[0, 1], "b. Bias by true magnitude, with 95% CI over complexes")
    t = slice_table(m, "magnitude", n_boot=300)
    order = ["|ddG|<=0.5", "0.5-1", "1-2", ">2"]
    t = t.set_index("magnitude").loc[[o for o in order if o in t.magnitude.values]].reset_index() \
        if "magnitude" in t.columns else t
    xs = np.arange(len(t))
    ax.bar(xs, t.bias, color=[DESTAB if b > 0 else STAB for b in t.bias], width=0.6, zorder=3)
    ax.errorbar(xs, t.bias, yerr=[t.bias - t.bias_lo, t.bias_hi - t.bias],
                fmt="none", ecolor=INK_2, capsize=4, linewidth=1.2, zorder=4)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=5)
    ax.set_xticks(xs, [f"{v}\n(n={n})" for v, n in zip(t.iloc[:, 0], t.n)], fontsize=8.5)
    ax.set_ylabel("bias  =  mean(predicted - true)")

    # (c) does per-complex performance track sample size, or difficulty?
    ax = _style(axes[1, 0], "c. Per-complex skill vs rows available (only n>=10 is scored)")
    per = pd.DataFrame([{"complex": c, **_stats(g)} for c, g in m.groupby("complex")])
    sc = per[per.n >= 10]
    ax.scatter(sc.n, sc.rho, s=28, c=sc.sd_true, cmap="viridis", zorder=3, edgecolor=SURFACE)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    r = stats.spearmanr(sc.n, sc.rho)
    ax.set_xlabel("rows for that complex"); ax.set_ylabel("per-complex Spearman")
    ax.text(0.97, 0.05, f"Spearman(n, rho) = {r.correlation:+.2f}  (p={r.pvalue:.2f})",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_2)
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.85)
    cb.set_label("that complex's ddG spread (sd)", fontsize=8.5)

    # (d) epistasis -- is the multi-point failure a non-additivity failure?
    ax = _style(axes[1, 1], "d. Double-mutant cycles: observed vs predicted epistasis")
    add = additivity_probe(m)
    if len(add) >= 5:
        ax.scatter(add.obs_epistasis, add.pred_epistasis, s=24, c=BLUE, alpha=0.7,
                   linewidth=0, zorder=3)
        lim2 = [-5.2, 2.8]
        ax.plot(lim2, lim2, color=INK, linewidth=1.2, zorder=4)
        ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
        ax.set_xlim(lim2); ax.set_ylim(lim2)
        rr = stats.spearmanr(add.obs_epistasis, add.pred_epistasis)
        ax.text(0.03, 0.95, f"n = {len(add)} cycles\nSpearman {rr.correlation:+.2f}\n"
                            f"observed sd {add.obs_epistasis.std():.2f}, "
                            f"predicted sd {add.pred_epistasis.std():.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
    ax.set_xlabel("observed epistasis (true double - sum of trues)")
    ax.set_ylabel("predicted epistasis")

    fig.tight_layout()
    path = out / "diagnostics.png"
    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(paths.ROOT)}")
    return str(path)

# The entry point lives at the very bottom on purpose. It used to sit directly under `main`,
# which is above `figures`, so `main()` executed before `def figures` had been evaluated and
# every invocation ended in `NameError: name 'figures' is not defined` -- after printing all
# the tables, so the crash looked cosmetic while in fact no figure was ever written.
if __name__ == "__main__":
    main()
