"""Head-to-head error analysis: the random forest against the residual fusion network.

The two are statistically tied on the headline metric -- per-complex Spearman 0.498 against
0.466, paired bootstrap +0.030 with CI [-0.008, +0.075] -- while the network wins on global
Spearman (0.509 against 0.398) and RMSE (1.597 against 1.674). A single number cannot say why,
and two separate error-analysis reports do not either, because the interesting quantity is the
*difference* slice by slice on the same rows.

Every slice therefore carries both models side by side, with the same imbalance guards the
single-model analysis uses: how many complexes contributed, what share the largest one holds, and
the slice's own label spread. A delta computed on 34 rows from 3 complexes is not a finding.

    python -m src.compare_models
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluate, paths
from .error_analysis import MIN_CX_FOR_CI, _stats, load

FOREST = "ov_both_geom"          # chem + geom + geomrev + mpnn, random forest
NET = "v2_h4_residual"           # ESM-2 + ProteinMPNN fusion, trained on the forest's residual


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Both models' predictions on the same rows, with the metadata every slice needs."""
    m = a.rename(columns={"y_pred": "p_forest"}).copy()
    m["p_net"] = b.set_index("row_id").loc[m.row_id, "y_pred"].to_numpy()
    m["r_forest"] = m.p_forest - m.y_true
    m["r_net"] = m.p_net - m.y_true
    return m


def slice_head_to_head(m: pd.DataFrame, by: str, min_rows: int = 15) -> pd.DataFrame:
    rows = []
    g = m.dropna(subset=[by])
    for level, sub in g.groupby(by, observed=True):
        if len(sub) < 5:
            continue
        f = _stats(sub.rename(columns={"p_forest": "y_pred"}))
        n = _stats(sub.rename(columns={"p_net": "y_pred"}))
        rows.append({
            by: level, "n": len(sub), "n_cx": sub.complex.nunique(),
            "top_cx": round(sub.complex.value_counts().iloc[0] / len(sub), 2),
            "sd_true": round(sub.y_true.std(), 2),
            "rho_forest": round(f["rho"], 3) if np.isfinite(f["rho"]) else np.nan,
            "rho_net": round(n["rho"], 3) if np.isfinite(n["rho"]) else np.nan,
            "d_rho": round(n["rho"] - f["rho"], 3) if np.isfinite(f["rho"]) and np.isfinite(n["rho"]) else np.nan,
            "rmse_forest": round(f["rmse"], 2), "rmse_net": round(n["rmse"], 2),
            "bias_forest": round(f["bias"], 2), "bias_net": round(n["bias"], 2),
            "trust": ("one-complex" if sub.complex.value_counts().iloc[0] / len(sub) > 0.6
                      else "thin" if len(sub) < min_rows or sub.complex.nunique() < MIN_CX_FOR_CI
                      else "ok"),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def report(forest: str = FOREST, net: str = NET, verbose: bool = True) -> dict:
    a, b = load(forest), load(net)
    m = paired(a, b[["row_id", "y_pred"]].merge(a[["row_id"]], on="row_id"))
    out = paths.REPORTS / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    tables = {}

    if verbose:
        print("=" * 108)
        print(f"HEAD TO HEAD   forest={forest}   net={net}")
        print("=" * 108)
        for nm, col in (("forest", "p_forest"), ("net", "p_net")):
            s = m[col]
            print(f"  {nm:<7} predictions: sd {s.std():.3f}  range [{s.min():+.2f}, {s.max():+.2f}]"
                  f"   below zero {int((s < 0).sum()):3d}")
        print(f"  {'truth':<7}            : sd {m.y_true.std():.3f}  "
              f"range [{m.y_true.min():+.2f}, {m.y_true.max():+.2f}]   below zero "
              f"{int((m.y_true < 0).sum()):3d}")
        print()
        r = stats.pearsonr(m.r_forest, m.r_net)[0]
        print(f"  residual correlation between the two models: {r:+.3f}")
        print(f"    (1.0 would mean identical errors and no value in combining them)")

    # --- slices -------------------------------------------------------------------------
    for col, label in [("direction", "direction of the true effect"),
                       ("magnitude", "magnitude of the true effect"),
                       ("mut_side", "which side was mutated"),
                       ("location_single", "interface location, single-point rows"),
                       ("is_single", "single vs multi-point")]:
        if col not in m.columns:
            continue
        t = slice_head_to_head(m, col)
        tables[col] = t
        t.to_csv(out / f"h2h_{col}.csv", index=False)
        if verbose:
            print()
            print("-" * 108)
            print(f"BY {label.upper()}")
            print("-" * 108)
            print(t.to_string(index=False))

    # --- per complex --------------------------------------------------------------------
    rows = []
    for cx, sub in m.groupby("complex"):
        if len(sub) < 10:
            continue
        f = _stats(sub.rename(columns={"p_forest": "y_pred"}))
        n = _stats(sub.rename(columns={"p_net": "y_pred"}))
        rows.append({"complex": cx, "n": len(sub), "sd_true": round(sub.y_true.std(), 2),
                     "rho_forest": round(f["rho"], 3), "rho_net": round(n["rho"], 3),
                     "d_rho": round(n["rho"] - f["rho"], 3)})
    per_cx = pd.DataFrame(rows).sort_values("d_rho")
    tables["per_complex"] = per_cx
    per_cx.to_csv(out / "h2h_per_complex.csv", index=False)

    if verbose:
        print()
        print("-" * 108)
        print("PER COMPLEX (>=10 rows): where each model wins")
        print("-" * 108)
        print("  five complexes the FOREST wins by most:")
        print(per_cx.head(5).to_string(index=False))
        print("\n  five complexes the NET wins by most:")
        print(per_cx.tail(5).to_string(index=False))
        w = (per_cx.d_rho > 0).sum()
        print(f"\n  net wins on {w} of {len(per_cx)} scored complexes; "
              f"median delta {per_cx.d_rho.median():+.3f}, mean {per_cx.d_rho.mean():+.3f}")
        print(f"  sign test p = {stats.binomtest(int(w), len(per_cx), 0.5).pvalue:.3f}")
        print(f"\nwrote {out.relative_to(paths.ROOT)}/")
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forest", default=FOREST)
    p.add_argument("--net", default=NET)
    a = p.parse_args()
    report(a.forest, a.net)


if __name__ == "__main__":
    main()
