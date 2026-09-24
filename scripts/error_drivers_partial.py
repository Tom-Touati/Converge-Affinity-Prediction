"""Which descriptors predict error ONCE THE LABEL IS CONTROLLED FOR.

`error_drivers.py` reports raw correlations between descriptors and error. Those are
dominated by one thing: |error| correlates with |ddG| at rho ~0.5 and signed error with ddG at
rho ~ -0.8, because the model regresses toward the mean. Every other descriptor that happens to
correlate with ddG will therefore appear to predict error, whether or not it carries any
information the model failed to use.

This separates the two. For each descriptor it reports:

  raw       Spearman(descriptor, error)
  partial   Spearman(descriptor, error) with the LABEL partialled out of both sides

The partial is computed on ranks, by regressing both the descriptor and the error on the
control (and its square, since the relationship is not linear) and correlating the residuals.
A descriptor that survives is one the model is getting wrong for a reason that is not simply
"this mutation had a large effect".

It then asks the question the correlations cannot: fit a small gradient-boosted model to
predict |error| from the descriptors, cross-validated by complex, and report how much of the
error is predictable at all. If |error| is predictable from descriptors the model already has,
that is unused signal; if it is not, the residual error is noise or lies outside these features.

    python scripts/error_drivers_partial.py [--run ...] [--net ...]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

MUT = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")


def partial_spearman(x, y, ctrl):
    """Spearman(x, y) with `ctrl` removed from both, on ranks."""
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(ctrl))
    if ok.sum() < 50:
        return np.nan
    rx, ry, rc = (rankdata(v[ok]) for v in (x, y, ctrl))
    # quadratic in the control: regression to the mean is not linear in the label
    C = np.column_stack([np.ones_like(rc), rc, rc ** 2])
    bx, _, _, _ = np.linalg.lstsq(C, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(C, ry, rcond=None)
    return float(spearmanr(rx - C @ bx, ry - C @ by).statistic)


def descriptors(ref: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=ref.index)
    parsed = []
    for rid in ref.index:
        muts = rid.split("|", 1)[1].split(",") if "|" in rid else []
        ok = [m for m in (MUT.match(x) for x in muts) if m]
        parsed.append({"n_points": len(muts),
                       "to_alanine": float(bool(ok) and ok[0].group(4) == "A"),
                       "wt_is_VIL": float(bool(ok) and ok[0].group(1) in "VIL"),
                       "to_gly_pro": float(bool(ok) and ok[0].group(4) in "GP")})
    d = d.join(pd.DataFrame(parsed, index=ref.index))
    size = ref.groupby("complex").size()
    spread = ref.groupby("complex").y_true.std()
    d["complex_rows"] = ref["complex"].map(size).astype(float)
    d["complex_spread"] = ref["complex"].map(spread).astype(float)
    d["dev_from_cx_mean"] = ref.y_true - ref["complex"].map(
        ref.groupby("complex").y_true.mean())
    for f, pre in (("data/features/chem_perturb.parquet", ""),
                   ("data/features/geom.parquet", "geom__")):
        p = pathlib.Path(f)
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        if "row_id" in t.columns:
            t = t.set_index("row_id")
        t = t.select_dtypes("number")
        t.columns = [pre + c for c in t.columns]
        d = d.join(t, how="left")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="E0a_rf_handcrafted_seed0")
    ap.add_argument("--net", default="cat128_reg2_l1")
    a, _ = ap.parse_known_args()

    ref = pd.read_csv(paths.REPORTS / "perturb_v3_base" / "predictions.csv").set_index("row_id")
    ref = ref.sort_index()
    X = descriptors(ref)
    preds = {"forest": pd.read_csv(paths.REPORTS / a.run / "predictions.csv")
             .set_index("row_id").y_pred}
    f = pathlib.Path(f"results/oof/{a.net}.csv")
    if f.exists():
        preds[a.net] = pd.read_csv(f).groupby("row_id").ddg_pred.mean()

    for tag, p in preds.items():
        idx = X.index.intersection(p.index)
        y = ref.y_true.loc[idx].values
        err = (p.loc[idx] - ref.y_true.loc[idx]).values
        rows = []
        for c in X.columns:
            x = X[c].loc[idx].values.astype(float)
            if np.nansum(~np.isnan(x)) < 100 or len(np.unique(x[~np.isnan(x)])) < 3:
                continue
            rows.append({
                "descriptor": c,
                "|err| raw": spearmanr(x, np.abs(err), nan_policy="omit").statistic,
                "|err| partial": partial_spearman(x, np.abs(err), np.abs(y)),
                "err raw": spearmanr(x, err, nan_policy="omit").statistic,
                "err partial": partial_spearman(x, err, y),
            })
        t = pd.DataFrame(rows).set_index("descriptor")
        t["keep"] = t[["|err| partial", "err partial"]].abs().max(axis=1)
        t = t.sort_values("keep", ascending=False).drop(columns="keep")
        print(f"\n=== {tag} — controlling for the label ===")
        print("raw = Spearman(descriptor, error). partial = the same with |ddG| (or ddG)")
        print("removed from both sides. A descriptor only matters if the PARTIAL survives.\n")
        print(t.head(12).round(3).to_string())

        # how much of |error| is predictable from descriptors the model already has?
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.model_selection import GroupKFold, cross_val_predict
            M = X.loc[idx].astype(float)
            g = ref["complex"].loc[idx].values
            for name, cols in (("all descriptors", list(M.columns)),
                               ("WITHOUT the label proxies",
                                [c for c in M.columns if c not in
                                 ("dev_from_cx_mean", "complex_spread")])):
                pr = cross_val_predict(HistGradientBoostingRegressor(max_iter=200,
                                                                    random_state=0),
                                       M[cols].values, np.abs(err),
                                       cv=GroupKFold(5), groups=g)
                r = spearmanr(pr, np.abs(err)).statistic
                print(f"  |error| predicted from {name:<26} Spearman {r:+.3f}")
        except Exception as e:  # noqa: BLE001
            print("  (error-predictability model skipped:", type(e).__name__, ")")


if __name__ == "__main__":
    main()
