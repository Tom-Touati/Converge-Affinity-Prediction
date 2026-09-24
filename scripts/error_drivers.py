"""Which descriptors predict a model's error, and do the models fail on the same things?

Part III of the error analysis found specific failures by hand -- valine, alanine scans,
low-spread complexes. This does it systematically: every available descriptor against every
model's error, ranked by effect size, with a column per model so a defect that is shared can
be told from one that belongs to a single architecture.

Two error definitions, because they answer different questions:

  |error|  where is the model IMPRECISE -- which rows are hard for it
  error    where is the model BIASED -- which rows it systematically over- or under-predicts

Spearman throughout: most of these descriptors are skewed, several are counts, and a rank
correlation does not assume the relationship is linear.

A descriptor that correlates with SIGNED error is the more serious finding. It means the
model could be improved by a monotone correction in that descriptor alone -- the information
is present and unused.

    python scripts/error_drivers.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths                 # noqa: E402

MUT = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")
MODELS = {
    # the forest is still the best overall model; struct_film_chem is the best NETWORK,
    # the current submission for that role -- see README's results table
    "forest":           ("reports", "E0a_rf_handcrafted_seed0"),
    "struct_film_chem": ("oof", "struct_film_chem"),
}
MIN_ABS = 0.05          # ignore correlations smaller than this in the summary


def predictions(kind: str, name: str) -> pd.Series:
    if kind == "reports":
        d = pd.read_csv(paths.REPORTS / name / "predictions.csv")
        return d.set_index("row_id").y_pred
    d = pd.read_csv(f"results/oof/{name}.csv")
    return d.groupby("row_id").ddg_pred.mean()      # average the seeds


def descriptors(ref: pd.DataFrame) -> pd.DataFrame:
    """Everything known about a row that is not the model's output."""
    d = pd.DataFrame(index=ref.index)
    d["true_ddg"] = ref.y_true
    d["abs_ddg"] = ref.y_true.abs()

    # from the row id: the mutation itself
    parsed = []
    for rid in ref.index:
        muts = rid.split("|", 1)[1].split(",") if "|" in rid else []
        ok = [m for m in (MUT.match(x) for x in muts) if m]
        parsed.append({"n_points": len(muts),
                       "to_alanine": float(bool(ok) and ok[0].group(4) == "A"),
                       "from_gly_pro": float(bool(ok) and ok[0].group(1) in "GP"),
                       "to_gly_pro": float(bool(ok) and ok[0].group(4) in "GP")})
    d = d.join(pd.DataFrame(parsed, index=ref.index))

    # how well represented this row's complex is
    size = ref.groupby("complex").size()
    spread = ref.groupby("complex").y_true.std()
    d["complex_rows"] = ref["complex"].map(size).astype(float)
    d["complex_spread"] = ref["complex"].map(spread).astype(float)
    d["complex_mean_ddg"] = ref["complex"].map(ref.groupby("complex").y_true.mean())
    # how far this row sits from its own complex's mean -- the quantity a per-complex
    # correlation is actually built from
    d["dev_from_complex_mean"] = ref.y_true - d.complex_mean_ddg

    for f, prefix in (("data/features/chem_perturb.parquet", ""),
                      ("data/features/geom.parquet", "geom__")):
        p = pathlib.Path(f)
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        if "row_id" in t.columns:
            t = t.set_index("row_id")
        t = t.select_dtypes("number")
        t.columns = [prefix + c for c in t.columns]
        d = d.join(t, how="left")
    return d


def main() -> None:
    ref = pd.read_csv(paths.REPORTS / "perturb_v3_base" / "predictions.csv").set_index("row_id")
    ref = ref.sort_index()
    X = descriptors(ref)
    preds = {}
    for tag, (kind, name) in MODELS.items():
        try:
            preds[tag] = predictions(kind, name)
        except FileNotFoundError:
            continue
    print(f"{len(X)} rows, {X.shape[1]} descriptors, {len(preds)} models: "
          f"{', '.join(preds)}\n")

    for what, fn in (("ABSOLUTE error  (where the model is imprecise)", np.abs),
                     ("SIGNED error    (where the model is biased)", lambda e: e)):
        rows = []
        for col in X.columns:
            x = X[col]
            if x.notna().sum() < 100 or x.nunique() < 3:
                continue
            rec = {"descriptor": col}
            for tag, p in preds.items():
                idx = X.index.intersection(p.index)
                e = fn((p.loc[idx] - ref.y_true.loc[idx]).values)
                xv = x.loc[idx].values
                ok = ~(np.isnan(xv) | np.isnan(e))
                rec[tag] = spearmanr(xv[ok], e[ok]).statistic if ok.sum() > 50 else np.nan
            rows.append(rec)
        t = pd.DataFrame(rows).set_index("descriptor")
        t["mean_abs"] = t.abs().mean(axis=1)
        t = t.sort_values("mean_abs", ascending=False)
        print(f"=== {what} ===")
        print("Spearman rho, descriptor vs error. Positive = larger descriptor, larger error.")
        print(t[t.mean_abs >= MIN_ABS].head(14).round(3).to_string())
        print()

    # do the models fail on the SAME rows?
    print("=== do the models make the same mistakes? ===")
    tags = list(preds)
    print("Spearman between models' |error| per row:")
    M = pd.DataFrame(index=tags, columns=tags, dtype=float)
    for a in tags:
        for b in tags:
            idx = preds[a].index.intersection(preds[b].index).intersection(ref.index)
            ea = (preds[a].loc[idx] - ref.y_true.loc[idx]).abs()
            eb = (preds[b].loc[idx] - ref.y_true.loc[idx]).abs()
            M.loc[a, b] = spearmanr(ea, eb).statistic
    print(M.round(3).to_string())
    print("\nHigh off-diagonal values mean the architectures are not failing independently,")
    print("so an ensemble of them cannot recover much -- the hard rows are hard for all.")


if __name__ == "__main__":
    main()
