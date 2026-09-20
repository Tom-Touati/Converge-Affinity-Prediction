"""Everything built but never run, plus a few combinations worth a night of CPU.

Each block is a question that is already answerable from code on disk and has simply not been
asked. Results append to reports/overnight.csv as they land, so a crash loses only the block it
was in.

    python -m src.overnight
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from . import evaluate, paths, splits, train

OUT = paths.REPORTS / "overnight.csv"
_rows = []


def record(name, question, rho, extra=""):
    _rows.append({"experiment": name, "question": question,
                  "per_cx_rho": None if rho is None else round(rho, 3), "note": extra})
    pd.DataFrame(_rows).to_csv(OUT, index=False)
    print(f"  {name:<34} rho {('%+.3f' % rho) if rho is not None else '  --  '}   {extra}",
          flush=True)


def fit(model, features, name, n_boot=0):
    out = train.run(model, features, name, n_boot=n_boot, verbose=False)
    return evaluate.metrics(out, 10)["per_complex_spearman"]


def main():
    t0 = time.perf_counter()
    print("=" * 100)
    print("OVERNIGHT QUEUE")
    print("=" * 100)

    # 1. The wild-type/mutant concatenation, built as pca_pair_rf and never run. Subtraction
    #    discards the site (82% of the difference vector is substitution identity); keeping both
    #    halves and reducing each to 16 components separately is the alternative.
    print("\n[1] wt/mut concatenation instead of subtraction")
    for feats, nm in [(["esm35M_pair"], "pair_alone"),
                      (["chem", "geom", "mpnn", "esm35M_pair"], "pair_plus_best")]:
        try:
            record(f"pca_pair_{nm}", "does keeping both halves beat subtracting them?",
                   fit("pca_pair_rf", feats, f"ov_pca_pair_{nm}"))
        except Exception as e:
            record(f"pca_pair_{nm}", "failed", None, f"{type(e).__name__}: {e}")

    # 2. ESM-2 150M: extracted as the CurrAb control and never scored. Answers whether CurrAb's
    #    failure is the fine-tuning or the base model, and whether capacity helps at all.
    print("\n[2] ESM-2 150M, the CurrAb control")
    if (paths.FEATURES / "esm150M.parquet").exists():
        for feats, nm in [(["esm150M@scalars"], "alone"),
                          (["chem", "geom", "mpnn", "esm150M@scalars"], "plus_best")]:
            try:
                record(f"esm150M_{nm}", "is CurrAb's failure the fine-tuning or the base model?",
                       fit("rf", feats, f"ov_esm150M_{nm}"))
            except Exception as e:
                record(f"esm150M_{nm}", "failed", None, f"{type(e).__name__}: {e}")
    else:
        record("esm150M", "cache missing", None, "extraction did not complete")

    # 3. Both geometry blocks together. geom_rev alone matches geom (0.258 vs 0.245) but loses
    #    badly in combination (0.369 vs 0.487) because its residue half duplicates chem while
    #    dropping the true side-chain contacts. Keeping both should recover that.
    print("\n[3] both geometry blocks")
    try:
        r = fit("rf", ["chem", "geom", "geomrev", "mpnn"], "ov_both_geom")
        record("both_geometry", "does the reversible block add to the original?", r)
    except Exception as e:
        record("both_geometry", "failed", None, f"{type(e).__name__}: {e}")

    # 4. Augmentation on both geometry blocks. Reversal cost 0.071 with geom alone and 0.037 with
    #    geomrev alone; with both present the reversible half may carry the sign while the
    #    original carries the magnitude.
    print("\n[4] reverse augmentation with both geometry blocks")
    for feats, nm in [(["chem", "geom", "geomrev", "mpnn"], "both_geom")]:
        try:
            out = train.run("rf", feats, f"ov_aug_{nm}", n_boot=0, verbose=False, augment=True)
            m = evaluate.metrics(out, 10)
            record(f"augmented_{nm}", "can reversible geometry pay for augmentation?",
                   m["per_complex_spearman"],
                   f"balanced sign acc {m['sign_acc_big_balanced']:.3f}")
        except Exception as e:
            record(f"augmented_{nm}", "failed", None, f"{type(e).__name__}: {e}")

    # 5. Ensemble the forest with the fusion network. They fail differently -- bagged trees
    #    threshold, the net extrapolates -- so an average may beat both even though the net is
    #    the weaker model.
    print("\n[5] forest + fusion-net ensemble")
    try:
        d = splits.load()
        rf = pd.read_csv(paths.REPORTS / "rf_leaf3_chem_geom_mpnn" / "predictions.csv")
        for net_name in ["fusion_d32_scalars", "arch_all_scalars", "arch_had_mut_scalars"]:
            f = paths.REPORTS / net_name / "predictions.csv"
            if not f.exists():
                continue
            nn_p = pd.read_csv(f)
            m = rf.merge(nn_p[["row_id", "y_pred"]], on="row_id", suffixes=("_rf", "_nn"))
            for w in (0.25, 0.5):
                m["y_pred"] = (1 - w) * m.y_pred_rf + w * m.y_pred_nn
                r = evaluate.metrics(m, 10)["per_complex_spearman"]
                record(f"ensemble_{net_name}_w{w}",
                       "do two models that fail differently combine?", r)
    except Exception as e:
        record("ensemble", "failed", None, f"{type(e).__name__}: {e}")

    print(f"\ndone in {(time.perf_counter()-t0)/60:.0f} min -> {OUT.relative_to(paths.ROOT)}")
    print(pd.DataFrame(_rows).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
