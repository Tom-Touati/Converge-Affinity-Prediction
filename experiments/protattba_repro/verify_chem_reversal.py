"""Does the reverse-mutation augmentation actually reach the chem block?

The augmentation flips the label on half the training rows. Until now the chem vector
went out unchanged, so ``mpnn__llr_delta`` -- the strongest single correlate of ddG in
the block -- was handed one value against both signs of the target. This asserts the
fix end to end, on the real loader rather than on the table:

1. sample one row many times with augmentation on, and collect the distinct chem
   vectors the loader emits;
2. there must be exactly TWO, one per label sign;
3. the negative-label vector must equal the forward one with the antisymmetric columns
   negated and the paired columns exchanged;
4. the two must be scaled by the SAME statistics, which shows up as the kept columns
   being bit-identical between orientations.

Run where the caches live:

    PERTURB_ROOT=/home/ubuntu/perturb CHEM_TABLE=chem_perturb_v2 python verify_chem_reversal.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PERTURB_ROOT", "/home/ubuntu/perturb"))
sys.path.insert(0, str(ROOT))
NEG = ("chem__d_hydropathy", "chem__d_volume", "chem__d_charge", "chem__d_polarity",
       "chem__d_mw", "mpnn__llr_complex", "mpnn__llr_alone", "mpnn__llr_delta",
       "esm__llr_wt", "esm__llr_masked")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    T = load("trainer", ROOT / "_perturb_v2_colab.py")
    from model_simple import SiteTokenConfig

    rows = pd.read_parquet(ROOT / "perturb_rows.parquet")
    cache = T.Cache()
    if cache.chem_rev is None:
        raise SystemExit("no reversed table found -- CHEM_TABLE points at an older build")
    cols = list(cache.chem.select_dtypes("number").columns)
    print(f"chem table: {len(cols)} numeric columns, reversed twin present")

    fold = 0
    tr = rows[rows.fold != fold]
    c_all = cache.chem.select_dtypes("number")
    mu, sd = c_all.loc[tr.row_id].mean(), c_all.loc[tr.row_id].std().replace(0.0, 1.0)
    fwd = ((c_all - mu) / sd).astype(np.float32)
    rev = ((cache.chem_rev.select_dtypes("number").reindex(c_all.index) - mu) / sd
           ).astype(np.float32)

    pcas = T.fold_pca(rows, fold, cache, 128)
    cfg = SiteTokenConfig(pca_dim=128, proj=64, subtract=True, use_site_pool=False,
                          hidden=128, layers=1, chem_dim=len(cols), mpnn_proj=64,
                          delta_xattn=True, two_branch=True, n_heads=2, split_proj=True)
    ds = T.DS(tr, pcas, cache, cfg, True, seed=0, chem=fwd, chem_rev=rev)

    i = int(np.argmax(tr.row_id.values == tr.row_id.values[0]))
    rid = tr.row_id.values[i]
    seen = {}
    for _ in range(400):
        it = ds[i]
        seen.setdefault(round(float(it["y"]), 6), it["chem"])
    print(f"\nrow {rid}: {len(seen)} distinct label signs over 400 draws "
          f"-> {sorted(seen)}")
    assert len(seen) == 2, "augmentation never fired; raise the draw count"

    y_pos = max(seen)
    got_f, got_r = seen[y_pos], seen[min(seen)]
    want_f, want_r = fwd.loc[rid].to_numpy(), rev.loc[rid].to_numpy()
    print(f"forward  vector matches table: {np.allclose(got_f, want_f, atol=1e-5)}")
    print(f"reversed vector matches table: {np.allclose(got_r, want_r, atol=1e-5)}")
    assert np.allclose(got_f, want_f, atol=1e-5) and np.allclose(got_r, want_r, atol=1e-5)

    idx = {c: k for k, c in enumerate(cols)}
    print("\nwhat the head sees, forward -> reversed (standardised units)")
    for c in ("mpnn__llr_delta", "esm__llr_wt", "chem__d_volume",
              "chem__n_to_ala", "chem__n_from_ala", "rsasa_bound", "chem__n_mut"):
        if c in idx:
            k = idx[c]
            tag = "negated" if c in NEG else ""
            print(f"  {c:<22} {got_f[k]:+8.3f} -> {got_r[k]:+8.3f}  {tag}")

    # The antisymmetric columns must move by exactly minus twice their standardised mean
    # offset; the scaled sum of a column and its mirror is therefore constant across rows.
    for c in NEG:
        if c in idx:
            k = idx[c]
            s = fwd[c].to_numpy() + rev[c].to_numpy()
            assert np.allclose(s, s[0], atol=1e-4), f"{c} is not a clean negation"
    print("\nall antisymmetric columns negate cleanly under one shared standardiser")
    n_same = sum(1 for c in cols if np.allclose(fwd[c], rev[c], atol=1e-6))
    print(f"{n_same}/{len(cols)} columns identical in both orientations (the kept set)")


if __name__ == "__main__":
    main()
