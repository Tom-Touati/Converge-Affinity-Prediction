"""What scale does each block arrive at, where they are concatenated?

Concatenating blocks of different magnitude hands the first linear layer a decision it never
made: the larger block dominates the gradient, and the smaller one is effectively switched
off until the weights spend capacity undoing the mismatch. This project has already been
caught by exactly that -- the wild-type binding-site pools arrived 7x larger than the
sequence edit (2.396 against 0.327), and the edit is the only part that varies between
mutations of the same complex.

Most blocks are protected by their own LayerNorm. The delta_xattn path deliberately is NOT:
its learned alpha sets how loudly the attended edit speaks next to the chem columns, and a
LayerNorm there would undo it. That makes this the one path where the scales have to be
checked rather than assumed.

Run on a box that holds the caches:

    PERTURB_ROOT=/home/ubuntu/perturb python probe_fusion_scale.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("PERTURB_ROOT", "/home/ubuntu/perturb"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    sys.path.insert(0, str(ROOT))
    T = load("trainer", ROOT / "_perturb_v2_colab.py")
    import pandas as pd

    rows = pd.read_parquet(ROOT / "perturb_rows.parquet")
    rows["ddg"] = rows.ddg.clip(-4, 4)
    cache = T.Cache()
    fold = 0
    train = rows[rows.fold != fold]
    pcas = T.fold_pca(rows, fold, cache, 128)
    chem = None
    if cache.chem is not None:
        c = cache.chem.reindex(rows.row_id).select_dtypes("number")
        chem = ((c - c.loc[train.row_id.values].mean())
                / c.loc[train.row_id.values].std().replace(0, 1.0))

    from model_simple import PerturbSiteToken, SiteTokenConfig
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-branch", action="store_true")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--alpha-init", type=float, default=-1.56)
    ap.add_argument("--edit-op", default="sub", choices=["sub", "mul"])
    a, _ = ap.parse_known_args()
    cfg = SiteTokenConfig(pca_dim=128, proj=64, subtract=True, use_site_pool=False,
                          hidden=128, layers=a.layers, dropout=0.35, input_noise=0.25,
                          feature_dropout=0.35, chem_dim=chem.shape[1] if chem is not None else 0,
                          mpnn_proj=64, delta_xattn=True, n_heads=2, split_proj=True,
                          two_branch=a.two_branch, pool_alpha_init=a.alpha_init,
                          edit_op=a.edit_op)
    net = PerturbSiteToken(cfg).eval()

    ds = T.DS(train, pcas, cache, cfg, False, seed=0, chem=chem)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False,
                                         collate_fn=T.collate, num_workers=0)
    batch = next(iter(loader))

    # rebuild the two blocks exactly as forward does, without the noise
    with torch.no_grad():
        which = ("wt", "mt") if cfg.two_branch else ("delta",)
        acc, n_sites = {k: 0 for k in which}, 0
        for side in ("ab", "ag"):
            kv = net.mpnn_proj(batch[f"struct_{side}"])
            site = batch[f"site_{side}"].unsqueeze(-1)
            for k in which:
                if k == "delta":
                    p_mt = net.proj_side[side](batch[f"seq_{side}_mt"])
                    p_wt = net.proj_side[side](batch[f"seq_{side}_wt"])
                    q = p_mt * p_wt if a.edit_op == "mul" else p_mt - p_wt
                else:
                    q = net.proj_side[side](batch[f"seq_{side}_{k}"])
                acc[k] = acc[k] + (net.attn(q, kv, batch[f"mask_{side}"]) * site).sum(1)
            n_sites = n_sites + site.sum(1)
        pooled = (torch.cat([acc["wt"], acc["mt"]], -1) if cfg.two_branch else acc["delta"])
        alpha = torch.nn.functional.softplus(net.pool_alpha)
        z = pooled / (n_sites.clamp(min=1.0) * alpha)
        if cfg.two_branch:
            z = net.branch_mix(z)
        ch = batch["chem"]

    def stat(name, t):
        t = t.float()
        print(f"  {name:<28} sd {t.std():7.4f}   |mean| {t.mean().abs():7.4f}   "
              f"max|x| {t.abs().max():7.3f}")

    print(f"fold {fold}, batch of {len(batch['y'])}, alpha = {float(alpha):.4f}\n")
    print("AT THE CONCATENATION POINT")
    stat("attended edit  (64)", z)
    stat("chem columns   (%d)" % ch.shape[1], ch)
    ratio = float(ch.std() / z.std().clamp(min=1e-9))
    print(f"\n  chem / edit standard-deviation ratio: {ratio:.2f}x")
    if ratio > 3 or ratio < 1 / 3:
        print("  MISMATCHED -- the first linear layer will be driven by one block")
    else:
        print("  comparable")

    print("\nWHERE THE EDIT'S SCALE COMES FROM")
    with torch.no_grad():
        side = "ab"
        raw = (batch[f"seq_{side}_mt"] - batch[f"seq_{side}_wt"])
        prj = (net.proj_side[side](batch[f"seq_{side}_mt"])
               - net.proj_side[side](batch[f"seq_{side}_wt"]))
        kv = net.mpnn_proj(batch[f"struct_{side}"])
        att = net.attn(prj, kv, batch[f"mask_{side}"])
    stat("PCA delta, per token (128)", raw)
    stat("after Linear(128->64)", prj)
    stat("after attention (residual)", att)
    stat("mpnn keys/values (64)", kv)
    print(f"\n  mutated residues per row: median "
          f"{float(n_sites.median()):.0f}, max {float(n_sites.max()):.0f}")


if __name__ == "__main__":
    main()
