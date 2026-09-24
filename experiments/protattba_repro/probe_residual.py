"""What does the attention's residual put into the pooled vector?

CrossAttn returns ``res_pre * q_in + res_post * attn(q_in, kv)``. Which term carries the
mutation depends entirely on what is sitting in the QUERY slot, and the two directions are
not mirror images:

``seq_to_struct``  the query is the sequence edit. The residual is the edit itself, so it
                   varies between mutations of one complex -- it is signal.
``struct_to_seq``  the query is the wild-type structure, IDENTICAL for every mutation of a
                   complex. The residual is then a per-complex constant.

For the two-branch models the constant cancels: the head reads mt - wt and the same ``kv``
sits in both. For the single-branch delta model it does NOT. There the difference is taken
inside the keys, the attention is applied once, and ``res_pre * kv`` survives into the
pooled vector as a term that is the same for every mutation of a complex.

That matters because the metric is per complex. A large constant there contributes nothing
a within-complex correlation can reward, while still consuming the head's dynamic range and
the pooling divisor. This measures the two terms at the point they are summed, and splits
each into the part that varies WITHIN a complex and the part that only separates complexes.

    PERTURB_ROOT=/home/ubuntu/perturb python probe_residual.py [--direction struct_to_seq]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("PERTURB_ROOT", "/home/ubuntu/perturb"))
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def split(v: np.ndarray, cx: np.ndarray) -> tuple[float, float]:
    """(within-complex sd, between-complex sd) of a batch of vectors."""
    v = v.reshape(len(v), -1)
    mu = np.zeros_like(v)
    for c in np.unique(cx):
        m = cx == c
        mu[m] = v[m].mean(0)
    return float((v - mu).std()), float(mu.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="struct_to_seq")
    ap.add_argument("--two-branch", action="store_true")
    a = ap.parse_args()

    T = load("trainer", ROOT / "_perturb_v2_colab.py")
    import pandas as pd
    from model_simple import PerturbSiteToken, SiteTokenConfig

    rows = pd.read_parquet(ROOT / "perturb_rows.parquet")
    rows["ddg"] = rows.ddg.clip(-4, 4)
    cache = T.Cache()
    train = rows[rows.fold != 0]
    pcas = T.fold_pca(rows, 0, cache, 128)
    c = cache.chem.select_dtypes("number")
    chem = ((c - c.loc[train.row_id].mean())
            / c.loc[train.row_id].std().replace(0, 1.0)).astype(np.float32)

    cfg = SiteTokenConfig(pca_dim=128, proj=64, subtract=True, use_site_pool=False,
                          hidden=128, layers=1, dropout=0.0, input_noise=0.0,
                          feature_dropout=0.0, chem_dim=chem.shape[1], mpnn_proj=64,
                          delta_xattn=True, n_heads=2, split_proj=True,
                          two_branch=a.two_branch, attn_direction=a.direction)
    net = PerturbSiteToken(cfg).eval()

    # complexes with several mutations each, so "within a complex" means something
    big = train.complex_key.value_counts()
    keep = train[train.complex_key.isin(big[big >= 8].index[:12])]
    ds = T.DS(keep, pcas, cache, cfg, False, seed=0, chem=chem)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(keep), shuffle=False,
                                         collate_fn=T.collate, num_workers=0)
    batch = next(iter(loader))
    cx = keep.complex_key.to_numpy()

    print(f"direction {a.direction}, two_branch {a.two_branch}, "
          f"{len(keep)} rows over {len(np.unique(cx))} complexes")
    print(f"res_pre {cfg.res_pre}, res_post {cfg.res_post}\n")

    tot_res = tot_att = 0
    with torch.no_grad():
        for side in ("ab", "ag"):
            kv = net.mpnn_proj(batch[f"struct_{side}"])
            seq = (net.proj_side[side](batch[f"seq_{side}_mt"])
                   - net.proj_side[side](batch[f"seq_{side}_wt"]))
            site = batch[f"site_{side}"].unsqueeze(-1)
            q_in, kv_in = ((kv, seq) if a.direction == "struct_to_seq" else (seq, kv))
            full = net.attn(q_in, kv_in, batch[f"mask_{side}"])
            resid = cfg.res_pre * q_in                 # the two terms of the return line
            att = full - resid
            tot_res = tot_res + (resid * site).sum(1)
            tot_att = tot_att + (att * site).sum(1)

    for name, v in (("residual  (res_pre * query)", tot_res),
                    ("attention (res_post * out)", tot_att),
                    ("their sum, what is pooled", tot_res + tot_att)):
        w, b = split(v.numpy(), cx)
        print(f"  {name:<30} sd {v.numpy().std():7.3f}   within-cx {w:7.3f}   "
              f"between-cx {b:7.3f}")
    w_r, _ = split(tot_res.numpy(), cx)
    w_a, _ = split(tot_att.numpy(), cx)
    print(f"\n  share of WITHIN-complex variation from the attention: "
          f"{w_a / max(w_a + w_r, 1e-9):.1%}")
    print(f"  residual magnitude / attention magnitude: "
          f"{tot_res.numpy().std() / max(tot_att.numpy().std(), 1e-9):.2f}x")


if __name__ == "__main__":
    main()
