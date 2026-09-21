"""AbRank-style pairwise ranking, implemented the way the paper trains it.

BACKBONE_COMPARISON.md records AbRank's strongest result: on their **Local Perturbation** split --
point mutations within a complex, which is exactly our task -- a ranking objective beats
regression for *every* model tested, and MINT swings from 0.43 to 0.78 AUC on the loss function
alone with the backbone unchanged. That is a larger effect than any backbone difference in the
whole document.

Our first attempt at a ranking loss scored 0.288 against Huber's 0.371, contradicting that. The
implementation was at fault, in a specific and measurable way.

**What was wrong: batching rows instead of sampling pairs.** The first version drew batches of
whole complexes and formed pairs inside each batch. With 54 complexes and a target batch of 64
rows that is roughly 25 gradient steps per epoch, each seeing pairs from only one to three
complexes. Counting the pairs that actually exist: **10,934 within-complex training pairs per
fold** at a 0.5 kcal/mol margin (14,582 with no margin). Sampling those directly gives ~170 steps
per epoch over the full diversity of complexes, rather than 25 over a handful.

**The objective.** For a pair (i, j) inside one complex, a logistic loss on the predicted
difference:

    L = softplus( -sign(y_i - y_j) * (f_i - f_j) )

Options, each switchable so the ablation is real:

* ``margin`` -- pairs closer than this in true ddG are dropped. Measurement noise is about 0.5
  kcal/mol, so below that the true ordering is close to a coin flip and the pair teaches noise.
* ``weighted`` -- scale each pair's loss by |y_i - y_j|, so a 4 kcal/mol gap matters more than a
  0.6 one. This is the LambdaRank idea: not all inversions cost the same.
* ``huber_w`` -- keep a regression term alongside. A pure ranking loss is scale-free, which
  discards the magnitudes entirely; at 750 rows those magnitudes are information we cannot spare,
  and AbRank trains on 380k assays where that trade is cheaper. It also removes the last anchor
  on output scale, which already produced a -6872 kcal/mol prediction once in this project.

Everything runs on the residual-on-forest setup, the only architectural change here whose paired
bootstrap clears zero.

    python -m src.rank_fusion --sweep
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import evaluate, paths, splits, train
from .fusion_arch import categoricals
from .fusion_net import (CLIP, SEQ_BLOCKS, SEQ_CACHE, STRUCT_BLOCKS, STRUCT_CACHE,
                         _standardise, _tokens)
from .fusion_v2 import _inner_forest_residuals
from .fusion_v3 import TriModalNet
from .model import MODELS
from .util import resolve_device


def build_pairs(y: np.ndarray, groups: np.ndarray, margin: float = 0.5):
    """Every within-group pair whose true values differ by more than ``margin``."""
    I, J = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        ii, jj = np.triu_indices(len(idx), 1)
        a, b = idx[ii], idx[jj]
        keep = np.abs(y[a] - y[b]) > margin
        if keep.any():
            I.append(a[keep])
            J.append(b[keep])
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(I), np.concatenate(J)


def _fit(tr, va, cfg, seed):
    s_tr, g_tr, ch_tr, c_tr, x_tr, y_tr, grp_tr = tr
    s_va, g_va, ch_va, c_va, x_va, y_va, _ = va
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = resolve_device(cfg.get("device", "auto"))
    net = TriModalNet(seq_dim=s_tr.shape[-1], struct_dim=g_tr.shape[-1],
                      chem_dim=ch_tr.shape[-1], d=cfg["d"], heads=cfg["heads"],
                      dropout=cfg["dropout"],
                      n_scalars=0 if x_tr is None else x_tr.shape[-1],
                      hadamard_pairs=cfg["pairs"], chem_token=cfg["chem_token"],
                      mut_token=cfg["mut_token"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    huber = nn.HuberLoss(delta=2.0)

    # Every tensor is created on the target device once, up front. The whole fold fits in a few
    # hundred MB, so there is no per-step host-to-device copy to pay for.
    T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt, device=dev)
    S, G, CH, X, Y = T(s_tr), T(g_tr), T(ch_tr), T(x_tr), T(y_tr)
    C = T(c_tr, torch.long)
    Sv, Gv, CHv, Xv, Yv = T(s_va), T(g_va), T(ch_va), T(x_va), T(y_va)
    Cv = T(c_va, torch.long)

    I, J = build_pairs(y_tr, grp_tr, cfg["margin"])
    if len(I) == 0:
        cfg = dict(cfg, huber_w=1.0, rank_w=0.0)

    def fwd(idx):
        return net(S[idx], G[idx], CH[idx], C[idx], None if X is None else X[idx])

    yv, yt = y_va, y_tr
    span = yt.max() - yt.min()
    lo_ok, hi_ok = yt.min() - span, yt.max() + span
    best, state, waited = -np.inf, None, 0
    n_rows = len(y_tr)
    for _ in range(cfg["epochs"]):
        net.train()
        order = rng.permutation(len(I)) if len(I) else np.zeros(0, int)
        # Cap the steps per epoch. Every pair step costs two forward passes (both halves), so
        # walking all ~11,000 pairs is about 28x the compute of row batching and made the first
        # run intractable. A fresh random subset each epoch still covers the pair space across
        # training while keeping an epoch comparable in cost to the regression baseline.
        n_steps = max(1, min(cfg["max_steps"], len(order) // cfg["batch"]))
        for k in range(n_steps):
            sel = order[k * cfg["batch"]:(k + 1) * cfg["batch"]]
            opt.zero_grad()
            loss = 0.0
            if cfg["rank_w"] > 0 and len(sel):
                pi = torch.as_tensor(I[sel], device=dev)
                pj = torch.as_tensor(J[sel], device=dev)
                fi, fj = fwd(pi), fwd(pj)
                dy = Y[pi] - Y[pj]
                w = dy.abs() if cfg["weighted"] else torch.ones_like(dy)
                r = (w * nn.functional.softplus(-torch.sign(dy) * (fi - fj))).sum() / w.sum()
                loss = loss + cfg["rank_w"] * r
            if cfg["huber_w"] > 0:
                ridx = torch.as_tensor(rng.choice(n_rows, min(cfg["batch"], n_rows),
                                                  replace=False), device=dev)
                loss = loss + cfg["huber_w"] * huber(fwd(ridx), Y[ridx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            pv = net(Sv, Gv, CHv, Cv, Xv).cpu().numpy()
        v = evaluate._safe_spearman(yv, pv)
        v = -np.inf if not np.isfinite(v) else v
        if cfg["huber_w"] > 0 and not (lo_ok <= pv.min() and pv.max() <= hi_ok):
            v = -np.inf
        if v > best + 1e-4:
            best, state, waited = v, {k2: t.clone() for k2, t in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    if state is not None:
        net.load_state_dict(state)
    return net.eval()


def run(cfg, seeds=(0, 1, 2, 3, 4)):
    d = splits.load()
    seq = _tokens(SEQ_CACHE, SEQ_BLOCKS, d.row_id)
    st = _tokens(STRUCT_CACHE, STRUCT_BLOCKS, d.row_id)
    cats = categoricals(d)
    y = d.ddG.to_numpy(np.float32)
    folds, clusters = d.fold.to_numpy(), d.cluster.to_numpy()
    groups = pd.factorize(d["#Pdb"])[0]

    blocks = ["chem", "geom", "geomrev", "mpnn"]
    Xs = np.nan_to_num(train.build_matrix(d, blocks).to_numpy(np.float32))
    chem = np.nan_to_num(train.build_matrix(d, ["chem"]).to_numpy(np.float32))

    oofs = []
    for seed in seeds:
        oof = np.full(len(d), np.nan, np.float32)
        for f in np.unique(folds):
            te = folds == f
            tr_all = ~te
            tc = pd.unique(clusters[tr_all])
            rng = np.random.default_rng(seed)
            vc = set(rng.choice(tc, min(max(3, len(tc) // 3), len(tc) - 1), replace=False))
            va = tr_all & np.isin(clusters, list(vc))
            tr = tr_all & ~va

            target, base_te = y.copy(), None
            if cfg["residual"]:
                idx_tr = np.where(tr_all)[0]
                inner = _inner_forest_residuals(Xs, y, idx_tr, clusters, seed)
                est = MODELS["rf"](seed).fit(Xs[tr_all], y[tr_all])
                base_te = est.predict(Xs[te])
                target[tr] = y[tr] - inner[tr]
                target[va] = y[va] - inner[va]

            s3, g3 = _standardise(seq[tr], seq[va], seq[te]), _standardise(st[tr], st[va], st[te])
            mu, sd = chem[tr].mean(0), chem[tr].std(0) + 1e-6
            ch3 = [np.clip((chem[m] - mu) / sd, -CLIP, CLIP) for m in (tr, va, te)]
            mu2, sd2 = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
            sc = [np.clip((Xs[m] - mu2) / sd2, -CLIP, CLIP) for m in (tr, va, te)]

            net = _fit((s3[0], g3[0], ch3[0], cats[tr], sc[0], target[tr], groups[tr]),
                       (s3[1], g3[1], ch3[1], cats[va], sc[1], target[va], groups[va]),
                       cfg, seed)
            with torch.no_grad():
                dev = next(net.parameters()).device
                T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
                p = net(T(s3[2]), T(g3[2]), T(ch3[2]),
                        T(cats[te], torch.long), T(sc[2])).cpu().numpy()
            oof[te] = p + base_te if base_te is not None else p
        oofs.append(oof)

    ens = pd.DataFrame({"row_id": d.row_id, "complex": d["#Pdb"], "cluster": d.cluster,
                        "fold": d.fold, "y_true": y.astype(float),
                        "y_pred": np.mean(oofs, axis=0)})
    m = evaluate.metrics(ens, 10)
    out_dir = paths.REPORTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": m, "ci": {}}, indent=2, default=float))
    (out_dir / "run.json").write_text(json.dumps({"name": cfg["name"], "model": "rank_fusion",
                                                  "config": cfg}, indent=2))
    return {"name": cfg["name"], "rank_w": cfg["rank_w"], "huber_w": cfg["huber_w"],
            "margin": cfg["margin"], "weighted": cfg["weighted"],
            "per_cx_rho": round(m["per_complex_spearman"], 3),
            "global_rho": round(m["global_spearman"], 3), "rmse": round(m["rmse"], 3)}


BASE = dict(heads=4, dropout=0.2, lr=3e-4, wd=1e-2, epochs=200, batch=64, patience=20, d=64,
            mut_token=True, residual=True, pairs=("sg",), chem_token=False,
            rank_w=1.0, huber_w=1.0, margin=0.5, weighted=False, max_steps=40,
            device="auto")


def sweep_configs():
    C = []
    def add(name, **kw):
        cfg = dict(BASE, name=name)
        cfg.update(kw)
        C.append(cfg)

    add("rk_huber_only", rank_w=0.0, huber_w=1.0)              # control: no ranking at all
    add("rk_rank_only", rank_w=1.0, huber_w=0.0)               # pure AbRank-style
    add("rk_both", rank_w=1.0, huber_w=1.0)                    # multi-task
    add("rk_both_weighted", rank_w=1.0, huber_w=1.0, weighted=True)
    add("rk_rank_heavy", rank_w=3.0, huber_w=1.0)
    add("rk_margin0", rank_w=1.0, huber_w=1.0, margin=0.0)     # every pair
    add("rk_margin1", rank_w=1.0, huber_w=1.0, margin=1.0)     # only well-separated pairs
    add("rk_both_chemhad", rank_w=1.0, huber_w=1.0, pairs=("sg", "cg"))
    return C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--max-steps", type=int, default=None,
                   help="pair-batches per epoch; raise it on a GPU, where the cap is not needed")
    a = p.parse_args()
    seeds = tuple(range(a.seeds))
    rows = []
    for cfg in sweep_configs():
        cfg = dict(cfg, device=a.device)
        if a.max_steps:
            cfg["max_steps"] = a.max_steps
        t0 = time.perf_counter()
        try:
            r = run(cfg, seeds)
        except Exception as e:
            r = {"name": cfg["name"], "per_cx_rho": None, "error": f"{type(e).__name__}: {e}"}
        r["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(r)
        print(f"{cfg['name']:<20} rho {str(r.get('per_cx_rho')):>6}  "
              f"global {str(r.get('global_rho')):>6}  rmse {str(r.get('rmse')):>6}  "
              f"{r['minutes']}m" + (f"  {r.get('error','')}" if r.get("error") else ""),
              flush=True)
        pd.DataFrame(rows).to_csv(paths.REPORTS / "rank_fusion_sweep.csv", index=False)
    t = pd.DataFrame(rows).sort_values("per_cx_rho", ascending=False, na_position="last")
    print()
    print("=" * 92)
    print("ABRANK-STYLE PAIRWISE RANKING -- every cell reported")
    print("=" * 92)
    print(t.to_string(index=False))
    print("\n  random forest                                 0.498")
    print("  same net, Huber only, row batches (v2)        0.466")
    print("  first rank attempt, complex batches (v2)      0.448")


if __name__ == "__main__":
    main()
