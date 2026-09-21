"""Four mechanistic hypotheses for closing the gap to the random forest.

The forest sits at 0.487; the best learned fusion so far is 0.408, and that is a maximum over
fourteen configurations so it is optimistic. Each hypothesis below has a reason to work that is
not "try another width".

**H1 -- give the network the features the forest actually has.** The forest's best run reads
chem + geom + geomrev + mpnn (49 columns); the network has only been given chem + geom + mpnn
(33). Part of the measured gap is simply a different feature set, and that is not a fair test of
the architecture.

**H2 -- put the chemistry into the mutation-token query.** The query is currently a sum of four
categorical embeddings, so it can express "Y to A" but not "large hydrophobic to small". Adding a
projection of the 21-dimensional chemistry vector lets the query carry the continuous properties
of the substitution, which is what the attention should be keyed on.

**H3 -- train on the metric.** The headline is per-complex Spearman, a ranking metric, and the
network is trained on Huber loss, which is a proxy. A within-complex pairwise logistic loss
optimises the ordering directly. AbRank reports ranking objectives beating regression by up to 12
AUC points under novel-antigen splits, and BACKBONE_COMPARISON.md records the objective moving
results more than the encoder does.

**H4 -- learn the forest's residual instead of competing with it.** Trees and gradient-trained
networks fail differently: trees threshold and cannot extrapolate, networks extrapolate and
overfit. Fitting the network to what the forest gets wrong is the residual-correction framing
PRIOR_WORK records CATH-ddG using to beat raw FoldX. The forest's in-fold predictions must come
from an **inner** cross-validation, never from the outer out-of-fold predictions: those were made
by forests that had seen the outer test fold, and training on them would leak it.

    python -m src.fusion_v2 --sweep
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
from .fusion_arch import CARDS, ArchNet, categoricals
from .fusion_net import (CLIP, SEQ_BLOCKS, SEQ_CACHE, STRUCT_BLOCKS, STRUCT_CACHE,
                         _standardise, _tokens)
from .model import MODELS
from .util import resolve_device


class ArchNetV2(ArchNet):
    """ArchNet with an optionally chemistry-aware mutation-token query (H2).

    Every argument is named rather than forwarded through ``*a, **kw``. The previous version
    accepted ``d`` both positionally and as a keyword and raised on construction -- the same
    duplicate-keyword mistake that silently emptied the first architecture sweep.
    """

    def __init__(self, seq_dim, struct_dim, d=64, heads=4, dropout=0.2, n_scalars=0,
                 hadamard=True, attn=False, mut_token=True, cats=False, query_dim=0):
        super().__init__(seq_dim, struct_dim, d=d, heads=heads, dropout=dropout,
                         n_scalars=n_scalars, hadamard=hadamard, attn=attn,
                         mut_token=mut_token, cats=cats)
        self.q_proj = nn.Linear(query_dim, d) if query_dim else None

    def forward(self, seq, struct, cats=None, scalars=None, qextra=None):
        s = self.seq_norm(self.seq_proj(seq))
        g = self.struct_norm(self.struct_proj(struct))
        parts = []
        if self.use_h:
            parts.append(self.had(s.mean(1) * g.mean(1)))
        if self.use_a:
            a, _ = self.attn(g, s, s, need_weights=False)
            parts.append(self.attn_norm(a).mean(1))
        if self.use_m or self.use_c:
            e = sum(emb(cats[:, i]) for i, emb in enumerate(self.cat_emb))
            if self.q_proj is not None and qextra is not None:
                e = e + self.q_proj(qextra)       # H2: continuous substitution properties
        if self.use_m:
            kv = torch.cat([s, g], dim=1)
            m, _ = self.m_attn(e.unsqueeze(1), kv, kv, need_weights=False)
            parts.append(self.m_norm(m).squeeze(1))
        if self.use_c:
            parts.append(e)
        if scalars is not None:
            parts.append(scalars)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def pairwise_rank_loss(pred, y, groups, margin_sd=0.5):
    """Within-complex pairwise logistic loss (H3).

    Only pairs inside a complex are compared, which is what per-complex Spearman measures, and
    pairs closer than ``margin_sd`` kcal/mol are dropped: measurement noise is around 0.5, so
    their ordering is close to random and training on them injects noise.
    """
    total, n = pred.new_zeros(()), 0
    for gid in torch.unique(groups):
        m = groups == gid
        if m.sum() < 2:
            continue
        p, t = pred[m], y[m]
        dp = p.unsqueeze(0) - p.unsqueeze(1)
        dt = t.unsqueeze(0) - t.unsqueeze(1)
        keep = dt.abs() > margin_sd
        if not keep.any():
            continue
        total = total + nn.functional.softplus(-torch.sign(dt[keep]) * dp[keep]).sum()
        n += int(keep.sum())
    if n == 0:
        # No comparable pair in this batch. Returning a bare zero detaches the graph and
        # backward() raises; scaling the predictions by zero keeps it attached.
        return pred.sum() * 0.0
    return total / n


def _complex_batches(groups, target_size):
    """Batches made of whole complexes, so a within-complex pairwise loss has pairs."""
    g = groups.cpu().numpy() if torch.is_tensor(groups) else np.asarray(groups)
    order = np.random.permutation(np.unique(g))
    batches, cur = [], []
    for gid in order:
        cur.append(np.where(g == gid)[0])
        if sum(len(c) for c in cur) >= target_size:
            batches.append(torch.as_tensor(np.concatenate(cur), dtype=torch.long))
            cur = []
    if cur:
        batches.append(torch.as_tensor(np.concatenate(cur), dtype=torch.long))
    return batches


def _fit(tr, va, cfg, seed):
    s_tr, g_tr, c_tr, x_tr, q_tr, y_tr, grp_tr = tr
    s_va, g_va, c_va, x_va, q_va, y_va, _ = va
    torch.manual_seed(seed)
    dev = resolve_device(cfg.get("device", "auto"))
    net = ArchNetV2(seq_dim=s_tr.shape[-1], struct_dim=g_tr.shape[-1], d=cfg["d"],
                    heads=cfg["heads"], dropout=cfg["dropout"],
                    n_scalars=0 if x_tr is None else x_tr.shape[-1],
                    hadamard=cfg["hadamard"], attn=cfg["attn"], mut_token=cfg["mut_token"],
                    cats=cfg["cats"],
                    query_dim=0 if q_tr is None else q_tr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    huber = nn.HuberLoss(delta=2.0)

    T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt, device=dev)
    s_tr, g_tr, x_tr, q_tr, y_tr = T(s_tr), T(g_tr), T(x_tr), T(q_tr), T(y_tr)
    s_va, g_va, x_va, q_va, y_va = T(s_va), T(g_va), T(x_va), T(q_va), T(y_va)
    c_tr, c_va = T(c_tr, torch.long), T(c_va, torch.long)
    grp_tr = T(grp_tr, torch.long)

    yv, yt = y_va.cpu().numpy(), y_tr.cpu().numpy()
    span = yt.max() - yt.min()
    lo_ok, hi_ok = yt.min() - span, yt.max() + span
    best, state, waited = -np.inf, None, 0
    for _ in range(cfg["epochs"]):
        net.train()
        # A within-complex ranking loss needs batches that actually contain within-complex
        # pairs. Random batches of 64 drawn over 54 complexes, then filtered to pairs more than
        # 0.5 kcal/mol apart, frequently contain none at all -- so for rank objectives the
        # batches are built from whole complexes instead.
        if cfg["loss"] in ("rank", "both"):
            batches = _complex_batches(grp_tr, cfg["batch"])
        else:
            perm = torch.randperm(len(y_tr))
            batches = [perm[i:i + cfg["batch"]] for i in range(0, len(y_tr), cfg["batch"])]
        for j in batches:
            opt.zero_grad()
            pred = net(s_tr[j], g_tr[j], c_tr[j],
                       None if x_tr is None else x_tr[j],
                       None if q_tr is None else q_tr[j])
            if cfg["loss"] == "rank":
                loss = pairwise_rank_loss(pred, y_tr[j], grp_tr[j])
            elif cfg["loss"] == "both":
                loss = huber(pred, y_tr[j]) + cfg.get("rank_w", 1.0) * \
                    pairwise_rank_loss(pred, y_tr[j], grp_tr[j])
            else:
                loss = huber(pred, y_tr[j])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(s_va, g_va, c_va, x_va, q_va).cpu().numpy()
        v = evaluate._safe_spearman(yv, pv)
        v = -np.inf if not np.isfinite(v) else v
        # a rank loss is scale-free, so the guard matters more here, not less
        if cfg["loss"] == "huber" and not (lo_ok <= pv.min() and pv.max() <= hi_ok):
            v = -np.inf
        if v > best + 1e-4:
            best, state, waited = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    if state is not None:
        net.load_state_dict(state)
    return net.eval()


def _inner_forest_residuals(X, y, folds_tr_idx, clusters, seed):
    """Forest predictions for training rows via an INNER cluster-grouped CV (H4).

    Using the outer out-of-fold predictions here would leak: those were produced by forests that
    had already seen the outer test fold.
    """
    pred = np.full(len(y), np.nan)
    cl = clusters[folds_tr_idx]
    uniq = pd.unique(cl)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    chunks = np.array_split(order, 3)
    for held in chunks:
        va = np.isin(cl, held)
        tr = ~va
        est = MODELS["rf"](seed).fit(X[folds_tr_idx][tr], y[folds_tr_idx][tr])
        idx = folds_tr_idx[va]
        pred[idx] = est.predict(X[idx])
    return pred


def run(cfg, seeds=(0, 1, 2, 3, 4), verbose=False):
    d = splits.load()
    seq = _tokens(SEQ_CACHE, SEQ_BLOCKS, d.row_id)
    st = _tokens(STRUCT_CACHE, STRUCT_BLOCKS, d.row_id)
    cats = categoricals(d)
    y = d.ddG.to_numpy(np.float32)
    folds, clusters = d.fold.to_numpy(), d.cluster.to_numpy()
    groups = pd.factorize(d["#Pdb"])[0]

    blocks = cfg.get("blocks") or ["chem", "geom", "geomrev", "mpnn"]
    scalars = np.nan_to_num(train.build_matrix(d, blocks).to_numpy(np.float32)) \
        if cfg.get("scalars") else None
    qextra = np.nan_to_num(train.build_matrix(d, ["chem"]).to_numpy(np.float32)) \
        if cfg.get("query_chem") else None

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

            base_tr = base_va = base_te = None
            target = y.copy()
            if cfg.get("residual"):
                Xs = np.nan_to_num(train.build_matrix(d, blocks).to_numpy(np.float32))
                idx_tr = np.where(tr_all)[0]
                inner = _inner_forest_residuals(Xs, y, idx_tr, clusters, seed)
                est = MODELS["rf"](seed).fit(Xs[tr_all], y[tr_all])
                base_te = est.predict(Xs[te])
                base_tr, base_va = inner[tr], inner[va]
                target = y.copy()
                target[tr] = y[tr] - base_tr
                target[va] = y[va] - base_va

            s_tr, s_va, s_te = _standardise(seq[tr], seq[va], seq[te])
            g_tr, g_va, g_te = _standardise(st[tr], st[va], st[te])
            sc = [None, None, None]
            if scalars is not None:
                mu, sd = scalars[tr].mean(0), scalars[tr].std(0) + 1e-6
                sc = [np.clip((scalars[m] - mu) / sd, -CLIP, CLIP) for m in (tr, va, te)]
            qx = [None, None, None]
            if qextra is not None:
                mu, sd = qextra[tr].mean(0), qextra[tr].std(0) + 1e-6
                qx = [np.clip((qextra[m] - mu) / sd, -CLIP, CLIP) for m in (tr, va, te)]

            net = _fit((s_tr, g_tr, cats[tr], sc[0], qx[0], target[tr], groups[tr]),
                       (s_va, g_va, cats[va], sc[1], qx[1], target[va], groups[va]), cfg, seed)
            with torch.no_grad():
                T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt, device=dev)
                p = net(T(s_te), T(g_te), T(cats[te], torch.long), T(sc[2]), T(qx[2])).cpu().numpy()
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
    (out_dir / "run.json").write_text(json.dumps({"name": cfg["name"], "model": "fusion_v2",
                                                  "config": cfg}, indent=2))
    return {"name": cfg["name"], "per_cx_rho": round(m["per_complex_spearman"], 3),
            "global_rho": round(m["global_spearman"], 3), "rmse": round(m["rmse"], 3)}


BASE = dict(heads=4, dropout=0.2, lr=3e-4, wd=1e-2, epochs=300, batch=64, patience=25, d=64,
            hadamard=True, attn=False, mut_token=True, cats=False, scalars=True,
            query_chem=False, residual=False, loss="huber", blocks=None)


def sweep_configs():
    C = []
    def add(name, **kw):
        cfg = dict(BASE, name=name)
        cfg.update(kw)
        C.append(cfg)

    add("v2_h1_allfeatures")                                   # H1 alone
    add("v2_h2_querychem", query_chem=True)                    # H1 + H2
    add("v2_h3_rankloss", loss="rank")                         # H1 + H3
    add("v2_h3_both", loss="both")                             # H1 + H3 (mixed objective)
    add("v2_h2h3", query_chem=True, loss="both")               # H1 + H2 + H3
    add("v2_h4_residual", residual=True)                       # H1 + H4
    add("v2_h4_residual_rank", residual=True, loss="both")     # H1 + H3 + H4
    add("v2_h4_residual_query", residual=True, query_chem=True)
    add("v2_all", residual=True, query_chem=True, loss="both")
    add("v2_all_d32", residual=True, query_chem=True, loss="both", d=32)
    return C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    seeds = tuple(range(a.seeds))
    rows = []
    for cfg in sweep_configs():
        t0 = time.perf_counter()
        try:
            r = run(cfg, seeds)
        except Exception as e:
            r = {"name": cfg["name"], "per_cx_rho": None, "global_rho": None,
                 "rmse": None, "error": f"{type(e).__name__}: {e}"}
        r["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(r)
        print(f"{cfg['name']:<26} rho {str(r['per_cx_rho']):>6}  "
              f"rmse {str(r['rmse']):>6}  {r['minutes']}m"
              + (f"  {r.get('error','')}" if r.get("error") else ""), flush=True)
        pd.DataFrame(rows).to_csv(paths.REPORTS / "fusion_v2_sweep.csv", index=False)
    t = pd.DataFrame(rows).sort_values("per_cx_rho", ascending=False, na_position="last")
    print()
    print("=" * 84)
    print("HYPOTHESES v2 -- every cell reported; the maximum over a sweep is not a result")
    print("=" * 84)
    print(t.to_string(index=False))
    print("\n  random forest, chem+geom+geomrev+mpnn   0.498")
    print("  random forest, chem+geom+mpnn           0.487")


if __name__ == "__main__":
    main()
