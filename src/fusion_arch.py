"""Architecture search over the multimodal fusion: which mechanism actually contributes?

``fusion_net.py`` runs a Hadamard path and a cross-attention path in parallel and concatenates
them, so a score from it cannot say which one earned the result. This module makes every
mechanism switchable and sweeps them, so each claim is backed by an ablation rather than by the
whole model being present.

**The mechanisms.**

*Hadamard.* Pool each modality to one vector, multiply element-wise, pass through an MLP.
Concatenation lets a head form additive combinations only; a product gives it multiplicative ones.
That is the physically motivated shape here -- the effect of a substitution is roughly *how
disruptive the swap is* times *how much the site cares*, and those are the two things the two
encoders separately describe.

*Structure-queries-sequence cross-attention.* Queries from the three ProteinMPNN tokens, keys and
values from the three ESM-2 tokens. ``h_V`` comes from backbone geometry alone, so it describes
the site and is constant across substitutions at that position; the sequence tokens are the only
ones that vary with the substitution. The site asks, the substitution answers.

*Mutation-token cross-attention.* A single learned query assembled from the mutation's categorical
embeddings, attending over all six modality tokens at once. This is the cleanest statement of the
task -- "here is the mutation, go and read what both encoders say about it" -- and unlike the
previous mechanism its query is not constant across substitutions at the same site.

*Categorical embeddings.* Learned vectors for wild-type residue, mutant residue, interface
location and which side of the interface was mutated. Multi-point rows get their own category
rather than a sentinel, because -1 in a count column is what produced an 11-sigma input and a
-6872 kcal/mol prediction once already.

Assay method is deliberately **not** embedded. It is experimental metadata, not a property of the
mutation, and a model that learns "this assay reads high" will not transfer to a new target.

    python -m src.fusion_arch --sweep
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
from .fusion_net import (CLIP, SEQ_BLOCKS, SEQ_CACHE, STRUCT_BLOCKS, STRUCT_CACHE,
                         _standardise, _tokens)

AAS = "ACDEFGHIKLMNPQRSTVWY"
LOCS = ["COR", "SUP", "RIM", "INT", "SUR"]
SIDES = ["antibody", "antigen", "both", "ab_vs_ab"]
#: wt residue, mutant residue, interface location, side. Each has one extra slot for
#: multi-point rows, which are a genuine category rather than a missing value.
CARDS = (21, 21, 6, 4)


def categoricals(d: pd.DataFrame) -> np.ndarray:
    from .structures import parse_mutations

    rows = []
    for r in d.itertuples():
        muts = parse_mutations(r.mutations)
        if len(muts) == 1:
            wt, mu = AAS.index(muts[0].wt), AAS.index(muts[0].mut)
            loc = LOCS.index(r.location) if r.location in LOCS else 5
        else:
            wt = mu = 20            # "multi-point", not a sentinel
            loc = 5
        rows.append([wt, mu, loc, SIDES.index(r.mut_side)])
    return np.asarray(rows, np.int64)


class ArchNet(nn.Module):
    def __init__(self, seq_dim, struct_dim, d=64, heads=4, dropout=0.2, n_scalars=0,
                 hadamard=True, attn=True, mut_token=False, cats=False):
        super().__init__()
        self.use_h, self.use_a, self.use_m, self.use_c = hadamard, attn, mut_token, cats
        self.seq_proj, self.struct_proj = nn.Linear(seq_dim, d), nn.Linear(struct_dim, d)
        self.seq_norm, self.struct_norm = nn.LayerNorm(d), nn.LayerNorm(d)

        width = 0
        if hadamard:
            self.had = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(d, d))
            width += d
        if attn:
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(d)
            width += d
        if cats or mut_token:
            self.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in CARDS])
        if mut_token:
            self.m_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.m_norm = nn.LayerNorm(d)
            width += d
        if cats:
            width += d
        self.head = nn.Sequential(nn.Linear(width + n_scalars, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, seq, struct, cats=None, scalars=None):
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
        if self.use_m:
            q = e.unsqueeze(1)                                  # the mutation, as one query
            kv = torch.cat([s, g], dim=1)                       # everything both encoders say
            m, _ = self.m_attn(q, kv, kv, need_weights=False)
            parts.append(self.m_norm(m).squeeze(1))
        if self.use_c:
            parts.append(e)
        if scalars is not None:
            parts.append(scalars)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _fit(tr, va, cfg, seed):
    (s_tr, g_tr, c_tr, x_tr, y_tr) = tr
    (s_va, g_va, c_va, x_va, y_va) = va
    torch.manual_seed(seed)
    net = ArchNet(s_tr.shape[-1], g_tr.shape[-1], cfg["d"], cfg["heads"], cfg["dropout"],
                  0 if x_tr is None else x_tr.shape[-1], cfg["hadamard"], cfg["attn"],
                  cfg["mut_token"], cfg["cats"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.HuberLoss(delta=2.0)
    T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt)
    s_tr, g_tr, x_tr, y_tr = T(s_tr), T(g_tr), T(x_tr), T(y_tr)
    s_va, g_va, x_va, y_va = T(s_va), T(g_va), T(x_va), T(y_va)
    c_tr, c_va = T(c_tr, torch.long), T(c_va, torch.long)

    yv, yt = y_va.numpy(), y_tr.numpy()
    span = yt.max() - yt.min()
    lo_ok, hi_ok = yt.min() - span, yt.max() + span
    best, state, waited = -np.inf, None, 0
    for _ in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(y_tr), cfg["batch"]):
            j = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = lossf(net(s_tr[j], g_tr[j], c_tr[j], None if x_tr is None else x_tr[j]),
                         y_tr[j])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(s_va, g_va, c_va, x_va).numpy()
        v = evaluate._safe_spearman(yv, pv)
        v = -np.inf if not np.isfinite(v) else v
        if not (lo_ok <= pv.min() and pv.max() <= hi_ok):
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


def run(cfg, seeds=(0, 1, 2, 3, 4), verbose=True):
    d = splits.load()
    seq = _tokens(SEQ_CACHE, SEQ_BLOCKS, d.row_id)
    st = _tokens(STRUCT_CACHE, STRUCT_BLOCKS, d.row_id)
    cats = categoricals(d)
    scalars = None
    if cfg.get("scalars"):
        scalars = np.nan_to_num(train.build_matrix(d, ["chem", "geom", "mpnn"]).to_numpy(np.float32))
    y = d.ddG.to_numpy(np.float32)
    folds, clusters = d.fold.to_numpy(), d.cluster.to_numpy()

    per_seed, oofs = [], []
    for seed in seeds:
        oof = np.full(len(d), np.nan, np.float32)
        for f in np.unique(folds):
            te = folds == f
            tr_all = ~te
            tc = pd.unique(clusters[tr_all])
            rng = np.random.default_rng(seed)
            n_va = max(3, len(tc) // 3)
            vc = set(rng.choice(tc, min(n_va, len(tc) - 1), replace=False))
            va = tr_all & np.isin(clusters, list(vc))
            tr = tr_all & ~va

            s_tr, s_va, s_te = _standardise(seq[tr], seq[va], seq[te])
            g_tr, g_va, g_te = _standardise(st[tr], st[va], st[te])
            if scalars is None:
                x_tr = x_va = x_te = None
            else:
                mu, sd = scalars[tr].mean(0), scalars[tr].std(0) + 1e-6
                x_tr, x_va, x_te = (np.clip((scalars[m] - mu) / sd, -CLIP, CLIP)
                                    for m in (tr, va, te))
            net = _fit((s_tr, g_tr, cats[tr], x_tr, y[tr]),
                       (s_va, g_va, cats[va], x_va, y[va]), cfg, seed)
            with torch.no_grad():
                T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt)
                oof[te] = net(T(s_te), T(g_te), T(cats[te], torch.long), T(x_te)).numpy()

        out = pd.DataFrame({"row_id": d.row_id, "complex": d["#Pdb"], "cluster": d.cluster,
                            "fold": d.fold, "y_true": y.astype(float), "y_pred": oof})
        per_seed.append(evaluate.metrics(out, 10)["per_complex_spearman"])
        oofs.append(oof)

    ens = pd.DataFrame({"row_id": d.row_id, "complex": d["#Pdb"], "cluster": d.cluster,
                        "fold": d.fold, "y_true": y.astype(float),
                        "y_pred": np.mean(oofs, axis=0)})
    m = evaluate.metrics(ens, 10)
    out_dir = paths.REPORTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out_dir / "predictions.csv", index=False)
    n_par = sum(p.numel() for p in ArchNet(
        seq.shape[-1], st.shape[-1], cfg["d"], cfg["heads"], cfg["dropout"],
        0 if scalars is None else scalars.shape[-1], cfg["hadamard"], cfg["attn"],
        cfg["mut_token"], cfg["cats"]).parameters())
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": m, "ci": {}}, indent=2, default=float))
    (out_dir / "run.json").write_text(json.dumps({"name": cfg["name"], "model": "fusion_arch",
                                                  "config": cfg, "params": n_par}, indent=2))
    return {"name": cfg["name"], "params": n_par,
            "hadamard": cfg["hadamard"], "attn": cfg["attn"],
            "mut_token": cfg["mut_token"], "cats": cfg["cats"], "scalars": cfg["scalars"],
            "per_cx_rho": round(m["per_complex_spearman"], 3),
            "seed_mean": round(float(np.mean(per_seed)), 3),
            "seed_sd": round(float(np.std(per_seed)), 4),
            "global_rho": round(m["global_spearman"], 3), "rmse": round(m["rmse"], 3)}


BASE = dict(heads=4, dropout=0.2, lr=3e-4, wd=1e-2, epochs=300, batch=64, patience=25, d=64)


def sweep_configs():
    C = []
    def add(name, **kw):
        # build defaults first, then override -- passing both as kwargs to dict() hands it the
        # same key twice and raises
        cfg = dict(BASE, name=name, hadamard=False, attn=False, mut_token=False,
                   cats=False, scalars=False)
        cfg.update(kw)
        C.append(cfg)
    # --- which mechanism carries the result? one at a time, then together ---
    add("arch_hadamard", hadamard=True)
    add("arch_attn", attn=True)
    add("arch_had_attn", hadamard=True, attn=True)
    add("arch_muttoken", mut_token=True)
    add("arch_cats", cats=True)
    # --- combinations ---
    add("arch_had_mut", hadamard=True, mut_token=True)
    add("arch_had_cats", hadamard=True, cats=True)
    add("arch_all", hadamard=True, attn=True, mut_token=True, cats=True)
    # --- width, on the two most promising shapes ---
    add("arch_hadamard_d32", hadamard=True, d=32)
    add("arch_had_mut_d32", hadamard=True, mut_token=True, d=32)
    add("arch_had_mut_d128", hadamard=True, mut_token=True, d=128)
    # --- and with the hand-crafted scalars appended, for the ceiling ---
    add("arch_had_mut_scalars", hadamard=True, mut_token=True, scalars=True)
    add("arch_all_scalars", hadamard=True, attn=True, mut_token=True, cats=True, scalars=True)
    add("arch_had_mut_scalars_d32", hadamard=True, mut_token=True, scalars=True, d=32)
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
        r = run(cfg, seeds)
        r["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(r)
        print(f"{cfg['name']:<28} rho {r['per_cx_rho']:+.3f}  "
              f"(seed {r['seed_mean']:+.3f}+-{r['seed_sd']:.3f})  "
              f"{r['params']:>7,}p  rmse {r['rmse']:.3f}  {r['minutes']}m", flush=True)
    t = pd.DataFrame(rows).sort_values("per_cx_rho", ascending=False)
    t.to_csv(paths.REPORTS / "fusion_arch_sweep.csv", index=False)
    print()
    print("=" * 110)
    print("FUSION ARCHITECTURE ABLATION -- every cell reported; the maximum is not selectable")
    print("=" * 110)
    print(t.to_string(index=False))
    print()
    print("  random forest, 33 hand-crafted features            0.487")
    print("  random forest, the same two pretrained encoders    0.362")


if __name__ == "__main__":
    main()
