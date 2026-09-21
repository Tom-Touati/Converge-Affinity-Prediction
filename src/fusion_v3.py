"""Chemistry as a third modality inside the fusion, not as a bypass to the head.

Where chem sits today. ``fusion_v2`` concatenates all 49 scalar columns -- 21 of them chem --
straight onto the final MLP's input. They therefore skip both fusion mechanisms: the Hadamard
product sees only sequence and structure, and the mutation-token attention reads only the six
encoder tokens. Chemistry influences the answer but never interacts with the other modalities
except additively, in the last layer.

**Why that is probably the wrong place for it.** The Hadamard product exists to express
"how disruptive is this substitution" times "how much does this site care", and the two encoders
were supposed to supply those two factors. But the sequence encoder does not: 82% of the variance
in its embedding difference is explained by the substitution type alone, and its leading
components are `n_to_ala` (r 0.71) and `d_mw` (r 0.67). ESM-2 *is* substitution chemistry, encoded
worse than `chem` encodes it directly. So the product that should matter is **chem x structure**,
not sequence x structure, and we have never formed it.

Three changes, each switchable:

* ``chem_proj`` -- chem gets its own projection into the shared space, like the other modalities.
* ``hadamard_pairs`` -- which products to form. ``sg`` is the current sequence x structure;
  ``cg`` is chem x structure; ``cs`` is chem x sequence. Several can run in parallel.
* ``chem_token`` -- chem also joins the key/value set the mutation-token query attends over, so
  the query can retrieve chemistry alongside the encoder representations.

Everything runs on top of the residual-on-forest setup, which is the configuration that works
(+0.094 over its own control, the only architectural gain in this project whose paired bootstrap
clears zero).

    python -m src.fusion_v3 --sweep
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
from .fusion_arch import CARDS, categoricals
from .fusion_net import (CLIP, SEQ_BLOCKS, SEQ_CACHE, STRUCT_BLOCKS, STRUCT_CACHE,
                         _standardise, _tokens)
from .fusion_v2 import _complex_batches, _inner_forest_residuals, pairwise_rank_loss
from .model import MODELS
from .util import resolve_device


class TriModalNet(nn.Module):
    """Sequence, structure and chemistry as three peers in the fusion."""

    def __init__(self, seq_dim, struct_dim, chem_dim, d=64, heads=4, dropout=0.2,
                 n_scalars=0, hadamard_pairs=("sg",), chem_token=True, mut_token=True):
        super().__init__()
        self.pairs, self.use_chem_token, self.use_m = tuple(hadamard_pairs), chem_token, mut_token
        self.seq_proj = nn.Linear(seq_dim, d)
        self.struct_proj = nn.Linear(struct_dim, d)
        self.chem_proj = nn.Linear(chem_dim, d)
        self.seq_norm, self.struct_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.chem_norm = nn.LayerNorm(d)

        width = 0
        self.had = nn.ModuleDict()
        for p in self.pairs:
            self.had[p] = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d, d))
            width += d
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in CARDS])
        if mut_token:
            self.m_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.m_norm = nn.LayerNorm(d)
            width += d
        self.head = nn.Sequential(nn.Linear(width + n_scalars, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, seq, struct, chem, cats, scalars=None):
        s = self.seq_norm(self.seq_proj(seq))            # (B, 3, d)
        g = self.struct_norm(self.struct_proj(struct))   # (B, 3, d)
        c = self.chem_norm(self.chem_proj(chem))         # (B, d)
        pooled = {"s": s.mean(1), "g": g.mean(1), "c": c}

        parts = [self.had[p](pooled[p[0]] * pooled[p[1]]) for p in self.pairs]
        if self.use_m:
            e = sum(emb(cats[:, i]) for i, emb in enumerate(self.cat_emb))
            kv = [s, g] + ([c.unsqueeze(1)] if self.use_chem_token else [])
            kv = torch.cat(kv, dim=1)
            m, _ = self.m_attn(e.unsqueeze(1), kv, kv, need_weights=False)
            parts.append(self.m_norm(m).squeeze(1))
        if scalars is not None:
            parts.append(scalars)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _fit(tr, va, cfg, seed):
    s_tr, g_tr, ch_tr, c_tr, x_tr, y_tr, grp_tr = tr
    s_va, g_va, ch_va, c_va, x_va, y_va, _ = va
    torch.manual_seed(seed)
    dev = resolve_device(cfg.get("device", "auto"))
    net = TriModalNet(seq_dim=s_tr.shape[-1], struct_dim=g_tr.shape[-1],
                      chem_dim=ch_tr.shape[-1], d=cfg["d"], heads=cfg["heads"],
                      dropout=cfg["dropout"],
                      n_scalars=0 if x_tr is None else x_tr.shape[-1],
                      hadamard_pairs=cfg["pairs"], chem_token=cfg["chem_token"],
                      mut_token=cfg["mut_token"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    huber = nn.HuberLoss(delta=2.0)
    T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt, device=dev)
    s_tr, g_tr, ch_tr, x_tr, y_tr = T(s_tr), T(g_tr), T(ch_tr), T(x_tr), T(y_tr)
    s_va, g_va, ch_va, x_va, y_va = T(s_va), T(g_va), T(ch_va), T(x_va), T(y_va)
    c_tr, c_va, grp_tr = T(c_tr, torch.long), T(c_va, torch.long), T(grp_tr, torch.long)

    yv, yt = y_va.cpu().numpy(), y_tr.cpu().numpy()
    span = yt.max() - yt.min()
    lo_ok, hi_ok = yt.min() - span, yt.max() + span
    best, state, waited = -np.inf, None, 0
    for _ in range(cfg["epochs"]):
        net.train()
        if cfg["loss"] in ("rank", "both"):
            batches = _complex_batches(grp_tr, cfg["batch"])
        else:
            perm = torch.randperm(len(y_tr))
            batches = [perm[i:i + cfg["batch"]] for i in range(0, len(y_tr), cfg["batch"])]
        for j in batches:
            opt.zero_grad()
            pred = net(s_tr[j], g_tr[j], ch_tr[j], c_tr[j],
                       None if x_tr is None else x_tr[j])
            if cfg["loss"] == "rank":
                loss = pairwise_rank_loss(pred, y_tr[j], grp_tr[j])
            elif cfg["loss"] == "both":
                loss = huber(pred, y_tr[j]) + pairwise_rank_loss(pred, y_tr[j], grp_tr[j])
            else:
                loss = huber(pred, y_tr[j])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(s_va, g_va, ch_va, c_va, x_va).cpu().numpy()
        v = evaluate._safe_spearman(yv, pv)
        v = -np.inf if not np.isfinite(v) else v
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
    scalars = Xs if cfg.get("scalars") else None

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

            target = y.copy()
            base_te = None
            if cfg.get("residual"):
                idx_tr = np.where(tr_all)[0]
                inner = _inner_forest_residuals(Xs, y, idx_tr, clusters, seed)
                est = MODELS["rf"](seed).fit(Xs[tr_all], y[tr_all])
                base_te = est.predict(Xs[te])
                target[tr] = y[tr] - inner[tr]
                target[va] = y[va] - inner[va]

            s3 = _standardise(seq[tr], seq[va], seq[te])
            g3 = _standardise(st[tr], st[va], st[te])
            mu, sd = chem[tr].mean(0), chem[tr].std(0) + 1e-6
            ch3 = [np.clip((chem[m] - mu) / sd, -CLIP, CLIP) for m in (tr, va, te)]
            sc = [None, None, None]
            if scalars is not None:
                mu2, sd2 = scalars[tr].mean(0), scalars[tr].std(0) + 1e-6
                sc = [np.clip((scalars[m] - mu2) / sd2, -CLIP, CLIP) for m in (tr, va, te)]

            net = _fit((s3[0], g3[0], ch3[0], cats[tr], sc[0], target[tr], groups[tr]),
                       (s3[1], g3[1], ch3[1], cats[va], sc[1], target[va], groups[va]),
                       cfg, seed)
            with torch.no_grad():
                T = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt, device=dev)
                p = net(T(s3[2]), T(g3[2]), T(ch3[2]), T(cats[te], torch.long), T(sc[2])).cpu().numpy()
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
    (out_dir / "run.json").write_text(json.dumps({"name": cfg["name"], "model": "fusion_v3",
                                                  "config": cfg}, indent=2))
    return {"name": cfg["name"], "pairs": "+".join(cfg["pairs"]),
            "chem_token": cfg["chem_token"],
            "per_cx_rho": round(m["per_complex_spearman"], 3),
            "global_rho": round(m["global_spearman"], 3), "rmse": round(m["rmse"], 3)}


BASE = dict(heads=4, dropout=0.2, lr=3e-4, wd=1e-2, epochs=300, batch=64, patience=25, d=64,
            mut_token=True, scalars=True, residual=True, loss="huber",
            pairs=("sg",), chem_token=False)


def sweep_configs():
    C = []
    def add(name, **kw):
        cfg = dict(BASE, name=name)
        cfg.update(kw)
        C.append(cfg)

    add("v3_base_sg")                                       # reproduces v2_h4_residual
    add("v3_cg", pairs=("cg",))                             # chem x structure instead
    add("v3_sg_cg", pairs=("sg", "cg"))                     # both products
    add("v3_all_pairs", pairs=("sg", "cg", "cs"))           # every pairwise product
    add("v3_chemtoken", chem_token=True)                    # chem joins the attention kv set
    add("v3_cg_chemtoken", pairs=("cg",), chem_token=True)
    add("v3_allpairs_chemtoken", pairs=("sg", "cg", "cs"), chem_token=True)
    add("v3_cg_d32", pairs=("cg",), d=32)
    add("v3_sg_cg_d32", pairs=("sg", "cg"), d=32)
    add("v3_cg_rank", pairs=("cg",), loss="both")
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
            r = {"name": cfg["name"], "per_cx_rho": None, "error": f"{type(e).__name__}: {e}"}
        r["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(r)
        print(f"{cfg['name']:<24} rho {str(r.get('per_cx_rho')):>6}  "
              f"global {str(r.get('global_rho')):>6}  rmse {str(r.get('rmse')):>6}  "
              f"{r['minutes']}m" + (f"  {r.get('error','')}" if r.get("error") else ""),
              flush=True)
        pd.DataFrame(rows).to_csv(paths.REPORTS / "fusion_v3_sweep.csv", index=False)
    t = pd.DataFrame(rows).sort_values("per_cx_rho", ascending=False, na_position="last")
    print()
    print("=" * 88)
    print("CHEMISTRY AS A THIRD MODALITY -- every cell reported")
    print("=" * 88)
    print(t.to_string(index=False))
    print("\n  random forest                       0.498")
    print("  v2_h4_residual (chem head-only)     0.466")


if __name__ == "__main__":
    main()
