"""The multimodal fusion network the assignment asks for: two pretrained encoders, learned fusion.

Both modalities enter as **representations from frozen pretrained models**, not as hand-crafted
descriptors:

* **Sequence** -- ESM-2 35M, three tokens per row, each 480-dimensional: the wild-type embedding
  at the mutated position, the mutant embedding, and their difference.
* **Structure** -- ProteinMPNN's encoder ``h_V``, three tokens per row, each 128-dimensional:
  conditioned on the whole complex, on the mutated chain alone, and their difference. ``h_V`` is
  the geometric graph network's own representation, taken before the 20-way output softmax that
  ``proteinmpnn.py`` reads.

Two fusion mechanisms, run in parallel and concatenated before the head.

**Hadamard.** Each modality is pooled to one vector and the two are multiplied element-wise.
Concatenation lets a head form only additive combinations of the two modalities; an element-wise
product gives it multiplicative ones directly. That matters here for a physical reason: the
effect of a substitution is roughly *how disruptive the swap is* times *how much the site cares*,
and those are the two things the two encoders separately describe.

**Cross-attention, structure querying sequence.** Queries come from the structure tokens, keys and
values from the sequence tokens. The direction is deliberate. ``h_V`` is computed from backbone
geometry alone, so it describes the *site* and is identical for every substitution at that
position -- 256 of 696 single-point rows share a site with another row. The sequence tokens are
the only ones that vary with the substitution. Letting the site ask the question and the
substitution answer it is the factorisation the data supports; the reverse direction would have a
constant query.

**Expected to lose, and built anyway.** The plain-MLP gate in ARCHITECTURE.md failed at 0 of 12
configurations, and this model carries roughly 72k parameters against ~750 training rows. The
brief asks for a justified multimodal fusion of pretrained encoders, and a well-built one that is
honestly evaluated against a simpler baseline is the deliverable, whichever way the number falls.

Early stopping uses an inner split **grouped by homology cluster**, never random: a constant
per-fold predictor already scores global rho -0.36 on this data, so a random validation split
would leak the homology the outer split exists to prevent.

    python -m src.fusion_net --sweep
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

SEQ_BLOCKS = ("wt_d", "mt_d", "d")          # wild type, mutant, difference
STRUCT_BLOCKS = ("hc", "ha", "hd")          # complex, chain alone, difference
SEQ_CACHE, STRUCT_CACHE = "esm35M_pair", "mpnnrep"


def _tokens(cache: str, prefixes, n_rows_index) -> np.ndarray:
    """(N, n_tokens, dim) from one cached feature block."""
    b = pd.read_parquet(paths.FEATURES / f"{cache}.parquet").loc[n_rows_index]
    mats = []
    for p in prefixes:
        cols = [c for c in b.columns if c.startswith(p) and c[len(p):].isdigit()]
        cols.sort(key=lambda c: int(c[len(p):]))
        mats.append(b[cols].to_numpy(np.float32))
    return np.stack(mats, axis=1)


class FusionNet(nn.Module):
    def __init__(self, seq_dim: int, struct_dim: int, d: int = 64, heads: int = 4,
                 dropout: float = 0.2, n_scalars: int = 0):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, d)
        self.struct_proj = nn.Linear(struct_dim, d)
        self.seq_norm = nn.LayerNorm(d)
        self.struct_norm = nn.LayerNorm(d)

        # path 1: element-wise product of the two pooled modalities
        self.hadamard = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

        # path 2: structure queries, sequence keys and values
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d)

        self.head = nn.Sequential(
            nn.Linear(2 * d + n_scalars, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, seq, struct, scalars=None):
        s = self.seq_norm(self.seq_proj(seq))          # (B, 3, d)
        g = self.struct_norm(self.struct_proj(struct))  # (B, 3, d)

        had = self.hadamard(s.mean(1) * g.mean(1))     # (B, d)
        att, _ = self.attn(g, s, s, need_weights=False)  # structure asks, sequence answers
        att = self.attn_norm(att).mean(1)               # (B, d)

        z = torch.cat([had, att] + ([scalars] if scalars is not None else []), dim=-1)
        return self.head(z).squeeze(-1)


def _standardise(tr: np.ndarray, *others):
    mu = tr.reshape(-1, tr.shape[-1]).mean(0)
    sd = tr.reshape(-1, tr.shape[-1]).std(0) + 1e-6
    return [(a - mu) / sd for a in (tr, *others)]


def _fit(seq_tr, st_tr, sc_tr, y_tr, seq_va, st_va, sc_va, y_va, cfg, seed):
    torch.manual_seed(seed)
    net = FusionNet(seq_tr.shape[-1], st_tr.shape[-1], cfg["d"], cfg["heads"],
                    cfg["dropout"], 0 if sc_tr is None else sc_tr.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.HuberLoss(delta=2.0)     # label noise ~0.5 kcal/mol; Huber limits outlier pull

    T = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32)
    seq_tr, st_tr, y_tr = T(seq_tr), T(st_tr), T(y_tr)
    seq_va, st_va, y_va = T(seq_va), T(st_va), T(y_va)
    sc_tr, sc_va = T(sc_tr), T(sc_va)

    n = len(y_tr)
    yv = y_va.numpy()
    # Select on validation SPEARMAN, not validation loss. Measured on this data: the network
    # overfits magnitudes hard (training Huber 1.43 -> 0.03) while validation loss is lowest at
    # epoch 1 and never beats it, so loss-based early stopping restores an untrained model --
    # which is exactly what the first smoke test did, scoring 0.038. Validation rank correlation
    # meanwhile climbs from 0.19 to 0.27 over the same run. The headline metric is rank-based, so
    # selecting on rank is both the honest choice and the one that works. The validation split is
    # carved from TRAINING clusters, so nothing about the test fold informs it.
    best, best_state, waited = -np.inf, None, 0
    for epoch in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            pred = net(seq_tr[idx], st_tr[idx], None if sc_tr is None else sc_tr[idx])
            loss = lossf(pred, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(seq_va, st_va, sc_va).numpy()
        v = evaluate._safe_spearman(yv, pv)
        v = -np.inf if not np.isfinite(v) else v
        if v > best + 1e-4:
            best, best_state, waited = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def run(cfg: dict, seeds=(0, 1, 2, 3, 4), verbose: bool = True) -> dict:
    d = splits.load()
    seq = _tokens(SEQ_CACHE, SEQ_BLOCKS, d.row_id)
    st = _tokens(STRUCT_CACHE, STRUCT_BLOCKS, d.row_id)
    scalars = None
    if cfg.get("scalars"):
        scalars = train.build_matrix(d, ["chem", "geom", "mpnn"]).to_numpy(np.float32)
        scalars = np.nan_to_num(scalars)
    y = d.ddG.to_numpy(np.float32)
    folds, clusters = d.fold.to_numpy(), d.cluster.to_numpy()

    per_seed, oofs = [], []
    for seed in seeds:
        oof = np.full(len(d), np.nan, np.float32)
        for f in np.unique(folds):
            te = folds == f
            tr_all = ~te
            # inner split GROUPED BY CLUSTER -- a random one would leak homology
            tr_clusters = pd.unique(clusters[tr_all])
            rng = np.random.default_rng(seed)
            # a third of the training clusters, never fewer than three: the first version held
            # out len//5, which on 10 training clusters is 2 clusters and 66 rows -- far too
            # noisy a signal to choose a checkpoint on
            n_va = max(3, len(tr_clusters) // 3)
            va_clusters = set(rng.choice(tr_clusters, min(n_va, len(tr_clusters) - 1),
                                         replace=False))
            va = tr_all & np.isin(clusters, list(va_clusters))
            tr = tr_all & ~va

            s_tr, s_va, s_te = _standardise(seq[tr], seq[va], seq[te])
            g_tr, g_va, g_te = _standardise(st[tr], st[va], st[te])
            if scalars is None:
                c_tr = c_va = c_te = None
            else:
                mu, sd = scalars[tr].mean(0), scalars[tr].std(0) + 1e-6
                c_tr, c_va, c_te = ((scalars[m] - mu) / sd for m in (tr, va, te))

            net = _fit(s_tr, g_tr, c_tr, y[tr], s_va, g_va, c_va, y[va], cfg, seed)
            with torch.no_grad():
                T = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32)
                oof[te] = net(T(s_te), T(g_te), T(c_te)).numpy()

        out = pd.DataFrame({"row_id": d.row_id, "complex": d["#Pdb"], "cluster": d.cluster,
                            "fold": d.fold, "y_true": y.astype(float), "y_pred": oof})
        per_seed.append(evaluate.metrics(out, 10))
        oofs.append(oof)
        if verbose:
            print(f"    seed {seed}: rho {per_seed[-1]['per_complex_spearman']:+.3f}", flush=True)

    ens = pd.DataFrame({"row_id": d.row_id, "complex": d["#Pdb"], "cluster": d.cluster,
                        "fold": d.fold, "y_true": y.astype(float),
                        "y_pred": np.mean(oofs, axis=0)})
    m = evaluate.metrics(ens, 10)
    name = cfg["name"]
    out_dir = paths.REPORTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out_dir / "predictions.csv", index=False)
    net_params = sum(p.numel() for p in FusionNet(
        seq.shape[-1], st.shape[-1], cfg["d"], cfg["heads"], cfg["dropout"],
        0 if scalars is None else scalars.shape[-1]).parameters())
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": m, "ci": {}}, indent=2, default=float))
    (out_dir / "run.json").write_text(json.dumps(
        {"name": name, "model": "fusion_net", "features": [SEQ_CACHE, STRUCT_CACHE],
         "config": cfg, "params": net_params, "seeds": list(seeds),
         "n_rows": len(d), "n_features": int(seq.shape[-1] + st.shape[-1])}, indent=2))
    return {
        "name": name, "params": net_params,
        "per_cx_rho": round(m["per_complex_spearman"], 3),
        "seed_mean": round(float(np.mean([p["per_complex_spearman"] for p in per_seed])), 3),
        "seed_sd": round(float(np.std([p["per_complex_spearman"] for p in per_seed])), 4),
        "global_rho": round(m["global_spearman"], 3), "rmse": round(m["rmse"], 3),
    }


BASE = dict(heads=4, dropout=0.2, lr=3e-4, wd=1e-2, epochs=300, batch=64, patience=25)

SWEEP = [
    dict(BASE, name="fusion_d32", d=32, scalars=False),
    dict(BASE, name="fusion_d64", d=64, scalars=False),
    dict(BASE, name="fusion_d64_drop05", d=64, dropout=0.5, scalars=False),
    dict(BASE, name="fusion_d128", d=128, scalars=False),
    dict(BASE, name="fusion_d64_scalars", d=64, scalars=True),
    dict(BASE, name="fusion_d32_scalars", d=32, scalars=True),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    seeds = tuple(range(a.seeds))
    rows = []
    for cfg in (SWEEP if a.sweep else SWEEP[1:2]):
        print(f"\n=== {cfg['name']}  d={cfg['d']} dropout={cfg['dropout']} "
              f"scalars={cfg['scalars']} ===", flush=True)
        t0 = time.perf_counter()
        r = run(cfg, seeds)
        r["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(r)
        print(f"  -> per-complex rho {r['per_cx_rho']:+.3f} "
              f"(seed mean {r['seed_mean']:+.3f} sd {r['seed_sd']:.3f}), "
              f"{r['params']:,} params, {r['minutes']} min", flush=True)
    t = pd.DataFrame(rows).sort_values("per_cx_rho", ascending=False)
    t.to_csv(paths.REPORTS / "fusion_sweep.csv", index=False)
    print()
    print("=" * 96)
    print("MULTIMODAL FUSION SWEEP -- all cells reported; the maximum is not a selectable result")
    print("=" * 96)
    print(t.to_string(index=False))
    print()
    print("  reference: random forest on 33 hand-crafted features = 0.487")
    print("             pretrained sequence + structure into a forest = 0.362")
    print(f"\nwrote {(paths.REPORTS / 'fusion_sweep.csv').relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
