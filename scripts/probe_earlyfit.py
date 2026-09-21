#!/usr/bin/env python3
"""Does the train/validation gap exist from the first steps, or does it open later?

Runs one configuration on ONE fold for a few dozen steps, evaluating train and validation
loss on the same objective every `--eval-every` steps, and writes a small CSV. Several of
these are meant to run at once (see --list), one process per configuration, because the
question is comparative and each run is seconds of work.

    python scripts/probe_earlyfit.py --list                 # names
    python scripts/probe_earlyfit.py --config full_heavy    # one probe
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import splits, train                                          # noqa: E402
from src.fusion_arch import categoricals                               # noqa: E402
from src.fusion_v2 import _inner_forest_residuals                      # noqa: E402
from src.model import MODELS                                           # noqa: E402
from src.rank_fusion import (BASE, CLIP, SEQ_BLOCKS, SEQ_CACHE, STRUCT_BLOCKS,  # noqa: E402
                             STRUCT_CACHE, _fit, _split_held, _standardise, _tokens)

PROBES = {
    "baseline":    {},
    "noise":       dict(noise_std=0.2),
    "fdrop":       dict(feat_drop=0.2),
    "aug_both":    dict(noise_std=0.2, feat_drop=0.2),
    "dropout":     dict(dropout=0.5),
    "wd":          dict(wd=1e-1),
    "small":       dict(d=32),
    "tiny":        dict(d=16),
    "full_light":  dict(noise_std=0.15, feat_drop=0.15, dropout=0.35, wd=3e-2, d=48),
    "full_heavy":  dict(noise_std=0.35, feat_drop=0.3, dropout=0.5, wd=1e-1, d=32),
    "lowlr":       dict(lr=5e-5),
    "huber_only":  dict(rank_w=0.0),

    # Component ablations. Each removes one path into the head, so the question "which part
    # is doing the memorising" gets an answer per part rather than in aggregate.
    #   no_attn      drop the mutation-token cross-attention   -20,864 params
    #   no_hadamard  drop the sequence-structure Hadamard      -12,416 params
    #   no_scalars   stop feeding the forest's own 49 features -3,136 params, but it is the
    #                only path that hands the net what the forest already models
    "no_attn":      dict(mut_token=False),
    "no_hadamard":  dict(pairs=()),
    "no_scalars":   {"_no_scalars": True},
    "only_scalars": dict(mut_token=False, pairs=()),   # floor: the head sees the 49 scalars only
    "no_seq":       {"_no_seq": True},
    "no_attn_noise": dict(mut_token=False, noise_std=0.2),
    "no_scalars_noise": {"_no_scalars": True, "noise_std": 0.2},
}
OUT = pathlib.Path("reports/probe_earlyfit")


def build(fold: int, seed: int):
    d = splits.load()
    seq = _tokens(SEQ_CACHE, SEQ_BLOCKS, d.row_id)
    st = _tokens(STRUCT_CACHE, STRUCT_BLOCKS, d.row_id)
    cats = categoricals(d)
    y = d.ddG.to_numpy(np.float32)
    folds, clusters, cx = d.fold.to_numpy(), d.cluster.to_numpy(), d["#Pdb"].to_numpy()
    groups = pd.factorize(d["#Pdb"])[0]
    Xs = np.nan_to_num(train.build_matrix(d, ["chem", "geom", "geomrev", "mpnn"]).to_numpy(np.float32))
    chem = np.nan_to_num(train.build_matrix(d, ["chem"]).to_numpy(np.float32))

    held = folds == fold
    tr = ~held
    va, te, _ = _split_held(held, clusters, cx, np.random.default_rng(seed))

    inner = _inner_forest_residuals(Xs, y, np.where(tr)[0], clusters, seed)
    est = MODELS["rf"](seed).fit(Xs[tr], y[tr])
    target = y.copy()
    target[tr] = y[tr] - inner[tr]
    target[va] = y[va] - est.predict(Xs[va])

    s3 = _standardise(seq[tr], seq[va], seq[te])
    g3 = _standardise(st[tr], st[va], st[te])
    mu, sd = chem[tr].mean(0), chem[tr].std(0) + 1e-6
    ch3 = [np.clip((chem[m] - mu) / sd, -CLIP, CLIP) for m in (tr, va, te)]
    mu2, sd2 = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    sc = [np.clip((Xs[m] - mu2) / sd2, -CLIP, CLIP) for m in (tr, va, te)]
    return ((s3[0], g3[0], ch3[0], cats[tr], sc[0], target[tr], groups[tr]),
            (s3[1], g3[1], ch3[1], cats[va], sc[1], target[va], groups[va]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="baseline")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1,
                   help="max_steps cannot exceed one epoch of pair batches "
                        "(~146 for fold 1), so going longer needs more epochs")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print(" ".join(PROBES))
        return
    if a.config not in PROBES:
        raise SystemExit(f"unknown probe {a.config!r}; pick from {' '.join(PROBES)}")

    tr, va = build(a.fold, a.seed)
    over = dict(PROBES[a.config])
    if over.pop("_no_scalars", False):
        # scalars enter the head by concatenation; _fit reads n_scalars off this tensor, so
        # None is what removes the path rather than a flag.
        tr, va = (t[:4] + (None,) + t[5:] for t in (tr, va))
        tr, va = tuple(tr), tuple(va)
    if over.pop("_no_seq", False):
        # zero the sequence tokens rather than deleting the arm: seq_proj still exists, so the
        # comparison isolates the information, not the parameter count.
        tr = (np.zeros_like(tr[0]),) + tr[1:]
        va = (np.zeros_like(va[0]),) + va[1:]
    # max_steps caps the epoch so `steps` is reached inside one pass; patience is disabled so
    # the probe always runs its full length rather than stopping early.
    cfg = dict(BASE, name=f"probe_{a.config}", rank_w=1.0, huber_w=1.0,
               epochs=a.epochs, patience=10 ** 6, max_steps=a.steps,
               eval_every=a.eval_every)
    cfg.update(over)
    _, hist = _fit(tr, va, cfg, a.seed)

    h = pd.DataFrame(hist)
    h["config"] = a.config
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_csv(OUT / f"{a.config}.csv", index=False)
    keep = ["step", "train_loss_eval", "val_loss_eval", "loss_gap", "train_rho", "val_rho"]
    print(f"=== {a.config}")
    print(h[keep].head(12).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
