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


def _objective(pred, Y, I, J, cfg, dev):
    """The training objective, evaluated on a whole split. Used for train-vs-validation loss.

    Same weights and same margin as training, so the two numbers are directly comparable.
    Returns a float; the rank term is skipped when its weight is zero or no pair survives.
    """
    total = 0.0
    if cfg["huber_w"] > 0:
        total += cfg["huber_w"] * float(nn.HuberLoss(delta=2.0)(pred, Y))
    if cfg["rank_w"] > 0 and len(I):
        pi = torch.as_tensor(I, device=dev)
        pj = torch.as_tensor(J, device=dev)
        dy = Y[pi] - Y[pj]
        w = dy.abs() if cfg["weighted"] else torch.ones_like(dy)
        r = (w * nn.functional.softplus(-torch.sign(dy) * (pred[pi] - pred[pj]))).sum() / w.sum()
        total += cfg["rank_w"] * float(r)
    return total


def _augment(t, cfg):
    """Augment an encoder's output in feature space, during training only.

    The loss curves are unambiguous about what limits this model: training Spearman reaches
    1.00 while validation sits near 0.2, so it memorises all 723-757 rows. There is no more
    data to give it -- SKEMPI's antibody subset is 997 rows -- so the alternative is to stop
    the same vector arriving twice. Two augmentations, both applied to the frozen encoders'
    outputs rather than to the structures, since the encoders are not being fine-tuned and
    re-running them per epoch would be absurd:

    * ``noise_std`` -- Gaussian jitter on the standardised representation. Inputs are already
      unit-scaled, so the value reads directly as a fraction of a standard deviation.
    * ``feat_drop`` -- zero whole embedding dimensions, shared across the batch, rescaling the
      survivors. Coarser than element-wise dropout on purpose: it forces the head to spread
      its reliance across dimensions instead of leaning on a few, which is the failure mode
      that memorisation of 750 rows actually takes.
    """
    if t is None:
        return t
    ns, fd = cfg.get("noise_std", 0.0), cfg.get("feat_drop", 0.0)
    if ns > 0:
        t = t + torch.randn_like(t) * ns
    if fd > 0:
        keep = (torch.rand(t.shape[-1], device=t.device) >= fd).to(t.dtype)
        t = t * keep / max(1e-6, 1.0 - fd)
    return t


def _split_held(held, clusters, cx, rng):
    """Halve the held-out fold: one half supervises early stopping, the other is scored.

    Prefers a CLUSTER-disjoint halving so the half used for checkpoint selection contains no
    homologue of the half being scored. That is not always possible: fold 0 of this dataset is
    a *single* cluster (1MLC_AB_E, 274 rows, 14 complexes, 27% of all data), so there is no
    cluster boundary to cut along and the split falls back to complexes. Where that happens
    the two halves are homologous by construction and early stopping is choosing a checkpoint
    against near-copies of the rows it will be scored on, which flatters that fold. ``how``
    records which case applied so the write-up can say so per fold rather than in general.
    """
    fc = pd.unique(clusters[held])
    if len(fc) > 1:
        vc = _half_complexes(held, clusters, fc, cx, rng)
        va = held & np.isin(clusters, list(vc))
        if va.any() and (held & ~va).any():
            return va, held & ~va, "cluster"
    fx = pd.unique(cx[held])
    pick = set(rng.permutation(fx)[:max(1, len(fx) // 2)])
    va = held & np.isin(cx, list(pick))
    if not va.any() or not (held & ~va).any():      # degenerate fold, keep it whole
        return held, held, "degenerate"
    return va, held & ~va, "complex"


def _half_complexes(tr_all, clusters, tc, cx, rng):
    """Pick whole clusters until validation holds about half the held-in COMPLEXES.

    Taking half the *clusters* is not the same thing and was badly behaved: clusters are very
    unequal, so a 50% cluster split put 473 validation rows against 250 training rows in fold 0
    -- validation larger than the training set it was meant to supervise. Accumulating whole
    clusters until the complex count crosses half gives what was actually asked for while
    keeping the cluster as the indivisible unit, so homologues never straddle the boundary.

    At least three clusters go to validation, and at least one is always left for training.
    """
    count = lambda v: len(np.unique(cx[tr_all & np.isin(clusters, list(v))])) if v else 0
    target = len(np.unique(cx[tr_all])) / 2
    vc, order = set(), list(rng.permutation(tc))
    for c in order:
        if len(vc) >= len(tc) - 1 or (len(vc) >= 3 and count(vc) >= target):
            break
        vc.add(c)
        last = c
    # A single cluster can hold many complexes, so the crossing step can overshoot badly --
    # fold 3 once landed on 26 validation complexes against 6 for training. Undo the last
    # addition when doing so lands nearer the target and still leaves three clusters.
    if len(vc) > 3:
        without = vc - {last}
        if abs(count(without) - target) < abs(count(vc) - target):
            vc = without
    return vc


def _row_batches(rng, n_rows, batch):
    """Yield row batches by cycling a fresh permutation, so every row is used equally often.

    The previous version called ``rng.choice(n_rows, batch, replace=False)`` independently at
    every step, so ``replace=False`` only held *within* a batch and rows had no epoch structure
    at all. Measured over one epoch at 40 steps: rows were visited between 0 and 14 times, and
    in fold 3, 23 of 681 rows were never seen. Cycling a permutation bounds the spread to one
    visit between the most- and least-seen row.
    """
    order = rng.permutation(n_rows)
    pos = 0
    while True:
        if pos + batch > len(order):
            order, pos = rng.permutation(n_rows), 0
        yield order[pos:pos + batch]
        pos += batch


def _fit(tr, va, cfg, seed):
    s_tr, g_tr, ch_tr, c_tr, x_tr, y_tr, grp_tr = tr
    s_va, g_va, ch_va, c_va, x_va, y_va, grp_va = va
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
    # Validation pairs exist only so the SAME objective can be evaluated on both sides. Until
    # now history held a training loss next to a validation rmse -- different quantities, so
    # the one question that matters ("is the train/validation gap there from step one?") could
    # not be read off it at all.
    Iv, Jv = build_pairs(y_va, grp_va, cfg["margin"])
    if len(I) == 0:
        cfg = dict(cfg, huber_w=1.0, rank_w=0.0)

    def fwd(idx, train=False):
        s_, g_, ch_ = S[idx], G[idx], CH[idx]
        x_ = None if X is None else X[idx]
        if train:
            s_, g_, ch_, x_ = (_augment(v, cfg) for v in (s_, g_, ch_, x_))
        return net(s_, g_, ch_, C[idx], x_)

    yv, yt = y_va, y_tr
    span = yt.max() - yt.min()
    lo_ok, hi_ok = yt.min() - span, yt.max() + span
    best, state, waited, gstep, stop = -np.inf, None, 0, 0, False
    n_rows = len(y_tr)
    ev_every = cfg.get("eval_every") or 0     # 0 keeps the original once-per-epoch cadence

    def check(ep, n_steps, ep_tot, ep_rank, ep_huber, seen):
        """Validate, checkpoint, and decide whether patience is exhausted.

        Hoisted out of the epoch loop so it can also run mid-epoch. With the step cap gone an
        epoch is 135-205 steps, i.e. each row is seen 12-17 times before validation is looked
        at even once -- so per-epoch early stopping cannot see, let alone act on, anything
        that happens while the model is memorising.
        """
        nonlocal best, state, waited
        net.eval()
        with torch.no_grad():
            fv = net(Sv, Gv, CHv, Cv, Xv)
            ft = net(S, G, CH, C, X)
            pv, pt = fv.cpu().numpy(), ft.cpu().numpy()
            # Both sides in eval mode on clean inputs: no dropout, no augmentation, whole set
            # rather than a batch. The training figure here is therefore NOT the running
            # training loss above -- it is the comparable one.
            tl = _objective(ft, Y, I, J, cfg, dev)
            vl = _objective(fv, Yv, Iv, Jv, cfg, dev)
        net.train()
        v = evaluate._safe_spearman(yv, pv)
        v = -np.inf if not np.isfinite(v) else v
        clipped = cfg["huber_w"] > 0 and not (lo_ok <= pv.min() and pv.max() <= hi_ok)
        if clipped:
            v = -np.inf
        hist.append({"epoch": ep, "step": gstep, "steps": n_steps,
                     "loss": ep_tot / max(seen, 1),
                     "rank_loss": ep_rank / max(seen, 1) if cfg["rank_w"] > 0 else np.nan,
                     "huber_loss": ep_huber / max(seen, 1) if cfg["huber_w"] > 0 else np.nan,
                     "val_rho": np.nan if not np.isfinite(v) else v,
                     "val_rmse": float(np.sqrt(np.mean((pv - yv) ** 2))),
                     "train_loss_eval": tl, "val_loss_eval": vl, "loss_gap": vl - tl,
                     "train_rho": evaluate._safe_spearman(yt, pt),
                     "rejected": bool(clipped)})
        if v > best + 1e-4:
            best, state, waited = v, {k2: t.clone() for k2, t in net.state_dict().items()}, 0
            return False
        waited += 1
        return waited >= cfg["patience"]
    rows_it = _row_batches(rng, n_rows, min(cfg["batch"], n_rows))
    hist = []
    for ep in range(cfg["epochs"]):
        net.train()
        order = rng.permutation(len(I)) if len(I) else np.zeros(0, int)
        # An epoch now means an epoch. ``max_steps`` was previously a hard default of 40, which
        # bound in every fold and silently discarded 3,856-9,906 of the pairs each pass (only
        # 20-40% were ever used). It is now an opt-in cap for a slow machine; left unset, the
        # loop walks every full batch. When there is no ranking term the pair pool is irrelevant
        # and the epoch is sized by rows instead.
        n_full = (len(order) // cfg["batch"] if (cfg["rank_w"] > 0 and len(order))
                  else max(1, n_rows // cfg["batch"]))
        n_steps = max(1, min(cfg["max_steps"] or n_full, n_full))
        ep_rank, ep_huber, ep_tot = 0.0, 0.0, 0.0
        for k in range(n_steps):
            sel = order[k * cfg["batch"]:(k + 1) * cfg["batch"]]
            opt.zero_grad()
            loss = 0.0
            if cfg["rank_w"] > 0 and len(sel):
                pi = torch.as_tensor(I[sel], device=dev)
                pj = torch.as_tensor(J[sel], device=dev)
                fi, fj = fwd(pi, train=True), fwd(pj, train=True)
                dy = Y[pi] - Y[pj]
                w = dy.abs() if cfg["weighted"] else torch.ones_like(dy)
                r = (w * nn.functional.softplus(-torch.sign(dy) * (fi - fj))).sum() / w.sum()
                loss = loss + cfg["rank_w"] * r
                ep_rank += float(r)
            if cfg["huber_w"] > 0:
                ridx = torch.as_tensor(next(rows_it), device=dev)
                h = huber(fwd(ridx, train=True), Y[ridx])
                loss = loss + cfg["huber_w"] * h
                ep_huber += float(h)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_tot += float(loss)
            gstep += 1
            if ev_every and gstep % ev_every == 0:
                if check(ep, n_steps, ep_tot, ep_rank, ep_huber, k + 1):
                    stop = True
                    break

        if not ev_every:
            if check(ep, n_steps, ep_tot, ep_rank, ep_huber, n_steps):
                break
        elif stop:
            break
    if state is not None:
        net.load_state_dict(state)
    return net.eval(), hist


def run(cfg, seeds=(0, 1, 2, 3, 4)):
    d = splits.load()
    seq = _tokens(SEQ_CACHE, SEQ_BLOCKS, d.row_id)
    st = _tokens(STRUCT_CACHE, STRUCT_BLOCKS, d.row_id)
    cats = categoricals(d)
    y = d.ddG.to_numpy(np.float32)
    folds, clusters = d.fold.to_numpy(), d.cluster.to_numpy()
    groups = pd.factorize(d["#Pdb"])[0]
    cx = d["#Pdb"].to_numpy()

    blocks = ["chem", "geom", "geomrev", "mpnn"]
    Xs = np.nan_to_num(train.build_matrix(d, blocks).to_numpy(np.float32))
    chem = np.nan_to_num(train.build_matrix(d, ["chem"]).to_numpy(np.float32))

    oofs, history, splitlog = [], [], []
    for seed in seeds:
        oof = np.full(len(d), np.nan, np.float32)
        for f in np.unique(folds):
            held = folds == f
            tr = ~held                     # the whole of the other folds now trains
            # Validation is half the complexes of the HELD-OUT fold, and the other half is
            # what gets scored. Training therefore keeps all k-1 folds instead of surrendering
            # 20-67% of its rows, which is what the previous inner split cost it.
            rng = np.random.default_rng(seed)
            va, te, how = _split_held(held, clusters, cx, rng)
            splitlog.append({"seed": seed, "fold": int(f), "how": how,
                             "val_rows": int(va.sum()), "test_rows": int(te.sum()),
                             "val_cx": int(len(np.unique(cx[va]))),
                             "test_cx": int(len(np.unique(cx[te]))),
                             "train_rows": int(tr.sum())})

            # residual=False by default, and it should stay that way. Training on
            # y - forest_prediction made every reported number uninterpretable: the net's own
            # validation correlation (~0.23) measured how well it ranked the FOREST's mistakes,
            # while the headline (~0.50) came from adding the forest back at inference -- a net
            # with zero skill still scored ~0.50, because 0 + forest_pred = forest_pred. The
            # forest was carrying the result and the net's real contribution was invisible.
            #
            # It did not even buy an easier target: sd(residual) measured 1.214 against
            # sd(ddG) 1.156 -- 105% of the original spread -- because the forest predicts a
            # narrow band near zero while the labels range far wider.
            target, base_te = y.copy(), None
            if cfg["residual"]:
                inner = _inner_forest_residuals(Xs, y, np.where(tr)[0], clusters, seed)
                est = MODELS["rf"](seed).fit(Xs[tr], y[tr])
                base_te = est.predict(Xs[te])
                target[tr] = y[tr] - inner[tr]
                # Validation now sits outside the training pool, so its forest baseline has to
                # come from the outer forest rather than the inner out-of-fold residuals.
                target[va] = y[va] - est.predict(Xs[va])

            s3, g3 = _standardise(seq[tr], seq[va], seq[te]), _standardise(st[tr], st[va], st[te])
            mu, sd = chem[tr].mean(0), chem[tr].std(0) + 1e-6
            ch3 = [np.clip((chem[m] - mu) / sd, -CLIP, CLIP) for m in (tr, va, te)]
            mu2, sd2 = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
            sc = [np.clip((Xs[m] - mu2) / sd2, -CLIP, CLIP) for m in (tr, va, te)]

            # use_scalars=False removes the bypass that hands the head the same 49 features
            # the forest already fitted. The net is trained on that forest's residual, so the
            # path lets it re-derive what has been subtracted out.
            if not cfg.get("use_scalars", True):
                sc = [None, None, None]
            net, hist = _fit((s3[0], g3[0], ch3[0], cats[tr], sc[0], target[tr], groups[tr]),
                             (s3[1], g3[1], ch3[1], cats[va], sc[1], target[va], groups[va]),
                             cfg, seed)
            for h in hist:
                h.update(seed=seed, fold=int(f), n_train=int(tr.sum()), n_val=int(va.sum()),
                         val_complexes=int(d.loc[va, "#Pdb"].nunique()),
                         train_complexes=int(d.loc[tr, "#Pdb"].nunique()))
            history.extend(hist)
            with torch.no_grad():
                dev = next(net.parameters()).device
                T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
                p = net(T(s3[2]), T(g3[2]), T(ch3[2]),
                        T(cats[te], torch.long),
                        None if sc[2] is None else T(sc[2])).cpu().numpy()
            oof[te] = p + base_te if base_te is not None else p
        oofs.append(oof)

    # Each seed scores only the half of the fold it did not validate on, so a row is predicted
    # by some seeds and not others -- nanmean, not mean, or every row a single seed skipped
    # would poison the whole column. Rows no seed ever scored are dropped and counted, because
    # a silently shrinking denominator is exactly how a metric stops being comparable.
    stack = np.vstack(oofs)
    seen = np.sum(~np.isnan(stack), axis=0)
    pred = np.full(stack.shape[1], np.nan)
    if (seen > 0).any():                      # nanmean warns on all-NaN columns; skip them
        pred[seen > 0] = np.nanmean(stack[:, seen > 0], axis=0)
    ens = pd.DataFrame({"row_id": d.row_id, "complex": d["#Pdb"], "cluster": d.cluster,
                        "fold": d.fold, "y_true": y.astype(float),
                        "y_pred": pred, "n_seeds_scored": seen})
    dropped = int((seen == 0).sum())
    ens = ens[seen > 0].reset_index(drop=True)
    print(f"    scored {len(ens)}/{len(d)} rows "
          f"({dropped} never in a test half), median "
          f"{int(np.median(seen[seen > 0])) if (seen > 0).any() else 0} seeds per scored row",
          flush=True)
    m = evaluate.metrics(ens)
    m["n_scored"] = len(ens)
    m["n_dropped"] = dropped

    # The forest scored on EXACTLY these rows. Each seed scores only the half of a fold it did
    # not validate on, so a net covers ~830 of 997 rows while the forest covers all of them --
    # and an earlier comparison made a net look like a winner (0.506 vs 0.498) purely on that
    # difference. Carrying the matched number in the same row of the table makes that mistake
    # hard to repeat.
    rf_m = {}
    rf_path = paths.REPORTS / "ov_both_geom" / "predictions.csv"
    if rf_path.exists():
        rf = pd.read_csv(rf_path)
        rf = rf[rf.row_id.isin(ens.row_id)]
        if len(rf):
            rf_m = evaluate.metrics(rf)
    out_dir = paths.REPORTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out_dir / "predictions.csv", index=False)
    hdf = pd.DataFrame(history)
    hdf.to_csv(out_dir / "history.csv", index=False)
    # The generalisation gap at the epoch each (seed, fold) actually checkpointed. This is the
    # number the regularisation ladder exists to move -- the headline can stay flat while the
    # gap closes, and that would still be informative, so report both rather than one.
    gap = tr_rho = float("nan")
    if not hdf.empty and hdf["val_rho"].notna().any():
        picks = hdf.dropna(subset=["val_rho"]).loc[
            hdf.dropna(subset=["val_rho"]).groupby(["seed", "fold"])["val_rho"].idxmax()]
        tr_rho, va_rho = picks["train_rho"].mean(), picks["val_rho"].mean()
        gap = tr_rho - va_rho
    plot_history(hdf, cfg["name"], out_dir / "losses.png")
    pd.DataFrame(splitlog).to_csv(out_dir / "splits.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": m, "ci": {}}, indent=2, default=float))
    (out_dir / "run.json").write_text(json.dumps({"name": cfg["name"], "model": "rank_fusion",
                                                  "config": cfg}, indent=2))
    return {"name": cfg["name"], "rank_w": cfg["rank_w"], "huber_w": cfg["huber_w"],
            "margin": cfg["margin"], "weighted": cfg["weighted"],
            "noise": cfg.get("noise_std", 0.0), "fdrop": cfg.get("feat_drop", 0.0),
            "dropout": cfg["dropout"], "wd": cfg["wd"], "d": cfg["d"],
            "scalars": cfg.get("use_scalars", True), "attn": cfg.get("mut_token", True),
            "had": bool(cfg.get("pairs", ())), "residual": cfg.get("residual", False),
            "train_rho": round(tr_rho, 3), "gap": round(gap, 3),
            "n_scored": len(ens),
            "per_cx_rho": round(m["per_complex_spearman"], 3),
            "per_cx_rho10": round(m["per_complex_spearman_min10"], 3),
            "global_rho": round(m["global_spearman"], 3),
            "conc": round(m.get("concordance_micro_m05", float("nan")), 3),
            "rmse": round(m["rmse"], 3),
            # the forest, same rows, same metric -- the only fair comparison
            "rf_per_cx_rho": round(rf_m["per_complex_spearman"], 3) if rf_m else None,
            "rf_per_cx_rho10": round(rf_m["per_complex_spearman_min10"], 3) if rf_m else None,
            "rf_conc": round(rf_m.get("concordance_micro_m05", float("nan")), 3) if rf_m else None,
            "beats_rf": (round(m["per_complex_spearman"] - rf_m["per_complex_spearman"], 3)
                         if rf_m else None)}


def plot_history(hdf, name, out_png):
    """Four panels of what training actually did, per fold, averaged over seeds.

    Curves are ragged: early stopping fires at a different epoch in every (seed, fold), so
    each line is drawn only as far as that fold's shortest-surviving seed, and the epoch each
    fold's checkpoint was taken from is marked. Without the marks the loss curves are easy to
    over-read -- the model returned is the argmax of panel 3, not the end of panel 1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hdf.empty:
        return
    folds = sorted(hdf.fold.unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    (a0, a1), (a2, a3) = ax

    for i, f in enumerate(folds):
        g = hdf[hdf.fold == f].groupby("epoch")
        c = cmap(i)
        m = g.mean(numeric_only=True)
        a0.plot(m.index, m["loss"], color=c, label=f"fold {f}")
        if m["rank_loss"].notna().any():
            a1.plot(m.index, m["rank_loss"], color=c, ls="-", label=f"fold {f} rank")
        if m["huber_loss"].notna().any():
            a1.plot(m.index, m["huber_loss"], color=c, ls=":", label=f"fold {f} huber")
        a2.plot(m.index, m["val_rho"], color=c, label=f"fold {f}")
        a3.plot(m.index, m["train_rho"], color=c, ls="-")
        a3.plot(m.index, m["val_rho"], color=c, ls="--")
        if m["val_rho"].notna().any():
            best_ep = m["val_rho"].idxmax()
            a2.axvline(best_ep, color=c, ls=":", alpha=0.5)
            a2.plot([best_ep], [m["val_rho"].max()], "o", color=c, ms=6)

    a0.set_title("training loss (total)")
    a1.set_title("loss components -- solid rank, dotted Huber")
    a2.set_title("validation Spearman (dot = checkpoint taken)")
    a3.set_title("train (solid) vs validation (dashed) Spearman")
    for a in (a0, a1, a2, a3):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    a0.set_ylabel("loss")
    a1.set_ylabel("loss")
    a2.set_ylabel("rho")
    a3.set_ylabel("rho")
    a0.legend(fontsize=8)
    a2.legend(fontsize=8)

    nv = hdf.groupby("fold")[["n_train", "n_val", "val_complexes", "train_complexes"]].first()
    sub = "  |  ".join(f"fold {f}: train {r.n_train}r/{r.train_complexes}cx, "
                       f"val {r.n_val}r/{r.val_complexes}cx" for f, r in nv.iterrows())
    fig.suptitle(name + "\n" + sub, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


BASE = dict(heads=4, dropout=0.2, lr=3e-4, wd=1e-2, epochs=200, batch=64, patience=20, d=64,
            mut_token=True, residual=False, pairs=("sg",), chem_token=False,
            rank_w=1.0, huber_w=1.0, margin=0.5, weighted=False, max_steps=None,
            noise_std=0.0, feat_drop=0.0, eval_every=0, use_scalars=True, device="auto")


def aug_configs():
    """Regularisation ladder, built so a gain can be attributed to one cause.

    The baseline `rk_both` memorises: training Spearman 1.00 against validation ~0.2. Three
    separate levers could close that, and changing them together would make any improvement
    uninterpretable, so each rung moves one thing:

      aug_none      the same model again, as the control every rung is measured against
      aug_noise     Gaussian jitter only
      aug_drop      feature dropout only
      aug_both      both augmentations, no change to capacity
      reg_dropout   head dropout 0.2 -> 0.5, no augmentation
      reg_wd        weight decay 1e-2 -> 1e-1, no augmentation
      reg_small     width 64 -> 32, no augmentation
      full_light    every lever, gently
      full_heavy    every lever, hard -- the "tough regularisation" end of the range

    Huber is kept alongside the rank term throughout. Dropping it removes the only anchor on
    output scale, which is what produced rk_rank_only's RMSE of 6.179 and, earlier in this
    project, a -6872 kcal/mol prediction.
    """
    C = []
    def add(name, **kw):
        # eval_every=10 is the substantive change, not a knob. With the step cap gone an epoch
        # is 135-205 steps, and a step-level trace shows validation peaking around step 40 and
        # decaying after. Per-epoch early stopping never evaluated before step 146, so it could
        # not checkpoint the peak: it was keeping a memorised model at val rho ~0.25 when ~0.48
        # was on the table. Every rung below is measured with that fixed, or the ladder would
        # only be comparing which regulariser best survives a broken checkpoint policy.
        cfg = dict(BASE, name=name, rank_w=1.0, huber_w=1.0, eval_every=10, patience=25)
        cfg.update(kw)
        C.append(cfg)

    add("aug_none")
    add("aug_noise",    noise_std=0.2)
    add("aug_drop",     feat_drop=0.2)
    add("aug_both",     noise_std=0.2, feat_drop=0.2)
    add("reg_dropout",  dropout=0.5)
    add("reg_wd",       wd=1e-1)
    add("reg_small",    d=32)
    add("full_light",   noise_std=0.15, feat_drop=0.15, dropout=0.35, wd=3e-2, d=48)
    add("full_heavy",   noise_std=0.35, feat_drop=0.3,  dropout=0.5,  wd=1e-1, d=32)
    # The control for the control: per-epoch stopping, so the size of the checkpoint-policy
    # effect is measured here rather than asserted.
    add("epoch_stop",   eval_every=0, patience=20)
    return C


def direct_configs():
    """The network predicting ddG directly, measured against the forest on identical rows.

    Residual learning is gone. It made every number uninterpretable: the net was trained on
    y - forest_prediction, so its validation correlation described how well it ranked the
    FOREST's errors, while the headline came from adding the forest back -- a net with no
    skill at all still scored ~0.50. It did not even simplify the target: sd(residual) was
    1.214 against sd(ddG) 1.156.

    So these configs stand on their own. The comparison that matters is against
    `ov_both_geom` restricted to the same rows, because each seed scores only the half of a
    fold it did not validate on, and an earlier comparison flattered the net purely by scoring
    830 rows against the forest's 997.

    The regularisation settings carry over what the ablation established, which was that the
    combination matters and the width reduction does not (ab_heavy_d64 0.506, ab_heavy_d32
    0.502). Both are kept so that holds without residual learning too.
    """
    HEAVY = dict(noise_std=0.35, feat_drop=0.3, dropout=0.5, wd=1e-1)
    C = []

    def add(name, **kw):
        cfg = dict(BASE, name=name, rank_w=1.0, huber_w=1.0, eval_every=10, patience=25,
                   residual=False)
        cfg.update(kw)
        C.append(cfg)

    add("dir_plain")                                   # no regularisation, the floor
    add("dir_heavy",        **HEAVY)                    # the ablation's winner, no residual
    add("dir_heavy_d32",    d=32, **HEAVY)
    add("dir_huber_only",   rank_w=0.0, **HEAVY)        # regression only, no ranking term
    add("dir_margin0",      margin=0.0, **HEAVY)        # every within-complex pair, unfiltered
    add("dir_noattn",       mut_token=False, **HEAVY)   # does the mutation token still carry it
    add("dir_nohadamard",   pairs=(), **HEAVY)
    add("dir_noscalars",    use_scalars=False, **HEAVY)
    return C


def ablation_configs():
    """Head ablations and regularisation, measured on the SAME footing.

    The ablations were run on a local probe -- one fold, one seed, CPU -- while full_heavy's
    0.502 came from the ladder at four folds and three seeds. Comparing those two numbers is
    not valid, and the combination they suggest (drop the scalars, keep the width, regularise
    hard) has never been run at all. Everything here goes through run() identically.

    HEAVY is full_heavy's settings minus the width change, so `d` can be varied independently
    and the question "is the dimension reduction doing the work, or the regularisation?" has
    an answer rather than a guess.
    """
    HEAVY = dict(noise_std=0.35, feat_drop=0.3, dropout=0.5, wd=1e-1)
    C = []

    def add(name, **kw):
        cfg = dict(BASE, name=name, rank_w=1.0, huber_w=1.0, eval_every=10, patience=25)
        cfg.update(kw)
        C.append(cfg)

    add("ab_control")                                            # d=64, scalars, no reg
    add("ab_heavy_d64",       **HEAVY)                           # reg only, width kept
    add("ab_heavy_d32",       d=32, **HEAVY)                     # = full_heavy, the reference
    add("ab_noscalars",       use_scalars=False)                 # the probe's winner, no reg
    add("ab_noscalars_heavy", use_scalars=False, **HEAVY)        # <- the proposed combination
    add("ab_noscalars_d32",   use_scalars=False, d=32, **HEAVY)  # both, to test sub-additivity
    add("ab_noattn",          mut_token=False)                   # probe said this collapses
    add("ab_nohadamard",      pairs=())
    return C


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
    p.add_argument("--aug-sweep", action="store_true",
                   help="regularisation and representation-augmentation ladder")
    p.add_argument("--ablation-sweep", action="store_true",
                   help="head ablations and regularisation on one footing")
    p.add_argument("--direct-sweep", action="store_true",
                   help="no residual learning: the net predicts ddG itself")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--only", default=None,
                   help="comma list of config names; regenerate a few without repeating the "
                        "whole 3.5-hour sweep")
    p.add_argument("--max-steps", type=int, default=None,
                   help="pair-batches per epoch; raise it on a GPU, where the cap is not needed")
    a = p.parse_args()
    seeds = tuple(range(a.seeds))
    rows = []
    want = set(a.only.split(",")) if a.only else None
    pool = (direct_configs() if a.direct_sweep
            else ablation_configs() if a.ablation_sweep
            else aug_configs() if a.aug_sweep else sweep_configs())
    configs = [c for c in pool if want is None or c["name"] in want]
    if want and not configs:
        raise SystemExit(f"no config matched {sorted(want)}; "
                         f"available: {[c['name'] for c in pool]}")
    for cfg in configs:
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
