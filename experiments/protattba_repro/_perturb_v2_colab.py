"""Train the parameter-economy perturbation model on the Colab GPU. Imports only model_v2.

Everything structural is shipped precomputed: the crop indices, chain types, original residue
indices, site flags and the cropped distance submatrix. The VM therefore never parses a PDB,
and ITW and MUT cannot end up with different crops because there is only one.

Only the fold-local PCA happens here, fit on the training fold's tokens and cached, because it
must not see the test complexes.

    python _perturb_v2_colab.py --exp v2_full --folds 0 1 2 3 4 --seeds 0
"""
import argparse, json, os, pickle, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model_v2 import PerturbV2, PerturbV2Config, distance_bins

# The simple model reads only seq_*_wt/mt, struct_* and site_*; collate produces the
# attention model's extra tensors too and it simply ignores them, so one trainer serves
# both and the two are trained on byte-identical batches.
try:
    from model_simple import (MLPConfig, PerturbMLP, PerturbSimple,
                              PerturbSiteToken, PerturbTwoTower, SimpleConfig,
                              SiteTokenConfig, TwoTowerConfig)
except ImportError:          # the VM may not have it yet
    PerturbSimple = SimpleConfig = PerturbMLP = MLPConfig = None
    PerturbTwoTower = TwoTowerConfig = None
    PerturbSiteToken = SiteTokenConfig = None

ARCH = {"v2": (lambda: PerturbV2, lambda: PerturbV2Config),
        "simple": (lambda: PerturbSimple, lambda: SimpleConfig),
        "mlp": (lambda: PerturbMLP, lambda: MLPConfig),
        "twotower": (lambda: PerturbTwoTower, lambda: TwoTowerConfig),
        "sitetok": (lambda: PerturbSiteToken, lambda: SiteTokenConfig)}

# The VM layout by default. PERTURB_ROOT points it at the local caches instead, so the
# model can be probed on real batches without a GPU session -- which is how the gradient
# check runs. PERTURB_TOKENIZER likewise, since locally the tokenizer comes from the hub
# rather than from a directory the bootstrap downloaded.
#: Which model encodes the ANTIBODY side. "esm" keeps the original behaviour; "antiberty"
#: reads the per-residue AntiBERTy cache instead. The antigen side is always ESM -- an
#: antibody model has nothing to say about an antigen, and 385 of 940 rows are mutated
#: there. Set from --seq-ab in main(), read by Cache.seq and fold_pca.
SEQ_AB = "esm"
#: Skip the fold-local PCA entirely and hand the model raw embeddings. Set from --no-pca.
NO_PCA = False

ROOT = Path(os.environ.get("PERTURB_ROOT", "/content/perturb"))
TOKENIZER = os.environ.get("PERTURB_TOKENIZER", str(ROOT / "model" / "esm2_650m"))
OUT = ROOT / "out"
PCA_DIR = ROOT / "pca_v2"
AA = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AA)}
# VAL_FRACTION is a fraction of COMPLEXES, and complexes range from 2 to 87 rows, so
# 0.10 produced validation sets of 73 to 169 rows across the five folds. Early stopping
# ran on that, and validation Pearson barely tracked test Pearson (fold 2: val +0.245 ->
# test +0.461; fold 1: val +0.641 -> test +0.266). 0.20 roughly doubles it.
BATCH, LR, WD, PATIENCE, VAL_FRACTION = 32, 3e-4, 1e-2, 10, 0.20
#: Gradient-norm clip. It sat at 5.0 while the measured norm was 5.0-5.3, so about
#: half of all steps were rescaled and half were not -- the most intermittent
#: setting available, and the one where clipping distorts AdamW's m/sqrt(v) most.
#: It also differed systematically between arms: the heavier regularisation levels
#: inject more noise, clip more often, and were being compared against configs that
#: clipped far less. Set it well above the norm to turn clipping off.
GRAD_CLIP = 5.0


def blosum62():
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    out = np.zeros((20, 20), np.float32)
    for i, a in enumerate(AA):
        for j, b in enumerate(AA):
            out[i, j] = m[a, b]
    return out / 4.0


class Cache:
    def __init__(self):
        self.store = np.load(ROOT / "esm2_650m_tokens.npy", mmap_mode="r")
        self.index = pickle.load(open(ROOT / "esm2_650m_index.pkl", "rb"))["index"]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self.crops = np.load(ROOT / "perturb_crops.npz", allow_pickle=False)
        self._mpnn = {}
        # The project's handcrafted columns, if they were shipped. Standardisation is
        # fold-local and happens in run_fold, not here.
        f = ROOT / "chem_perturb.parquet"
        self.chem = pd.read_parquet(f).set_index("row_id") if f.exists() else None

    def tokens(self, s):
        k = self.tok(s, return_tensors="np")["input_ids"][0].astype(np.int32).tobytes()
        off, n = self.index[k]
        return np.asarray(self.store[off:off + n], np.float32)[1:-1]

    def _load_antiberty(self):
        """Per-residue AntiBERTy, antibody side only. Keyed on sha1 of the sequence.

        Positions outside a variable domain are exact zeros in this cache: AntiBERTy saw
        ~110-130 residue variable domains and our antibody side is the chains concatenated
        at median 429. 1200 of 1202 mutated antibody residues fall inside one, so the
        zeros cost almost nothing -- but they are zeros, not small numbers, so a consumer
        can tell silence from a weak signal.
        """
        import hashlib
        t = ROOT / "antiberty_tokens.npy"
        i = ROOT / "antiberty_index.pkl"
        if not (t.exists() and i.exists()):
            raise SystemExit(f"--seq-ab antiberty needs {t.name} and {i.name} on the VM")
        self._ab_store = np.load(t, mmap_mode="r")
        self._ab_index = pickle.load(open(i, "rb"))["index"]
        self._ab_key = lambda x: hashlib.sha1(x.encode()).hexdigest()
        print(f"  antiberty cache: {len(self._ab_index)} sequences, "
              f"dim {self._ab_store.shape[1]}", flush=True)

    def antiberty(self, s):
        if not hasattr(self, "_ab_store"):
            self._load_antiberty()
        off, n = self._ab_index[self._ab_key(s)]
        return np.asarray(self._ab_store[off:off + n], np.float32)

    def seq(self, side, s):
        """The encoder for this side. ESM everywhere unless the antibody side was switched."""
        if side == "ab" and SEQ_AB == "antiberty":
            return self.antiberty(s)
        return self.tokens(s)

    def mpnn(self, key):
        if key not in self._mpnn:
            z = np.load(ROOT / "mpnn_per_residue" / f"{key}.npz", allow_pickle=False)
            self._mpnn[key] = (z["h_ab"].astype(np.float32), z["h_ag"].astype(np.float32))
        return self._mpnn[key]

    def crop(self, row_id):
        g = lambda f: self.crops[f"{row_id}|{f}"]
        return dict(ab_idx=g("ab_idx"), ag_idx=g("ag_idx"), ab_chain=g("ab_chain"),
                    ag_chain=g("ag_chain"), ab_res=g("ab_res"), ag_res=g("ag_res"),
                    ab_site=g("ab_site"), ag_site=g("ag_site"),
                    dist=g("dist").astype(np.float32),
                    ab_wt_aa=g("ab_wt_aa"), ab_mt_aa=g("ab_mt_aa"),
                    ag_wt_aa=g("ag_wt_aa"), ag_mt_aa=g("ag_mt_aa"))


class Identity:
    """Stands in for a fitted PCA when --no-pca is set.

    The DS calls .transform on whatever fold_pca returns, so the cheapest way to skip the
    projection is to return something that does nothing rather than branch at every call
    site. n_components_ is reported so the per-fold log still prints a width.
    """

    def __init__(self, dim):
        self.n_components_ = dim
        self.explained_variance_ratio_ = np.ones(1)

    def transform(self, x):
        return np.asarray(x, np.float32)


def fold_pca(rows, fold, cache, dim, seed=0):
    """One PCA per side per modality, fit on the training fold's CROPPED tokens only.

    The sides used to share a sequence PCA, which was fine while both were ESM. They cannot
    once the antibody side is AntiBERTy: 512 dimensions against 1280, and two different
    models' output spaces have no common basis to find. Fitting per side is also the more
    honest default even for ESM -- antibody variable domains and antigens are different
    distributions, and one basis over both spends its components on telling them apart.

    The cache file name carries the encoder, so switching encoders cannot silently reuse a
    basis fitted for the other one.
    """
    if NO_PCA:
        # Widths are read off the cache itself rather than assumed: the antibody side is
        # 512 under AntiBERTy and 1280 under ESM, and the model has to be told which.
        r0 = rows.iloc[0]
        w_ab = cache.seq("ab", r0.ab_wt).shape[1]
        w_ag = cache.seq("ag", r0.ag_wt).shape[1]
        w_st = cache.mpnn(r0.complex_key)[0].shape[1]
        print(f"  fold {fold}: NO PCA -- raw widths ab {w_ab}, ag {w_ag}, struct {w_st}",
              flush=True)
        return Identity(w_ab), Identity(w_ag), Identity(w_st)
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    f = PCA_DIR / f"fold{fold}_pca{dim}_{SEQ_AB}.joblib"
    if f.exists():
        return joblib.load(f)
    from sklearn.decomposition import PCA
    tr = rows[rows.fold != fold]
    rng = np.random.default_rng(seed)
    per_side, str_rows = {"ab": [], "ag": []}, []
    for r in tr.itertuples():
        c = cache.crop(r.row_id)
        for side, col in (("ab", r.ab_wt), ("ag", r.ag_wt)):
            t = cache.seq(side, col)
            idx = c[f"{side}_idx"]
            take = idx[idx < len(t)]
            if len(take):
                per_side[side].append(
                    t[take][rng.choice(len(take), min(len(take), 24), replace=False)])
        h_ab, h_ag = cache.mpnn(r.complex_key)
        for h, idx in ((h_ab, c["ab_idx"]), (h_ag, c["ag_idx"])):
            take = idx[idx < len(h)]
            if len(take):
                str_rows.append(h[take][rng.choice(len(take), min(len(take), 24), replace=False)])

    fits = {}
    for side in ("ab", "ag"):
        X = np.concatenate(per_side[side])
        fits[side] = PCA(n_components=min(dim, X.shape[1]), random_state=seed).fit(X)
        print(f"  fold {fold}: PCA seq[{side}] {X.shape} -> {fits[side].n_components_} "
              f"({fits[side].explained_variance_ratio_.sum():.0%})", flush=True)
    Xt = np.concatenate(str_rows)
    p_str = PCA(n_components=min(dim, Xt.shape[1]), random_state=seed).fit(Xt)
    print(f"  fold {fold}: PCA struct {Xt.shape} -> {p_str.n_components_} "
          f"({p_str.explained_variance_ratio_.sum():.0%})", flush=True)
    out = (fits["ab"], fits["ag"], p_str)
    joblib.dump(out, f)
    return out


class DS(Dataset):
    def __init__(self, frame, pcas, cache, cfg, augment, seed, memo=None, chem=None):
        self.f = frame.reset_index(drop=True)
        self.p_ab, self.p_ag, self.p_str = pcas
        self.c, self.cfg = cache, cfg
        self.augment, self.rng = augment, np.random.default_rng(seed)
        self.bl = blosum62()
        self.memo = memo if memo is not None else {}
        self.chem = chem          # already standardised, indexed by row_id

    def seq_at(self, s, idx, side="ab"):
        # memoised per (sequence, side): the same string could in principle appear on both
        # sides, and it must not then be projected with the wrong basis
        key = (s, side)
        if key not in self.memo:
            p = self.p_ab if side == "ab" else self.p_ag
            self.memo[key] = p.transform(self.c.seq(side, s)).astype(np.float32)
        t = self.memo[key]
        take = np.clip(idx, 0, len(t) - 1)
        return t[take]

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        r = self.f.iloc[i]
        c = self.c.crop(r.row_id)
        ab_wt, ag_wt, ab_mt, ag_mt = r.ab_wt, r.ag_wt, r.ab_mt, r.ag_mt
        y, swap = float(r.ddg), False
        if self.augment and self.rng.random() < 0.5:
            ab_wt, ab_mt = ab_mt, ab_wt
            ag_wt, ag_mt = ag_mt, ag_wt
            y, swap = -y, True                # the crop is unchanged, per the spec

        s_ab_wt = self.seq_at(ab_wt, c["ab_idx"], "ab")
        s_ab_mt = self.seq_at(ab_mt, c["ab_idx"], "ab")
        s_ag_wt = self.seq_at(ag_wt, c["ag_idx"], "ag")
        s_ag_mt = self.seq_at(ag_mt, c["ag_idx"], "ag")

        h_ab, h_ag = self.c.mpnn(r.complex_key)
        key = (r.complex_key, "str")
        if key not in self.memo:
            self.memo[key] = (self.p_str.transform(h_ab).astype(np.float32),
                              self.p_str.transform(h_ag).astype(np.float32))
        p_ab, p_ag = self.memo[key]
        t_ab = p_ab[np.clip(c["ab_idx"], 0, len(p_ab) - 1)]
        t_ag = p_ag[np.clip(c["ag_idx"], 0, len(p_ag) - 1)]

        def blos(site, n, wt_letters, mt_letters):
            """Letters come from the crop, already aligned to this side's site order."""
            a = np.zeros((n, 40), np.float32); b = np.zeros((n, 40), np.float32)
            for j, p in enumerate(np.flatnonzero(site)):
                if j >= len(wt_letters):
                    continue
                x, zz = str(wt_letters[j]), str(mt_letters[j])
                if swap:
                    x, zz = zz, x            # reverse-mutation augmentation
                rx = self.bl[AA_INDEX[x]] if x in AA_INDEX else np.zeros(20, np.float32)
                rz = self.bl[AA_INDEX[zz]] if zz in AA_INDEX else np.zeros(20, np.float32)
                a[p] = np.concatenate([rx, rx]); b[p] = np.concatenate([rx, rz])
            return a, b

        b_ab_wt, b_ab_mt = blos(c["ab_site"], len(c["ab_idx"]), c["ab_wt_aa"], c["ab_mt_aa"])
        b_ag_wt, b_ag_mt = blos(c["ag_site"], len(c["ag_idx"]), c["ag_wt_aa"], c["ag_mt_aa"])
        chem = (self.chem.loc[r.row_id].values.astype(np.float32)
                if self.chem is not None else np.zeros(0, np.float32))
        return dict(chem=chem,
                    seq_ab_wt=s_ab_wt, seq_ab_mt=s_ab_mt, seq_ag_wt=s_ag_wt, seq_ag_mt=s_ag_mt,
                    struct_ab=t_ab, struct_ag=t_ag,
                    blosum_ab_wt=b_ab_wt, blosum_ab_mt=b_ab_mt,
                    blosum_ag_wt=b_ag_wt, blosum_ag_mt=b_ag_mt,
                    site_ab=c["ab_site"], site_ag=c["ag_site"],
                    chain_ab=c["ab_chain"].astype(np.int64),
                    chain_ag=c["ag_chain"].astype(np.int64),
                    res_ab=c["ab_res"].astype(np.int64), res_ag=c["ag_res"].astype(np.int64),
                    dist=c["dist"], y=np.float32(y), row_id=r.row_id)


def collate(items):
    L = max(x["seq_ab_wt"].shape[0] for x in items)
    M = max(x["seq_ag_wt"].shape[0] for x in items)
    out = {}
    def st3(k, n):
        a = np.zeros((len(items), n, items[0][k].shape[1]), np.float32)
        for i, x in enumerate(items):
            a[i, :x[k].shape[0]] = x[k]
        return torch.from_numpy(a)
    def st2(k, n, dt=np.float32):
        a = np.zeros((len(items), n), dt)
        for i, x in enumerate(items):
            a[i, :len(x[k])] = x[k]
        return torch.from_numpy(a)
    for side, n in (("ab", L), ("ag", M)):
        for k in (f"seq_{side}_wt", f"seq_{side}_mt", f"struct_{side}",
                  f"blosum_{side}_wt", f"blosum_{side}_mt"):
            out[k] = st3(k, n)
        out[f"site_{side}"] = st2(f"site_{side}", n)
        out[f"chain_{side}"] = st2(f"chain_{side}", n, np.int64)
        out[f"res_{side}"] = st2(f"res_{side}", n, np.int64)
        m = np.zeros((len(items), n), np.float32)
        for i, x in enumerate(items):
            m[i, :x[f"seq_{side}_wt"].shape[0]] = 1
        out[f"mask_{side}"] = torch.from_numpy(m)
    d = np.full((len(items), L, M), 99.0, np.float32)
    for i, x in enumerate(items):
        dd = x["dist"]; d[i, :dd.shape[0], :dd.shape[1]] = dd
    out["dist"] = torch.from_numpy(d)
    out["dist_bins"] = distance_bins(torch.from_numpy(d))
    if len(items[0].get("chem", ())):
        out["chem"] = torch.from_numpy(np.stack([x["chem"] for x in items]))
    out["y"] = torch.tensor([x["y"] for x in items])
    out["row_id"] = [x["row_id"] for x in items]
    return out


def per_complex_rho(pred, true, keys, min_rows=5):
    """Mean Spearman within a complex, which is the quantity the model is judged on.

    Selecting on pooled Pearson over the validation rows rewards telling complexes apart --
    between-complex variance is about 45% of the label variance -- and that is not what the
    model is for. Complexes with fewer than min_rows carry no reliable rank, so they are
    dropped; if nothing survives, fall back to pooled Pearson rather than return nothing.
    """
    df = pd.DataFrame({"p": pred, "t": true, "k": keys})
    rs = []
    for _, g in df.groupby("k"):
        if len(g) >= min_rows and g.t.std() > 0 and g.p.std() > 0:
            r = g.p.corr(g.t, method="spearman")
            if pd.notna(r):
                rs.append(float(r))
    return float(np.mean(rs)) if rs else pear(pred, true)


# Prefix per group, per architecture. The simple model's modules are named differently,
# and grouping it with the attention model's names silently logged zeros for every group
# while the total was fine -- a breakdown that looks like a starved network but is only a
# name mismatch. Each arch gets its own map and anything unmatched lands in "other", so a
# module that is added later cannot vanish from the accounting.
GRAD_GROUPS_V2 = {"head": "head.", "attn": "branch.attn", "fuse": "branch.fuse",
                  "film": "branch.gamma", "film_b": "branch.beta",
                  "red_seq": "branch.red_seq", "red_str": "branch.red_str",
                  "blosum": "branch.mut", "chain": "branch.chain"}
GRAD_GROUPS_SIMPLE = {"head": "mlp.", "film": "gamma", "film_b": "beta",
                      "red_seq": "proj_seq", "red_str": "proj_str"}
GRAD_GROUPS_MLP = {"head": "net."}
GRAD_GROUPS_TT = {"head": "mlp.", "proc": "proc"}
GRAD_GROUPS_ST = {"head": "mlp."}
GRAD_GROUPS = GRAD_GROUPS_V2


def grad_report(model, clip):
    """Gradient health for one step, measured BEFORE clipping.

    A difference-of-branches head can starve without anything in the loss curve showing it:
    if f_mut - f_itw is mostly noise the head still gets a large gradient while the branch
    that has to produce the edit gets very little. Per-group norms make that visible.
    """
    sq, dead, n = 0.0, 0, 0
    per = {k: 0.0 for k in GRAD_GROUPS}
    per["other"] = 0.0
    for name, prm in model.named_parameters():
        n += prm.numel()
        if prm.grad is None:
            dead += prm.numel()
            continue
        g = prm.grad.detach()
        g2 = float(g.pow(2).sum())
        sq += g2
        dead += int((g.abs() < 1e-12).sum())
        for k, pref in GRAD_GROUPS.items():
            if name.startswith(pref):
                per[k] += g2
                break
        else:
            per["other"] += g2
    total = sq ** 0.5
    out = {"grad_norm": total, "grad_clipped": 1.0 if total > clip else 0.0,
           "grad_dead_frac": dead / max(n, 1)}
    for k, v in per.items():
        out["g_" + k] = v ** 0.5
    return out


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def history(exp, row):
    p = OUT / f"perturb_{exp}" / "history.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "step", "steps", "loss", "val_rho", "val_cx_rho", "train_rho",
            "val_rmse", "seconds", "fold", "seed", "n_train", "n_val", "name",
            "host_gb", "gpu_gb", "grad_norm", "grad_clipped", "grad_dead_frac",
            "g_head", "g_attn", "g_fuse", "g_film", "g_film_b", "g_red_seq",
            "g_red_str", "g_blosum", "g_chain", "g_proc", "g_other"]
    if not p.exists():
        p.write_text(",".join(cols) + "\n")
    with open(p, "a") as f:
        f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


def sweep_row(exp, note, params=None):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "rank_fusion_sweep.csv"
    rows = {}
    if f.exists():
        for r in pd.read_csv(f).to_dict("records"):
            rows[r.get("name")] = r
    rows[f"perturb_{exp}"] = {"name": f"perturb_{exp}", "note": note, "resid": False,
                              "min_grp": 5, "dataset": "skempi_abag", "model": "perturb_v2",
                              "d": 64, "drop": 0.2, "wd": 0.01, "params": params or ""}
    pd.DataFrame(list(rows.values())).to_csv(f, index=False)


def run_fold(rows, fold, seed, cfg, cache, exp, device, max_epochs, augment=True,
             select_on="per_complex", make_model=PerturbV2):
    torch.manual_seed(seed); np.random.seed(seed)
    pcas = fold_pca(rows, fold, cache, cfg.pca_dim, seed=0)
    test = rows[rows.fold == fold]
    tr_all = rows[rows.fold != fold]
    # Hold out complexes until the ROW target is met, not a fixed count of complexes.
    # Complexes run from 2 to 87 rows, so taking 20% of them took 311 of 752 rows on the
    # smoke run -- 41%, starving training to 441 rows -- while 10% had given as few as 73.
    # Either way the split size was whatever the shuffle happened to pick.
    cx = sorted(tr_all.complex_key.unique())
    rng = np.random.default_rng(seed); rng.shuffle(cx)
    sizes = tr_all.complex_key.value_counts()
    target, taken, val_cx = VAL_FRACTION * len(tr_all), 0, set()
    for c in cx:
        if taken >= target or len(val_cx) >= len(cx) - 1:
            break
        val_cx.add(c); taken += int(sizes[c])
    train = tr_all[~tr_all.complex_key.isin(val_cx)]
    val = tr_all[tr_all.complex_key.isin(val_cx)]

    memo = {}                                  # one PCA cache shared by all three splits
    chem = None
    if getattr(cfg, "chem_dim", 0) and cache.chem is not None:
        # Mean and sd from the TRAINING folds only. Standardising over everything would let
        # the held-out complexes set the scale the model is trained on -- small, silent, and
        # exactly the kind of leakage that makes a number look better than it is.
        c_all = cache.chem
        mu = c_all.loc[tr_all.row_id].mean()
        sd = c_all.loc[tr_all.row_id].std().replace(0.0, 1.0)
        chem = ((c_all - mu) / sd).astype(np.float32)
    mk = lambda f, a: DataLoader(DS(f, pcas, cache, cfg, a, seed, memo, chem),
                                 batch_size=BATCH, shuffle=a, collate_fn=collate,
                                 num_workers=0)
    tr, va, te = mk(train, augment), mk(val, False), mk(test, False)

    if getattr(cfg, "group_reduce", 0) and not cfg.in_seq:
        # the model cannot know these; they come from the cache the DS reads
        r0 = rows.iloc[0]
        cfg.in_seq = int(cache.seq("ab", r0.ab_wt).shape[1])
        cfg.in_str = int(cache.mpnn(r0.complex_key)[0].shape[1])
        print(f"  group_reduce: seq {cfg.in_seq} -> "
              f"{cfg.in_seq // cfg.group_reduce * cfg.group_out}, "
              f"struct {cfg.in_str} -> {cfg.in_str // cfg.group_reduce * cfg.group_out}",
              flush=True)
    model = make_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)  # WD/LR set from argv
    lf = torch.nn.MSELoss()
    sweep_row(exp, f"fold {fold} seed {seed} running", n_params)

    best, state, bad, t0 = -np.inf, None, 0, time.time()
    for ep in range(max_epochs):
        model.train(); tot, ns = 0.0, 0
        gstat = {}
        for b in tr:
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss = lf(model(x), b["y"].to(device))
            opt.zero_grad(); loss.backward()
            for gk, gv in grad_report(model, GRAD_CLIP).items():
                gstat[gk] = gstat.get(gk, 0.0) + gv
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += float(loss.detach()); ns += 1
        gstat = {gk: gv / max(ns, 1) for gk, gv in gstat.items()}
        model.eval(); p, t, vids = [], [], []
        with torch.no_grad():
            for b in va:
                x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                p.append(model(x).cpu().numpy()); t.append(b["y"].numpy())
                vids += b["row_id"]
        pv, tv = np.concatenate(p), np.concatenate(t)
        r_pool = pear(pv, tv)
        r_cx = per_complex_rho(pv, tv, [i.rsplit("|", 1)[0] for i in vids])
        r = r_cx if select_on == "per_complex" else r_pool
        import resource
        history(exp, dict(epoch=ep, step=(ep + 1) * max(ns, 1), steps=ns,
                          loss=round(tot / max(ns, 1), 5),
                          val_rho="" if np.isnan(r_pool) else round(r_pool, 5),
                          val_cx_rho="" if np.isnan(r_cx) else round(r_cx, 5),
                          train_rho="",
                          **{gk: round(gv, 5) for gk, gv in gstat.items()},
                          val_rmse=round(float(np.sqrt(((pv - tv) ** 2).mean())), 5),
                          seconds=round(time.time() - t0, 1), fold=fold, seed=seed,
                          n_train=len(train), n_val=len(val), name=f"perturb_{exp}",
                          host_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, 2),
                          gpu_gb=round(torch.cuda.max_memory_allocated()/1e9, 2)
                                  if torch.cuda.is_available() else 0.0))
        if ep % 5 == 0:
            print(f"    fold {fold} seed {seed} ep {ep:3d} "
                  f"val r {r_pool:+.4f} cx {r_cx:+.4f} | "
                  f"grad {gstat.get('grad_norm', 0):.3f} "
                  f"head {gstat.get('g_head', 0):.3f} attn {gstat.get('g_attn', 0):.3f} "
                  f"clip {gstat.get('grad_clipped', 0):.0%} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if np.isnan(r):
            r = -np.inf
        if r > best:
            best, bad = r, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state:
        model.load_state_dict(state)
    model.eval(); ids, yp, yt = [], [], []
    with torch.no_grad():
        for b in te:
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            yp.append(model(x).cpu().numpy()); yt.append(b["y"].numpy()); ids += b["row_id"]
    return ids, np.concatenate(yt), np.concatenate(yp), ep + 1, (time.time()-t0)/60, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="v2_full")
    ap.add_argument("--rows", default=str(ROOT / "perturb_rows.parquet"))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--arch", default="v2", choices=["v2", "simple", "mlp", "twotower", "sitetok"],
                    help="which model; both read the same batches")
    ap.add_argument("--select-on", default="per_complex",
                    choices=["per_complex", "pooled"],
                    help="what early stopping maximises on the validation fold")
    ap.add_argument("--seq-ab", default="esm", choices=["esm", "antiberty"],
                    help="which model encodes the antibody side; the antigen is always ESM")
    ap.add_argument("--single-only", action="store_true",
                    help="keep only rows with exactly one mutated residue")
    ap.add_argument("--no-pca", action="store_true",
                    help="skip the fold-local PCA and hand the model raw embeddings; the "
                         "model must then reduce them itself")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="gradient-norm clip; 20 is effectively off here")
    ap.add_argument("--wd", type=float, default=None,
                    help="AdamW weight decay; the ladder has always used 1e-2")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--clip", type=float, default=4.0,
                    help="clip |ddG| to this, in training and in scoring; 0 disables")
    a = ap.parse_args()

    global SEQ_AB, NO_PCA
    SEQ_AB = a.seq_ab
    NO_PCA = a.no_pca
    ov = json.loads(a.overrides)
    augment = ov.pop("_augment", True)
    global GRAD_GROUPS
    GRAD_GROUPS = {"simple": GRAD_GROUPS_SIMPLE, "mlp": GRAD_GROUPS_MLP,
                   "twotower": GRAD_GROUPS_TT,
                   "sitetok": GRAD_GROUPS_ST}.get(a.arch, GRAD_GROUPS_V2)
    model_cls, cfg_cls = (f() for f in ARCH[a.arch])
    if model_cls is None:
        raise SystemExit(f"--arch {a.arch} not available: model_simple.py is missing")
    cfg = cfg_cls(**ov)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = pd.read_parquet(a.rows)
    if a.single_only:
        # k is read from the crop file, which is the same source the model pools from, so
        # "one mutated residue" means the same thing here and in the forward pass.
        cr = np.load(ROOT / "perturb_crops.npz", allow_pickle=False)
        k = np.array([int(cr[f"{r}|ab_site"].sum() + cr[f"{r}|ag_site"].sum())
                      for r in rows.row_id])
        n0 = len(rows)
        rows = rows[k == 1].reset_index(drop=True)
        print(f"single-point only: {len(rows)} of {n0} rows, "
              f"{rows.complex_key.nunique()} complexes", flush=True)
    if a.clip:
        # Clip before anything splits the table, so the training target, the validation
        # signal and the reported metric are the same quantity. Clipping only the training
        # labels would leave the RMSE dominated by the rows the model was told to ignore.
        n = int((rows.ddg.abs() > a.clip).sum())
        rows["ddg"] = rows.ddg.clip(-a.clip, a.clip)
        print(f"clipped |ddG| to {a.clip}: {n} of {len(rows)} rows "
              f"({100 * n / len(rows):.1f}%)", flush=True)
    global LR, WD, PATIENCE, GRAD_CLIP
    if a.wd is not None:
        WD = a.wd
    if a.lr is not None:
        LR = a.lr
    if a.patience is not None:
        PATIENCE = a.patience
    if a.grad_clip is not None:
        GRAD_CLIP = a.grad_clip
    print(f"  optim: lr {LR}, weight_decay {WD}, patience {PATIENCE}, "
          f"grad_clip {GRAD_CLIP}", flush=True)

    cache = Cache()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== {a.exp} === {len(rows)} rows, device {device}", flush=True)

    recs, oof, done = [], [], set()
    rp, op = OUT / f"{a.exp}_results.csv", OUT / f"{a.exp}_oof.csv"
    if rp.exists():
        prev = pd.read_csv(rp); recs = prev.to_dict("records")
        done = {(int(r["fold"]), int(r["seed"])) for r in recs}
        if op.exists():
            oof = [pd.read_csv(op)]
        print(f"resuming: {sorted(done)} already done", flush=True)

    for fold in a.folds:
        for seed in a.seeds:
            if (fold, seed) in done:
                print(f"  fold {fold} seed {seed}: reused", flush=True); continue
            ids, yt, yp, eps, mins, npar = run_fold(rows, fold, seed, cfg, cache, a.exp,
                                                    device, a.max_epochs, augment,
                                                    a.select_on, model_cls)
            recs.append(dict(exp=a.exp, split="frozen5", fold=fold, seed=seed,
                             n_train=int((rows.fold != fold).sum()), n_test=len(ids),
                             pearson=pear(yp, yt),
                             rmse=float(np.sqrt(((yp - yt) ** 2).mean())),
                             mae=float(np.abs(yp - yt).mean()),
                             params=npar, train_minutes=round(mins, 2), epochs=eps))
            oof.append(pd.DataFrame(dict(row_id=ids, ddg_true=yt, ddg_pred=yp,
                                         fold=fold, seed=seed)))
            print(f"  fold {fold} seed {seed}: pearson {recs[-1]['pearson']:+.3f} "
                  f"rmse {recs[-1]['rmse']:.3f} | {eps} ep {mins:.1f} min "
                  f"| {npar:,} params", flush=True)
            pd.DataFrame(recs).to_csv(rp, index=False)
            pd.concat(oof).to_csv(op, index=False)
    sweep_row(a.exp, "done")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
