"""Train the perturbation model on the Colab GPU. Standalone: imports only model.py.

Everything the VM needs is shipped rather than recomputed there: the resolved row table (sites
already located inside the concatenated sequences, so no PDB parsing), the ProteinMPNN
per-residue cache, the Ca distance cache, and the ESM token cache that was extracted here.

Writes the same artefacts the local runner does -- one row per (exp, fold, seed) and the
out-of-fold predictions -- plus a per-epoch history in the dashboard's schema, so a run is
watchable while it is running.

    python _perturb_colab.py --exp full --folds 0 1 2 3 4 --seeds 0 1 2
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import PerturbConfig, PerturbModel, distance_bins

ROOT = Path("/content/perturb")
ESM_NPY = ROOT / "esm2_650m_tokens.npy"
ESM_IDX = ROOT / "esm2_650m_index.pkl"
ROWS = ROOT / "perturb_rows.parquet"
MPNN = ROOT / "mpnn_per_residue"
DIST = ROOT / "perturb_dist"
OUT = ROOT / "out"
PCA_DIR = ROOT / "pca"

AA = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AA)}
BATCH, LR, WD, PATIENCE, VAL_FRACTION = 32, 3e-4, 1e-2, 10, 0.10


def blosum62():
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    out = np.zeros((20, 20), np.float32)
    for i, a in enumerate(AA):
        for j, b in enumerate(AA):
            out[i, j] = m[a, b]
    return out / 4.0


class Cache:
    """ESM tokens, MPNN fields and distance matrices, each loaded once."""

    def __init__(self):
        self.store = np.load(ESM_NPY, mmap_mode="r")
        self.index = pickle.load(open(ESM_IDX, "rb"))["index"]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(ROOT / "model" / "esm2_650m"))
        self._mpnn, self._dist = {}, {}

    def tokens(self, s):
        key = self.tok(s, return_tensors="np")["input_ids"][0].astype(np.int32).tobytes()
        off, n = self.index[key]
        return np.asarray(self.store[off:off + n], np.float32)[1:-1]   # drop <cls>/<eos>

    def mpnn(self, key):
        if key not in self._mpnn:
            z = np.load(MPNN / f"{key}.npz", allow_pickle=False)
            self._mpnn[key] = (z["h_ab"].astype(np.float32), z["h_ag"].astype(np.float32))
        return self._mpnn[key]

    def dist(self, key):
        if key not in self._dist:
            self._dist[key] = np.load(DIST / f"{key}.npy")
        return self._dist[key]


def fold_pca(rows, fold, cache, n_components, seed=0):
    """PCA fit on the training fold's sequences only, then cached."""
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    out = PCA_DIR / f"fold{fold}_pca{n_components}.joblib"
    if out.exists():
        return joblib.load(out)
    from sklearn.decomposition import PCA
    train = rows[rows.fold != fold]
    seqs = sorted(set(train.ab_wt) | set(train.ag_wt) | set(train.ab_mt) | set(train.ag_mt))
    rng = np.random.default_rng(seed)
    chunks = []
    for s in seqs:
        t = cache.tokens(s)
        chunks.append(t[rng.choice(len(t), min(len(t), 64), replace=False)])
    X = np.concatenate(chunks)
    pca = PCA(n_components=n_components, random_state=seed).fit(X)
    joblib.dump(pca, out)
    print(f"  fold {fold}: PCA on {X.shape[0]} residues from {len(seqs)} training sequences, "
          f"{pca.explained_variance_ratio_.sum():.1%} variance", flush=True)
    return pca


class DS(Dataset):
    """One split's rows. The PCA cache is SHARED across train/val/test.

    It used to be per instance, so the same transformed sequence was held three times over.
    That is ~1 GB of host RAM for nothing, and host RAM is the plausible OOM here, not VRAM.
    """

    def __init__(self, frame, pca, cache, cfg, augment, seed, pc_cache=None):
        self.f = frame.reset_index(drop=True)
        self.pca, self.c, self.cfg = pca, cache, cfg
        self.augment, self.rng = augment, np.random.default_rng(seed)
        self.bl = blosum62()
        self._pc = pc_cache if pc_cache is not None else {}

    def pc(self, s):
        if s not in self._pc:
            self._pc[s] = self.pca.transform(self.c.tokens(s)).astype(np.float32)
        return self._pc[s]

    def __len__(self):
        return len(self.f)

    def rows_blosum(self, L, sites, wt, mt):
        a = np.zeros((L, 40), np.float32); b = np.zeros((L, 40), np.float32)
        for p, x, y in zip(sites, wt, mt):
            if p < L:
                rx = self.bl[AA_INDEX[x]] if x in AA_INDEX else np.zeros(20, np.float32)
                ry = self.bl[AA_INDEX[y]] if y in AA_INDEX else np.zeros(20, np.float32)
                a[p] = np.concatenate([rx, rx]); b[p] = np.concatenate([rx, ry])
        return a, b

    def __getitem__(self, i):
        r = self.f.iloc[i]
        ab_wt, ag_wt, ab_mt, ag_mt = r.ab_wt, r.ag_wt, r.ab_mt, r.ag_mt
        wt, mt, y = list(r.wt_aa), list(r.mt_aa), float(r.ddg)
        if self.augment and self.rng.random() < 0.5:
            ab_wt, ab_mt = ab_mt, ab_wt
            ag_wt, ag_mt = ag_mt, ag_wt
            wt, mt, y = mt, wt, -y

        s_ab_wt, s_ab_mt = self.pc(ab_wt), self.pc(ab_mt)
        s_ag_wt, s_ag_mt = self.pc(ag_wt), self.pc(ag_mt)
        L, M = len(s_ab_wt), len(s_ag_wt)

        h_ab, h_ag = self.c.mpnn(r.complex_key)
        t_ab = np.zeros((L, h_ab.shape[1]), np.float32); t_ab[:min(L, len(h_ab))] = h_ab[:L]
        t_ag = np.zeros((M, h_ag.shape[1]), np.float32); t_ag[:min(M, len(h_ag))] = h_ag[:M]

        sa = [int(x) for x in str(r.sites_ab).split(",") if x != ""]
        sg = [int(x) for x in str(r.sites_ag).split(",") if x != ""]
        site_ab = np.zeros(L, np.float32); site_ag = np.zeros(M, np.float32)
        for p in sa:
            if p < L:
                site_ab[p] = 1
        for p in sg:
            if p < M:
                site_ag[p] = 1

        b_ab_wt, b_ab_mt = self.rows_blosum(L, sa, wt, mt)
        b_ag_wt, b_ag_mt = self.rows_blosum(M, sg, wt, mt)

        d0 = self.c.dist(r.complex_key)
        d = np.full((L, M), 99.0, np.float32)
        d[:min(L, d0.shape[0]), :min(M, d0.shape[1])] = d0[:L, :M]
        near_ab = np.maximum((d.min(1) < 10).astype(np.float32), site_ab)
        near_ag = np.maximum((d.min(0) < 10).astype(np.float32), site_ag)
        return dict(seq_ab_wt=s_ab_wt, seq_ab_mt=s_ab_mt, seq_ag_wt=s_ag_wt, seq_ag_mt=s_ag_mt,
                    struct_ab=t_ab, struct_ag=t_ag, blosum_ab_wt=b_ab_wt, blosum_ab_mt=b_ab_mt,
                    blosum_ag_wt=b_ag_wt, blosum_ag_mt=b_ag_mt, site_ab=site_ab, site_ag=site_ag,
                    near_ab=near_ab, near_ag=near_ag, dist=d, y=np.float32(y), row_id=r.row_id)


def collate(items):
    L = max(x["seq_ab_wt"].shape[0] for x in items)
    M = max(x["seq_ag_wt"].shape[0] for x in items)
    out = {}
    def st(key, n, dim):
        a = np.zeros((len(items), n, dim), np.float32)
        for i, x in enumerate(items):
            a[i, :x[key].shape[0]] = x[key]
        return torch.from_numpy(a)
    for side, n in (("ab", L), ("ag", M)):
        for k in (f"seq_{side}_wt", f"seq_{side}_mt"):
            out[k] = st(k, n, items[0][k].shape[1])
        out[f"struct_{side}"] = st(f"struct_{side}", n, items[0][f"struct_{side}"].shape[1])
        for k in (f"blosum_{side}_wt", f"blosum_{side}_mt"):
            out[k] = st(k, n, 40)
        for k in (f"site_{side}", f"near_{side}"):
            a = np.zeros((len(items), n), np.float32)
            for i, x in enumerate(items):
                a[i, :len(x[k])] = x[k]
            out[k] = torch.from_numpy(a)
        m = np.zeros((len(items), n), np.float32)
        for i, x in enumerate(items):
            m[i, :x[f"seq_{side}_wt"].shape[0]] = 1
        out[f"mask_{side}"] = torch.from_numpy(m)
    d = np.full((len(items), L, M), 99.0, np.float32)
    for i, x in enumerate(items):
        dd = x["dist"]; d[i, :dd.shape[0], :dd.shape[1]] = dd
    out["dist_bins"] = distance_bins(torch.from_numpy(d))
    out["y"] = torch.tensor([x["y"] for x in items])
    out["row_id"] = [x["row_id"] for x in items]
    return out


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def history(exp, row):
    p = OUT / f"perturb_{exp}" / "history.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "step", "steps", "loss", "val_rho", "train_rho", "val_rmse",
            "seconds", "fold", "seed", "n_train", "n_val", "name", "host_gb", "gpu_gb"]
    if not p.exists():
        p.write_text(",".join(cols) + "\n")
    with open(p, "a") as f:
        f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


def sweep_row(exp, note):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "rank_fusion_sweep.csv"
    rows = {}
    if f.exists():
        for r in pd.read_csv(f).to_dict("records"):
            rows[r.get("name")] = r
    rows[f"perturb_{exp}"] = {"name": f"perturb_{exp}", "note": note, "resid": False,
                              "min_grp": 5, "dataset": "skempi_abag", "model": "perturb"}
    pd.DataFrame(list(rows.values())).to_csv(f, index=False)


def run_fold(rows, fold, seed, cfg, cache, exp, device, max_epochs, augment=True):
    torch.manual_seed(seed); np.random.seed(seed)
    pca = fold_pca(rows, fold, cache, cfg.seq_dim, seed=0)
    test = rows[rows.fold == fold]
    tr_all = rows[rows.fold != fold]
    cx = sorted(tr_all.complex_key.unique())
    rng = np.random.default_rng(seed); rng.shuffle(cx)
    val_cx = set(cx[:max(1, int(round(len(cx) * VAL_FRACTION)))])
    train = tr_all[~tr_all.complex_key.isin(val_cx)]
    val = tr_all[tr_all.complex_key.isin(val_cx)]

    pc_cache = {}          # one PCA cache for all three splits of this fold
    mk = lambda f, a: DataLoader(DS(f, pca, cache, cfg, a, seed, pc_cache), batch_size=BATCH,
                                 shuffle=a, collate_fn=collate, num_workers=0)
    tr, va, te = mk(train, augment), mk(val, False), mk(test, False)
    model = PerturbModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lf = torch.nn.MSELoss()
    sweep_row(exp, f"fold {fold} seed {seed} running")

    best, state, bad, t0 = -np.inf, None, 0, time.time()
    for ep in range(max_epochs):
        model.train(); tot, ns = 0.0, 0
        for b in tr:
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss = lf(model(x), b["y"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss.detach()); ns += 1
        model.eval(); p, t = [], []
        with torch.no_grad():
            for b in va:
                x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                p.append(model(x).cpu().numpy()); t.append(b["y"].numpy())
        pv, tv = np.concatenate(p), np.concatenate(t)
        r = pear(pv, tv)
        import resource
        host_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        if ep % 5 == 0:
            print(f"      mem: host {host_gb:.1f} GB peak, gpu {gpu_gb:.1f} GB peak", flush=True)
        history(exp, dict(epoch=ep, step=(ep + 1) * max(ns, 1), steps=ns,
                          loss=round(tot / max(ns, 1), 5),
                          val_rho="" if np.isnan(r) else round(r, 5), train_rho="",
                          val_rmse=round(float(np.sqrt(((pv - tv) ** 2).mean())), 5),
                          seconds=round(time.time() - t0, 1), fold=fold, seed=seed,
                          n_train=len(train), n_val=len(val), name=f"perturb_{exp}",
                          host_gb=round(host_gb, 2), gpu_gb=round(gpu_gb, 2)))
        if ep % 5 == 0:
            print(f"    fold {fold} seed {seed} epoch {ep:3d} val r {r:+.4f} "
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
    return ids, np.concatenate(yt), np.concatenate(yp), ep + 1, (time.time() - t0) / 60, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="full")
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--clip", type=float, default=4.0)
    a = ap.parse_args()

    cfg = PerturbConfig(**json.loads(a.overrides))
    augment = json.loads(a.overrides).get("_augment", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = pd.read_parquet(a.rows)
    if a.clip:
        rows["ddg"] = rows.ddg.clip(-a.clip, a.clip)
    cache = Cache()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== {a.exp} === {len(rows)} rows, device {device}, cfg {cfg}", flush=True)

    # Resume: a Colab runtime is reclaimed as soon as the local keep-alive dies, which has
    # happened four times. Completed (fold, seed) pairs are restored from whatever was pulled
    # down before the last death, so a reclaim costs the fold in flight rather than the run.
    recs, oof = [], []
    done = set()
    rp, op = OUT / f"{a.exp}_results.csv", OUT / f"{a.exp}_oof.csv"
    if rp.exists():
        prev = pd.read_csv(rp)
        recs = prev.to_dict("records")
        done = {(int(r["fold"]), int(r["seed"])) for r in recs}
        if op.exists():
            oof = [pd.read_csv(op)]
        print(f"resuming: {sorted(done)} already done", flush=True)

    for fold in a.folds:
        for seed in a.seeds:
            if (fold, seed) in done:
                print(f"  fold {fold} seed {seed}: reused from disk", flush=True)
                continue
            ids, yt, yp, eps, mins, n_params = run_fold(
                rows, fold, seed, cfg, cache, a.exp, device, a.max_epochs, augment)
            recs.append(dict(exp=a.exp, split="frozen5", fold=fold, seed=seed,
                             n_train=int((rows.fold != fold).sum()), n_test=len(ids),
                             pearson=pear(yp, yt),
                             rmse=float(np.sqrt(((yp - yt) ** 2).mean())),
                             mae=float(np.abs(yp - yt).mean()),
                             params=n_params, train_minutes=round(mins, 2), epochs=eps))
            oof.append(pd.DataFrame(dict(row_id=ids, ddg_true=yt, ddg_pred=yp,
                                         fold=fold, seed=seed)))
            print(f"  fold {fold} seed {seed}: pearson {recs[-1]['pearson']:+.3f} "
                  f"rmse {recs[-1]['rmse']:.3f} | {eps} epochs {mins:.1f} min", flush=True)
            pd.DataFrame(recs).to_csv(OUT / f"{a.exp}_results.csv", index=False)
            pd.concat(oof).to_csv(OUT / f"{a.exp}_oof.csv", index=False)
    sweep_row(a.exp, "done")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
