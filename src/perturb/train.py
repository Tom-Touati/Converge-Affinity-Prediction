"""Train and evaluate the perturbation model on the frozen 5-fold split.

The fixed recipe from section 1, applied identically to every experiment: AdamW at 3e-4,
weight decay 1e-2, dropout 0.2, batch 32, at most 60 epochs, early stopping on validation
Pearson, seeds 0/1/2, reverse-mutation augmentation on. No hyperparameter search, so an
ablation's effect is the ablation and not a tuning difference.

Three details that decide whether the numbers mean anything:

* **The validation split is carved from the training fold by complex**, 10% of training
  complexes, not 10% of rows. Splitting by row would put a complex on both sides and early
  stopping would select on data it had trained on.
* **Reverse-mutation augmentation** swaps the wild-type and mutant sequences, swaps the BLOSUM
  pair, and negates the label. It is applied to training batches only; validation and test are
  always the forward direction, so the metric stays comparable to every other run in the repo.
* **PCA comes from the fold**, fit on training sequences only, so the reduction cannot see the
  test complexes.

Run: ``python -m src.perturb.run --config configs/<exp>.yaml --fold 0 --seed 0``
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import paths
from src.perturb import data as D
from src.perturb.model import PerturbConfig, PerturbModel, distance_bins

#: The project's dashboard polls <HISTORY_ROOT>/rank_fusion_sweep.csv and then
#: <HISTORY_ROOT>/<name>/history.csv, so writing into that layout makes a perturbation run
#: visible live with no change to the dashboard itself.
HISTORY_ROOT = paths.REPORTS
SWEEP_CSV = HISTORY_ROOT / "rank_fusion_sweep.csv"

MAX_EPOCHS = 60
BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
VAL_FRACTION = 0.10


class PerturbDataset(Dataset):
    """Rows resolved to arrays. Heavy lookups are memoised per sequence, not per row."""

    def __init__(self, rows, pca, tokenizer, store, index, cfg: PerturbConfig,
                 augment: bool = False, seed: int = 0):
        self.rows, self.pca, self.tok = rows, pca, tokenizer
        self.store, self.index, self.cfg = store, index, cfg
        self.augment = augment
        self.blosum = D.blosum62()
        self.rng = np.random.default_rng(seed)
        self._seq_cache: dict[str, np.ndarray] = {}
        self._dist_cache: dict[str, np.ndarray] = {}

    def seq_tokens(self, s: str) -> np.ndarray:
        if s not in self._seq_cache:
            from src.perturb.esm_tokens import tokens_for
            t = tokens_for(s, self.tok, self.store, self.index)[1:-1]   # drop <cls>/<eos>
            self._seq_cache[s] = self.pca.transform(t).astype(np.float32)
        return self._seq_cache[s]

    def distances(self, key: str) -> np.ndarray:
        if key not in self._dist_cache:
            self._dist_cache[key] = np.load(D.DIST_DIR / f"{key}.npy")
        return self._dist_cache[key]

    def blosum_rows(self, length: int, sites, wt, mt) -> tuple[np.ndarray, np.ndarray]:
        """(wild-type pair, mutant pair) per position; the ITW pair is (a, a)."""
        b_wt = np.zeros((length, 40), np.float32)
        b_mt = np.zeros((length, 40), np.float32)
        for pos, a, b in zip(sites, wt, mt):
            if pos >= length:
                continue
            ra = self.blosum[D.AA_INDEX[a]] if a in D.AA_INDEX else np.zeros(20, np.float32)
            rb = self.blosum[D.AA_INDEX[b]] if b in D.AA_INDEX else np.zeros(20, np.float32)
            b_wt[pos] = np.concatenate([ra, ra])
            b_mt[pos] = np.concatenate([ra, rb])
        return b_wt, b_mt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ab_wt, ag_wt, ab_mt, ag_mt = r.ab_wt, r.ag_wt, r.ab_mt, r.ag_mt
        wt_aa, mt_aa, y = r.wt_aa, r.mt_aa, r.ddg

        # reverse mutation: swap the branches, swap the BLOSUM pair, negate the label
        if self.augment and self.rng.random() < 0.5:
            ab_wt, ab_mt = ab_mt, ab_wt
            ag_wt, ag_mt = ag_mt, ag_wt
            wt_aa, mt_aa = mt_aa, wt_aa
            y = -y

        s_ab_wt, s_ab_mt = self.seq_tokens(ab_wt), self.seq_tokens(ab_mt)
        s_ag_wt, s_ag_mt = self.seq_tokens(ag_wt), self.seq_tokens(ag_mt)
        l_ab, l_ag = len(s_ab_wt), len(s_ag_wt)

        from src.fusion.mpnn_per_residue import load as load_mpnn
        e = load_mpnn(r.complex_key)
        t_ab = e["h_ab"][:l_ab]
        t_ag = e["h_ag"][:l_ag]
        t_ab = np.pad(t_ab, ((0, max(0, l_ab - len(t_ab))), (0, 0)))[:l_ab]
        t_ag = np.pad(t_ag, ((0, max(0, l_ag - len(t_ag))), (0, 0)))[:l_ag]

        site_ab = np.zeros(l_ab, np.float32); site_ag = np.zeros(l_ag, np.float32)
        for p in r.sites_ab:
            if p < l_ab:
                site_ab[p] = 1.0
        for p in r.sites_ag:
            if p < l_ag:
                site_ag[p] = 1.0

        b_ab_wt, b_ab_mt = self.blosum_rows(l_ab, r.sites_ab, wt_aa, mt_aa)
        b_ag_wt, b_ag_mt = self.blosum_rows(l_ag, r.sites_ag, wt_aa, mt_aa)

        d = self.distances(r.complex_key)
        d = d[:l_ab, :l_ag]
        if d.shape != (l_ab, l_ag):
            pad = np.full((l_ab, l_ag), 99.0, np.float32)
            pad[:d.shape[0], :d.shape[1]] = d
            d = pad
        near_ab = (d.min(axis=1) < 10.0).astype(np.float32) if d.size else np.zeros(l_ab, np.float32)
        near_ag = (d.min(axis=0) < 10.0).astype(np.float32) if d.size else np.zeros(l_ag, np.float32)
        # the mutated site is always poolable, even if it is far from the interface
        near_ab = np.maximum(near_ab, site_ab)
        near_ag = np.maximum(near_ag, site_ag)

        return {
            "seq_ab_wt": s_ab_wt, "seq_ab_mt": s_ab_mt,
            "seq_ag_wt": s_ag_wt, "seq_ag_mt": s_ag_mt,
            "struct_ab": t_ab, "struct_ag": t_ag,
            "blosum_ab_wt": b_ab_wt, "blosum_ab_mt": b_ab_mt,
            "blosum_ag_wt": b_ag_wt, "blosum_ag_mt": b_ag_mt,
            "site_ab": site_ab, "site_ag": site_ag,
            "near_ab": near_ab, "near_ag": near_ag,
            "dist": d, "y": np.float32(y), "row_id": r.row_id,
        }


def collate(items):
    """Pad to the longest member and carry the mask explicitly."""
    l_ab = max(x["seq_ab_wt"].shape[0] for x in items)
    l_ag = max(x["seq_ag_wt"].shape[0] for x in items)
    out: dict[str, torch.Tensor] = {}

    def stack(key, length, dim):
        a = np.zeros((len(items), length, dim), np.float32)
        for i, x in enumerate(items):
            v = x[key]; a[i, : v.shape[0]] = v
        return torch.from_numpy(a)

    for side, length in (("ab", l_ab), ("ag", l_ag)):
        for k in ("seq_{}_wt", "seq_{}_mt"):
            out[k.format(side)] = stack(k.format(side), length, items[0][k.format(side)].shape[1])
        out[f"struct_{side}"] = stack(f"struct_{side}", length, items[0][f"struct_{side}"].shape[1])
        for k in ("blosum_{}_wt", "blosum_{}_mt"):
            out[k.format(side)] = stack(k.format(side), length, 40)
        for k in ("site_{}", "near_{}"):
            a = np.zeros((len(items), length), np.float32)
            for i, x in enumerate(items):
                v = x[k.format(side)]; a[i, : len(v)] = v
            out[k.format(side)] = torch.from_numpy(a)
        m = np.zeros((len(items), length), np.float32)
        for i, x in enumerate(items):
            m[i, : x[f"seq_{side}_wt"].shape[0]] = 1.0
        out[f"mask_{side}"] = torch.from_numpy(m)

    d = np.full((len(items), l_ab, l_ag), 99.0, np.float32)
    for i, x in enumerate(items):
        dd = x["dist"]; d[i, : dd.shape[0], : dd.shape[1]] = dd
    out["dist_bins"] = distance_bins(torch.from_numpy(d))
    out["y"] = torch.tensor([x["y"] for x in items])
    out["row_id"] = [x["row_id"] for x in items]
    return out


def register_run(name: str, note: str = "running") -> None:
    """One row per run in the file the dashboard reads first, written before the first epoch.

    The dashboard only fetches history for names that this file already lists, so registering
    after training would leave the run invisible for exactly the stretch worth watching.
    """
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = {}
    if SWEEP_CSV.exists():
        try:
            for r in pd.read_csv(SWEEP_CSV).to_dict("records"):
                rows[r.get("name")] = r
        except Exception:
            rows = {}
    rows[name] = {"name": name, "note": note, "resid": False, "min_grp": 5,
                  "dataset": "skempi_abag", "model": "perturb"}
    pd.DataFrame(list(rows.values())).to_csv(SWEEP_CSV, index=False)


def append_history(name: str, row: dict) -> None:
    """One line per epoch, in the dashboard's history.csv schema."""
    path = HISTORY_ROOT / name / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "step", "steps", "loss", "val_rho", "train_rho", "val_rmse",
            "seconds", "fold", "seed", "n_train", "n_val", "name"]
    if not path.exists():
        path.write_text(",".join(cols) + chr(10))
    with open(path, "a") as f:
        f.write(",".join(str(row.get(c, "")) for c in cols) + chr(10))


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def train_fold(rows, fold: int, cfg: PerturbConfig, seed: int, device: str,
               max_epochs: int = MAX_EPOCHS, verbose: bool = True,
               history_name: str | None = None):
    """One (fold, seed). Returns (row_ids, y_true, y_pred, epochs, minutes, n_params)."""
    from transformers import AutoTokenizer

    from src.perturb.esm_tokens import ESM_DIR, load

    torch.manual_seed(seed); np.random.seed(seed)
    store, index, _ = load()
    tok = AutoTokenizer.from_pretrained(str(ESM_DIR))
    pca = D.fit_fold_pca(rows, fold, n_components=cfg.seq_dim)

    test = [r for r in rows if r.fold == fold]
    train_all = [r for r in rows if r.fold != fold]

    # validation carved by COMPLEX, not by row
    cx = sorted({r.complex_key for r in train_all})
    rng = np.random.default_rng(seed)
    rng.shuffle(cx)
    n_val = max(1, int(round(len(cx) * VAL_FRACTION)))
    val_cx = set(cx[:n_val])
    train = [r for r in train_all if r.complex_key not in val_cx]
    val = [r for r in train_all if r.complex_key in val_cx]

    mk = lambda rs, aug: PerturbDataset(rs, pca, tok, store, index, cfg, augment=aug, seed=seed)
    dl = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh, collate_fn=collate,
                                   num_workers=0)
    tr_dl, va_dl, te_dl = dl(mk(train, True), True), dl(mk(val, False), False), dl(mk(test, False), False)

    model = PerturbModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    run_name = f"perturb_{history_name}" if history_name else None
    if run_name:
        register_run(run_name)

    best, best_state, bad, t0 = -np.inf, None, 0, time.time()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        for b in tr_dl:
            y = b["y"].to(device)
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach()); n_steps += 1

        model.eval(); p, t = [], []
        with torch.no_grad():
            for b in va_dl:
                x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                p.append(model(x).cpu().numpy()); t.append(b["y"].numpy())
        pv, tv = np.concatenate(p), np.concatenate(t)
        r = pearson(pv, tv)
        if run_name:
            append_history(run_name, {
                "epoch": epoch, "step": (epoch + 1) * max(n_steps, 1), "steps": n_steps,
                "loss": round(epoch_loss / max(n_steps, 1), 5),
                "val_rho": round(r, 5) if not np.isnan(r) else "",
                "train_rho": "",
                "val_rmse": round(float(np.sqrt(np.mean((pv - tv) ** 2))), 5),
                "seconds": round(time.time() - t0, 1), "fold": fold, "seed": seed,
                "n_train": len(train), "n_val": len(val), "name": run_name})
        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:3d}  val pearson {r:+.4f}", flush=True)
        if np.isnan(r):
            r = -np.inf
        if r > best:
            best, bad = r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); ids, yp, yt = [], [], []
    with torch.no_grad():
        for b in te_dl:
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            yp.append(model(x).cpu().numpy()); yt.append(b["y"].numpy()); ids += b["row_id"]
    return (ids, np.concatenate(yt), np.concatenate(yp), epoch + 1,
            (time.time() - t0) / 60, n_params, model)
