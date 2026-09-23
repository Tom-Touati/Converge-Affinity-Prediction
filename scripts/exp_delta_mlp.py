"""The shortest path from ESM to ddG: pool the delta at the mutated residues, fit an MLP.

No attention, no structure, no FiLM, no BLOSUM, no crop geometry -- one vector per row and
a regressor on top. It exists to answer what the whole perturbation architecture is worth
against the simplest thing that uses the same information, and it runs on a CPU in under a
minute because the feature matrix is 940 x 512.

    per side:  mean over the MUTATED residues of  PCA(esm_mutant) - PCA(esm_wild_type)
    concat the antibody and antigen sides   ->  512 features
    Ridge / MLP  ->  ddG

The PCA is fit on the TRAINING folds only, once per fold, on cropped tokens. Fitting it on
everything would let the test complexes shape the basis the model sees, which is the
quietest kind of leakage: nothing errors and every number comes out better.

Ridge is reported alongside the MLP on purpose. If the two agree, whatever the MLP is doing
is linear in these features and its extra capacity is decoration.

    python scripts/exp_delta_mlp.py [--pca 256] [--clip 4]
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

ESM_DIR = paths.FEATURES / "esm_per_residue"
CROPS = paths.FEATURES / "perturb_crops.npz"
ROWS = paths.ROOT / "experiments" / "protattba_repro" / "cache" / "perturb_rows.parquet"


def token_reader():
    store = np.load(ESM_DIR / "esm2_650m_tokens.npy", mmap_mode="r")
    index = pickle.load(open(ESM_DIR / "esm2_650m_index.pkl", "rb"))["index"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    def read(seq: str) -> np.ndarray:
        key = tok(seq, return_tensors="np")["input_ids"][0].astype(np.int32).tobytes()
        off, n = index[key]
        return np.asarray(store[off:off + n], np.float32)[1:-1]   # drop CLS/EOS
    return read


def raw_deltas(rows: pd.DataFrame, crops, read) -> dict:
    """Per row and side, the 1280-d ESM difference AT the mutated residues.

    Kept at full width so the PCA can be refit per fold without re-reading the cache; the
    projection to `pca` dimensions happens inside the fold loop.
    """
    out = {}
    for r in rows.itertuples():
        per_side = {}
        for side, wt_col, mt_col in (("ab", "ab_wt", "ab_mt"), ("ag", "ag_wt", "ag_mt")):
            idx = crops[f"{r.row_id}|{side}_idx"]
            site = crops[f"{r.row_id}|{side}_site"].astype(bool)
            try:
                wt, mt = read(getattr(r, wt_col)), read(getattr(r, mt_col))
            except KeyError:
                per_side[side] = None
                continue
            keep = idx < min(len(wt), len(mt))
            take, s = idx[keep], site[: len(idx)][keep]
            if not len(take) or not s.any():
                per_side[side] = None            # this side carries no mutation
                continue
            per_side[side] = (mt[take][s] - wt[take][s]).mean(axis=0)
        out[r.row_id] = per_side
    return out


def features(ids, deltas, pca, dim):
    """(n, 2 * dim). A side with no mutation on it contributes zeros, not noise."""
    X = np.zeros((len(ids), 2 * dim), np.float32)
    for i, rid in enumerate(ids):
        for j, side in enumerate(("ab", "ag")):
            v = deltas[rid][side]
            if v is not None:
                X[i, j * dim:(j + 1) * dim] = pca.transform(v[None, :])[0]
    return X


def per_complex(d: pd.DataFrame) -> float:
    v = [float(np.corrcoef(g.y_pred, g.y_true)[0, 1]) for _, g in d.groupby("complex")
         if len(g) >= 5 and g.y_true.std() > 0 and g.y_pred.std() > 0]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--clip", type=float, default=4.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    rows = pd.read_parquet(ROWS)
    rows["ddg"] = rows.ddg.clip(-a.clip, a.clip)
    rows["complex"] = rows.row_id.str.rsplit("|", n=1).str[0]
    crops = np.load(CROPS, allow_pickle=False)
    print(f"{len(rows)} rows, {rows['complex'].nunique()} complexes, "
          f"clip +-{a.clip}, PCA {a.pca}\n")
    print("reading ESM deltas at the mutated residues ...", flush=True)
    deltas = raw_deltas(rows, crops, token_reader())

    preds = []
    for fold in sorted(rows.fold.unique()):
        tr = rows[rows.fold != fold]
        te = rows[rows.fold == fold]
        # PCA on TRAINING rows' deltas only
        pool = np.stack([v for rid in tr.row_id for v in
                         (deltas[rid]["ab"], deltas[rid]["ag"]) if v is not None])
        pca = PCA(n_components=min(a.pca, pool.shape[1], len(pool)),
                  random_state=0).fit(pool)
        Xtr = features(tr.row_id.tolist(), deltas, pca, a.pca)
        Xte = features(te.row_id.tolist(), deltas, pca, a.pca)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        ytr = tr.ddg.values

        ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr, ytr)
        p_ridge = ridge.predict(Xte)
        # several seeds averaged: a single MLP on 752 rows is mostly initialisation
        p_mlp = np.mean([
            MLPRegressor(hidden_layer_sizes=(a.hidden,), alpha=1.0, max_iter=2000,
                         early_stopping=True, n_iter_no_change=25, random_state=s)
            .fit(Xtr, ytr).predict(Xte)
            for s in range(a.seeds)], axis=0)
        preds.append(pd.DataFrame(dict(row_id=te.row_id.values, complex=te["complex"].values,
                                       fold=fold, y_true=te.ddg.values,
                                       ridge=p_ridge, mlp=p_mlp)))
        print(f"  fold {fold}: n_train {len(tr)} n_test {len(te)}  "
              f"PCA explains {pca.explained_variance_ratio_.sum():.0%}", flush=True)

    d = pd.concat(preds, ignore_index=True)
    print(f"\n{'model':<28} {'per-cx r':>9} {'pooled':>8} {'rmse':>7} {'bias':>7}")
    for col, label in (("ridge", f"Ridge on delta PCA-{a.pca}"),
                       ("mlp", f"MLP({a.hidden}) on delta PCA-{a.pca}")):
        x = d.rename(columns={col: "y_pred"})
        e = x.y_pred - x.y_true
        print(f"{label:<28} {per_complex(x):>+9.3f} "
              f"{np.corrcoef(x.y_pred, x.y_true)[0,1]:>+8.3f} "
              f"{np.sqrt((e**2).mean()):>7.3f} {e.mean():>+7.3f}")
    cm = d.groupby("complex").y_true.transform("mean")
    e = cm - d.y_true
    print(f"{'[predict complex mean]':<28} {0.0:>+9.3f} "
          f"{np.corrcoef(cm, d.y_true)[0,1]:>+8.3f} {np.sqrt((e**2).mean()):>7.3f}")

    out = paths.ROOT / "results" / "oof" / f"delta_mlp_pca{a.pca}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.rename(columns={"mlp": "ddg_pred", "y_true": "ddg_true"}).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
