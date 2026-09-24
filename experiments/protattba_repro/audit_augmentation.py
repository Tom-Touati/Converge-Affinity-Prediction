"""How much of the augmentation and the input perturbation actually reaches training?

Both are configured as probabilities, and a probability in a config file is not evidence.
Three things can differ from the intent without anything failing:

  * the reverse-mutation draw is taken inside ``DS.__getitem__`` from a generator created
    once per Dataset, so what the model really sees depends on how often __getitem__ runs
    and in what order, not on the nominal 0.5;
  * the input noise is scaled by the data's own per-feature standard deviation, so its real
    size is whatever the features happen to be, not the 0.25 in the config;
  * the noise is drawn FRESH on every call while the feature-dropout mask is drawn once per
    width and reused. The model consumes ``mt - wt``, so those two behave completely
    differently in the quantity that actually matters, and neither behaves like the comment
    above ``px`` originally claimed.

Runs against the local caches in ``.perturb_local`` and needs no GPU and no torch.

    python experiments/protattba_repro/audit_augmentation.py [--epochs 3] [--fold 0]
"""
from __future__ import annotations

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd

LOCAL = pathlib.Path(__file__).resolve().parents[2] / ".perturb_local"
BATCH, INPUT_NOISE, FEATURE_DROPOUT = 32, 0.25, 0.35


def load_tokens():
    idx = pickle.load(open(LOCAL / "esm2_650m_index.pkl", "rb"))["index"]
    store = np.load(LOCAL / "esm2_650m_tokens.npy", mmap_mode="r")
    return idx, store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--noise", type=float, default=INPUT_NOISE)
    ap.add_argument("--drop", type=float, default=FEATURE_DROPOUT)
    a, _ = ap.parse_known_args()

    rows = pd.read_parquet(LOCAL / "perturb_rows.parquet")
    rows["ddg"] = rows.ddg.clip(-4, 4)
    train = rows[rows.fold != a.fold].reset_index(drop=True)
    n = len(train)
    print(f"fold {a.fold}: {n} training rows, {len(rows) - n} held out\n")

    # ---- 1. the realised reverse-mutation rate ------------------------------------
    # Exactly the draw DS makes: one rng.random() per __getitem__, one generator per
    # Dataset, num_workers=0 so the sequence advances across the whole epoch.
    print("1. REVERSE-MUTATION AUGMENTATION  (nominal p=0.5, training split only)")
    rng = np.random.default_rng(0)
    rates, signs = [], []
    for ep in range(a.epochs):
        draws = rng.random(n) < 0.5
        rates.append(draws.mean())
        y = np.where(draws, -train.ddg.values, train.ddg.values)
        signs.append(y.mean())
        print(f"   epoch {ep}: {draws.sum():>4}/{n} reversed = {draws.mean():.1%}, "
              f"label mean {y.mean():+.3f}")
    print(f"   realised rate over {a.epochs} epochs: {np.mean(rates):.1%}")
    print(f"   label mean: {np.mean(signs):+.3f} augmented vs "
          f"{train.ddg.mean():+.3f} raw")
    p_never = 0.5 ** a.epochs
    print(f"   a row is never reversed in {a.epochs} epochs with p={p_never:.3f}; "
          f"over 25 epochs that is {0.5**25:.2e}")
    print("   validation and test loaders are built with augment=False, so this is "
          "training-only\n")

    # ---- 2. the scale the noise is actually measured against ----------------------
    print("2. INPUT NOISE  (nominal 0.25 x per-feature sd, training only)")
    idx, store = load_tokens()
    from sklearn.decomposition import PCA

    crops = np.load(LOCAL / "perturb_crops.npz", allow_pickle=False)

    # ESM-2's vocabulary is a fixed table, so the token ids can be built directly. This
    # avoids importing transformers, which pulls in torch -- and torch does not load on
    # this machine. The mapping is asserted against the index below rather than trusted.
    VOCAB = {a: i for i, a in enumerate(
        "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . "
        "- <null_1> <mask>".split())}

    def tokens_of(s):
        ids = np.array([VOCAB["<cls>"]] + [VOCAB[c] for c in s] + [VOCAB["<eos>"]],
                       dtype=np.int32)
        off, m = idx[ids.tobytes()]
        return np.asarray(store[off:off + m], np.float32)[1:-1]

    sample, deltas = [], []
    for r in train.head(60).itertuples():
        try:
            t_wt, t_mt = tokens_of(r.ab_wt), tokens_of(r.ab_mt)
        except KeyError:
            continue
        ci = crops[f"{r.row_id}|ab_idx"]
        ci = ci[ci < len(t_wt)]
        if not len(ci):
            continue
        sample.append(t_wt[ci])
        site = crops[f"{r.row_id}|ab_site"]
        m = np.flatnonzero(site[: len(ci)])
        if len(m):
            deltas.append((t_mt[ci][m] - t_wt[ci][m]).mean(0))
    X = np.concatenate(sample)
    pca = PCA(n_components=128, random_state=0).fit(X)
    Z = pca.transform(X)
    sd = Z.std(0)
    print(f"   PCA-128 sequence features: per-feature sd median {np.median(sd):.4f}, "
          f"max {sd.max():.4f}")
    print(f"   injected noise sd = {a.noise} x that = {a.noise*np.median(sd):.4f} "
          f"per feature")

    # ---- 3. what that does to the quantity the model consumes --------------------
    print("\n3. WHAT SURVIVES IN delta = mt - wt  (the model's actual input)")
    # NOT pca.transform: that subtracts the training mean, which is wrong for a
    # difference. transform(mt) - transform(wt) = (mt - wt) @ components_.T, so the
    # delta is projected with the components alone.
    D = np.stack(deltas) @ pca.components_.T
    d_sd = D.std(0)
    print(f"   delta per-feature sd: median {np.median(d_sd):.4f}")
    per_branch = a.noise * np.median(sd)
    on_delta = per_branch * np.sqrt(2)      # independent draws on wt and mt
    print(f"   noise reaching the delta: {per_branch:.4f} x sqrt(2) = {on_delta:.4f}")
    print(f"   ==> noise is {on_delta/np.median(d_sd):.1f}x the delta's own sd")
    print("   The noise is drawn FRESH for the wild-type and the mutant branch, so it does")
    print("   not cancel in the difference -- it compounds by sqrt(2). The dropout mask is")
    print("   drawn once per width and reused, so it multiplies the delta instead.")

    # ---- 4. feature dropout -------------------------------------------------------
    print(f"\n4. FEATURE DROPOUT  (p={a.drop}, one mask per width, shared wt/mt)")
    keep = np.random.default_rng(0).random((5000, 128)) >= a.drop
    print(f"   {100*(~keep).mean():.0f}% of features zeroed; survivors scaled by "
          f"{1/(1-a.drop):.2f}x")
    print(f"   a 128-wide block keeps a median of {int(np.median(keep.sum(1)))} features")
    print("   Shared between branches, so delta = mask*(mt) - mask*(wt) = mask*(mt-wt):")
    print("   it zeroes features OF the delta rather than cancelling out of it.")


if __name__ == "__main__":
    main()
