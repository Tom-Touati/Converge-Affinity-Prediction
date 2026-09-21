#!/usr/bin/env python3
"""Distribution of antigen TM-scores, and what it implies for the difficulty tiers.

Four panels, because the headline question -- "why is every complex `hard`?" -- is not
answerable from the raw distribution alone. It needs the pairs split by whether they can
ever land on opposite sides of the frozen split.
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import splits                                         # noqa: E402
from src.tmscore import MATRIX_PATH, TM_THRESHOLD              # noqa: E402

M = pd.read_csv(MATRIX_PATH, index_col=0)
d = splits.load()
cx = d.drop_duplicates("#Pdb").set_index("#Pdb")
keys = [k for k in M.index if k in cx.index]
M = M.loc[keys, keys]
cluster = cx.loc[keys, "cluster"]
fold = cx.loc[keys, "fold"]
rows_per_cx = d.groupby("#Pdb").size()

iu = np.triu_indices(len(keys), 1)
tm = M.to_numpy()[iu]
same_cluster = (cluster.to_numpy()[:, None] == cluster.to_numpy()[None, :])[iu]
same_fold = (fold.to_numpy()[:, None] == fold.to_numpy()[None, :])[iu]

fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
bins = np.linspace(0, 1, 41)

ax[0].hist(tm, bins=bins, color="#4c78a8")
ax[0].axvline(TM_THRESHOLD, color="#d62728", ls="--")
ax[0].set_title(f"all {len(tm):,} antigen pairs\n{(tm > TM_THRESHOLD).sum()} above 0.8")

ax[1].hist(tm[~same_cluster], bins=bins, color="#54a24b", alpha=.85,
           label=f"different cluster (n={(~same_cluster).sum()})")
ax[1].hist(tm[same_cluster], bins=bins, color="#e45756", alpha=.85,
           label=f"same cluster (n={same_cluster.sum()})")
ax[1].axvline(TM_THRESHOLD, color="#333", ls="--")
ax[1].set_title("by homology cluster")
ax[1].legend(fontsize=8)

ax[2].hist(tm[~same_fold], bins=bins, color="#54a24b", alpha=.85,
           label=f"across folds (n={(~same_fold).sum()})")
ax[2].hist(tm[same_fold], bins=bins, color="#e45756", alpha=.85,
           label=f"within a fold (n={same_fold.sum()})")
ax[2].axvline(TM_THRESHOLD, color="#333", ls="--")
hi_cross = int(((tm > TM_THRESHOLD) & ~same_fold).sum())
ax[2].set_title(f"by fold -- {hi_cross} cross-fold pairs above 0.8")
ax[2].legend(fontsize=8)

# per complex: the best structural match available in its fold's TRAINING set
best, labels = [], []
for k in keys:
    tr = [c for c in keys if cx.loc[c, "fold"] != cx.loc[k, "fold"]]
    best.append(float(M.loc[k, tr].max()) if tr else 0.0)
    labels.append(k)
best = np.array(best)
ax[3].hist(best, bins=bins, color="#b279a2")
ax[3].axvline(TM_THRESHOLD, color="#d62728", ls="--")
ax[3].set_title(f"best match in TRAINING data, per complex\nmax = {best.max():.3f}, "
                f"{(best > TM_THRESHOLD).sum()} of {len(best)} above 0.8")

for a in ax:
    a.set_xlabel("TM-score (antigen chains, normalised by shorter)")
    a.set_ylabel("pairs")
    a.grid(alpha=.3)
fig.suptitle("Antigen structural similarity in the SKEMPI antibody subset -- "
             "high-similarity pairs exist, but the cluster-grouped split keeps them "
             "on the same side of every fold boundary", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = pathlib.Path("reports/figures/tm_distribution.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}")

print(f"\nall pairs           n={len(tm):5d}  median {np.median(tm):.3f}  "
      f">0.8: {(tm > TM_THRESHOLD).sum()}")
print(f"same cluster        n={same_cluster.sum():5d}  median {np.median(tm[same_cluster]):.3f}  "
      f">0.8: {int((tm[same_cluster] > TM_THRESHOLD).sum())}")
print(f"different cluster   n={(~same_cluster).sum():5d}  median {np.median(tm[~same_cluster]):.3f}  "
      f">0.8: {int((tm[~same_cluster] > TM_THRESHOLD).sum())}")
print(f"cross-fold pairs    n={(~same_fold).sum():5d}  >0.8: {hi_cross}")
print(f"\nbest training-set match per complex: min {best.min():.3f}  "
      f"median {np.median(best):.3f}  max {best.max():.3f}")
top = np.argsort(-best)[:6]
for i in top:
    print(f"    {labels[i]:18s} best cross-fold TM {best[i]:.3f}  "
          f"({rows_per_cx.get(labels[i], 0)} rows)")
