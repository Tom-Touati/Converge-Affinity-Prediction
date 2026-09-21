#!/usr/bin/env python3
"""How many structurally similar measurements each complex has -- with and without the split.

The tier definition counts *measurements*, not complexes: a target is "easy" when 50 or more
training rows come from complexes above TM 0.8. Plotting that count two ways shows why every
complex here lands in "hard". The similar data exists in quantity; the cluster-grouped split
puts all of it on the same side of the fold boundary.
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import splits                                    # noqa: E402
from src.tmscore import EASY_MIN, MATRIX_PATH, TM_THRESHOLD  # noqa: E402

M = pd.read_csv(MATRIX_PATH, index_col=0)
d = splits.load()
cx = d.drop_duplicates("#Pdb").set_index("#Pdb")
keys = [k for k in M.index if k in cx.index]
M = M.loc[keys, keys]
rows_per_cx = d.groupby("#Pdb").size()

whole, train_only, n_sim_cx = [], [], []
for k in keys:
    sim = [c for c in keys if c != k and M.loc[k, c] > TM_THRESHOLD]
    whole.append(int(rows_per_cx.reindex(sim).fillna(0).sum()))
    n_sim_cx.append(len(sim))
    tr = [c for c in sim if cx.loc[c, "fold"] != cx.loc[k, "fold"]]
    train_only.append(int(rows_per_cx.reindex(tr).fillna(0).sum()))
whole, train_only, n_sim_cx = map(np.array, (whole, train_only, n_sim_cx))

tier = lambda v: "hard" if v == 0 else ("easy" if v >= EASY_MIN else "medium")
tw = pd.Series([tier(v) for v in whole]).value_counts()
tt = pd.Series([tier(v) for v in train_only]).value_counts()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
bins = np.histogram_bin_edges(np.concatenate([whole, [0]]), bins=20)

ax[0].hist(whole, bins=bins, color="#4c78a8", edgecolor="#22303f")
ax[0].axvline(EASY_MIN, color="#d62728", ls="--", label=f"easy threshold ({EASY_MIN})")
ax[0].set_title("IGNORING the split\nsimilar measurements available anywhere in SKEMPI-AB")
ax[0].legend(fontsize=8)

ax[1].hist(train_only, bins=bins, color="#e45756", edgecolor="#4a2222")
ax[1].axvline(EASY_MIN, color="#333", ls="--")
ax[1].set_title("UNDER the cluster-grouped split\nsimilar measurements in that fold's training set")

lab = ["hard", "medium", "easy"]
x = np.arange(3)
ax[2].bar(x - 0.19, [tw.get(l, 0) for l in lab], 0.38, label="ignoring the split", color="#4c78a8")
ax[2].bar(x + 0.19, [tt.get(l, 0) for l in lab], 0.38, label="under the split", color="#e45756")
ax[2].set_xticks(x)
ax[2].set_xticklabels([f"hard\n(0)", f"medium\n(1-{EASY_MIN - 1})", f"easy\n({EASY_MIN}+)"])
ax[2].set_title("tier assignment, both ways")
ax[2].legend(fontsize=8)
for i, l in enumerate(lab):
    ax[2].text(i - 0.19, tw.get(l, 0) + 0.6, str(tw.get(l, 0)), ha="center", fontsize=9)
    ax[2].text(i + 0.19, tt.get(l, 0) + 0.6, str(tt.get(l, 0)), ha="center", fontsize=9)

for a in ax[:2]:
    a.set_xlabel(f"measurements from complexes with TM > {TM_THRESHOLD}")
    a.set_ylabel("complexes")
ax[2].set_ylabel("complexes")
for a in ax:
    a.grid(alpha=.3)
fig.suptitle("Structurally similar support per complex. The data exists -- the split removes "
             "all of it from training.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = pathlib.Path("reports/figures/tm_support.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}\n")

print(f"similar MEASUREMENTS per complex, ignoring the split:")
print(f"   min {whole.min()}  median {int(np.median(whole))}  max {whole.max()}  "
      f"| zero for {int((whole == 0).sum())} of {len(whole)} complexes")
print(f"   tiers: {dict(tw)}")
print(f"under the split: all {int((train_only == 0).sum())} of {len(train_only)} are zero")
print(f"\nsimilar COMPLEXES per complex: median {int(np.median(n_sim_cx))}, max {n_sim_cx.max()}")
big = np.argsort(-whole)[:6]
for i in big:
    print(f"   {keys[i]:18s} {whole[i]:4d} similar measurements across {n_sim_cx[i]:2d} complexes "
          f"-- all in fold {cx.loc[keys[i], 'fold']}")
