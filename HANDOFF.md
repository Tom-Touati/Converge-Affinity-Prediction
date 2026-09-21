# Handoff — state of the project, and what a fresh session needs to know

Written 2026-09-21, after the session that removed residual learning. Read this before
trusting any number in the older documents: several headline figures in `README.md`,
`ARCHITECTURE.md` and `ERROR_ANALYSIS.md` predate the corrections below and are wrong.

---

## 0. Read this first — the dataset changed on 2026-09-21

**Censored affinities are now dropped.** SKEMPI records a detection limit as `">1e-6"` and
parses it to a bare number, so the row means "Kd is at least this" and the pipeline read "Kd
is exactly this". Those rows are one-directionally biased: they average **+2.09 kcal/mol**
against +0.97 for the rest, because a censored affinity means binding too weak to measure.

    997 rows, 54 complexes   ->   940 rows, 53 complexes   (72 measurements, 4I77_HL_Z lost)

The cost is large, and in the direction that matters:

| | before (997 rows) | after (940 rows) |
|---|---|---|
| forest, cluster split | 0.388 | **0.239** |
| forest, complex split | 0.413 | **0.354** |
| concordance, cluster | 0.735 | **0.668** |

Those censored rows were the easiest in the dataset to rank — large, one-directional effects
— so 6% of the data was inflating the headline by 0.149. **Every network number in §1 and §6
below was produced on the 997-row dataset and needs regenerating.** The forest numbers here
are current; the net numbers are not.

Dropping is a holding policy, not the right answer. See §9.

---

## 1. The result

**The random forest is the model. The neural network does not beat it, on either split.**

Cluster-grouped split (the reported one), all models scored on identical rows:

| model | per-complex ρ | vs forest |
|---|---|---|
| random forest, `chem+geom+geomrev+mpnn` | **0.376** | — |
| best net (`dir_nohadamard`) | 0.308 | −0.069 |
| worst net (`dir_noattn`) | 0.197 | −0.179 |

Complex-level split (looser, secondary):

| model | per-complex ρ | vs forest |
|---|---|---|
| random forest | **0.413** | — |
| best net (`dir_plain_complex`) | 0.308 | −0.105 |

The looser split helps the forest (+0.025) and does nothing for the net (0.308 either way).
The network is not being held back by the harshness of the evaluation.

---

## 2. The correction that invalidates older numbers

Every earlier "the net wins" result — 0.466, 0.502, 0.506 — came from **residual learning**.
The net was trained on `y − forest_prediction` and the forest was added back at inference:

```python
target[tr] = y[tr] - inner[tr]      # what the net learns
oof[te]    = p + base_te            # what gets scored
```

A net with **zero** skill still scored ~0.50, because `0 + forest_pred = forest_pred`. The
net's own validation correlation was ~0.23 and described how well it ranked the *forest's
mistakes*. Standing alone, the same architecture scores 0.308.

It did not even buy an easier target: on fold 1, `sd(residual)` = 1.214 against `sd(ddG)` =
1.156 — **105%** of the original spread, because the forest predicts a narrow band near zero
while the labels range much wider.

**`residual=False` is now the default and must stay that way.** See the memory note
`no-residual-learning.md`.

---

## 3. Quote the metric definition with the number

"0.5" was the most flattering of six defensible values for the *same* forest predictions:

| definition | value |
|---|---|
| per-complex ρ, ≥10 mutations (27 of 54 complexes) | 0.498 |
| per-cluster ρ | 0.500 |
| global (pooled) ρ | 0.433 |
| **per-complex ρ, ≥5 mutations (34 of 54)** — current default | **0.388** |
| per-complex ρ, no threshold (all 54) | 0.354 |
| pairwise concordance, margin 0.5 (48 complexes, every row) | 0.735 |

`evaluate.MIN_GROUP` moved from 10 to 5. The old rule dropped 27 of 54 complexes and the
dropped ones carried the **highest** error (rmse 2.27 for complexes with 1–5 mutations
against 1.33 for 11–20). The seven complexes it hid average ρ **−0.036**, three of them
anti-correlated. `per_complex_spearman_min10` is still reported so older numbers stay
readable — check the `min_group` field before comparing two runs.

`evaluate.pairwise_concordance` has no small-group floor (defined from n ≥ 2) and covers 48
complexes where Spearman covers 34. It is also what the ranking loss optimises.

---

## 4. Where the model actually fails

Correlation alone hides this — `rmse/sd` compares the error against predicting that slice's
own mean, so ≥ 1 means the model adds nothing *within* the group.

| slice | n | macro ρ | rmse | rmse/sd |
|---|---|---|---|---|
| destabilising | 544 | +0.439 | 1.78 | **1.13** |
| neutral | 323 | +0.071 | 0.75 | **3.08** |
| stabilising | 130 | **−0.045** | 2.49 | **2.74** |
| antibody-side | 584 | +0.308 | 1.54 | **1.00** |
| antigen-side | 332 | +0.531 | 1.57 | 0.84 |
| single-point | 696 | +0.485 | 1.40 | 0.87 |
| multi-point | 301 | +0.135 | 2.11 | 0.94 |

Destabilising mutations show a respectable ρ while the magnitudes are worse than a constant.
Antibody-side — 584 of 997 rows, the affinity-maturation case — is exactly as good as the
group mean. Stabilising mutations, the ones that matter for design, are anti-correlated.

Caveat: the slice mean is an **oracle** baseline (you would have to know the true class to
use it). Unconditioned, the model is at rmse/sd = 1.647/1.823 = 0.90, so it does beat the
global mean.

---

## 5. Structural difficulty: the tiering is degenerate here

Implemented the published SKEMPI tiering in `src/tmscore.py` (easy ≥50 structurally similar
training measurements, medium 1–50, hard 0) with real TM-align over **antigen chains**
(aligning whole complexes would put nearly every antibody pair above 0.8 on framework alone).

All 54 complexes come out **hard**, and it is the split, not a bug:

```
all antigen pairs     1431   median TM 0.311   >0.8: 118
same cluster           168   median TM 0.965   >0.8: 117
different cluster     1263   median TM 0.307   >0.8:   1
cross-fold pairs      1033                     >0.8:   0
```

Best training-set match any complex gets is **0.636**. The homology clustering finds the
structural twins and puts them in one fold, so none can ever be training data for another.

`intrinsic_tier` (structural support *anywhere*, ignoring the split) does vary — 21 easy, 25
medium, 8 hard. Counter-intuitively the 8 intrinsically hard complexes score **best**
(+0.608 vs +0.441 easy, bootstrap +0.167 CI [+0.059, +0.270]). The mechanism is **not
established**; what is established is that the tiers differ in composition — easy is 72%
antibody-side and 34% multi-point, hard is 56% antigen-side and 12% multi-point. Cluster size
does *not* explain it (corr(ρ, log cluster rows) = −0.148).

This is a large part of why published SKEMPI numbers sit above ours. Under the complex-level
split, 42 of 54 complexes gain a TM > 0.8 twin in training (worst 1.000) against 0 of 54
under cluster grouping.

---

## 6. What was tried and did not work

- **Ranking loss (AbRank-style)** — at full pair coverage the best config ties the no-rank
  control (0.421 vs 0.420); everything else is worse, down to 0.205 for pure ranking, which
  collapses because the Huber term is the only anchor on output scale.
- **ESM-2 650M** — raw concatenation scores 0.262 against 0.500, but that is dilution: 3,853
  ESM columns against 49 informative ones, with `max_features="sqrt"` sampling ~62 per split.
  PCA-32 gives 0.473 against a matched `rf20` control at 0.499 — delta −0.026, CI
  [−0.065, +0.013], **does not clear zero**. A 19× larger PLM does not rescue the sequence
  modality. Seven probes now agree.
- **Regularisation and augmentation** — tuned against residual targets, does not transfer.
  `dir_plain` (no regularisation at all) is top or near-top on both splits.
- **`fusion_v3`** — had never run: `run()` referenced `dev`, which is local to `_fit`, so
  every config raised `NameError` on its first fold. Fixed; best config 0.449, still behind.

What *does* replicate: the mutation-token attention is load-bearing (`dir_noattn` is worst on
both splits), and removing the scalar bypass hurts (`dir_noscalars`).

---

## 7. Infrastructure gotchas that will waste a day if rediscovered

**The Colab CLI (`google-colab-cli`, run from WSL):**

- **Nothing can be gated on an exit code.** `colab exec` returns 0 when the remote script
  raises; `colab status` returns 0 for a session that does not exist. Check output text.
  `scripts/colab_run.sh` does this correctly — use it rather than writing a new runner.
- **Subprocess output does not reach the stream.** `colab exec` relays only what the kernel
  process itself writes, so `subprocess.run` output vanishes. `colab_job.py:sh()` pipes and
  re-prints.
- **Never reinstall the CLI mid-session** — it kills the keep-alive daemon `colab new`
  spawned and the VM idles out.
- **Pin `jupyter-kernel-client==0.9.0`.** The CLI declares it unpinned and 1.0.2 renamed
  `KernelClient` → `JupyterKernelClient`, breaking every `exec`.
- **`python3 -m venv --system-site-packages` fails on Colab** in `ensurepip`, and a venv is
  pointless there anyway. Use `PYTHON_BIN=$(which python3)`.
- **Filter both `torch==` and `numpy==` from requirements on Colab.** Both pins are Windows
  MSVC artefacts; `numpy==1.26.4` downgrades Colab's numpy 2 and breaks the preinstalled
  `jax`, which `transformers` imports transitively, killing every ESM-2 step.
- **Push cached features rather than re-extracting** (`push_features` in `colab_run.sh`).
  `/content` is wiped with the session; re-extraction costs ~20 min per run.
- **`colab download` works while the kernel is BUSY** — this is what makes the live dashboard
  possible.

**Local tooling:**

- `tail -f` is a **no-op on `/mnt/c`** — WSL's DrvFs has no inotify. Poll instead.
- `grep` block-buffers when stdout is not a TTY; use `--line-buffered`.
- `json.dumps` writes bare `NaN`, which `JSON.parse` rejects — the dashboard serialiser
  converts NaN/inf to null. **`curl` will not catch this**, because it never parses the body.

---

## 8. Tools built this session

- **`scripts/dashboard/`** — live view, `python scripts/dashboard/serve.py` or
  `preview_start` with `.claude/launch.json`. Three tabs: Training (per-step curves, pulled
  off a *running* VM), Error analysis (direction, single/multi-point, antibody/antigen side,
  per-cluster, support, structural tiers, per-complex scatter), Runs (all ~100 runs, newest
  first, with `resid` and `min_grp` columns so two different metrics are never read off one
  line).
- **`scripts/probe_earlyfit.py`** — runs configurations in parallel for a few dozen steps and
  records train and validation loss on the *same* objective. Use it before committing GPU
  time; but note its one-fold one-seed results **did not replicate** (`no_scalars` scored
  0.499 on the probe and 0.378 on the real sweep), so treat it as a smoke test, not evidence.
- **`src/tmscore.py`** — TM-align tiering, matrix cached at `data/tm_matrix.csv`.
- **`src/splits.py --grouping {cluster,complex,random}`** — the looser split writes
  `folds_complex.csv` and never touches the frozen `folds.csv`.
- **`scripts/colab_run.sh <stage>...`** — the only runner that should be used.

---

## 9. Open and unresolved

- **No paired bootstrap on the net-vs-forest gaps.** Needed before writing "clearly worse"
  rather than "worse". `evaluate.paired_bootstrap(a, b, n_boot=2000)`.
- **`AI_PROMPTS.md` is a required deliverable and is days stale.**
- **`README.md`, `ARCHITECTURE.md`, `ERROR_ANALYSIS.md` carry pre-correction numbers.**
- **Censored affinities: dropped, and that is a placeholder.** `src/data.py` removes any row
  whose affinity was recorded as a detection limit (72 measurements, 57 modelling rows, one
  whole complex). Dropping throws away real evidence — "binds worse than X" is informative,
  and it is exactly the strongly-destabilising end of the range the model is worst at. The
  right treatment is **censored regression** (a one-sided loss that penalises predicting below
  the bound but not above it) or a ranking constraint, either of which uses the row as the
  bound it actually is. `--keep-censored` reproduces the old behaviour and exists only for
  that; it is wrong and should not be used for a reported number.
- **The net sweeps need re-running on the 940-row dataset.** Every `dir_*` figure predates the
  drop.
- The intrinsic-tier reversal (§5) has a measured effect and no established mechanism.
- `thoughts.md` is Tom's personal notes, swept into a commit by an early `git add -A`. Decide
  whether it should stay tracked.
