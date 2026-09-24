# Antibody–antigen ΔΔG: a gated multimodal model

Predicting how a mutation changes antibody–antigen binding free energy, from **sequence and
3D structure**, on the antibody–antigen subset of SKEMPI 2.0.

**The model is `l1_gated`: a learned per-channel gate over ESM-2 and ProteinMPNN
representations, read at the mutated residues. 48,001 trainable parameters, both encoders
frozen.**

It was chosen over cross-attention, FiLM and plain concatenation — and *how* it was chosen is
why this README is organised the way it is. On **940 rows across 53 complexes**, with a
measured seed-to-seed spread of 0.028–0.117, most architectural differences in this problem
are smaller than the noise. Establishing which differences are real became the substance of
the work.

| | |
| --- | --- |
| **The model, specified in full** | [docs/MODEL.md](docs/MODEL.md) |
| **Why each choice was made** | [docs/JUSTIFICATIONS.md](docs/JUSTIFICATIONS.md) |
| **Every architecture tried (45 runs)** | [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md) |
| **Error analysis** | [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) |
| **AI prompt history** | [AI_PROMPTS.md](AI_PROMPTS.md) |
| Results · Running it · Hardware · Limitations | below |

---

## 1. The model

```
ESM-2 650M per residue  -> PCA-128 -.
                                     >- Linear(128->64), shared -> site_mean at mutated residues
ProteinMPNN encoder_h_V -> PCA-128 -'

z_seq = LayerNorm( site_mean(mutant) - site_mean(wild-type) )      64   the edit
z_str = LayerNorm( site_mean(structure at those residues) )        64   the pocket
chem                                                               26   substitution chemistry

g = sigmoid( W [ z_seq ; z_str ; chem ] )          per-channel gates, W: 154 -> 128
z = [ g[:64]*z_seq ; g[64:]*z_str ; chem ]                        154
    Linear(154->128) -> GELU -> Dropout(0.35) -> Linear(128->1)
```

**The fusion.** Each of the 128 channels gets its own gate, and the gate is computed from
**both modalities and the chemistry jointly**. The model therefore chooses, per row and per
channel, how much sequence and how much structure to admit — conditioned on what the mutation
is. Concatenation mixes at a fixed ratio; attention lets tokens query each other but pays for
a full attention block. Gating sits between them, at 19,840 parameters.

**Three representation decisions that matter more than the fusion does:**

- **A mutation is a contrast, so its representation is a contrast** —
  `site_mean(mutant) − site_mean(wild-type)`, with the mutant sequence **re-embedded**, not
  edited in latent space. Editing a latent vector assumes the encoder is linear in the
  substitution; it is not.
- **Read at the mutated residues, not across the interface.** Measured: pooling structure over
  the whole binding area instead costs ~0.05. ProteinMPNN's signal is local, and averaged over
  ~85 residues what survives is nearly constant per complex.
- **Every block gets its own LayerNorm** (`elementwise_affine=False`, zero parameters). The
  pools arrive ~7× larger than the edit (2.396 vs 0.327); normalised jointly, the pools set the
  scale and the edit — the only part that varies between mutations of one complex — is crushed.

Full specification, parameter breakdown and training protocol: **[docs/MODEL.md](docs/MODEL.md)**.

## 2. How the fusion was chosen

Every mechanism below is the *same network* with only the fusion swapped. The row that makes
the rest interpretable is the control — `st64_noattn`, the identical network with the
structure branch **deleted**.

| fusion mechanism | params | per-cx r | vs control |
| --- | --- | --- | --- |
| **gated fusion** ← the model | 48,001 | **+0.300** | **+0.088** |
| plain concatenation | 36,353 | +0.293 | +0.081 |
| cross-attention, structure → sequence | 77,441 | +0.241 | +0.029 |
| FiLM on the structure delta | 52,993 | +0.227 | +0.015 |
| cross-attention, weighted residual 0.2/0.8 | 77,441 | +0.208 | −0.004 |
| cross-attention, sequence → structure | 77,441 | +0.199 | −0.013 |
| cross-attention, separate MPNN projection | 77,441 | +0.192 | −0.020 |
| **no fusion (control)** | 36,481 | **+0.212** | — |
| antibody↔antigen attention (v2/v3/v4 family) | 51k–811k | +0.112 … +0.200 | below control |

**The attention attempts, and why they were abandoned.** Cross-attention over ProteinMPNN was
built five ways — both directions, weighted residual connections, a separate structure
projection, and with the fold-local PCA removed so attention had raw embeddings to work with.
**Four of the five score at or below a model with the attention deleted**, at roughly twice the
parameters.

Antibody↔antigen cross-attention — ProtAttBA's mechanism, and the one this project originally
planned — was measured too: it is the entire v2/v3/v4 family, with rotary position embeddings,
a distance bias on the attention logits and a chain embedding. It reaches +0.112 to +0.200,
*below* the no-fusion control, and the largest model in the project (810,886 parameters) scores
+0.200.

On 752 training rows per fold, a full attention block cannot pay for itself. Gating and
concatenation are the two cheapest mechanisms and the only two that clear the control
convincingly.

**Gating over concatenation — and what that claim rests on.** +0.300 against +0.293 is +0.007,
*inside* the noise. Gating is **not** measurably more accurate. It is chosen because it is a
genuine conditional fusion, is no worse, and is **twice as stable**: seed spread 0.028 against
0.063, with its worst seed (+0.235) above concatenation's worst (+0.210).

### The training objective — a within-complex ranking loss was tried and did not improve

The headline metric is a *within-complex* correlation, so the obvious move is to optimise it
directly: an AbRank-style pairwise logistic loss over pairs of mutations on the same complex
(`src/fusion_v2.py::pairwise_rank_loss`). It cancels the per-complex offset exactly and is
invariant to per-complex rescaling, which matters when 47.7 % of SKEMPI temperatures are
assumed rather than measured.

It does not help.

| objective | per-complex ρ |
| --- | --- |
| no-rank control (regression only) | 0.420 |
| best ranking config, full pair coverage | 0.421 |
| ranking mixed at lower weight | 0.288 |
| **pure ranking, no regression term** | **0.170** (RMSE 4.20) |

*Measured on the pre-correction 997-row dataset, so these are not comparable to the table in
§3 — the comparison against its own control is internal and valid.*

At best it **ties** the control. Pure ranking collapses, and the mechanism is clear: the
regression term is the only anchor on output scale, and without it the model is free to drift
anywhere that preserves order — RMSE 4.20 against ~1.5.

The deeper reason it cannot help is that **19,834 pairs from 997 rows is 19.9× the examples
and zero new information.** Pairs are a re-expression of the same labels, not augmentation.

**What survives.** A ranking loss is the only route to the **80 non-binders and 86 censored
rows** — measurements with no usable ΔΔG but a known ordering, and the censored rows are
exactly the ones the regression target forced this project to drop (997 → 940). It is deferred
on a narrow trigger: build it for *data recovery*, not for a score gain.

## 3. Results

Grouped 5-fold cross-validation, **no complex shared between folds**. `make report` and
`scripts/final_table.py` regenerate everything; no number here was typed by hand.

```
model                                 ens  per seed  spread  n  neg    bal  bal-cal   stab
l1_gated  (the model)               +0.300    +0.249   0.028  3    7  0.463    0.459   0.25
plain concatenation                 +0.293    +0.239   0.063  3    6  0.439    0.457   0.19
cross-attention (best of five)      +0.241    +0.182   0.046  3    9  0.454    0.445   0.35
no-fusion control                   +0.212    +0.212     -    1    9  0.392    0.423   0.13
--- baseline ---
E0a_rf_handcrafted (chem+geom+MPNN) +0.381    +0.379   0.032  3    6  0.444    0.498   0.06
--- floor ---
[complex mean only]                 +0.000                         0.580            0.58
```

`ens` averages seeds then scores once; `per seed` scores each separately. **`spread` is the
number to read every comparison against.** Quoting one without the other is how a model appears
to gain 0.05 by being written up differently — `l1_gated` is +0.300 as an ensemble and +0.249
as the mean of its seeds, from identical predictions.

**Two rows to read before the model's.**

**The floor.** Predicting each complex's own mean — never looking at the mutation — scores
**+0.672 pooled Pearson, 1.144 RMSE, 0.580 balanced accuracy**, beating everything here on all
three, while scoring **+0.000** per complex. Between-complex variance dominates this dataset,
so pooled correlation, RMSE and accuracy largely measure whether a model can identify the
complex. That is why per-complex is the headline and why `report_runs.py` prints this floor
beneath every comparison.

**The random forest baseline beats the neural model**, +0.381 against +0.300 — about 2.5× the
seed spread, one of the few gaps here that clears the noise. It uses 49 handcrafted columns
spanning both modalities (substitution chemistry; interface geometry; ProteinMPNN
log-likelihoods) and no learned representation at all. Reported rather than buried; §5 says
what it means.

### Generalisation under a homology split

`frozen5` withholds whole complexes; the `cluster` split withholds whole homology clusters.
Three seeds each, per-complex Spearman:

| model | by complex | by cluster | retained |
| --- | --- | --- | --- |
| random forest, handcrafted | 0.418 | 0.224 | **54 %** |
| pooled ESM + ProteinMPNN forest | 0.366 | 0.182 | 50 % |
| our network | 0.246 | 0.084 | **34 %** |

Everything degrades; the network degrades most, and 11 of its complexes finish with a
*negative* within-complex correlation. **The embedding-based model was leaning on homology the
cluster split withholds.** This is the single most important caveat on the neural result.

### Where the neural model is better

| | balanced acc | stabilising recall |
| --- | --- | --- |
| `l1_gated` | **0.463** | **0.25** |
| random forest | 0.444 | **0.06** |
| random forest, quantile-calibrated | 0.498 | 0.29 |

The forest finds **7 of 126** stabilising mutations. The neural model finds four times as many.
For affinity maturation — *finding* affinity-improving mutations rather than ranking known ones
— that matters more than the correlation gap.

The forest's weakness is a thresholding artifact, not a modelling failure: it compresses into
50 % of the label's range, so cutting at the label edges starves the minority class. Quantile
calibration lifts its recall 0.06 → 0.29 at **zero cost to ranking** (monotone, so per-complex
Spearman is unchanged at +0.3613). Even calibrated it does not reach the best network's 0.48.

## 4. Evaluation and error analysis

**[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md)** — 19 sections in three parts. The essentials:

- **Pooled Pearson mostly measures complex identity.** The floor beats every model on it.
- **The seed spread exceeds most architectural effects** — 0.117 per complex, 0.376 pooled
  within a single fold. The whole ladder fits inside it.
- **Regression to the mean is the dominant error structure**: ρ ≈ −0.8 between signed error and
  true ΔΔG *for every architecture*, and ρ ≈ −0.58 even within a complex. One fact explains the
  class bias, the valine failure and the compressed output range together.
- **Imbalance bias.** The average stabilising mutation is predicted *destabilising*, and within
  that class r = 0.007. X→A is 46 % of the data and the model is twice as good on it. Valine is
  under-predicted by 1.4 kcal/mol and ranked backwards. 21 of 53 complexes are too small to
  enter the headline metric at all.
- **It is not a pipeline bug.** Across 940 rows and 1,726 mutated sites: zero residue
  mismatches, and ‖t_mut − t_wt‖ puts the mutated sites in the top-k on 97 % of sides at a
  median of 22× the median residue. `make align`.

Two conclusions in this project were drawn from partial folds and **reversed** when the final
fold landed. Both are recorded in [AI_PROMPTS.md](AI_PROMPTS.md).

## 5. Limitations

1. **Data volume is the binding constraint.** 752 training rows per fold, 32 scorable
   complexes. No regularisation setting removed the seed variance: raising dropout 75 %, noise
   150 % and weight decay tenfold cost 0.037–0.043 on two architectures and reduced variance on
   neither.
2. **The neural model loses to a random forest** by 0.081, and retains 34 % under a homology
   split against the forest's 54 %. On this much data, 49 informative columns beat a learned
   representation.
3. **Pretrained sequence embeddings do not clear zero on their own**, across six probes and
   three protein language models. 26 chemistry columns moved the network +0.191 → +0.266 —
   larger than any architectural change measured. AntiBERTy, antibody-specific, lost to ESM-2
   by 0.046 on a matched control.
4. **There is no mutant structure.** ProteinMPNN sees the wild-type backbone only, so the
   structure term is identical for every mutation of a complex. The sharpest structural
   limitation of the design.
5. **Label noise caps the achievable correlation** — within-complex label sd has a median of
   1.104 kcal/mol.
6. **Three complexes hold 24 % of the rows**, so any pooled statistic is partly about them.
7. **Only 8 of 45 configurations have three complete seeds.** The rest are single-seed and
   their individual numbers are not interpretable at the resolution the tables print them.

## 6. Next steps, in the order I would do them

1. **Read the gates.** No checkpoints were saved, so *what the model gates on* — how much
   sequence versus structure, for which mutations — cannot be answered from the artifacts. This
   turns the fusion mechanism from a score into an explanation, and it is the first thing to do.
2. **Blend the forest with the network.** They agree only 0.42–0.54 on which rows are hard, and
   a fixed 25 % blend lifts per-complex r from +0.397 to +0.418–0.424 across two different
   networks. The weight was not tuned on held-out data, so validate it — but it is better
   evidenced than any architecture change here.
3. **Share the input-noise draw across branches.** Measured: the noise is drawn independently
   per branch, so it compounds by √2 and reaches **0.8× the delta's own sd** — at the heaviest
   setting, more noise than signal. Two-line change, and it would explain why heavier
   regularisation kept costing accuracy.
4. **Give the network the forest's geometry columns.** The largest measured gain in the project
   came from adding chemistry; interface geometry is the untried other half.
5. **Ordinal loss instead of MSE.** MSE asks the model to reproduce a value whose repeat
   measurements disagree by ~1.1 kcal/mol. A CORAL head is implemented (`ordinal=2`) and
   verified; it never completed a run.
6. **Three seeds minimum for any future claim**, and report the spread.
7. **More data before more architecture.** 261 AB645/AB1101 rows are built, cached and
   leakage-filtered, and have never been trained on.

## 7. Setup and running

```bash
pip install -r requirements.txt          # Python 3.10; torch pinned, see requirements.txt
make data                                # SKEMPI -> data/processed/*.parquet   (~65 s)
make splits                              # frozen homology folds -> data/folds.csv  (~28 s)
make features                            # geometry, ESM, ProteinMPNN caches
make ladder                              # every rung, each writing reports/<name>/
make report                              # the results table, all runs on one truth
make errors                              # slice tables and diagnostics
make align                               # verify each mutation is where we index it
make test                                # split-integrity and harness tests
```

`scripts/final_table.py` produces the results table above; `scripts/bias_analysis.py` and
`scripts/error_drivers.py` produce Part III of the error analysis. `data/folds.csv` is
committed and frozen — nothing downstream regenerates it.

**Known environment issue:** `tests/test_perturb_*.py` require torch, which fails to load on
the development machine (8 GB RAM, `WinError 1114`). The other tests pass; the torch tests run
wherever torch loads.

## 8. Hardware and runtime

Development: **Intel Core i7-8650U, 4 cores @ 1.9 GHz, 8 GB RAM, no GPU**, Windows 10.
Neural training: **Colab T4**, driven from the CLI in `experiments/protattba_repro/`.

| step | where | wall clock |
| --- | --- | --- |
| `src.data` — parse, verify 2,109 positions, dedup | CPU | ~65 s |
| `src.splits` — 1,431 pairwise alignments + leakage report | CPU | ~28 s |
| `src.features.geometry` — Shrake-Rupley over 54 complexes | CPU | ~5.5 min |
| ESM-2 650M per residue, 448,772 tokens | T4 | ~4 min (1,998 tok/s) |
| the same | CPU | ~160 min |
| ProteinMPNN per residue, 54 complexes | CPU | ~7 min |
| one forest rung, 5 folds × 3 seeds | CPU | ~20 s |
| **`l1_gated`, 5 folds × 3 seeds** | **T4** | **~25 min** |
| the full neural ladder as run (~45 configurations) | T4 | ~14 h |
| `pytest tests` (non-torch) | CPU | ~43 s |

Colab sessions are reclaimed after roughly an hour, shorter than several runs. The harness
resumes at `(fold, seed)` granularity and `experiments/protattba_repro/supervise.sh` rebuilds a
reclaimed session and relaunches its queue unattended.

## 9. Repository layout

```
src/
  data.py splits.py evaluate.py    the pipeline, the frozen split, the harness
  features/                        chem, geometry, esm2, proteinmpnn — each cached by row id
  fusion/                          the forest baselines
  perturb/                         the neural models and their data plumbing
scripts/
  report_runs.py                   every run on one common truth, with the floor beneath it
  final_table.py                   the results table above
  bias_analysis.py                 error analysis part III — imbalance
  error_drivers.py                 error analysis part III — descriptors vs error
experiments/protattba_repro/       Colab harness, external-baseline reproduction, audits
configs/                           one yaml per rung
reports/ results/                  generated tables, predictions, out-of-fold scores
ERROR_ANALYSIS.md AI_PROMPTS.md    error analysis, prompt history
docs/MODEL.md                      the model, in full
docs/JUSTIFICATIONS.md             why each choice was made
docs/ARCHITECTURES.md              all 45 configurations
docs/                              plan, design rationale, decisions, EDA, prior work
```

Every run records its config, git SHA, seed and wall-clock into `results/results.csv`.
