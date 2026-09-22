# Reproducing ProtAttBA on S1131

An isolated reproduction of the sequence-only ΔΔG model from *"Sequence-Only Prediction of
Binding Affinity Changes: A Robust and Interpretable Model for Antibody Engineering"*
(Bioinformatics 2025, [btaf446](https://academic.oup.com/bioinformatics/article/41/8/btaf446/8229597),
[arXiv:2505.20301](https://arxiv.org/abs/2505.20301)), code at
[code4luck/ProtAttBA](https://github.com/code4luck/ProtAttBA), pinned at `9adbf98`.

Two jobs:

1. **A sequence-only baseline** to fuse with the structure-based ladder later. Frozen
   ESM2-650M plus a trainable cross-attention head, no structure at all.
2. **A check on this project's infrastructure** — that an external model's code runs here and
   lands on its published number, so a gap in one of our own runs is attributable to us.

Nothing under `src/`, `configs/`, `data/` or the existing `Makefile` targets was modified.
The upstream checkout is gitignored; the two files this reproduction consumes are vendored
under `upstream/`.

---

## Target

Table 1, S1131 row, ESM2 column:

| metric | published |
|---|---|
| PCC | 0.84 ± 0.05 |
| Spearman ρ | 0.75 ± 0.06 |
| RMSE | 1.31 ± 0.09 kcal/mol |

---

## Step 0 — the published number is internally consistent, and the split is pinned

Before spending GPU time, `verify_published.py` checks Table 1 against the authors' own
artefacts. The repo commits `cross_validation/results/S1131_results.csv`: a fold id, a row
index, a prediction and a label for all 1131 rows, i.e. the complete output of one 10-fold run.

```
[1] fold assignment      KFold(10, shuffle=True, random_state=3407) matches exactly, all 1131 rows
[2] label alignment      test_indices indexes S1131.csv row order, max |drift| = 5.5e-07
[3] scoring their own shipped predictions, per fold:

    metric   reproduced from shipped preds   Table 1 (ESM2, S1131)   match
    pcc     0.8370 +/- 0.0501                0.84 +/- 0.05            yes
    rho     0.7497 +/- 0.0648                0.75 +/- 0.06            yes
    rmse    1.3063 +/- 0.0856                1.31 +/- 0.09            yes
```

Three things this buys:

- **Table 1 is a per-fold mean with a population standard deviation** (ddof=0). Not stated in
  the paper; ddof=1 gives 0.053 for PCC, which rounds the same but is worth pinning down.
- **The split is not a guess.** `SEED=3407` from `bash_cross-validation.sh` reproduces their
  fold assignment for all 1131 rows, so our run uses the identical one and no gap can be
  blamed on fold luck.
- **Per-example comparison is possible.** All 1131 of their predictions can be compared row by
  row against ours, not just in aggregate.

---

## What S1131 actually is, and why that matters for this project

The model sees four sequences and a label per row, and nothing else. `utils/common.py` reads
only `PDB, mutation, a, b, a_mut, b_mut, ddG`, and `get_s1131_data` hands the head
`(a, b, a_mut, b_mut, ddG)`. Their code calls `a` the antibody stream and `b` the antigen
stream, but they are just the two partner sides of a complex.

Row 0, as the model receives it:

| field | value |
|---|---|
| PDB / partners | 1A22, chains A_B |
| mutation | `A:C171A` |
| `a` / `a_mut` | 191 residues, differing at exactly one position (C→A at index 181) |
| `b` / `b_mut` | 238 residues, byte-identical |
| ddG | +1.010 kcal/mol |

Shape of the set:

| | |
|---|---|
| rows | 1131, every one a single-point mutation |
| distinct complexes | 112 |
| rows per complex | median 1, max 190 |
| concentration | 3 complexes hold 45% of rows, top 10 hold 66% |
| ddG | mean +1.24, sd 2.45; 842 destabilising, 275 stabilising, 14 exactly zero |

**S1131 contains no antibody-antigen complexes.** SKEMPI 2.0 annotates complex class in
`Hold_out_type`, and that is exactly the column `src/data.py` uses to build this project's
dataset (`raw["Hold_out_type"].str.contains("AB/AG")`). Applied to S1131's 112 PDB ids:

```
Pr/PI     990 rows      protease / protease-inhibitor
<blank>  1429 rows      unannotated (hormone-receptor, barnase-barstar, colicin-Im9, ...)
AB/AG       0 rows
```

The three largest complexes are all protease-inhibitor systems: 3SGB (190 rows, protease B
with turkey ovomucoid inhibitor), 1PPF (171, elastase with OMTKY3) and 1R0R (152, subtilisin
Carlsberg with OMTKY3). The canonical antibody-lysozyme complexes 1DQJ and 1VFB *are* labelled
AB/AG in SKEMPI and are **absent** from S1131. And there is **zero PDB overlap** between
S1131's 112 complexes and this project's 53.

So the paper's title is about antibody engineering, and their AB645/AB1101 sets are antibody
data, but the S1131 row of Table 1 is measured on general protein-protein binding with the
antibody cases removed. Reproducing it is still a valid check on their method and on this
infrastructure, which is what it is used for here. It is **not** an antibody baseline, and its
0.84 must not be quoted next to this project's numbers as though it were one.

### The antibody sets are the ones worth having, and they overlap us heavily

`AB645.csv` and `AB1101.csv` carry real antibody columns
(`antibody_light_seq`, `antibody_heavy_seq`, `antigen_a_seq`, `antigen_b_seq`):

| set | rows | complexes | complexes shared with our 53 | their rows from shared complexes | our rows from shared complexes |
|---|---|---|---|---|---|
| AB645 | 645 | 25 | 19 | 499 of 645 | 625 of 940 |
| AB1101 | 1100 | 28 | 20 | 656 of 1100 | 636 of 940 |

Two consequences, both for later rather than here:

- These are the right sequence-only baseline for this project, not S1131.
- Two thirds of our rows sit in complexes they also train on, so a published AB645/AB1101
  number is **not** a clean external comparison for us, and any fusion that trains on them
  must keep the frozen cluster split or it will leak.

---

## Which ESM2 checkpoint

The paper never says, and `model/readme.txt` only says to download weights from Hugging Face.
It is pinned twice over by their own config:

- `bash_cross-validation.sh` sets `MODEL_LOCATE="./model/esm2_650m"`.
- the same script sets `HIDDEN_SIZE=1280`, wired straight into every head dimension. Of the
  ESM2 family only 650M has hidden size 1280 — 35M is 480, 150M is 640, 3B is 2560 — so
  nothing else would build.

So `facebook/esm2_t33_650M_UR50D`, confirmed on download: 33 layers, hidden 1280, rotary
positions, 652.4 M parameters. `check_encoder_assumptions.py` asserts the 1280, so a wrong
checkpoint fails loudly.

---

## What was run: their code, with the frozen encoder lifted out of the loop

Running `cross_validation/src_s1131/trainer.py` literally as written is not a bounded
debugging problem on this hardware — it is arithmetically out of reach. It calls ESM2-650M
four times per example on every step of every epoch. Over S1131's 711k residues that is about
4 CPU-hours per epoch, and about **1700 hours** for a 10-fold run. There is no local GPU path
either: the checkpoint is 2.6 GB in fp32 and the laptop's GTX 1050 has 2 GB total.

With `--freeze_backbone` the encoder is recomputing a constant, so it was cached once. S1131's
four sequence columns hold 4524 cells but only **1021 distinct sequences** — 112 PDB ids share
wild-type chains, and `b_mut == b` for 894 of the 1131 rows — so this is 127k residues of work
once rather than 711k per epoch per fold.

`check_encoder_assumptions.py` measures the three properties that make this an identity rather
than an approximation, against the downloaded checkpoint:

| property | why it matters | measured |
|---|---|---|
| no train-mode stochasticity | `SeqBindModel` holds the encoder as a submodule, so `model.train()` puts it in train mode | ESM2 ships both dropout probabilities at 0.0; two train-mode passes agree bit-for-bit, and match eval mode |
| no cross-example coupling | a batch-dependent embedding could not be cached per sequence | 0 BatchNorm modules in the encoder |
| padding invariance | their collator pads to the batch maximum, so a sequence appears at different widths across epochs | max 5.0e-06 absolute, 3.9e-05 relative, on real tokens |

Everything that decides the number is then **imported from the checkout, not copied**:

| their module | what it decides |
|---|---|
| `utils.common` | how S1131.csv becomes four sequence lists |
| `utils.data_split` | the fold assignment |
| `dataset.WrapperDataset` | tokenisation, padding, batching, shuffling |
| `model.SeqBindModel` | the architecture |
| `model_module.rope_attn` | the rotary cross-attention block |
| `litmodel.LitModel` | loss, metrics, optimiser |

The single substitution is rebinding the name `EsmModel` inside their already-imported `model`
module to a cache-backed stand-in. No upstream file is edited.

Hyperparameters are `bash_cross-validation.sh` verbatim (seed 3407, 120 max epochs, patience
20, lr 3e-5, batch 12, hidden 1280, 4 heads, dropout 0.1, MSE, frozen backbone) with
`trainer.py`'s defaults for what the script leaves unset. The shipped script names AB1101; the
README says to use it "with different args" for the other sets and `src_s1131` carries
identical defaults, so these are the S1131 values too.

### Deviations, all echoed at the top of every run

- frozen ESM2-650M served from a precomputed cache rather than recomputed in the loop
  (identity under `--freeze_backbone`, per the table above)
- `num_workers=0` rather than 4 (Windows spawn cost; the sampler runs in the parent process
  either way, so batch composition and order are unchanged)
- `--grad-checkpoint` recomputes the cross-attention blocks in backward when VRAM is tight.
  Compute-for-memory only: `checkpoint` re-runs under the same RNG state, so the dropout masks
  and therefore the gradients are unchanged, and **batch size stays at their 12**. That last
  point matters — see below.

---

## Two findings in their code, both load-bearing

### 1. Model selection runs on the test fold

`WrapperDataset.__init__` does

```python
self.train_dataset, self.val_dataset = self.get_dataset(train_idxes, test_idxes, ...)
```

so `val_dataset` **is** the test fold, and `get_val_loader` and `get_test_loader` return the
same object. `trainer.py` then monitors `val_pearson_corr` for both `EarlyStopping` and
`ModelCheckpoint`, loads `best_model_path`, and calls `trainer.test` on that same fold.

So the reported epoch is the one that scored best *on the test fold*, chosen out of up to 120.
This is the configuration that produced the published 0.84, so reproducing it is the task —
but it is not a held-out number, and it is not the number this project would have to beat.
`run_cv.py --protocol honest` therefore also runs the identical folds with the monitored split
carved out of the training rows, using their own unused `get_K_fold_with_test_generator`
(train_ratio 0.875). That helper reuses the same `KFold` call, so the ten test folds stay
row-for-row identical — asserted in code, not assumed.

### 2. The attention mask is filled with `1e-10`, not `-inf`

In `rope_attn.MutilHeadSelfAttn.self_attn`:

```python
attn = attn.masked_fill(mask == 0, 1e-10)
```

Padded keys therefore keep a real share of the softmax mass instead of being excluded. The
padded *values* are exactly zero — every consumer of the encoder output is an `AttnTransform`
whose softmax is `masked_fill_(~mask, -inf)` — so the leak contributes zero vectors, but it
still dilutes the real keys in proportion to how many pad positions the batch happens to have.

A prediction is therefore **not independent of its batch's padding width**. That is why the
batch size and iteration order are reproduced exactly rather than chosen for convenience, and
why shrinking the batch to fit a small GPU would not have been a neutral change.

---

## Their pinned environment cannot be installed on Colab, and why that turned out not to matter

ProtAttBA asks for `python==3.10, pytorch==2.1.2, pytorch-cuda=11.8`. That is exactly what the
local venv uses. Colab, however, now ships **python 3.13** with torch 2.11 and transformers
5.16, and their 2024 pins (protobuf 3.19.6, llvmlite 0.43.0, `rjieba`, `progressbar2`) have no
3.13 wheels, so `pip install -r requirments.txt` falls back to building from source and dies
with *"Getting requirements to build wheel did not run successfully"*. Their requirements file
simply cannot be installed on that runtime.

So only the packages the S1131 path actually imports were installed, and Colab's own stack
provided the rest. Nothing in `src_s1131`, `model_module` or `utils` touches xgboost,
matplotlib, seaborn, llvmlite, rjieba, progressbar2, lxml or protobuf.

That leaves a genuine risk: transformers 5.16 is two major versions past their 4.40.2, and if
its ESM2 implementation differed, every number downstream would be measuring a different
encoder. `crosscheck_embeddings.py` settles it by re-embedding three sequences — the shortest,
the median and the longest in the set — on the VM and comparing against the cache built locally
under their own pins:

```
reference built under python 3.10.11, torch 2.1.2+cu118, transformers 4.40.2
this runtime   python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1

  sequence 2  len 965  tokens 967
    mean   reference -0.000815   here -0.000815   |diff| 7.57e-11
    std    reference +0.275025   here +0.275025   |diff| 7.51e-11
    min    reference -9.370749   here -9.370749   |diff| 9.54e-07
    max    reference +3.176334   here +3.176334   |diff| 4.77e-07
    first_token_head max |diff| over 8 components 9.69e-08
```

Agreement to ~1e-10 on the distribution moments and ~1e-6 at the extremes is float
reassociation, not an implementation change. The forced library drift does not touch the
encoder, so the run is attributable.

One incidental note from the load report, identical on both stacks: `EsmModel` adds a
`pooler.dense` that ESM2's checkpoint does not contain, so it is randomly initialised on every
load. Their code only ever reads `last_hidden_state`, so it is never used — but it is the kind
of warning that is worth having checked rather than scrolled past.

---

## Hardware

| | |
|---|---|
| local | Intel i7-8650U, 4 cores, 7.9 GB RAM, GTX 1050 2 GB (driver 461.40, CUDA 11.2) |
| local GPU verdict | runs the cu118 build via CUDA minor-version compatibility, but PyTorch gets ~900 MB after the CUDA context and Windows display; OOM at batch 12 even with attention recomputed in backward |
| remote | Colab T4 16 GB, via this project's own `scripts/colab_run.sh` conventions |
| environment | separate Python 3.10 venv, torch 2.1.2+cu118, transformers 4.40.2, pytorch-lightning 2.2.5 — the project's own `requirements.txt` and `environment` are untouched |

Wall clock:

| step | time | where |
|---|---|---|
| clone, venv, deps, ESM2-650M download (2.6 GB) | ~25 min | local |
| embedding cache, 1021 sequences / 129k tokens | 159 min | local CPU, 43 tok/s |
| embedding cache, same work | ~3 min | Colab T4 |
| 10-fold CV, both protocols | see below | Colab T4 |

The 159-minute local extraction is the single number that most argues for doing this on the
GPU box: the same work is roughly 50× faster on a T4, and it is a prerequisite for every
subsequent run.

---

## Results

<!-- RESULTS -->
*Pending: the 10-fold runs are executing on Colab. This section is filled by `compare.py`,
which prints the published table, our two protocols, the training-mean floor, and per-example
agreement with their shipped predictions.*

---

## Reproducing

```bash
# local: environment, weights, cache, then the published-number check
python -m venv .venv_protattba && .venv_protattba/Scripts/pip install -r requirements_protattba.txt
git clone https://github.com/code4luck/ProtAttBA.git ProtAttBA && git -C ProtAttBA checkout 9adbf98
python fetch_esm2.py
python verify_published.py
python check_encoder_assumptions.py
python extract_embeddings.py --device cpu     # 159 min here; use --device cuda on a real GPU

# the run itself
python run_cv.py --protocol upstream --device gpu
python run_cv.py --protocol honest   --device gpu
python compare.py
```

On a machine that cannot hold it, the same thing on a Colab T4:

```bash
bash colab_run.sh upstream honest
```

---

## For the main ladder — suggestions, not implemented here

- **The cross-attention head is a candidate rung.** Antibody-as-query/antigen-as-key and the
  reverse, over frozen per-residue embeddings, is a different inductive bias from this
  project's current fusion: it never forms a single pooled pair vector before the interaction
  is modelled. `HANDOFF.md` §6 records that the mutation-token attention is load-bearing and
  that `dir_noattn` is worst on both splits, which points the same way.
- **The cache pattern is reusable.** A frozen encoder inside a training loop is pure waste;
  `extract_embeddings.py`'s keying on token ids with a hard error on a miss is a safe shape
  for that, and `check_encoder_assumptions.py` is the test that licenses it.
- **Their split is row-wise with no grouping.** 1131 rows over 112 PDB ids, shuffled by row,
  so a test row's complex is essentially always in training. That is the main reason their 0.84
  is not comparable to this project's cluster-grouped numbers, and it is a property of the
  benchmark rather than a defect in their code. Any fused model must keep being scored on the
  frozen cluster split.
- **AB645 / AB1101 were out of scope** and are the obvious follow-up; the harness takes a
  `--data-name` change and their `src_ab645` / `src_ab1101` modules are already in the
  checkout.

## Looking at runs: TensorBoard

The custom dashboard is replaced by TensorBoard. `scripts/to_tensorboard.py` is a one-way
mirror of `reports/` -- it reads the CSVs the trainers already write and changes nothing, so
it can be re-run at any time and nothing depends on it.

```bash
python scripts/to_tensorboard.py          # the live comparison: perturb_* and the E0 forests
tensorboard --logdir runs/tb              # http://localhost:6006
```

Older families are added by substring (`python scripts/to_tensorboard.py arch_ ab_`), and
`--all` mirrors all 131. The default is deliberately small: TensorBoard plots every run in
the log directory checked, so mirroring everything reproduces exactly the problem the old
dashboard had.

What is where:

| tab | shows |
|---|---|
| TIME SERIES / SCALARS | `val/pearson`, `val/rmse`, `train/loss` per epoch, one series per fold |
| | `test/*` on the run summary, indexed by fold, so step 3 is fold 3 |
| | `oof/*` pooled over all held-out rows, including `bias` and `rmse_debiased` |
| IMAGES | predicted-vs-true scatter with the fitted offset; per-complex Pearson bars |
| HPARAMS | every run in one sortable table -- this replaces the old Runs tab |
| TEXT | the per-fold results table |

Run naming is flat, `<run>` and `<run>__fold<k>`, not nested: TensorBoard derives a run name
from its path with `os.sep`, so a nested layout produces `perturb_v2_full\fold0` on Windows
and a backslash inside the filter box's regex is unusable. Flat, typing `perturb_v2_full`
selects the summary and all five folds, `perturb_v2_full$` just the summary, and `fold0`
overlays fold 0 across every run.

`oof/rmse_debiased` next to `oof/label_sd` is the one to watch: equal means the model is no
better than predicting that run's own mean, whatever its correlation says.
