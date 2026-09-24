# What I would do next, and why

Ordered by expected value per unit of effort. Each item names the measurement that motivates
it, because this project's own evidence is unusually specific about where the constraint is.

**The constraint is not capacity, architecture, or label noise.** An 810,886-parameter model
scores +0.200 and a 41,792-parameter one scores +0.205. Four of five cross-attention variants
score at or below a model with the fusion deleted. Measurement noise is 0.240 kcal/mol,
implying a ceiling of 0.979 on per-complex Pearson against our 0.381. What remains is **data
volume and what the representations were pretrained to know** — 752 training rows per fold, 32
scorable complexes, and embeddings that retain 34 % of their score when homologues are
withheld against handcrafted features' 54 %.

Everything below follows from that.

---

# Tier 1 — cheap, and each closes a loop this project opened

## 1. Share the input-noise draw across branches

**Two-line change.** The noise is currently drawn independently for the wild-type and mutant
branches, so it does not cancel in `mutant − wild-type` — it compounds by √2 and reaches
**0.8× the edit's own standard deviation** (edit sd 0.128; 0.25 × 0.282 × √2 = 0.0998). At the
heaviest regularisation tried it would exceed the signal.

**Why it matters beyond the fix:** it is the most credible explanation for the otherwise
strange result that heavier regularisation *cost* 0.037–0.043 on two architectures while
reducing seed variance on neither. Measured by `audit_augmentation.py`; see
`docs/JUSTIFICATIONS.md §B2`.

## 2. Save checkpoints and read the fusion gates

No weights were saved, so the obvious question about a gated model — *what does it gate on* —
cannot be answered from the artifacts. One re-run fixes it, and it converts the fusion
mechanism from a score into a figure: how much sequence versus structure, for which mutation
types, for which complexes.

Do this together with **fixing the gated projection bug** (`gated_fusion` was missing from the
condition that builds the structure projection, so gated runs shared one map between ESM-2 and
ProteinMPNN). Both are one run.

## 3. Blend the forest with the network

They agree only **0.42–0.54** on which rows are hard, and four of four z-scored blends beat the
forest — best +0.424 against +0.397. The weight was not tuned on held-out data, so validate it
properly; but it is better evidenced than any architecture change in this project, and it costs
one cross-validation loop.

## 4. Give the network the forest's interface-geometry columns

The largest single gain measured anywhere here was adding 26 chemistry columns to the network
(+0.191 → +0.266). The forest's other block — rSASA, burial, contact counts — has never been
given to the network. It is the untried half of the one intervention that demonstrably worked.

## 5. Normalise the target within cluster — measured, and it moved

**Reported result: gated `chem+geom`, cluster-scaled, +0.241 → +0.299.** Standardising the
regression target within homology cluster rather than globally, so the model is asked for the
*within-cluster* ranking directly instead of having to learn each cluster's offset and spread
on the way there.

That it helps is consistent with the largest effect in `ERROR_ANALYSIS.md`. Regression to the
mean is the dominant error structure at **ρ ≈ −0.8** against the label, and it survives
*within* a complex at **ρ ≈ −0.58** against deviation from the complex mean — so it is eating
the very quantity the headline metric is built from. The complex's own mean ΔΔG correlates
with signed error at **−0.547** for the submitted model. A model spending capacity on
reproducing between-cluster location and scale is spending it on the part of the problem that
the complex-mean floor already solves for free (+0.672 pooled, §8 of the error analysis).

**The caution is specific, and it is the reason this is not yet a headline number.** The
scaling statistics must be fitted on **training folds only**, exactly as the PCA bases and the
chemistry standardisation already are. Fit them on all rows and each test complex contributes
its own mean and spread to its own normalisation — which is the complex-mean floor smuggled in
as preprocessing, and that floor scores **+0.672 pooled Pearson** while never looking at the
mutation. A leak here would look like a large, clean gain. Before this is reported as a
result it needs: the statistics demonstrably fold-local, three seeds, and the standard table
from `scripts/final_table.py`.

**Status.** The +0.241 → +0.299 pair is from a run that is not in `results/oof/`, on one
configuration. By this project's own rule — established after two conclusions drawn from
partial folds were both wrong — that makes it a lead worth an evening, not a claim.

---

# Tier 2 — an antibody–antigen-specific joint pretraining

**This is the main proposal, and it has two halves that only work together: more data, and a
sequence–structure alignment trained on antibody–antigen interfaces specifically.**

The constraint is 752 labelled rows. Everything in Tier 1 spends that budget more carefully;
this changes how much labelled data is needed, by moving representation learning onto
**unlabelled antibody–antigen structure**, of which there is roughly an order of magnitude
more than we have labels for.

It also answers a defect this project measured rather than assumed. Our two encoders occupy
**unrelated representation spaces**, and the only thing joining them is a pair of
`Linear(128 → 64)` maps trained on 752 ΔΔG values. That is very little supervision for a large
alignment problem, and the consequence is visible: the model's performance tracks structural
proximity to training at Spearman **+0.494**, and collapses to **+0.060** on complexes with no
structural relative. It has learned to recognise, not to generalise.

## 6. Why it must be antibody–antigen-specific, not generic protein

A generic sequence–structure alignment — over the whole PDB — would be dominated by globular
protein cores, which is the wrong regime three times over:

- **Antibody binding is loop-mediated.** The CDRs are hypervariable, conformationally flexible
  and solvent-exposed. A generic objective sees them as a small, noisy minority; here they are
  the entire question.
- **The framework is near-constant across antibodies.** Two unrelated antibodies sit at 70–80 %
  sequence identity, which is why this project's homology clustering links antibodies at 90 %
  rather than the usual 30 % (`src/splits.py`). A generic contrastive objective would spend
  most of its capacity on framework it could memorise, and would call two unrelated antibodies
  the same thing.
- **Epitopes are discontinuous.** The antigen side of the interface is assembled from residues
  distant in sequence. An alignment trained on contiguous local neighbourhoods will not
  represent that.

Our own encoder comparison is consistent with this and is currently under-explained.
**AntiBERTy lost to ESM-2 by 0.046** — but in a setup where the antigen stayed ESM-2 and *one
shared projection served both sides*, so an antibody-specific prior was being asked to live in
a space fitted to a general one. That comparison should be re-run after the alignment below
exists, not before.

## 7. The pretraining task

**Data.** SAbDab (~7,000 antibody structures, continuously updated), restricted to those with a
bound antigen; AbDb for cleaned, numbered Fv pairs; and the antibody–antigen subset of the PDB
beyond SKEMPI. Leakage control is non-negotiable and this project already has the machinery for
it: exclude, per fold, any pretraining complex whose PDB id or homology cluster appears in that
fold's test set (`src/fusion/sampling.py::exclude_leaked`, and the cluster definition in
`src/splits.py`).

**The unit is an interface neighbourhood, not a residue.** For each contact across the
interface, take the local neighbourhood on both sides. That makes the positive pair
*"this CDR loop against this epitope patch"* rather than *"this residue in this fold"*, which
is the relation ΔΔG actually depends on.

**Objective — two terms, and the second is the one that diagnoses.**

1. **Contrastive alignment.** Project the ESM-2 view and the ProteinMPNN view of the same
   neighbourhood into one space and train symmetric InfoNCE so matched views attract:

   ```
   z_s = f_seq(ESM(nbhd))     z_t = f_str(MPNN(nbhd))
   L   = InfoNCE(z_s, z_t) + InfoNCE(z_t, z_s)
   ```

   **Negative sampling is where the antibody specificity is enforced.** Negatives drawn from
   *other complexes* teach complex identity — the exact failure in ERROR_ANALYSIS §8, where
   predicting a complex's mean scores +0.672 pooled and +0.000 per complex. Negatives drawn
   from **within the same complex**, and ideally from **within the same CDR**, force the
   representation to be position-specific. That is a design decision this project's evidence
   makes for us.

2. **Bidirectional cross-modality prediction.** Train ESM → MPNN and MPNN → ESM on the same
   neighbourhoods under a regression loss. What the model *fails* to predict is the useful
   part: if structure is near-perfectly predictable from sequence, the structure branch adds
   nothing a language model does not already encode — a direct answer to a question our fusion
   ladder could only answer as a score. If it is not, the **residual**
   `MPNN(x) − predict_from_seq(x)` isolates the structure-specific information, and *that*
   is what should be fused rather than the raw embedding.

   This is the sharper version of the fusion question. The ladder compared six ways of mixing
   two representations and found the mechanism did not matter; this asks how much
   non-redundant information there is to mix, and would **explain** that null rather than
   record it.

**Then** freeze `f_seq` and `f_str` and drop them into `cat128_reg2_l1` in place of the two
projections currently fitted to 752 labels. Everything downstream is unchanged, so the
comparison is clean.

## 8. How to tell whether it worked

The lesson of this project is that a mechanism means nothing without a control, so:

- **Primary:** the same model with the aligned projections, against the same model with
  randomly initialised ones, at **three seeds**. Anything smaller than the seed spread is not
  a result.
- **The test that actually matters:** performance on the **hard tier** — the 19.8 % of rows
  with no near-identical training twin, where the model currently scores **+0.060**. If
  pretraining is doing what it is supposed to, that number moves. If only the easy tier
  improves, the alignment has learned recognition again and should be rejected.
- **A diagnostic that costs nothing:** re-measure Spearman(proximity, per-complex r). It is
  +0.494 now. Pretraining that generalises should *reduce* it.
- **Reconstruction R² sliced by burial**, from term 2. Prediction: good on exposed residues,
  poor on buried and tightly packed ones — which are exactly the rows the model fails on
  (§16, valine, under-predicted by 1.4 kcal/mol and ranked backwards). If that holds, it both
  validates the diagnostic and names the next target.

**Risks, stated.** Alignment is not ΔΔG prediction — a representation can be perfectly aligned
and carry nothing about mutational sensitivity. Contrastive objectives are sensitive to the
negative distribution, which is why the sampling above is specified rather than left to a
default. And SAbDab is itself redundant: many entries are the same antibody against the same
antigen, so it must be clustered before it is counted as data, exactly as our 53 complexes
were.

# Tier 3 — more data, which everything else points at

## 9. Synthetic mutations — FoldX as a mutant-structure generator, and as a labeller

Deferred earlier behind a trigger: *build it only if a learning curve shows data volume is the
constraint*. Everything in `ERROR_ANALYSIS.md` Part III says it is. This is the item that
turns that finding into training rows, and it has three distinct uses that are worth separating
because they fix different things and cost different amounts.

### 9a. FoldX ΔΔG as one input column — the cheapest, do it first

One scalar per row, into the forest and into the network's `chem` block. `skempi-foldx` ships
precomputed values, so FoldX need not run at all for the rows we already have.

The reason to expect it to pay is in `docs/ABSCI_DATASET.md §9.2`. On split-by-complex
fivefold CV over full SKEMPI v2.0 (n = 5,729), FoldX has the **best Spearman of any individual
model, 0.526** — above a pretrained geometric GNN's 0.525 — while having the **worst MAE and
RMSE in the table**. It ranks well and is calibrated badly. Our headline metric is a rank
correlation, so we want exactly the half of FoldX that is good. Note the caveat before
budgeting on it: that table is full SKEMPI, not the antibody–antigen subset, and empirical
force fields are generally reported as weaker on AB/AG interfaces than on globular ones. The
honest version of this item is *measure FoldX on our 940 rows first*, as its own row in
`scripts/final_table.py`, beside the complex-mean floor.

### 9b. Synthesise the mutant *structure* — this is the one that closes an architectural hole

`docs/MODEL.md` names the sharpest limitation of the submitted design in one line: **there is
no mutant structure**. ProteinMPNN sees the wild-type backbone only, so `z_str` is *identical
for every mutation of a given complex*. The structure modality cannot currently distinguish two
mutations on the same complex at all — it contributes a per-complex constant, which is precisely
the thing the per-complex metric is designed to cancel.

FoldX `RepairPDB` → `BuildModel` produces a mutant structure: side chains repacked on a fixed
backbone. Re-run ProteinMPNN on it and the structure term becomes a **delta**,
`site_mean(proj_str(struct_mt)) − site_mean(proj_str(struct_wt))`, mirroring the sequence edit
in step 3 of the forward pass rather than sitting beside it as a constant. That is a genuine
architectural change reachable without any new labels, and it is testable against a control
that is already measured: the no-fusion model at +0.212 and the submitted model at +0.293.

**Where it will be least trustworthy is where we most need it.** BuildModel repacks side chains
and does not model backbone motion. Our own error analysis says absolute error rises with
interface contacts (ρ 0.317 for the submitted model) and with ΔrSASA on mutation (0.283) — the
buried, tightly-packed sites where a fixed backbone is the worst assumption, and where Gly and
Pro substitutions change what the backbone can do. So the per-tier and per-descriptor breakdown
matters more for this item than the headline does.

### 9c. A pseudo-labelled mutation pool — pretraining, not a target

Enumerate mutations at interface positions SKEMPI never measured, label them with FoldX, and
use them to **pretrain**, then fine-tune on the real labels. Three conditions, each of which
this project has already paid to learn:

- **Do not use a learned labeller trained on SKEMPI.** ThermoMPNN, GearBind or our own forest
  would pseudo-label rows whose information came from complexes sitting in our *test* folds.
  FoldX is an empirical force field, not fitted to SKEMPI, which is the specific reason it is
  the right tool for this job and a learned ΔΔG predictor is not.
- **Pseudo-labels cap the student at the teacher.** Trained *as the target*, the ceiling is
  FoldX. As a pretraining task with a real-label fine-tune, it buys a representation rather
  than a score, which is the only form worth doing at 752 training rows per fold.
- **Do not reintroduce the class imbalance.** Reverse-mutation augmentation moved the training
  label mean from **+0.720 to −0.031** and that is most of why the submitted model finds three
  times as many stabilising mutations as the forest (0.19 against 0.06). A synthetic pool
  enumerated naively will be overwhelmingly destabilising and will hand that back.

### The acceptance test, for all three

**The hard tier, not the average.** Complexes with no structural relative in training score
**+0.060** against **+0.367** for those with a near-identical twin, and the easy tier is 75 % of
test rows (§13). Synthetic data drawn from the complexes we already have adds rows in the
region that is already easy. If FoldX augmentation moves the average and leaves the hard tier
where it is, it bought retrieval, not generalisation — and §11 below is the finding that says
that is the likely outcome unless the synthetic complexes are new *structure space*.

## 10. The AB645 / AB1101 rows already built

261 rows are built, cached and leakage-filtered (homology models of our own complexes removed),
and have never been trained on. They need an ESM re-extract that includes their sequences.

## 11. Target new *structure space*, not new measurements

The sharpest data finding here: **how often a complex appears barely predicts performance**
(Spearman +0.04 / +0.09), while **how structurally close it is to training does** (+0.17 /
+0.37). Ten complexes with no structural relative in training score 0.190 against 0.404 for
those with three or more.

So "more data" should mean **complexes in untested regions of structure space**, not more
mutations on complexes we already have. That distinction is worth making to whoever funds the
next assay plate.

## 12. A ranking loss, for data recovery only

Measured and rejected on score: at full pair coverage it ties its control (0.421 vs 0.420) and
pure ranking collapses to 0.170 because the regression term is the only anchor on output scale.
The efficiency argument fails too — 19,834 pairs from 997 rows is 19.9× the examples and zero
new information.

**But it is the only objective that can consume a measurement with an ordering and no usable
value**, which is the 80 non-binders and 86 censored rows — and the censored rows are exactly
the ones the regression target forced this project to drop. Build it for recovery, never for a
score.

---

# Evaluation work, which would change how all of the above is read

## 13. Select and report on the hard tier

**75 % of test rows have a near-identical training twin** (median TM 0.991); only 19.8 % are
genuinely hard. On those the forest scores +0.270 and the network +0.117. A model intended for
novel targets should be **selected** on the hard tier, not on the average — selecting on the
average selects for retrieval.

## 14. Weight the per-complex mean, or state that it is unweighted

It currently treats a 7-row complex with 0.78 label spread identically to an 87-row one
spanning 2 kcal/mol, and it is computed over **32 of 53 complexes**. Row-weighting, or a
minimum-spread filter, would both be defensible. Neither is applied; that is a judgment call
and should be stated as one.

## 15. Three seeds minimum, and publish the spread

Only 8 of 45 configurations here have three complete seeds. The measured spread reaches 0.117
per complex and 0.376 pooled within a single fold. Most published ΔΔG comparisons would not
survive this standard, which is itself worth saying.
