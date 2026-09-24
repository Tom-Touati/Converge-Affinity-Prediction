# Why each choice was made

The assignment grades *justifiable choices* above everything else. This document states, for
each modelling, augmentation and evaluation decision: **what was chosen, what else was
available, why, and what evidence supports it** — including the decisions taken on judgment
with no evidence behind them, which are marked as such.

`docs/decisions.md` is the chronological log of decisions as they were taken.
`docs/ARCHITECTURES.md` is what was built and what it scored. This is the reasoning.

---

# A. Modelling

## A1. Predict ΔΔG by regression; build an ordinal head; do not use a 3-way softmax

**Chosen.** Regression on clipped ΔΔG as the primary target, with a CORAL ordinal head
implemented as an alternative.

**Why.** The argument for binning is real and is recorded in `thoughts.md`: classification
*neutralises measurement error*, because a label that is only accurate to ~1 kcal/mol does not
support a squared loss that tries to reproduce it exactly. Our own data agrees — within-complex
label sd has a median of **1.104 kcal/mol**, and SKEMPI's repeated `(complex, mutation)` pairs
(123 groups, 258 rows) disagree at a scale that caps any achievable correlation.

Regression stayed primary for one reason: **the evaluation is a ranking**. Per-complex
correlation and concordance need a continuous score, and binning to three classes discards the
ordering inside each bin — exactly the information the metric rewards. Binning would improve
the loss's robustness while degrading the thing being measured.

The ordinal head is the resolution rather than a compromise: it trains on which side of −0.5
and +0.5 a mutation falls (the robustness argument) while **one shared scalar drives both
thresholds**, so the output is still a continuous ranking (the metric argument). Two extra
parameters.

**Why not a 3-way softmax.** A softmax over three ordered classes is free to assign a mutation
high probability of *stabilising* and *destabilising* simultaneously, with neutral in between —
incoherent for an ordered target. CORAL's shared direction makes
`P(y > −0.5) ≥ P(y > +0.5)` hold for every input by construction, verified on every row.

**Evidence.** Head implemented and verified (ordered thresholds, monotonicity on all rows,
ranking identical to the underlying scalar). **The runs did not complete** — this is a
justified design, not a measured result, and is not claimed as one.

## A1b. A within-complex ranking loss was built, measured, and rejected

**Chosen.** Regression (MSE/Huber) as the training objective. A pairwise ranking loss exists
in the codebase (`src/fusion_v2.py::pairwise_rank_loss`) and is not used.

**Why it was the obvious thing to try.** The headline metric is a *within-complex*
correlation, and a within-complex pairwise loss optimises it directly rather than through a
proxy. It has three real merits, none of them aesthetic: it cancels the per-complex offset
exactly (a constant-per-fold predictor scores global ρ −0.36, so that shortcut is live); it is
invariant to per-complex monotone rescaling, which matters because **47.7 % of SKEMPI
temperatures are assumed rather than measured**; and it is the only objective that can consume
a measurement with an ordering but no usable value.

**Why it was rejected.** Measured, AbRank-style, at several mixing weights:

| objective | per-complex ρ | RMSE |
| --- | --- | --- |
| no-rank control (regression only) | 0.420 | ~1.5 |
| best ranking config, full pair coverage | 0.421 | — |
| ranking mixed at lower weight (`v2_h3_rankloss`) | 0.288 | 1.97 |
| pure ranking, no regression term (`rk_rank_only`) | **0.170** | **4.20** |

*n = 997, the pre-correction dataset — not comparable to the current 940-row tables, but the
comparison against its own control is internal and valid.*

At full pair coverage it **ties** the control, 0.421 against 0.420. Everything else is worse.

**The mechanism of the collapse is worth stating**, because it is not a tuning failure: a
ranking loss constrains only *order*, so the regression term is the only anchor on output
scale. Remove it and the model drifts anywhere order-preserving — RMSE 4.20 against ~1.5, on a
label whose own sd is 1.79.

**And the reason it could not have helped much.** 19,834 pairs from 997 rows is **19.9× the
examples and zero new information**. Pairs are a re-expression of the existing labels, not
augmentation. The efficiency argument was tested and not supported.

**What survives, and why it stays in the next steps.** The ranking loss is the only route to
the **80 non-binders and 86 censored rows** — measurements with a known ordering but no usable
ΔΔG. The censored rows are precisely the ones the regression target forced this project to drop
(997 → 940, and the forest fell 0.388 → 0.239 on the cluster split when they went). So it is
deferred on a narrow trigger: build it for **data recovery**, never for a score gain.

Two cost notes recorded at the time, for whoever builds it: the top 3 complexes hold 43.2 % of
pairs against 22.8 % of rows, so pairs need 1/C(n,2) weighting; and near-ties are noise, so
filtering to |Δ| > 1 keeps 10,703 pairs (54 %).

## A2. Frozen encoders, small trained heads

**Chosen.** ESM-2 and ProteinMPNN frozen; only heads of 15k–85k parameters train.

**Why.** 752 training rows per fold. Fine-tuning a 650M-parameter encoder on that is not a
risk judgment, it is arithmetic.

**Evidence.** The whole capacity ladder: an 810,886-parameter model scores **+0.200** and a
41,792-parameter one scores **+0.205**. Removing an entire head layer (31 % of a model) cost
nothing measurable. Capacity was never the binding constraint, so spending it on the encoder
would not have helped either.

## A3. ESM-2 650M for sequence, ProteinMPNN for structure

**Chosen.** As above. Alternatives considered: ESM-2 35M/150M, AntiBERTy, CurrAb, ESM-IF1,
SaProt.

**Why ProteinMPNN.** An inverse-folding model scores *this residue in this pocket*, which is
the conditional the mutation question asks. It is also cheap and runs on CPU.

**Why ESM-2 rather than an antibody-specific model.** Measured, not assumed:
**AntiBERTy lost to ESM-2 by 0.046** on a matched control (`st64_nopool_chem_abty` +0.166 vs
`st64_nopool_chem_esm` +0.212). CurrAb and ESM-2 35M also failed to clear zero as additions to
the forest. `thoughts.md` asks "how many of the proteins and antigens did the model actually
previously see?" — a fair worry, and the homology-split result is the closest answer available:
networks built on these embeddings retain **34 %** of their score when homologues are withheld
against the forest's **54 %**, so a substantial part of what the embeddings contribute *is*
recognition rather than physics.

## A4. Represent the mutation as a difference of embeddings at the mutated residues

**Chosen.** `site_mean(mutant) − site_mean(wild-type)` over the mutated residues, with the
mutant sequence **re-embedded** rather than edited in latent space.

**Why.** ΔΔG is a contrast between two states, so the representation should be a contrast too.
Re-embedding is the honest version: an edited latent vector assumes the encoder is linear in
the substitution, which it is not.

**Why pool at the mutated residues rather than over the interface.** Measured:
`area_concat` (+0.227) against `cat128_reg2_l1` (+0.293) — pooling structure over the whole
binding area costs ~0.05. ProteinMPNN's signal is local; averaged over ~85 residues it washes
out and what survives is nearly constant per complex.

**Why multi-point rows average their k tokens rather than being dropped.** 272 rows, 29 % of
the data, and 7 complexes have no single-point row at all. Averaging is a choice and not
obviously the right one — ΔΔG is not additive in the mutation count
(corr(k, ΔΔG) = −0.151 overall, −0.292 among multi-point rows) — but dropping them would remove
seven complexes entirely. **Evidence it is imperfect:** multi-point rows are 80 % worse on MAE
than single-point at the same correlation (ERROR_ANALYSIS §17).

## A4b. Dimensionality reduction: PCA, a learned grouped projection, and neither

ESM-2 emits 1280 dimensions per residue against 752 training rows per fold. Something has to
reduce it, and the choice is not free — it decides what reaches the model before the model
sees anything. Three approaches were measured.

**1. Fold-local PCA** (the default). Fitted on the training folds only, separately per side,
refit for every fold so nothing leaks. 128 components keep ~81–90 % of variance.

**2. A learned grouped projection** (`GroupedReduce`). No PCA at all: the raw 1280-d embedding
is reduced *inside* the network as ten independent 128 → 16 blocks. Block-diagonal rather than
dense for an arithmetic reason — a dense `Linear(1280, 160)` is 204,800 parameters against
**20,480**, and on 752 rows the dense version is not a reduction at all.

**3. No reduction** — raw embeddings straight into a forest.

| reduction | model | per-cx r | seeds |
| --- | --- | --- | --- |
| fold-local PCA-128 | site-token net | +0.219 / +0.212 | 1 + 1 |
| **grouped projection, no PCA** | the same net | **+0.274 / +0.199** | 1 + 1 |
| grouped projection + heavier reg | the same net | +0.193 | 3 |
| PCA-128 | pooled-delta MLP | +0.157 | 1 |
| PCA-256 | pooled-delta MLP | +0.159 | 1 |

Averaging the two seeds each: **grouped projection +0.236, PCA-128 +0.216.**

**Why PCA is kept anyway.** The +0.020 is inside the noise, and the two seeds of the grouped
projection are **+0.274 and +0.199** — a 0.075 spread on an identical configuration, which is
the widest we measured anywhere. The grouped projection was reported as a +0.062 win before its
replicate arrived; it did not survive. PCA-128 is retained because it is cheaper, because it is
fitted per fold with an auditable variance-explained figure, and because nothing measured beats
it outside the noise.

**PCA-128 versus PCA-256** makes no difference either: +0.157 against +0.159 in the MLP family.

**In the forest, reduction actively hurts.** Pooled ESM + ProteinMPNN scores **0.366** raw and
**0.303** through PCA-128 — reduction costs 0.063. The forest's `max_features="sqrt"` already
samples a subset per split, so PCA removes information the tree ensemble was handling on its
own. Separately measured earlier: raw ESM concatenation scores 0.262 against a 0.500 control,
which is dilution rather than reduction failure — 3,853 ESM columns against 49 informative
ones — and PCA-32 gives 0.473 against a matched 0.499 control, delta −0.026, CI
[−0.065, +0.013], **not clearing zero**.

**The conclusion across all three.** How the embedding is reduced changes results by less than
the seed noise in the network, and reduction is a net negative in the forest. This is another
instance of the pattern in §A2 and §C1 — the representation pipeline is not where the
performance is.

## A5. The random forest is a BASELINE, not the submission

**Chosen.** `E0a_rf_handcrafted` — chemistry + interface geometry + ProteinMPNN — is built,
tuned and reported as the reference the multimodal model is measured against. The submitted
model is `l1_gated` ([docs/MODEL.md](MODEL.md)).

**Why a baseline at all, and why this one.** A multimodal deep model on 940 rows is only worth
building if it beats what the same features support without one. The forest is the strongest
thing that can be built on this data cheaply: it consumes both modalities (chemistry is
sequence-derived, geometry and ProteinMPNN are structural), it trains in ~20 seconds on CPU,
and tree ensembles are the appropriate hypothesis class for a few hundred rows and 49
informative columns. Anything weaker would have been a straw man.

**It is a demanding baseline, and that is the point.** It scores **+0.381** against the
network's **+0.300** — about 2.5× the seed spread, one of the few gaps in this project that
clears the noise — and retains **54 %** under the homology split where the network retains
**34 %**. A baseline that the deep model comfortably beat would have told us far less.

**What the comparison establishes.** Three things, none of which would be visible without it:

- On 752 training rows per fold, **49 informative columns outperform a learned
  representation**. That is the headline finding about this problem, not about this model.
- **The networks are leaning on homology** and the forest much less so — visible only because
  both were run on both splits.
- The neural model is nevertheless **better where it matters for design**: balanced accuracy
  0.463 against 0.444, and stabilising recall **0.25 against 0.06**. The forest finds 7 of 126
  affinity-improving mutations. That asymmetry is the strongest argument for keeping the
  multimodal model in the picture at all, and it is invisible without the baseline to contrast.

**Why the forest is not submitted despite scoring higher.** The assignment asks for a
multimodal sequence-and-structure model and states that state-of-the-art performance is not
expected. Submitting the baseline because it wins the headline metric would answer a different
question, discard the fusion work that the task is actually about, and hide the class-recall
result above. The forest's number is reported prominently instead — in the results table, in
the limitations, and here — because a submission that buries its own baseline is not worth
reading.

**The baseline's own defect, for completeness.** The forest has the worst stabilising recall in
the project. Quantile calibration lifts it 0.06 → 0.29 at zero cost to ranking (§C5), so the
version anyone should actually use is the calibrated one.

## A6. Fusion by feature concatenation, after measuring five alternatives

**Chosen.** Gating or plain concatenation.

**Why.** Not aesthetics — the control. `st64_noattn` (+0.212) is the same network with fusion
deleted, and **four of five cross-attention variants score at or below it** while costing twice
the parameters. Gating (+0.300) and concatenation (+0.293) are the two cheapest mechanisms and
the only two that clear the control convincingly.

Antibody↔antigen cross-attention — the mechanism `thoughts.md` proposed and ProtAttBA's own —
was measured as family A and reaches +0.112 to +0.200, below the simpler families.

---

# B. Augmentation and regularisation

## B1. Reverse-mutation augmentation, p = 0.5

**Chosen.** With probability 0.5 per row per epoch, swap wild-type and mutant and negate the
label.

**Why.** ΔΔG is antisymmetric by definition: reversing a mutation negates its free-energy
change. This is free, exactly correct label information — not a heuristic — and it attacks the
class imbalance at its root. **70 % of rows are destabilising**, so a model can profit from
guessing "destabilising"; augmentation removes that incentive.

**Evidence it does what it claims** (`audit_augmentation.py`, measured not assumed): the
realised rate is 51 %, and the training label mean moves from **+0.720 to −0.031**. Every row
is seen both ways — the chance a row is never reversed across 25 epochs is 3×10⁻⁸. Applied to
training only; validation and test loaders are built with `augment=False`.

**Its known cost.** Earlier measurement: per-complex ρ 0.487 → 0.416, but *balanced* sign
accuracy 0.512 → 0.681. It trades ranking for balance. Given §14 — the model predicts the
average stabilising mutation as destabilising — that is the right trade for this task, but it
is a trade and is recorded as one.

## B2. Input noise and feature dropout, shared mask

**Chosen.** Gaussian noise at 0.10–0.40 × per-feature sd; feature dropout 0.20–0.50, mask drawn
once per width and **shared between the wild-type and mutant branches**.

**Why shared.** The model consumes `mutant − wild-type`. A mask applied to both gives
`mask·(mt − wt)` — it zeroes features *of the difference* rather than corrupting it.

**Evidence, and a defect found by measuring it.** The mask behaves as intended. **The noise
does not.** It is drawn fresh on each branch, so it does not cancel — it compounds by √2 and
reaches **0.8× the delta's own standard deviation** at the default setting, and would exceed
the signal at the heaviest setting tried. The code comment claimed both were shared; it was
true of the mask and false of the noise.

This is the most likely explanation for why heavier regularisation kept *costing* accuracy
(−0.037 and −0.043 on two architectures) without reducing seed variance. **Sharing the noise
draw is the top untried lever** and is a two-line change.

## B3. Gradient clipping at 5.0, later made configurable

**Chosen.** Global-norm clip, threshold now a flag.

**Why it became a problem.** The threshold sat at 5.0 while the measured gradient norm was
5.0–5.3, so roughly half of all steps were rescaled and half were not — the most intermittent
setting available, and one that differed systematically between arms, since heavier
regularisation produces noisier gradients and clips more often.

**Evidence, which contradicted the expectation.** Turning clipping off (threshold 20) *cost*
0.028 (`cat128_reg2_l1` +0.293 → `cat128_reg2_l1_noclip` +0.265). Clipping was helping. The
confound was real, the direction was the opposite of the one assumed, and it is now a flag so
the question can be asked rather than inherited.

## B4. Labels clipped to ±4 kcal/mol

**Chosen.** Clip in training *and* in scoring, so every model is compared on one truth.

**Why.** `thoughts.md` flags "extreme values in positive ddG". The tail is real and the model
cannot reach it — §7 shows the ten worst residuals are dominated by +7.2 to +7.9 hotspots
predicted at +1.5 to +2.4. Clipping stops a handful of unreachable rows dominating a squared
loss.

**Why it must also apply to scoring.** It caught a real error: the forest was being compared at
RMSE 1.532 against networks at 1.5–1.6 while being scored on *unclipped* labels. On the common
truth it is 1.335. `report_runs.py` now enforces one truth for every run.

## B5. Augmentations considered and not done

- **Rotating the 3D structure** (`thoughts.md`). ProteinMPNN's `encoder_h_V` is computed from
  distances and relative geometry and is already rotation-invariant, so rotation augmentation
  would produce identical features. Correctly motivated for a coordinate-consuming model; a
  no-op for this one.
- **Predicting the mutant structure and using it as a feature** (`thoughts.md`). This is the
  right instinct — §"the mutant has no structure" in `docs/ARCHITECTURE.md` identifies it as
  the sharpest limitation, since the structure branch is *identical* for every mutation of a
  complex. Not done: folding 940 mutants was outside the compute budget, and a predicted
  structure's error would be correlated with exactly the buried, tightly-packed sites the model
  already fails on (§16, valine). Listed as a next step rather than attempted badly.
- **FoldX pseudo-labels.** Deferred behind a trigger: only worth it if a learning curve shows
  data volume is the constraint. Everything in Part III says it is, so this is now live.

---

# C. Evaluation and error analysis

## C1. Per-complex correlation is the headline, not pooled

**Chosen.** Mean within-complex Pearson/Spearman over complexes with ≥5 rows.

**Why.** Measured, and it is the single most important methodological finding here: predicting
**each complex's own mean** — a model that never looks at the mutation — scores **+0.672 pooled
Pearson, 1.144 RMSE, 0.580 balanced accuracy**, beating every model in the project on all
three, while scoring **+0.000** per complex. Between-complex variance dominates, so pooled
metrics largely measure whether a model can identify the complex.

**Consequence.** Every table leads with per-complex, and `report_runs.py` prints that floor
beneath every comparison so a reader can see what the pooled column is worth.

**The cost of this choice, stated.** Only **32 of 53 complexes** have ≥5 rows, so the headline
is computed over 60 % of complexes, systematically the larger ones, and it weights a 7-row
complex the same as an 87-row one. Row-weighting or a minimum-spread filter would both be
defensible; neither is applied, and that is a judgment call rather than an evidenced one.

## C2. Splits: by complex, and by homology cluster

**Chosen.** Report both. `frozen5` withholds whole complexes; `cluster` withholds homology
clusters.

**Why both.** `thoughts.md` asks for "5 fold based on homology clusters, very tough standard".
It is the right standard and the project measured why: 42 of 54 complexes gain a TM > 0.8
training twin under complex grouping, against 0 of 54 under cluster grouping. But the cluster
split yields only 4 usable folds and much smaller test sets, so the two are not directly
comparable and both are reported with that stated.

**What it bought.** The clearest architectural finding in the project: the forest retains
**54 %** under the homology split, the networks **34 %**. Without the harder split, the
networks look merely worse; with it, they are revealed to be leaning on homology.

## C3. Three seeds minimum, and report the spread

**Chosen.** Report ensemble-of-seeds *and* per-seed mean with max − min.

**Why.** Measured: seed spread reaches **0.117** per complex and **0.376** pooled within a
single fold. The entire architectural ladder fits inside that. Two conclusions in this project
were drawn from partial folds and **reversed** when the last fold arrived.

**The honest state.** Only **8 of 45** configurations have three complete seeds; **37 have
one**. Their individual numbers are not interpretable at the resolution the tables print them,
which is why `ARCHITECTURES.md` opens with that warning rather than burying it.

## C4. Metrics beyond correlation

**Chosen.** Report per-complex r and ρ, count of complexes with *negative* within-complex
correlation, pooled r, RMSE raw and debiased, sign accuracy raw and **class-balanced**,
within-complex pairwise concordance, and 3-class balanced accuracy with per-class recall.

**Why so many.** Each hides a different failure. Accuracy rewards guessing the majority
(classes are 13/34/53). Correlation hides the forest finding 7 of 126 stabilising mutations.
RMSE is dominated by between-complex structure. Concordance — "of two mutations on this
complex, does the model order them correctly" — is the question a designer actually asks, and
no single number substitutes for it.

## C5. Calibrate before thresholding

**Chosen.** Cut the score at its own quantiles, matched to training class frequencies, rather
than at the label edges.

**Why.** The forest compresses into 50 % of the label's sd, so its predictions rarely reach
below −0.5 and cutting at the label edges starves the minority class. Calibration is
**monotone**, so per-complex correlation is provably unchanged.

**Evidence.** Stabilising recall **0.06 → 0.29**, balanced accuracy **0.439 → 0.500**,
per-complex Spearman +0.3613 → +0.3613. Free.

## C6. Verify the pipeline before believing the ceiling

**Chosen.** Check alignment before attributing a ~0.3 ceiling to the model.

**Why.** Three different bugs — an off-by-one from an insertion code, a chain mix-up, and
applying the substitution to SEQRES instead of the ATOM-derived sequence — all look identical
from a loss curve, and all would mean the model was learning from noise.

**Evidence.** Across all 940 rows and 1,726 mutated sites: **zero** residue mismatches, WT and
MT sequences differ only at the recorded sites, and `‖t_mut − t_wt‖` puts the mutated sites in
the top-k on **97 %** of sides at a median of **22×** the median residue. `make align`.

A real defect was found and fixed by this class of check: `Row.wt_aa` was indexed by
rank-within-a-side, so 82 of 940 rows carried another mutation's residue in their BLOSUM
features.

## C7. Slice the errors by every descriptor, not just the ones that occurred to us

**Chosen.** Correlate every available descriptor against every model's signed and absolute
error (`scripts/error_drivers.py`).

**Why.** Hand-picked slices find what the analyst already suspects. `thoughts.md` lists four
worries — unbalanced labels, extreme positive ΔΔG, multi-point vs single, antibody-side vs
antigen-side — and a systematic sweep can confirm those *and* surface what was not on the list.

**What it surfaced that hand-picking did not.** Signed error correlates with true ΔΔG at
**ρ ≈ −0.8 for every architecture**, and with deviation from the complex's own mean at
**ρ ≈ −0.58**. Regression to the mean is not one failure among several — it is the dominant
error structure, and it explains the class bias, the valine failure and the compressed output
range as one phenomenon. Rows in larger complexes are systematically under-predicted
(ρ ≈ −0.33, all four models), which is representation bias changing the *direction* of error.

It also showed the two networks agree on which rows are hard at **0.866** while the forest
agrees with them only **0.42–0.54** — which is why the fusion ladder went nowhere and why
blending the forest with a network beats both.

## C8. What `thoughts.md` asked for that is still open

- **Test-retest error as an explicit ceiling.** The ingredients exist (123 repeated
  `(complex, mutation)` groups, 258 rows; within-complex label sd median 1.104) but the
  implied ceiling on achievable correlation is not computed and reported as a line on every
  chart. It should be.
- **Difficulty scored by similarity to training points.** Per-complex size and label spread are
  analysed; nearest-neighbour distance in feature space to the training fold is not.
- **Train vs validation vs test loss curves.** Logged per epoch into `history.csv` and mirrored
  to TensorBoard, but not presented as a figure in any document.
- **Generalisation check on another dataset.** AB645/AB1101 rows are built, cached and
  leakage-filtered (`src/perturb/extra_rows.py`, homology models of our own complexes removed)
  but have never been trained or tested on.
