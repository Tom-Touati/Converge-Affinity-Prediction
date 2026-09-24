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

---

# Tier 2 — pretraining, which is the principled answer to the real constraint

The constraint is 752 labelled rows. Every item above spends that budget more carefully; these
change how much labelled data is needed in the first place, by moving representation learning
onto **unlabelled** structures — of which there are thousands (SAbDab holds ~7,000 antibody
structures; the PDB holds far more protein–protein interfaces).

This is also a direct response to a defect this project measured rather than assumed: our two
encoders occupy **unrelated representation spaces**, and the only thing aligning them is a pair
of `Linear(128 → 64)` maps trained on 752 ΔΔG labels. That is a very small amount of
supervision for a very large alignment problem.

## 5. CLIP-style contrastive alignment of sequence and structure

**The idea.** Take a residue (or a local neighbourhood) in a known complex. It has an ESM-2
embedding and a ProteinMPNN embedding — two views of the same physical object. Train two
projection heads so that **matched views attract and mismatched views repel**, with the usual
symmetric InfoNCE objective over a batch:

```
z_s = f_seq(ESM(residue i))          z_t = f_str(MPNN(residue i))
L   = InfoNCE(z_s, z_t) + InfoNCE(z_t, z_s)      temperature-scaled cosine
```

Negatives are other residues in the batch. Then **freeze** `f_seq` and `f_str` and drop them
into the model in place of the two projections currently learned from ΔΔG.

**Why this should help here specifically:**

- It learns the alignment from **unlabelled structure**, so it is not rationed by our 940 rows.
  The thing we cannot afford to learn from labels is exactly the thing this learns for free.
- It attacks the measured homology problem. Our embeddings retain 34 % under a homology split
  because much of what they carry is *recognition*. A contrastive objective over many complexes
  rewards the part of the sequence representation that is **predictive of local geometry** —
  which is physics — and is indifferent to the part that merely identifies the protein.
- The negatives can be chosen to target our failures. Sampling negatives from *the same
  complex* forces the alignment to be residue-specific rather than complex-specific, which is
  the precise failure mode §8 of the error analysis identifies: predicting a complex's mean
  scores +0.672 pooled and +0.000 per complex.

**Sensible ablations**, since this project's own lesson is that mechanisms must be priced
against controls: random negatives versus same-complex negatives; residue-level versus
neighbourhood-level views; and the alignment heads frozen versus fine-tuned on ΔΔG.

**Risk, stated plainly.** Alignment is not the same objective as ΔΔG prediction. A
representation can be perfectly aligned and carry nothing about mutational sensitivity. The
honest test is the same control structure used throughout: does the aligned model beat the same
model with randomly initialised projections, at three seeds.

## 6. Bidirectional cross-modality prediction

**The idea.** Instead of pulling the two modalities together, train each to **predict the
other**: a head that maps ESM-2 → ProteinMPNN embedding and another that maps ProteinMPNN →
ESM-2, on unlabelled complexes, under a regression loss.

**Why this is more interesting than it first sounds.** What the model *fails* to predict is the
useful part. If structure can be predicted from sequence almost perfectly, the structure branch
is adding nothing a language model does not already encode — which would be a direct, testable
answer to the question this project could only answer as a score ("what does ProteinMPNN add
beyond ESM?"). If it cannot, the **residual** `MPNN(x) − predict_from_seq(x)` isolates the
structure-specific information, and *that residual* is what should be fused rather than the raw
embedding.

**It is a sharper version of the fusion question.** Our fusion ladder compared six ways of
mixing two representations and found the mechanism did not matter. This asks a prior question —
how much non-redundant information is there to mix — and the answer would explain the ladder's
null result rather than just recording it.

**Concretely:** train on SAbDab complexes, measure per-residue reconstruction R², and report it
sliced by burial and interface proximity. My expectation, from §A3's observation that
ProteinMPNN's signal is local: reconstruction will be good on exposed residues and poor on
buried, tightly-packed ones — which are exactly the rows the model fails on (§16, valine).

## 7. Antibody-specific pretraining, done properly

AntiBERTy lost to ESM-2 by 0.046 on a matched control here. That is a real result but a narrow
one: the antigen stayed ESM-2 either way, and **one shared projection served both sides**, so
two unrelated spaces were being forced through one map. Either of the two objectives above
removes that confound, at which point an antibody-specific encoder deserves a second test — its
prior should be better on CDR loops, which is where the mutations are.

---

# Tier 3 — more data, which everything else points at

## 8. FoldX pseudo-labels

Deferred earlier behind a trigger: *build it only if a learning curve shows data volume is the
constraint*. Everything in `ERROR_ANALYSIS.md` Part III says it is. `skempi-foldx` ships
precomputed values, so FoldX never has to run. It gives a graded, structure-aware signal across
the whole label range — strictly more informative than the distal-neutral augmentation it was
weighed against, which only ever adds a point mass at zero.

## 9. The AB645 / AB1101 rows already built

261 rows are built, cached and leakage-filtered (homology models of our own complexes removed),
and have never been trained on. They need an ESM re-extract that includes their sequences.

## 10. Target new *structure space*, not new measurements

The sharpest data finding here: **how often a complex appears barely predicts performance**
(Spearman +0.04 / +0.09), while **how structurally close it is to training does** (+0.17 /
+0.37). Ten complexes with no structural relative in training score 0.190 against 0.404 for
those with three or more.

So "more data" should mean **complexes in untested regions of structure space**, not more
mutations on complexes we already have. That distinction is worth making to whoever funds the
next assay plate.

## 11. A ranking loss, for data recovery only

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

## 12. Select and report on the hard tier

**75 % of test rows have a near-identical training twin** (median TM 0.991); only 19.8 % are
genuinely hard. On those the forest scores +0.270 and the network +0.117. A model intended for
novel targets should be **selected** on the hard tier, not on the average — selecting on the
average selects for retrieval.

## 13. Weight the per-complex mean, or state that it is unweighted

It currently treats a 7-row complex with 0.78 label spread identically to an 87-row one
spanning 2 kcal/mol, and it is computed over **32 of 53 complexes**. Row-weighting, or a
minimum-spread filter, would both be defensible. Neither is applied; that is a judgment call
and should be stated as one.

## 14. Three seeds minimum, and publish the spread

Only 8 of 45 configurations here have three complete seeds. The measured spread reaches 0.117
per complex and 0.376 pooled within a single fold. Most published ΔΔG comparisons would not
survive this standard, which is itself worth saying.
