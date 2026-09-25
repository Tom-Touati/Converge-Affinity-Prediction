# Antibody–antigen ΔΔG: a frozen-encoder multimodal model

Predicting how a mutation changes antibody–antigen binding free energy, from **sequence and
3D structure**, on the antibody–antigen subset of SKEMPI 2.0.

**The model is `struct_film_chem`: a sequence-only backbone — LayerNorm, subtract the mutant
embedding from the wild-type at the mutated residue, one shared linear layer, sum — modulated
by a single FiLM gate built from pooled ProteinMPNN structure at that same site, with
substitution chemistry and interface geometry concatenated on top. 50,497 trainable
parameters, both encoders frozen.**

**The mechanism that won was a gate, not plain concatenation and not attention.** An earlier
sweep of 45 configurations — cross-attention in five variants, antibody↔antigen attention
across the interface, a two-tower model, a learned reduction replacing the PCA, binding-area
pooling, an ordinal head — found plain concatenation (`cat128_reg2_l1`, still in the table
below) ahead of everything else, including a gated-fusion run that turned out to be sharing
one projection across modalities by mistake. Fixing that bug and testing gating properly
changed the result: a corrected gated fusion (`gated_cg_clusterscale`) reached +0.344, ahead
of plain concatenation's +0.293. A follow-up sweep of 11 ways to inject structure into a
sequence-only design — four concatenation variants, a learned gate, four attention variants
including a nearest-antibody-residue pairing by wild-type 3D distance with rotary position
encoding on both sides, and two raw-scalar injections — settled it further: FiLM won outright,
ahead of every attention mechanism tried anywhere in the project's history. **Not one attention
mechanism beat gating or concatenation in either sweep.** The largest attention model in the
project (810,886 parameters) scores +0.200.

On **940 rows across 53 complexes**, with a measured seed spread of 0.028–0.117, most
architectural differences in this problem are smaller than the noise. Establishing which
differences are real — and which of this project's own earlier claims did not survive that
test — became the substance of the work.

| | |
| --- | --- |
| **The submitted model, `struct_film_chem`, specified in full** | [docs/DETAILS.md](docs/DETAILS.md#struct_film_chem) |
| **The model this superseded, specified in full** | [docs/MODEL.md](docs/MODEL.md) |
| **Why each choice was made** | [docs/JUSTIFICATIONS.md](docs/JUSTIFICATIONS.md) |
| **What I would do next, and why** | [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) |
| **Every architecture tried (45 runs, plus the structural-injection sweep that followed)** | [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md) |
| **Error analysis** | [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) |
| **AI prompt history** | [AI_PROMPTS.md](AI_PROMPTS.md) |
| Results · Running it · Hardware · Limitations | below |

---

## 1. The model

```
model                                 ens  per seed  spread  n  neg    bal  bal-cal   stab
E0a_rf_handcrafted (chem+geom+MPNN) +0.381    +0.379   0.032  3    6  0.444    0.498   0.06
struct_film_chem (seq, FiLM+chem)   +0.369    +0.314   0.092  3    3  0.521    0.530   0.24
gated_cg_clusterscale               +0.344    +0.299   0.107  3    5  0.521    0.547   0.33
struct_film_site (seq + FiLM only)  +0.340    +0.284   0.103  3    3  0.476    0.497   0.24
gated_cg_feats                      +0.340    +0.306   0.058  3    4  0.487    0.511   0.28
l1_gated        (ESM+MPNN, gated)   +0.300    +0.249   0.028  3    7  0.463    0.459   0.25
cat128_reg2_l1  (ESM+MPNN, concat)  +0.293    +0.239   0.063  3    6  0.439    0.457   0.19
st64_nopca_grouped                  +0.274    +0.274     -    1    6  0.508    0.506   0.48
gf_reg2         (heavier reg)       +0.270    +0.235   0.055  3    7  0.460    0.470   0.21
mut_pair_ffn_sub (sequence only)    +0.262    +0.232   0.094  3    7  0.449    0.454   0.25
delta_xattn2    (cross-attn, RoPE)  +0.239    +0.216   0.031  3    7  0.398    0.420   0.18
st64_film_struct (FiLM)             +0.227    +0.227     -    1    7  0.455    0.448   0.29
st64_noattn     (no fusion control) +0.212    +0.212     -    1    9  0.392    0.423   0.13
st64_xattn_rev_nopca (cross-attn)   +0.204    +0.182   0.046  3    9  0.454    0.445   0.35
[complex mean only] — the floor     +0.000                         0.580            0.58
```

**`struct_film_chem` is now the best neural network in the project**, +0.076 ens over the
previously-best net (`cat128_reg2_l1`) and ahead of every gated-fusion variant tried. It is
also, by parameter count, one of the *simplest*: a sequence-only backbone (LayerNorm, subtract
mutant from wild-type at the mutated residue, one shared linear layer, sum) with a single FiLM
gate — the pooled ProteinMPNN embedding at the mutated site scales and shifts the sequence
signal rather than being concatenated to it — plus the 39-column chem/geometry/zero-shot block
concatenated on top. No attention anywhere. Full architecture and the ablation that found it
are in [docs/DETAILS.md](docs/DETAILS.md#struct_film_chem).

Two honest caveats before treating this as settled. **Seed spread is 0.092**, wider than
`gated_cg_feats`'s 0.058 — the +0.369 ensemble number is real, but any single training run of
this model should be read as +0.314 ± 0.09, not +0.369. **Stabilising recall is 0.24**, well
below `st64_nopca_grouped`'s 0.48 — this model is not the answer if the downstream task is
*finding* affinity-improving mutations rather than ranking known ones; that recommendation is
unchanged from before.

**Now measured on a real homology-cluster split** (17 clusters from 53 complexes,
`data/cluster_folds.csv`, 4 folds greedily balanced by row count — `experiments/protattba_repro/
_perturb_v2_colab.py --fold-map`): ensemble **+0.272**, 74% of the standard-split score, seed
spread 0.034. `gated_cg_clusterscale` was run the same way for comparison: +0.214, 62%
retained, spread 0.113 — `struct_film_chem` degrades less under the harder split, not just
scores higher on the easier one. This is a **different cluster assignment and a different
metric convention** (ensemble Pearson here; the "Generalisation under a homology split" table
below reports per-complex Spearman on whatever split produced it, and this document does not
establish the two splits are identical) — read as a second, independent cluster-holdout
measurement, not a replacement number for that table.

The full bias/error-analysis battery in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) has not been
re-run against the new model; those numbers there still describe `cat128_reg2_l1` and are
flagged as pending an update, not silently carried over.

`ens` averages the seeds then scores once — what you would ship. `per seed` scores each seed
separately — what one training run gives you. **`spread` is max − min across seeds, and it is
the number to read every comparison against.** Reporting only one of these is how a model
appears to gain 0.05 by being written up differently: `l1_gated` is +0.300 as an ensemble and
+0.249 as the mean of its seeds, from the same predictions.
ESM-2 650M per residue, per side (ab, ag)     -> Linear(128->64), own map per side  proj_side
GELU, LayerNorm the mutant and wild-type projections SEPARATELY, then:
    LN(mt) - LN(wt)  ->  Linear(64->64)  ->  GELU  ->  sum over the mutated residues
    summed over ab and ag                                                       z_seq   64

ProteinMPNN encoder_h_V, per side  -> Linear(128->64), own map        -> GELU -> LayerNorm
    mean-pooled at the mutated residues, summed over ab and ag                  z_st    64
gamma, beta = Linear(64->128)(z_st).chunk(2)          <- FiLM, not concatenation
z_seq = (1 + gamma) * z_seq + beta

z = [ z_seq ; chem + interface geometry + ESM/ProteinMPNN zero-shot scores ]     103
    Linear(103->128) -> GELU -> Dropout(0.35) -> Linear(128->1)
```

**The fusion, and one weight-sharing decision that matters.** Sequence and structure each get
**their own** first linear layer, and each of those is separate again per side (antibody,
antigen) — four independent `Linear(128 → 64)` maps in total, never one shared across
modalities. The wild-type/mutant pair within one side's sequence DOES share a map, and that
sharing is necessary — the subtraction only means anything if both land in one space. Sharing
*across* modalities would not be: ESM-2 and ProteinMPNN are never subtracted from one another,
and forcing them through one map buys nothing while costing the ability to scale each
modality independently.

This is not a hypothetical. An earlier draft submitted a gated-fusion model that scored +0.300,
and it turned out to be sharing one projection between the two encoders — its config requested
a separate structure projection and the model silently ignored it. Fixing the bug and testing
gating properly was flagged here as "a next step, not a result"; it has since been done.
**`gated_cg_clusterscale`, the corrected version, reaches +0.344** — ahead of plain
concatenation's +0.293, and briefly the best network in the project before the FiLM sweep
below found something better still. The lesson was not "gating doesn't help"; it was that the
first measurement of it was measuring a bug.

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

### Augmentation and input perturbation

Three are applied, all training-only, and they are not interchangeable — what matters is what
each does to `mutant − wild-type`, the quantity the model actually consumes.

| | setting | effect on the edit |
| --- | --- | --- |
| reverse-mutation | p = 0.5 | exact: ΔΔG is antisymmetric, so swapping negates the label |
| feature dropout | 0.35, mask shared across branches | `mask·(mt − wt)` — zeroes features *of* the edit |
| Gaussian input noise | 0.25 × per-feature sd, drawn **per branch** | does **not** cancel; compounds by √2 |

**Reverse-mutation is exact label information, not a heuristic.** It moves the training label
mean from **+0.720 to −0.031**, removing the incentive to guess "destabilising" on a set that
is 70 % destabilising. Measured realised rate 51 %; a row escapes reversal across 25 epochs
with probability 3×10⁻⁸.

**The Gaussian noise is the one that misbehaves.** Because it is drawn independently on the
wild-type and mutant branches it compounds rather than cancels, reaching **0.8× the edit's own
standard deviation** — the edit has sd 0.128, the noise contributes 0.25 × 0.282 × √2 = 0.0998.
At the heaviest regularisation tried it would exceed the signal. This is the most likely reason
heavier regularisation kept *costing* accuracy without reducing seed variance, and sharing the
draw is a two-line change (next steps §3). Measured by
`experiments/protattba_repro/audit_augmentation.py`.

Full specification, parameter breakdown and training protocol: **[docs/MODEL.md](docs/MODEL.md)**.

## 2. How the fusion was chosen

Every mechanism below is the *same network* with only the fusion swapped — this table predates
`struct_film_chem`'s different base architecture (§1) and is kept as its own controlled
ablation rather than merged with the sweep that found FiLM, above. The row that makes the rest
interpretable is the control — `st64_noattn`, the identical network with the structure branch
**deleted**.

| fusion mechanism | params | per-cx r | vs control |
| --- | --- | --- | --- |
| plain concatenation *(the model, at the time this table was built)* | 36,353 | +0.293 | +0.081 |
| gated fusion *(shared one projection across modalities — a bug; fixed and re-run in §1: +0.344)* | 48,001 | +0.300 | +0.088 |
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

**Concatenation over gating, and why the scores do not decide it.** Gated fusion reads +0.300
against concatenation's +0.293 — a +0.007 difference, well inside both seed spreads, so the
scores do not separate them. The deciding factor is architectural: **the gated runs were
sharing one `Linear(128 → 64)` between ESM-2 and ProteinMPNN.** Their config set
`mpnn_proj=64` and the model silently ignored it, because `gated_fusion` was missing from the
condition that builds the structure projection. Two encoders with unrelated output spaces were
being forced through one map.

Concatenation gives each modality its own projection, costs nothing measurable, and is
smaller (36,353 against 48,001). The gated model's tighter seed spread (0.028 against 0.063) is
its one real advantage and it is **not** claimable either, since it came from the buggy
configuration. The bug is fixed in `model_simple.py`; re-running gated fusion with a real
structure projection is a next step, not a result.

### Dimensionality reduction — PCA and a learned grouped projection are equivalent

ESM-2 emits 1280 dimensions per residue against 752 training rows, so something must reduce
it. Three approaches, same model otherwise:

| reduction | per-cx r | seeds |
| --- | --- | --- |
| fold-local PCA-128 | +0.219 / +0.212 | 1 + 1 |
| learned grouped projection, ten 128→16 blocks, no PCA | +0.274 / +0.199 | 1 + 1 |
| PCA-256 instead of PCA-128 (MLP family) | +0.159 vs +0.157 | 1 |

Two seeds each: **grouped projection +0.236, PCA-128 +0.216** — a +0.020 difference against a
0.075 spread *within* the grouped projection's own two seeds. PCA is kept because it is
cheaper, auditable per fold, and nothing beats it outside the noise. The grouped projection is
block-diagonal deliberately: 20,480 parameters against 204,800 for a dense `Linear(1280, 160)`.

**In the forest, reduction actively hurts** — pooled ESM + ProteinMPNN scores 0.366 raw and
0.303 through PCA-128. Full detail in [docs/JUSTIFICATIONS.md §A4b](docs/JUSTIFICATIONS.md).

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
struct_film_chem  <- the model      +0.369    +0.314   0.092  3    3  0.521    0.530   0.24
gated_cg_clusterscale (fixed gating) +0.344    +0.299   0.107  3    5  0.521    0.547   0.33
cat128_reg2_l1  (previous model)    +0.293    +0.239   0.063  3    6  0.439    0.457   0.19
l1_gated       (shared-proj bug)    +0.300    +0.249   0.028  3    7  0.463    0.459   0.25
st64_xattn_rev (best of five xattn) +0.241    +0.241     -    1    4  0.444    0.441   0.44
st64_noattn    (no-fusion control)  +0.212    +0.212     -    1    9  0.392    0.423   0.13
--- baseline ---
E0a_rf_handcrafted (chem+geom+MPNN) +0.381    +0.379   0.032  3    6  0.444    0.498   0.06
--- floor ---
[complex mean only]                 +0.000                         0.580            0.58
```

Run names, not descriptions, so each row can be checked against `results/oof/<name>.csv`.
The two single-seed rows are marked `-` in `spread` and should be read as indicative only.

`ens` averages seeds then scores once; `per seed` scores each separately. **`spread` is the
number to read every comparison against.** Quoting one without the other is how a model appears
to gain 0.05 by being written up differently — the model reads +0.369 as an ensemble and
+0.314 as the mean of its seeds, from identical predictions.

**Two rows to read before the model's.**

**The floor.** Predicting each complex's own mean — never looking at the mutation — scores
**+0.672 pooled Pearson, 1.144 RMSE, 0.580 balanced accuracy**, beating everything here on all
three, while scoring **+0.000** per complex. Between-complex variance dominates this dataset,
so pooled correlation, RMSE and accuracy largely measure whether a model can identify the
complex. That is why per-complex is the headline and why `report_runs.py` prints this floor
beneath every comparison.

**The random forest is a baseline, not a competing submission** — it exists to establish what
these features support *without* a learned representation, and it is reported prominently
because a submission that buries its own baseline is not worth reading.

It beats the neural model, +0.381 against +0.369 — a gap of 0.012 against a seed spread of
0.092, roughly an **eighth** of the spread. That gap was 2.5× the spread against the previous
best net; it is not established against this one, and this is worth stating plainly rather
than keeping the old framing with a new number slotted in. What is still true regardless: it
uses 49 handcrafted columns spanning both modalities (substitution chemistry; interface
geometry; ProteinMPNN log-likelihoods), no learned representation at all, and trains in ~20
seconds on CPU.

That comparison is the point of running it. It establishes that on 752 training rows per fold,
49 informative columns hold their own against a learned representation — a finding about *the
problem*, not about this particular network — and, by being run on both splits (below), that
the networks lean on homology far more than the forest does. It simultaneously shows where the
neural model *is* clearly better: stabilising recall 0.24 against 0.06, a gap of four-to-one
that neither model's seed variance touches. A weaker baseline would have told us none of this.
§5 and [docs/JUSTIFICATIONS.md §A5](docs/JUSTIFICATIONS.md) say more.

### Generalisation under a homology split

`frozen5` withholds whole complexes; the `cluster` split withholds whole homology clusters.
Three seeds each, per-complex Spearman:

| model | by complex | by cluster | retained |
| --- | --- | --- | --- |
| random forest, handcrafted | 0.418 | 0.224 | **54 %** |
| pooled ESM + ProteinMPNN forest | 0.366 | 0.182 | 50 % |
| our network (`cat128_reg2_l1`) | 0.246 | 0.084 | **34 %** |

Everything degrades; the network degrades most, and 11 of its complexes finish with a
*negative* within-complex correlation. **The embedding-based model was leaning on homology the
cluster split withholds.** This is the single most important caveat on the neural result, and
it is measured on `cat128_reg2_l1` specifically, on this table's own split and metric
(Spearman). `struct_film_chem` has since been run on a homology-cluster split too — a
separately-built one (§1: `data/cluster_folds.csv`, ensemble Pearson, not directly comparable
to this table's numbers) — and the caveat holds there as well: it retains 74% of its standard-
split score, degrading less than `gated_cg_clusterscale`'s 62% but still degrading. The
expectation in this paragraph is no longer unmeasured for the current model, even though it
is not measured on *this specific* split.

### Where the neural model is better

| | balanced acc | stabilising recall |
| --- | --- | --- |
| `struct_film_chem` (the model) | 0.521 | **0.24** |
| `cat128_reg2_l1` (previous model) | 0.439 | 0.19 |
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
- **With the label partialled out, the two models fail on different axes.** The forest
  mis-weights **burial** — at equal true ΔΔG it predicts buried, highly-contacting sites higher
  (partial ρ −0.59 on distance to partner, +0.50 on contacts, invisible in the raw +0.005 and
  −0.12). The network mis-weights the **substitution**: it over-predicts alanine (+0.57) and
  under-predicts large volume and mass changes (−0.52, −0.60). That is the mechanism behind
  their low error agreement, and behind the blend beating both.
- **The forest's errors are predictable at ρ = 0.416 from features it already has**,
  cross-validated by complex. Unused signal, by definition.
- **Imbalance bias.** The average stabilising mutation is predicted *destabilising*, and within
  that class r = 0.007. X→A is 46 % of the data and the model is twice as good on it. Valine is
  under-predicted by 1.4 kcal/mol and ranked backwards. 21 of 53 complexes are too small to
  enter the headline metric at all.
- **Proximity predicts performance; frequency does not.** How many rows a complex has barely
  correlates with per-complex r (+0.04 forest / +0.14 model); how structurally close it is to
  training does (+0.17 / **+0.49**). Complexes with no structural relative in training — 10 of
  53, 24 % of rows — score **0.132** on the model against 0.432 for those with three or more.
- **Three quarters of test rows have a near-identical training twin** (median TM 0.991), and
  only 19.8 % are genuinely hard. On those hard rows the forest scores +0.270 and the model
  **+0.060** — so the headline numbers are 75 % weighted toward near-retrieval, and the honest
  expectation on a novel target is far lower.
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
2. **The error is systematically anti-correlated with the label's sign, and this is the most
   actionable defect.** Spearman(sign(ΔΔG), signed error) = **−0.61** for the submitted
   model and **−0.61** for the forest. Concretely:

   | label | n | mean ΔΔG | model mean error | forest |
   | --- | --- | --- | --- | --- |
   | stabilising (ΔΔG < 0) | 250 | −0.809 | **+1.113** | +1.169 |
   | near-neutral (\|ΔΔG\| ≤ 0.5) | 315 | +0.046 | +0.422 | +0.419 |
   | destabilising (ΔΔG > 0) | 657 | +1.562 | −0.756 | −0.531 |

   Stabilising mutations are over-predicted by about **+1.1 kcal/mol** and destabilising ones
   under-predicted by **0.5–0.7**. Crucially, the correlation between label sign and
   *\|error\|* is ≈ 0 (−0.06 for this model, −0.08 to +0.10 across all four): the model is **not less precise** on stabilising
   mutations, it is precise and **systematically wrong in direction**. For a designer that is
   worse than noise, because a confidently wrong sign is actionable in the wrong direction.

   This is shrinkage toward a training mean that is 70 % destabilising, and it is the same
   phenomenon as the class bias in §4 and the compressed output range. Reverse-mutation
   augmentation already attacks it — it moves the training label mean from +0.720 to −0.031 —
   and it is not enough.
3. **The neural model loses to a random forest** by 0.081, and retains 34 % under a homology
   split against the forest's 54 %. On this much data, 49 informative columns beat a learned
   representation.
4. **Pretrained sequence embeddings do not clear zero on their own**, across six probes and
   three protein language models. 26 chemistry columns moved the network +0.191 → +0.266 —
   larger than any architectural change measured. AntiBERTy, antibody-specific, lost to ESM-2
   by 0.046 on a matched control.
5. **There is no mutant structure.** ProteinMPNN sees the wild-type backbone only, so the
   structure term is identical for every mutation of a complex. The sharpest structural
   limitation of the design.
6. **Label noise is NOT the binding constraint — measured, and this corrects an earlier claim
   in this repository.** SKEMPI measures 107 (complex, mutation) pairs more than once; pooled
   within-group sd is **0.240 kcal/mol**, and 43 of those groups span different publications at
   sd 0.227, so the estimate reflects genuine independent replication. Against a within-complex
   signal sd of 1.178 that implies a ceiling of **0.979** on per-complex Pearson. We reach
   0.381 — **39 % of what is achievable**. Earlier text cited a within-complex label sd of
   1.104 as the noise floor; that figure is the spread of *different* mutations, which is
   signal, not noise. See [ERROR_ANALYSIS §23](ERROR_ANALYSIS.md).
7. **Three complexes hold 24 % of the rows**, so any pooled statistic is partly about them.
8. **Only 8 of 45 configurations have three complete seeds.** The rest are single-seed and
   their individual numbers are not interpretable at the resolution the tables print them.

## 6. Next steps, in the order I would do them

1. **Read the gates.** No checkpoints were saved, so *what the model gates on* — how much
   sequence versus structure, for which mutations — cannot be answered from the artifacts. This
   turns the fusion mechanism from a score into an explanation, and it is the first thing to do.
2. **Blend the forest with the network.** They agree only 0.42–0.54 on which rows are hard, and
   a fixed 25 % blend lifts per-complex r from +0.397 to +0.416 with the submitted model, and
   +0.424 with `st64_nopca_grouped`. The weight was not tuned on held-out data, so validate it — but it is better
   evidenced than any architecture change here.
3. **Normalise the target within cluster.** Reported at **+0.241 → +0.299** on gated
   `chem+geom`. It is consistent with the largest effect in the error analysis — regression to
   the mean at ρ ≈ −0.8, surviving *within* complex at ρ ≈ −0.58 — since a model normalised per
   cluster no longer spends capacity reproducing each cluster's own location and spread. **The
   scaling statistics must be fitted on training folds only**: fitted on everything, each test
   complex supplies its own mean to its own normalisation, which is the complex-mean floor
   (+0.672 pooled) smuggled in as preprocessing and would look like a large clean gain. That
   run is not in `results/oof/`, so by this project's own three-seed rule it is a lead, not yet
   a result.
4. **Share the input-noise draw across branches.** Measured: the noise is drawn independently
   per branch, so it compounds by √2 and reaches **0.8× the delta's own sd** — at the heaviest
   setting, more noise than signal. Two-line change, and it would explain why heavier
   regularisation kept costing accuracy.
5. **Give the network the forest's geometry columns.** The largest measured gain in the project
   came from adding chemistry; interface geometry is the untried other half.
6. **Synthesise mutations — FoldX for the mutant *structure* first, labels second.** The
   sharpest limitation of the submitted model is that **there is no mutant structure**:
   ProteinMPNN sees the wild-type backbone only, so the structure term is identical for every
   mutation of a complex and contributes a per-complex constant. FoldX `BuildModel` produces a
   mutant structure, which turns that term into a *delta* mirroring the sequence edit — an
   architectural gain needing no new labels. Separately, FoldX has the best Spearman of any
   individual model on full SKEMPI (0.526) while having the worst RMSE, so it is worth one
   input column, and it can pseudo-label mutations SKEMPI never measured. It is the right
   labeller precisely because it is *not* fitted to SKEMPI — a learned ΔΔG predictor would leak
   our own test complexes into the synthetic set. Use pseudo-labels to pretrain, not as the
   target, and keep the synthetic pool class-balanced or it hands back the imbalance that
   reverse-mutation augmentation removed (+0.720 → −0.031).
7. **Ordinal loss instead of MSE.** MSE asks for a point value on a target whose repeat
   measurements disagree by **0.240 kcal/mol** — small, so this is an argument about the *shape*
   of the loss rather than about noise. It is worth trying because the decision the labels are
   used for is a three-way one, and because MSE is what shrinks predictions toward the mean. A
   CORAL head is implemented (`ordinal=2`) and verified; it never completed a run.
8. **Three seeds minimum for any future claim**, and report the spread.
9. **More data — but new *structure space*, not new measurements.** How often a complex
   appears barely predicts performance (Spearman +0.04 / +0.09); how structurally close it is
   to training does (+0.17 / +0.37). 261 AB645/AB1101 rows are built, cached and
   leakage-filtered, and have never been trained on. This is also the acceptance test for
   item 6: synthetic mutations on complexes we already have add rows where the model is already
   strong (**hard tier +0.060 against easy +0.367**), so if FoldX augmentation moves the
   average and leaves the hard tier flat, it bought retrieval, not generalisation.

**The main proposal is bigger than any of these, and it has two halves that only work
together: more data, and a sequence–structure alignment pretrained specifically on
antibody–antigen interfaces.** Our two encoders occupy unrelated spaces joined only by a pair
of `Linear(128 → 64)` maps fitted to 752 labels, and the consequence is measured: performance
tracks structural proximity to training at Spearman **+0.494** and collapses to **+0.060** on
complexes with no structural relative. The model learned to recognise, not to generalise.

A *generic* protein alignment would not fix it — it would be dominated by globular cores, while
antibody binding is loop-mediated, the framework is near-constant across unrelated antibodies
(which is why this project clusters antibodies at 90 % identity rather than 30 %), and epitopes
are discontinuous. [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) specifies the AB/AG-specific
version: interface neighbourhoods as the unit, negatives drawn from within the same complex and
CDR, a contrastive term plus a cross-modality prediction term whose *residual* isolates what
structure adds beyond sequence — and the acceptance test, which is the **hard tier**, not the
average.

## 7. Setup and running

```bash
pip install -r requirements.txt          # Python 3.10; torch pinned, see requirements.txt
make data                                # SKEMPI -> data/processed/*.parquet   (~65 s)
make splits                              # frozen homology folds -> data/folds.csv  (~28 s)
make features                            # geometry, ESM, ProteinMPNN caches
make ladder                              # the CLASSICAL ladder (mean -> GBT -> forest baseline)
make report                              # the results table, all runs on one truth
make errors                              # slice tables and diagnostics
make align                               # verify each mutation is where we index it
make test                                # split-integrity and harness tests
```

**To train the submitted model, `struct_film_chem`.** The neural runs are driven separately,
from `experiments/protattba_repro/`, on a GPU box rather than the dev machine — a Colab T4 for
the earlier 45-configuration sweep, an EC2 g4dn.xlarge (also a T4) for `struct_film_chem` and
the structural-injection sweep that found it, since Colab's session limits made the longer
sweep impractical. The command:

```bash
python _perturb_v2_colab.py --exp struct_film_chem --arch sitetok \
  --folds 0 1 2 3 4 --seeds 0 1 2 --wd 0.1 --grad-clip 10 --clip 4 --select-on per_complex \
  --patience 15 \
  --overrides '{"pca_dim": 128, "proj": 64, "hidden": 128, "layers": 1, "dropout": 0.35, "input_noise": 0.0, "feature_dropout": 0.0, "chem_dim": 39, "split_proj": true, "n_heads": 2, "mut_pair_ffn": true, "mut_pair_op": "sub", "mut_pair_act": true, "mut_pair_struct_inject": "film_site"}'
```

with `CHEM_TABLE=chem_perturb_v2` in the environment (the 39-column chem/geometry/zero-shot
table; `scripts/build_perturb_features.py` builds it). It writes `out/struct_film_chem_results.
csv` and per-row predictions, collected into `results/oof/struct_film_chem.csv`, then published
into `reports/perturb_struct_film_chem/` via `python -m src.perturb.publish struct_film_chem
--name perturb_struct_film_chem --model perturb_v3` — the format every table and every
error-analysis script in this repository reads. Re-running is safe: a configuration with five
folds already on disk is skipped rather than retrained.

`scripts/final_table.py` produces the results table above; it defaults `SUBMITTED` to
`struct_film_chem`. `scripts/error_drivers.py`, which contributes to Part III of the error
analysis, now compares `struct_film_chem` against the forest and three older nets by name. The
deeper numbered sections of `ERROR_ANALYSIS.md` itself have not been re-run against the new
model — flagged in §1 above rather than left silently stale. `data/folds.csv` is committed and
frozen — nothing downstream regenerates it.

**Known environment issue:** `tests/test_perturb_*.py` require torch, which fails to load on
the development machine (8 GB RAM, `WinError 1114`). The other tests pass; the torch tests run
wherever torch loads.

## 8. Hardware and runtime

Development: **Intel Core i7-8650U, 4 cores @ 1.9 GHz, 8 GB RAM, no GPU**, Windows 10.
Neural training: **Colab T4** for the 45-configuration sweep that produced `cat128_reg2_l1`;
**two EC2 g4dn.xlarge instances (also T4)** for `struct_film_chem` and everything after it,
once Colab's session limits made a longer sweep impractical. Both driven from the same CLI in
`experiments/protattba_repro/`.

| step | where | wall clock |
| --- | --- | --- |
| `src.data` — parse, verify 2,109 positions, dedup | CPU | ~65 s |
| `src.splits` — 1,431 pairwise alignments + leakage report | CPU | ~28 s |
| `src.features.geometry` — Shrake-Rupley over 54 complexes | CPU | ~5.5 min |
| ESM-2 650M per residue, 448,772 tokens | T4 | ~4 min (1,998 tok/s) |
| the same | CPU | ~160 min |
| ProteinMPNN per residue, 54 complexes | CPU | ~7 min |
| one forest rung, 5 folds × 3 seeds | CPU | ~20 s |
| `cat128_reg2_l1`, 5 folds × 3 seeds (previous model) | T4 (Colab) | ~25 min |
| the 45-configuration ladder that found it | T4 (Colab) | ~14 h |
| **`struct_film_chem`, 5 folds × 3 seeds** | **T4 (EC2 g4dn.xlarge)** | **~25 min, measured** |
| the 11-way structural-injection sweep that found it | T4 (EC2 g4dn.xlarge) ×2 | not precisely timed -- run interleaved with other experiments over several hours, not as a clean isolated sweep |
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
