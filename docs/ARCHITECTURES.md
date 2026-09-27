# Every architecture tried, what it tested, and what it showed

`docs/ARCHITECTURE.md` is the design rationale written *before* the ladder ran — what 750
training rows can support, the pipeline, the parameter budget. This is the record of what was
actually built and measured afterwards.

**45 neural configurations completed at least one full 5-fold run**, plus the forest ladder.
Every number is per-complex Pearson on the frozen by-complex split, scored against one common
truth by `scripts/report_runs.py`. Seeds are averaged where more than one exists; the count is
in the table.

> **Families A–D are the first 45 runs. [Family E](#family-e--the-site-pair-ladder-and-how-the-submitted-model-changed)
> covers a later phase** (~30 further 3-seed configurations plus a 200-trial single-seed
> screen) which produced `struct_film_chem`, still the submitted model.
> **[Family F](#family-f--injecting-where-a-mutation-sits-relative-to-the-binding-site) covers
> a later phase still** — checkpoint transfer from full SKEMPI, and a sweep of features
> describing a mutation's own position relative to the binding site — all of it negative once
> measured correctly, including a real confound (§F.4) caught in the process of checking an
> apparent win. **[Family G](#family-g--training-objectives-and-a-factorial-sweep-of-the-leaders-own-architecture)
> covers a later phase still** — ranking-loss objectives, a depth × fusion-mechanism factorial
> sweep on the leader's own two branches, and an early/mid/late fusion-stage ablation. The
> leader's own FiLM-at-the-end design beats every alternative fusion timing and every simpler
> combining rule tried; a depth-2 architecture change and a tied-init fix for `split_mutwt`
> both looked promising at 3 seeds and did NOT survive an 8-seed re-measurement (§G.2). One
> training-objective change did survive it — a soft-Kendall-tau ranking term added to the
> regression loss — and is now part of the submitted recipe (DETAILS.md). Where families
> disagree, the later one is the later measurement and says so explicitly.

> **Read every gap against the seed spread.** Measured at **0.028 to 0.117** depending on
> configuration (ERROR_ANALYSIS §9). Most rows in this document are separated by less than
> that. The families tell a story; individual orderings inside a family mostly do not.

---

## Summary — the whole ladder in one view

| family | best | params | per-cx r | what the family established |
| --- | --- | --- | --- | --- |
| **A.** full attention model (v2/v3/v4) | `full` | 810,886 | +0.200 | capacity is not the constraint |
| **B.** pooled-delta MLPs | `mlp_delta_chem_reg` | 85,376 | +0.266 | chem features carry the gain |
| **C.** site-token fusion ladder | `cat128_reg2_l1` *(the model, at the time of this table)* | 36,353 | +0.293 | fusion *mechanism* is not the constraint |
| **D.** random forest on 49 columns | `E0a_rf_handcrafted` | — | **+0.381** | the **baseline** the model is measured against |

The largest model in the project (810k parameters) scores **+0.200**. The best neural model
(48k) scores **+0.300**. The forest, with no learned representation at all, scores **+0.381**.

---

## A. The full attention model — v2, v3, v4

**Antibody↔antigen cross-attention** over the cropped interface (`cross_chain=True`,
`attn(h_ab, h_ag, ...)`), with rotary position embedding, a distance bias on the attention
logits, a chain-identity embedding, and BLOSUM features on the mutated residues. This is the
architecture the plan proposed, and it is the mechanism `thoughts.md` asked for — "encode the
ab binding site sequence and the ag too, then run cross attention between ag to ab, for both
mutated and not". It is also ProtAttBA's mechanism.

**So antibody↔antigen attention is measured, and this family is the measurement.**

| run | params | per-cx r | what it changed |
| --- | --- | --- | --- |
| `full` | 810,886 | +0.200 | the full model, widest setting |
| `v5_simple` | 41,792 | +0.205 | **20× smaller, same score** |
| `v3_sitemean` | 59,281 | +0.172 | pooling at mutated residues instead of over the crop |
| `v4_noblosum` | 48,465 | +0.165 | BLOSUM removed |
| `v4_noblosum_reg` | 48,465 | +0.150 | and regularised |
| `v3_site` | 51,089 | +0.129 | site-restricted attention |
| `v3_reg` | 51,089 | +0.133 | regularised |
| `v3_base` | 51,089 | +0.126 | baseline |
| `v3_indep` | 51,089 | +0.118 | independent branches |
| `v2_full` | 51,089 | +0.112 | the original |

**What it established.** An 810k-parameter model and a 42k one score the same. Every component
this family added — RoPE, distance bias, chain embedding, BLOSUM — was removed in later
families without loss, several with gains. On 752 training rows, the architecture was
elaborate in ways the data could not pay for.

## B. Pooled-delta MLPs

Strip everything: pool the ESM delta at the mutated residues, LayerNorm, MLP. The point was to
price the architecture by removing it.

| run | params | per-cx r | what it changed |
| --- | --- | --- | --- |
| `mlp_delta_chem_reg` | 85,376 | **+0.266** | + 26 chemistry columns |
| `mlp_delta_pca256_h128_reg` | 82,048 | +0.191 | wider head, no chem |
| `mlp_delta_pca256_reg` | 32,832 | +0.159 | PCA-256 |
| `mlp_delta_pca128_reg` | 16,448 | +0.157 | PCA-128 |
| `mlp_delta_pca256` | 32,832 | +0.144 | unregularised |
| `mlp_chem_only` | 20,096 | **−0.005** | chemistry alone, no embeddings |

**What it established, and it is the most useful result in the document.** Adding 26 chemistry
columns moved the same network **+0.191 → +0.266**. That single feature change is larger than
any architectural difference measured anywhere in this project.

But `mlp_chem_only` scores **−0.005**. The chemistry columns are worth nothing on their own in
a network — they are *complementary* to the embeddings, not a substitute. (In the forest the
same columns are the backbone, which is itself informative about what each model class can do
with them.)

## C. The site-token fusion ladder

The main family. Sequence read at the mutated residues as `site_mean(mutant) − site_mean(wild
type)`; structure read from ProteinMPNN; the two combined by one of six mechanisms. All share
a config so exactly one thing changes per run.

### C1. Fusion mechanism — the question the project was built to answer

| run | params | per-cx r | mechanism |
| --- | --- | --- | --- |
| `cat128_reg2_l1` *(the model, at the time of this table)* | 36,353 | +0.293 | plain concatenation, one-layer head, **separate projection per modality** |
| `l1_gated` | 48,001 | +0.300 | gated fusion — but see the note below: it shares one projection across modalities |
| `st64_gated_fusion` | 64,513 | +0.282 | gated fusion, two-layer head |
| `st64_film_struct` | 52,993 | +0.227 | FiLM on the structure delta |
| `st64_xattn_rev` | 77,441 | +0.241 | cross-attention, structure → sequence |
| `st64_xattn_r28` | 77,441 | +0.208 | weighted residual 0.2 / 0.8 |
| `st64_xattn_r01` | 77,441 | +0.199 | sequence → structure |
| `st64_chem_xattn_sep` | 77,441 | +0.192 | separate ProteinMPNN projection |
| `st64_xattn_r37` | 77,441 | +0.172 | residual 0.3 / 0.7 |
| **`st64_noattn`** | **36,481** | **+0.212** | **the control — fusion deleted** |

**The control is the point.** Four of the five cross-attention variants score *at or below* a
model with the attention removed, while costing twice the parameters. Gating and concatenation
clear it; they are also the two cheapest mechanisms. Cross-attention over ProteinMPNN — the
most elaborate thing built — does not pay for itself.

**A defect in the gated runs, found late.** Every `gated_fusion` run in this table projected
ESM-2 and ProteinMPNN through the **same** `Linear(128 → 64)`. Their configs set
`mpnn_proj=64`; the model ignored it, because `gated_fusion` was missing from the condition
that constructs the structure projection. So the gated numbers describe a model sharing one map
between two unrelated representation spaces, and the +0.007 it holds over concatenation — along
with its tighter seed spread — cannot be attributed to gating. The condition is fixed in
`model_simple.py`. Re-running gated fusion correctly is an open item; the submitted model is the
concatenation variant, which gives each modality its own projection.

### C2. Representation choices

| run | params | per-cx r | what it changed |
| --- | --- | --- | --- |
| `st64_nopca_grouped` | 61,057 | +0.274 | no PCA; learned block-diagonal reduction |
| `st64_nopca_s1` | 61,057 | +0.199 | *the same thing, second seed* |
| `nopca_reg2` | 61,057 | +0.233 | and regularised harder |
| `st64_nopool_chem` | 36,481 | +0.219 | binding-site pools removed |
| `st_pool64` | 74,113 | +0.174 | binding-site pools kept |
| `st_proj_sub_nopool` | 49,537 | +0.171 | projected subtraction |
| `area_gated` | 48,001 | +0.255 | structure pooled over the binding *area* |
| `area_concat` | 36,353 | +0.227 | the same, concatenated |

**Two cautions live in this block.** `st64_nopca_grouped` at +0.274 and `st64_nopca_s1` at
+0.199 are the *identical configuration at a different seed* — a 0.075 gap, and the single
clearest illustration of why this document warns about reading individual rows. Dropping the
PCA was reported as a +0.062 win before the replicate arrived; it does not survive.

Pooling structure over the whole binding area rather than at the mutated residues costs
~0.05 (`area_concat` +0.227 vs `cat128_reg2_l1` +0.293). ProteinMPNN's signal here is local;
averaged over ~85 interface residues it washes out.

### C3. Features and encoders

| run | params | per-cx r | what it changed |
| --- | --- | --- | --- |
| `st64_chem` | 52,865 | +0.204 | + 26 chemistry columns |
| `st64_nochem` | 49,537 | +0.185 | without them |
| `st64_nopool_chem_esm` | 36,481 | +0.212 | ESM-2 650M on the antibody side |
| `st64_nopool_chem_abty` | 36,481 | +0.166 | **AntiBERTy** on the antibody side |

**An antibody-specific language model lost to a general one by 0.046** on a matched control,
in a setting where the antigen stayed ESM-2 either way and one shared projection served both
sides — so the two encoders landed in unrelated spaces. That is the fairest reading available
and it is not favourable to AntiBERTy here.

### C4. Optimisation

| run | params | per-cx r | what it changed |
| --- | --- | --- | --- |
| `cat128_reg2_l1` | 36,353 | +0.293 | gradient clip 5 (default) |
| `cat128_reg2_l1_noclip` | 36,353 | +0.265 | clipping effectively off |
| `gf_reg2` | 64,513 | +0.270 | heavier regularisation |
| `st64_gated_fusion` | 64,513 | +0.282 | the same at baseline regularisation |

Turning clipping off *costs* 0.028 — the opposite of what was expected when the threshold was
found to sit exactly at the typical gradient norm. Heavier regularisation costs 0.037–0.043 on
two architectures and reduces seed variance on neither.

### C5. Implemented, not yet measured at three seeds

- **Antibody↔antigen cross-attention inside the cheap family** (`ab_ag_attn` in
  `model_simple`). The mechanism itself is *not* untested — family A above is exactly that, at
  +0.112 to +0.200. What is untested is whether it does better inside the 36-48k site-token
  model than it did inside the 51-810k v2 model, with the components that family A was
  carrying (RoPE, distance bias, chain embedding, BLOSUM) removed. Built and forward-tested;
  the runs did not complete before the session budget ran out.
- **Ordinal head** (`ordinal=2`, CORAL) — two thresholds driven by one shared scalar so they
  cannot contradict each other, with three-class metrics logged per epoch. Verified to train
  and to keep its thresholds ordered; the runs did not complete.

Both are in the code and queued in `experiments/protattba_repro/_run_ladder.py`. Neither has a
result, and neither is claimed as one.

## D. The forest ladder

| run | per-cx ρ (3 seeds) | features |
| --- | --- | --- |
| `E0a_rf_handcrafted` | **0.418** | chemistry + interface geometry + ProteinMPNN |
| `E0c_rf_pooled_esm_mpnn` | 0.366 | pooled ESM + ProteinMPNN |
| `E0c_..._pca128` | 0.303 | the same, PCA-128 |
| `E0b_rf_pooled_esm` | 0.273 | pooled ESM only |
| `E0e_mean` | undefined | predicts the training mean |

Adding the structure encoder to pooled sequence embeddings is worth **+0.093**. Replacing
pooled embeddings with handcrafted columns is worth a further **+0.052**.

---

## What the whole ladder says

**The simplest fusion won, and it was not chosen for being simple.** Every elaboration here was
built first and priced afterwards against the thing it replaced. Concatenation survived because
nothing beat it outside the seed spread — not five cross-attention variants, not
antibody↔antigen attention at up to 22× the parameters, not FiLM, not gating, not a learned
reduction replacing the PCA. `ERROR_ANALYSIS.md §25` collects the mechanism.

One number is worth stating because it cuts against the easy reading: across all 45
configurations Spearman(parameter count, per-complex r) is **+0.276**, so bigger models score
slightly *better* on average. That correlation is manufactured by the deliberately crippled
ablations at the small end — `mlp_chem_only` has 20,096 parameters and scores −0.005. The
claim is not "smaller is better"; it is that **at equal information, added mechanism did not
pay for itself**.


1. **Capacity is not the constraint.** 810k parameters and 42k score the same; removing an
   entire head layer (31 % of a model) cost nothing measurable.
2. **Fusion mechanism is not the constraint.** Four of six mechanisms sit at or below the
   no-fusion control. The two that win are the two simplest.
3. **Features are the constraint.** The one change larger than any architectural difference
   was adding 26 chemistry columns (+0.075), and the best model overall uses no learned
   representation at all.
4. **Most of this table is inside the noise.** Two runs of one identical configuration differ
   by 0.075. Any ordering here read at finer resolution than that is not a finding.

---

## Family E — the site-pair ladder, and how the submitted model changed

The 45 runs above ended with plain concatenation (`cat128_reg2_l1`, +0.293) as the best
network, and the conclusion that *fusion mechanism is not the constraint*. A later phase
(2026-09-24/25, on EC2 rather than Colab) built a different backbone and then swept how to
attach structure to it. **The conclusion survived in a stronger form than it was stated, and
one part of it was wrong.**

### E.1 The backbone: drop everything except the mutated residue

| run | params | per-cx r | what it tests |
| --- | --- | --- | --- |
| `mut_pair_ffn` (multiply) | 28,993 | +0.014 | LN(mt) × LN(wt) at mutated residues |
| **`mut_pair_ffn_sub`** | **28,993** | **+0.262** | LN(mt) − LN(wt), otherwise identical |

Per side, at each mutated residue only: project, LayerNorm mutant and wild type separately,
**subtract**, one shared Linear, GELU, sum over the mutated positions. No structure, no chem,
no attention, 28,993 parameters — and +0.262, which at the time was third in the project.

**The multiply/subtract pair is the single largest controlled effect measured anywhere in this
project**: +0.014 against +0.262 from changing one operator. Two mechanisms, both checkable:
an elementwise product cannot distinguish "both channels silent" from "one channel active, one
silent" — same near-zero output, opposite meaning — and its gradient with respect to one side
*is* the other side's value, so a near-zero channel on one side starves the gradient reaching
the other. A subtraction has neither property.

This was then confirmed at scale: a 200-trial grid (10 injection modes × 2 operators × 2 chem
states × 5 regularisation presets, single seed each) had **`sub` ahead of `mul` in 91 of 100
matched pairs, mean advantage +0.149**, and `chem=39` ahead of `chem=0` in **96 of 100**.

### E.2 Eleven ways to attach structure to it

Each row is the same backbone plus one mechanism, 3 seeds, full 5 folds:

| mechanism | per-cx r | note |
| --- | --- | --- |
| **FiLM at the mutated site + 39-col chem** (`struct_film_chem`) | **+0.369** | **the submitted model** |
| FiLM, ESM-IF1 in place of ProteinMPNN | +0.360 | best class separation of anything measured |
| FiLM at the mutated site, no chem block | +0.340 | |
| learned gate at the mutated site | +0.318 | |
| raw scalars: 5 ProteinMPNN zero-shot scores | +0.328 | no learned structure encoder at all |
| raw scalars: 7 handcrafted geometry columns | +0.315 | |
| concat, pooled at the mutated site | +0.294 | |
| concat, pooled over the whole crop | +0.282 | |
| nearest-ab-residue structural difference, concatenated | +0.277 | |
| nearest-ab-residue difference, cross-attended (RoPE both sides) | +0.239 | |
| cross-attention, sequence queries structure | +0.253 | |
| cross-attention, structure queries sequence | +0.206 | worst of the eleven |

**Gating beat concatenation, and concatenation beat attention — every time.** Family C
concluded "fusion mechanism is not the constraint" because nothing beat plain concatenation.
With a backbone that works, the ordering separates cleanly, and the part of the earlier
conclusion that was wrong is the implied *nothing can beat concatenation*: FiLM does, by
+0.075 over the plain-concat variant of the same backbone. What survives, and is now much
better evidenced, is **attention being the worst mechanism available** — bottom two of eleven
here, and four of five below the no-fusion control in family C.

That two raw-scalar injections (+0.328, +0.315) beat every attention variant and every
concatenation of a *learned* structural embedding is the sharpest form of the project's
recurring finding: what reaches the model matters more than how it is combined.

### E.3 Rotary position encoding, and why it did almost nothing

| run | per-cx r |
| --- | --- |
| cross-attention baseline (no RoPE) | +0.151 |
| + RoPE, base 10000 (the language-model default) | +0.185 |
| + RoPE, base 200 (retuned) on the chem-query model | +0.302 vs +0.279 without |

Keyed on true residue index, not crop slot — measured on real crops, consecutive slots sit 1
to 40 residues apart, so rotating by slot would encode a spacing that does not exist. A
per-chain offset was needed too: the index restarts at 0 in every chain, so an H30 and an L30
were being handed the same phase.

**The default base is wrong for this data by a wide margin.** Same-chain separations here have
median 29 and 90th percentile 75. At base 10000, **7 of 16 frequency channels turn less than
half a radian across that entire range** — constants, carrying nothing — and 5 more wrap and
alias. Four channels were doing anything. At base 200 the series spans 1–170: 8 usefully
tuned, none dead. Every gap in the table above is still inside the seed spread, so this is a
diagnostic finding rather than a result: the encoding was present and three-quarters idle, and
saying so is worth more than the +0.023.

### E.3b Widening the PCA to 256 — a clean loss

| run | per-cx r | seed spread | negative complexes |
| --- | --- | --- | --- |
| `struct_film_chem`, PCA-128 | **+0.369** | 0.092 | **3** |
| `struct_film_chem`, PCA-256 | +0.305 | 0.123 | 8 |

Doubling the fold-local PCA width costs 0.064, widens the seed spread, and nearly triples the
complexes that finish with a *negative* within-complex correlation. At 752 training rows per
fold, 256 components per side is more basis than the labels can constrain, and the extra
directions are fitted to fold-specific noise — which is exactly what a rise in both seed
variance and negative-complex count looks like.

Worth recording because the obvious reading of "89% of variance retained at 128" is that more
components must help. They do not; the retained-variance number describes the *inputs*, and
what binds here is the supervision.

It also exposed a latent bug worth naming: `mpnn_proj` had been built assuming the structure's
post-PCA width always equals `pca_dim`. PCA cannot exceed the raw input's width, so with
ProteinMPNN's 128-d encoder the structure PCA silently capped at 128 while the layer expected
256. This was invisible for the whole project because the default `pca_dim` *is* 128 — the two
were equal by coincidence, not by construction. Fixed with `struct_pca_dim`, computed by the
trainer from the fitted PCA and verified byte-identical at the default.

### E.4 Reference checks — what the floor actually is

Structure-free, chem-free, deliberately naive:

| check | per-cx r |
| --- | --- |
| ab/ag nearest-neighbour product, per residue | +0.073 |
| pooled-ab × pooled-ag interaction | −0.093 |
| the same with a `mul` outer combine | −0.009 |

All at or below zero. Worth keeping because they price the rest: the fusion mechanisms in
family C and E.2 are doing real work, and a naive interaction term is not a cheap substitute
for any of them.

---

## Family F — injecting where a mutation sits relative to the binding site

A later phase again, following two questions in sequence: does the ~300 non-antibody complexes
in the rest of SKEMPI transfer anything to the AB/AG task, and can the model be told directly
where a mutation sits relative to the interface, rather than leaving it to infer that from
pooled structure. Both were tried on the real 940-row AB/AG set, 5 folds × 3 seeds unless noted.

### F.1 Splitting the mutant/wild-type projection — a real regression

`mut_pair_ffn`'s subtraction, `LN(mt) − LN(wt)`, projects both sides through the SAME map
(`tok_proj`) before the LayerNorm. This project's own rule — never share the first linear layer
between modalities — argues for splitting it, exactly as it argued for splitting ab/ag and
structure. Tried directly: mutant and wild-type each get their own `Linear(128 → 64)`, 4 total
(ab_mt, ab_wt, ag_mt, ag_wt) instead of 2.

| run | per-cx r | seed spread | negative complexes |
| --- | --- | --- | --- |
| `struct_film_chem` (shared mt/wt projection) | **+0.369** | 0.092 | 3 |
| `split_mutwt` (separate mt/wt projection) | +0.294 | **0.237** | 4 |

The sharpest instability measured anywhere in this project — seed spread more than doubles.
Unlike ab/ag or sequence/structure, mutant and wild-type are not two different modalities to
keep apart; they are the same modality at two time points, and the subtraction only means
anything if both land in the identical learned space. Splitting the projection lets them drift
apart across training, so `LN(mt) − LN(wt)` increasingly compares two different bases rather
than measuring what changed. The "don't share weights across modalities" rule has a real
exception, and this is it.

### F.2 Ten ways to add whole-crop binding-site context — none beat the leader

Ten mechanisms added ON TOP of `struct_film_chem`'s existing site-pooled FiLM gate, each
injecting binding-site context pooled over the WHOLE crop rather than just the mutated residue
— concatenation, a second FiLM stage, a learned gate, cross-attention, distance-weighted
pooling, standard deviation of the local embedding, the nearest-neighbour structural
difference, and three pure-geometry scalars (crop size, tightest contact anywhere in the crop,
mean contact distance). Full 15-combo runs each:

| mechanism | per-cx r | ens | negative complexes |
| --- | --- | --- | --- |
| `struct_film_chem` (no addition) | +0.314 | **+0.369** | 3 |
| `mean_dist_crop` (pure geometry) | +0.296 | +0.361 | 3 |
| `crop_concat` | +0.269 | +0.360 | 3 |
| `crop_std` | +0.276 | +0.347 | 2 |
| `nn_diff_crop` | +0.267 | +0.343 | 3 |
| `min_dist_crop` (pure geometry) | +0.311 | +0.341 | 3 |
| `crop_gate` | +0.285 | +0.338 | 5 |
| `crop_film2` | +0.271 | +0.335 | 3 |
| `crop_wpool` | +0.266 | +0.327 | 5 |
| `crop_size` (pure geometry) | +0.267 | +0.318 | 4 |
| `crop_xattn` | +0.272 | +0.303 | 6 |

None improve on the leader. The closest is pure geometry with no learned structure at all
(`mean_dist_crop`, a scalar), matching this project's recurring finding that what reaches the
model matters more than how elaborately it is fused. `crop_xattn` is worst — attention loses
again, the same pattern in every fusion sweep this project has run. Read together with F.3
below: whole-crop context added AFTER the backbone does not help, but a per-residue feature
added INTO the mutation's own representation does.

### F.3 Ten features encoding a mutation's position relative to the binding site

A different injection point: instead of pooling structure over the crop and concatenating it
after `mut_pair_ffn` runs (F.2's approach), compute one feature PER MUTATED RESIDUE from
tensors already in the batch — the crop's own Cα distance matrix, site/mask flags, residue
indices, no new precomputation — and add it directly to the mutant projection `p_mt`, before
the LayerNorm and the mt−wt subtraction:

```
p_mt = p_mt + sigmoid(gate) * Linear(1 -> 64)(feature)      # gate initialised at sigmoid(-4) = 0.018
```

The gate is a single learned scalar, starting closed so a brand-new signal does not perturb an
otherwise-working representation before there is any gradient evidence it helps — the model
starts at approximately the ungated baseline and only opens the gate if the feature earns it.

Ten features tried, all through the same mechanism: the mutated residue's own minimum distance
to the nearest partner-chain residue, a hard- and a soft-cutoff local contact count, its
fractional position along its own chain, the percentile rank of its own interface distance
among all same-side crop residues, its distance relative to the crop's own mean, a
sequence-local density count, the actual structural embedding of its single nearest partner
residue, a smooth interface indicator, and the row's own mutation count. Full 15-combo runs:

| mechanism | per-cx r | ens | negative complexes | wc s\|d |
| --- | --- | --- | --- | --- |
| `burial_rank` | +0.311 | **+0.370** | 3 | **0.795** |
| `struct_film_chem` (no addition) | +0.314 | +0.369 | 3 | 0.758 |
| `nearest_partner_struct` | +0.306 | +0.366 | 5 | 0.794 |
| `rel_seq_pos` | +0.311 | +0.365 | 3 | 0.786 |
| `is_at_interface` | +0.311 | +0.364 | 3 | 0.790 |
| `contacts_soft` | +0.299 | +0.362 | 2 | 0.792 |
| `n_mut_row` | +0.304 | +0.359 | 4 | 0.779 |
| `contacts_8a` | +0.308 | +0.355 | 5 | 0.792 |
| `local_seq_density` | +0.295 | +0.346 | 4 | 0.790 |
| `dist_to_site` | +0.287 | +0.343 | 4 | 0.777 |
| `dist_delta_mean` | +0.271 | +0.340 | 4 | **0.165 spread** |

`wc s|d` is the within-complex AUC ranking a stabilising mutation above a destabilising one of
the SAME complex — identity-free, prevalence-free, the hardest of this project's ranking
metrics. **Every mode except `local_seq_density` and `dist_delta_mean` improves it over the
leader** in this table, several by a wide margin, even where ensemble Pearson is roughly
level — but §F.4 below found that this table's own baseline was wrong, and the improvement
does not survive the correction. `dist_delta_mean` is the clearest failure regardless: worst
ensemble score and, at 0.165, the widest seed spread of anything in this table.

Raw distance (`dist_to_site`) underperforms its own normalised form (`burial_rank`) by 0.027
ens here. Complexes vary enormously in interface size and packing, so a fixed distance means
different things in different complexes — 6 Å is buried in a tight, small interface and
essentially the contact surface in a large, loose one. Converting to a percentile rank within
the crop's own residues removes that confound, and is the single largest difference between
any two of the ten modes — though, again, see §F.4 for why this ranking should not be trusted
without the correction below.

### F.4 A confound found while running F.3 — and why the apparent win did not survive it

Every run in F.2 and F.3 above used `split_proj=true`, this project's own standing convention
for `struct_film_chem`. That flag was believed to control only sequence's per-side projection —
but the fix in E that gave structure its own per-side map (§E.2's own history) reused the SAME
flag rather than adding a new one, so `split_proj=true` has been silently splitting structure's
projection too, in every run since that fix landed. `struct_film_chem`'s own 50,497 parameters
are only reachable with structure's projection SHARED; `split_proj=true` under the current code
produces a 58,689-parameter model with structure split — the same architecture separately
measured in E.2 as `struct_film_chem_splitstruct`, **+0.354 ens, a 0.015 regression** against
the true leader.

Every number in F.2 and F.3 was therefore measured on a base architecture already 0.015 worse
than `struct_film_chem`, without that being visible in the tables above — they read as
comparisons against the leader, but the actual comparison was against a handicapped variant of
it. Fixed with a new `split_struct` field (`SiteTokenConfig`) that decouples the two: `None`
defers to `split_proj` exactly as before, so every existing config and every number in E and in
F.1–F.3 above is unaffected as a *record*; an explicit `True`/`False` overrides structure's
projection independently of sequence's. Re-running `burial_rank`, the best of F.3's ten, on the
TRUE `struct_film_chem` base (`split_struct: false`):

| run | params | per-cx r | ens | seed spread |
| --- | --- | --- | --- | --- |
| `struct_film_chem` (true base, no addition) | 50,497 | +0.314 | **+0.369** | 0.092 |
| `burial_rank` on the SPLIT-STRUCT base (F.3's number) | 58,818 | +0.311 | +0.370 | 0.087 |
| `burial_rank` on the TRUE base (`split_struct: false`) | 50,626 | +0.292 | +0.335 | **0.147** |

**The apparent tie was an artefact of the wrong baseline, not a real improvement.** On the
architecture `struct_film_chem` actually is, `burial_rank` is a clear regression — 0.034 ens
below the leader, with the widest seed spread measured for any variant of this backbone in the
project. It was not improving the real model; it was compensating for an unrelated regression
(split structure) that it happened to be measured on top of, landing at roughly the split
architecture's score by coincidence of two effects working in opposite directions on two
different problems. **`struct_film_chem` remains the submitted model.** None of the other nine
modes in F.3 has been re-verified on the true base, but the mechanism that seemed most promising
did not survive the correction, and the confound applies identically to all ten — the F.3 table
should be read as measuring a different (weaker) base architecture than the one it appears to
compare against, not retried in the hope a different mode fares better.

**Worth keeping regardless of the negative result:** the `split_proj`/`split_struct` coupling
was a real bug affecting every run in this project since structure's split was added in Family E
— not just this sweep — and the decoupling is a genuine fix, independent of whether any feature
in F.3 turns out to help. It is also a second instance of this project's most persistent lesson:
comparing a promising result against the wrong baseline is easy to do by accident, and checking
which exact architecture a number was measured against is not optional.

### F.5 Checkpoint transfer from full SKEMPI — two ways tried, both short of training on AB/AG alone

Separately from F.1–F.4: does pretraining on the ~300 non-antibody SKEMPI complexes help,
either mixed into every batch or as initialisation before fine-tuning on AB/AG. Full-SKEMPI
infrastructure (5,748 rows, 343 complexes) built earlier in the project made both cheap to try.

| approach | ens | note |
| --- | --- | --- |
| `struct_film_chem` (AB/AG only, with chem) | **+0.369** | the leader |
| pretrain (backbone) → fine-tune with chem (partial load) | +0.362 | ties the leader; see below |
| `control_nochem` (AB/AG only, no chem, matched architecture) | +0.304 | the fair baseline for the two below |
| pretrain (backbone, no chem) → fine-tune, no chem | +0.272 | below its own matched baseline |
| batch-mixing curriculum (AB/AG share annealed 50%→100%) | +0.241 | worst of the three |

Pretraining excludes each AB/AG fold's own held-out rows from the pretrain pool (a `--fold-map`
assigning AB/AG rows their real CV fold and every other SKEMPI row a sentinel that never
matches), so the fine-tune test fold is never seen during pretraining. Checkpoint transfer uses
`--init-ckpt-dir`/`--save-ckpt-dir`, added to `_perturb_v2_colab.py` for this.

The first pretrain→fine-tune attempt (+0.272) looked like a clean negative result — pretraining
hurt. It was really about chemistry: full SKEMPI's cache only has the 21 base chemistry columns
computed for all 5,748 rows, not the 33/39-column enriched table AB/AG's own cache has, so a
checkpoint trained with chemistry there cannot be loaded into a model expecting AB/AG's richer
block — the two head widths differ. Restoring chemistry via a name-and-shape PARTIAL load (chem
only touches the head's first Linear, which necessarily differs in width and re-initialises;
every backbone tensor — the part chemistry never touches — transfers exactly) reaches +0.362,
tying the leader and improving `wc s|d` (0.799 vs 0.758). The batch-mixing curriculum, tried
first and independently, scores lowest of the three and was not worth revisiting with the same
fix, since it mixes the two populations throughout training rather than sequencing them —
there is no separate "backbone" stage to protect from the chemistry mismatch.

---

## Family G — training objectives, and a factorial sweep of the leader's own architecture

A later phase again, on the true `struct_film_chem` base (`split_struct: false`) throughout.
Two questions: does the LOSS the leader is trained with matter as much as its architecture,
and does deepening or re-fusing the architecture itself — not adding a new feature, just
changing how the existing two branches combine — beat it.

### G.1 An AbRank-style ranking loss, mixed in at several weights

This project already measured a within-complex pairwise ranking loss once (JUSTIFICATIONS.md
§A1b) and rejected it — on the pre-correction 997-row dataset, with the OLDER `fusion_v2`
backbone, not the current leader. Re-measured on `struct_film_chem` itself, `MSE + λ ·
pairwise_rank_loss` (softplus logistic on pairs with |Δtrue| > 0.5 kcal/mol, never replacing
the regression term — pure ranking has no anchor on output scale, the same finding as before):

| weight λ | ens | seed spread |
| --- | --- | --- |
| 0 (control, reproduces the leader exactly) | +0.369 | 0.092 |
| 0.2 | +0.365 | 0.095 |
| 0.5 | +0.358 | 0.102 |
| **1.0** | **+0.371** | 0.125 |

Non-monotonic: 0.2 and 0.5 both underperform the control, 1.0 edges above it. The same
non-monotonic pattern (a low mixing weight hurting more than a high one) appeared in the
original AbRank measurement too. A parallel sweep of `1 - Pearson correlation` within each
complex (`--corr-weight`, a direct differentiable surrogate for the headline metric instead of
a ranking-margin term) is monotonically WORSE as its weight increases (+0.363, +0.352, +0.350
at 0.2/0.5/1.0) — correlation-as-loss does not help at any weight tried.

### G.2 Ten ranking methods at a fixed weight, and a re-measurement that mattered

Ten formulations of "make correct relative order more likely," all mixed at λ=1.0, replacing
just the pairwise term above: a hinge margin instead of the logistic; the logistic weighted by
|Δtrue| ("severity"); an "equal-pairs attract" term giving the |Δtrue| ≤ 0.5 bin its own loss
instead of excluding it; a smooth (tanh-based) Kendall-tau-style concordance; the classification-
labels-as-ranking-metric idea directly — 5 ordinal bins (≪, <, ≈, >, ≫ at ±0.5/±1.5 kcal/mol)
via both a cumulative-threshold BCE and an RBF-softmax cross-entropy; a soft-rank Spearman
correlation; ListMLE (Plackett-Luce negative log-likelihood of the group's true order); and a
triplet hinge anchored on each complex's own most-stabilising and most-destabilising row.

| method | ens | per-seed mean | seed spread |
| --- | --- | --- | --- |
| `class5_ce` (RBF softmax, 5 bins) | **+0.401** | +0.310 | 0.150 |
| `kendall_soft` | +0.391 | **+0.326** | 0.087 |
| `class5_ord` (cumulative BCE, 5 bins) | +0.389 | +0.304 | 0.149 |
| `severity` | +0.388 | +0.299 | 0.154 |
| `spearman_soft` | +0.382 | +0.325 | 0.116 |
| `equal_attract` | +0.373 | +0.312 | 0.101 |
| `triplet` | +0.373 | +0.312 | 0.115 |
| `binary` (= G.1's λ=1.0) | +0.371 | +0.315 | 0.125 |
| `hinge` | +0.370 | +0.309 | 0.123 |
| `struct_film_chem` (no addition) | +0.369 | +0.314 | 0.092 |
| `listmle` | +0.369 | +0.310 | 0.089 |

Every one of the ten ties or beats the leader on ensemble at 3 seeds. `kendall_soft` looked
like the one worth trusting — best per-seed mean, and the only method with a TIGHTER seed
spread than the leader's own. **An 8-seed rerun, `kendall_soft` against the leader on the
identical 8 seeds, found the tight spread did not survive:**

| run | seeds | ens | per-seed mean | seed spread |
| --- | --- | --- | --- | --- |
| `kendall_soft` | 8 | **+0.379** | **+0.290** | 0.217 |
| `struct_film_chem` | 8 | +0.368 | +0.281 | 0.217 |

Both land at the SAME spread once measured on enough seeds — 0.087 was itself a lucky draw of
3 seeds, not a property of the method, and the leader's own true spread (0.217) is more than
double what its 3-seed table (0.092) suggested throughout every OTHER comparison in this
document. What survives: a small, consistent +0.01 edge on both ensemble and per-seed mean, on
the same 8 seeds — real, but a modest finding, not the standout it looked like at 3 seeds. The
lesson generalises past this one method: **a 3-seed spread this project has quoted everywhere
is a floor on the true variability, not an estimate of it**, and a result that depends on being
in the tail of that floor should be re-measured at more seeds before being trusted.

**Applying that lesson to the two other candidates this sweep produced** (§G.3's best depth
config, §G.5's `split_mutwt_tied_init`): both re-measured at 8 seeds, matched against the
plain leader on the identical seeds.

| run | seeds | ens | per-seed mean | seed spread |
| --- | --- | --- | --- | --- |
| `kendall_soft` | 8 | **+0.379** | **+0.290** | 0.217 |
| `struct_film_chem` (plain regression) | 8 | +0.368 | +0.281 | 0.217 |
| both-branches-depth-2 (§G.3) | 8 | +0.359 | +0.298 | **0.148** |
| `split_mutwt_tied_init` (§G.5) | 8 | +0.353 | +0.250 | 0.247 |

Neither survives as an ensemble improvement — both land BELOW the plain leader once matched
on 8 seeds, not just inside its noise. The depth-2 config's profile is genuinely different
(highest per-seed mean of the four, tightest spread by a wide margin) but its ensemble is
lower, and this project ships the ensemble (README's own `ens` vs `per seed` distinction) —
by that standard it is a worse choice, an interesting reliability/ensemble trade-off rather
than a candidate. `split_mutwt_tied_init` is worse on every column. **`kendall_soft` is the
only one of the three that beats the leader on the metric that matters, is now part of the
submitted training recipe** (DETAILS.md's `struct_film_chem` section), and the other two are
recorded here as negative results, the same way §Family F's confound was.

### G.3 Depth × fusion, a factorial sweep on the leader's two branches

Two structural questions, crossed: does making the sequence-delta MLP (`pair_ffn`) or the
structure MLP deeper help, and does FiLM's gate beat simpler ways to combine the two branches
once both are fully pooled. `seq_mlp_depth`/`struct_mlp_depth` ∈ {1 (leader), 2, 3}; `fuse_mode`
∈ {film (leader), multiply, add, concat}. Full 15-combo runs:

| config | ens | seed spread | wc s\|d |
| --- | --- | --- | --- |
| **both depth 2, FiLM** | **+0.372** | 0.103 | **0.795** |
| `struct_film_chem` (both depth 1, FiLM) | +0.369 | 0.092 | 0.758 |
| struct depth 2, FiLM | +0.367 | 0.086 | 0.801 |
| seq depth 2, FiLM | +0.360 | 0.098 | 0.755 |
| struct depth 2, multiply | +0.359 | 0.129 | 0.812 |
| both depth 3, FiLM | +0.353 | 0.130 | 0.758 |
| both depth 2, multiply | +0.339 | 0.101 | 0.760 |
| seq depth 3, FiLM | +0.338 | 0.100 | 0.775 |
| seq depth 2, multiply | +0.321 | 0.137 | 0.776 |
| both depth 1, multiply | +0.263 | 0.129 | 0.760 |

Separately, `fuse_mode` alone (both branches at depth 1, matching the leader exactly except
the combining rule): **FiLM +0.369 > concat +0.351 > add +0.337 > multiply +0.263** — the same
ordering holds throughout the depth sweep above (every multiply row is well below its FiLM
counterpart at the same depth), confirming this is about the mechanism, not an interaction with
depth. FiLM's combined multiplicative-and-additive gate is doing real work no single simpler
rule reproduces.

**Depth 2 on both branches together is the best single configuration measured in this
project** (+0.372 ens) and also has the best `wc s|d` of anything in this table (0.795,
struct-depth-2-alone reaches 0.801, the single highest). Depth 3 reverses the gain on both
axes — 752 training rows per fold cannot support a third layer on top of the frozen encoders.
The improvement is inside the seed-spread floor established in §G.2 (0.217, not the 0.09ish
these 3-seed numbers show), so it is reported as a candidate worth an 8-seed confirmation, not
a settled win.

### G.4 Early vs mid vs late fusion — WHEN structure enters, not just how

A different axis: at what point does structure enter the sequence computation, relative to the
per-residue mutant/wild-type delta. **Late** (`struct_film_chem` itself, unchanged): the
sequence branch — LN(mt) − LN(wt) → pair_ffn → sum over sites — never sees structure at all;
FiLM combines the two only after BOTH are fully pooled into one vector each. **Early**: the
wild-type structure embedding at each residue is added to both p_mt and p_wt BEFORE LayerNorm
and the subtraction — structure shapes what "the same residue" means before any delta exists.
**Mid**: structure is added to the per-residue delta AFTER the subtraction and `pair_ffn`, but
before summing over the mutated sites — reweighting an already-computed "what changed" signal,
one residue at a time.

| fusion stage | ens | seed spread | neg | wc s\|d |
| --- | --- | --- | --- | --- |
| **late** (`struct_film_chem`) | **+0.369** | 0.092 | 3 | 0.758 |
| mid | +0.312 | 0.126 | 6 | 0.772 |
| early | +0.302 | 0.176 | 3 | 0.741 |

Late fusion wins clearly, and mid beats early. The ordering makes sense in hindsight: late
fusion lets the sequence branch finish computing an already-meaningful "what changed" signal
before structure conditions the final decision. Mid fusion perturbs that signal after it exists
— a smaller intervention. Early fusion perturbs BOTH p_mt and p_wt with the identical
wild-type structure embedding before their SEPARATE LayerNorms; because LayerNorm normalises
each vector by its own mean and variance, adding an identical vector to two different inputs
does not cancel in the subtraction the way it might if this were linear, and empirically this
is the most damaging place to introduce structure of the three tried. **The leader's own design
— fuse only after both branches are fully formed — is not an arbitrary choice among these
three; it is the best of the three by a wide margin, wider than any other single ablation in
this document moved the leader.**

### G.5 Two smaller checks: a wild-type binding-strength feature, and concat instead of subtract

`struct_bind_feat`: the wild-type complex's own ProteinMPNN log-probability (`mpnn__logp_wt_
complex`, a proxy for how strong the wild-type binding already is), concatenated onto z_st
before FiLM rather than left in the tail chem block. **+0.335 ens** — a regression, and with a
wide seed spread (0.171). Structure's own pooled representation does not benefit from a scalar
that chemistry already carries at the tail.

`mut_pair_op="concat"`: replace the fixed LN(mt) − LN(wt) subtraction with [LN(mt); LN(wt)],
2w-wide, reduced back to w by `pair_ffn`'s own first layer instead of a rule fixed in advance.
**+0.368 ens** — ties the leader (within noise), at the cost of 4,096 more parameters for the
wider first layer. Letting the network learn how to combine mt and wt does not beat simply
subtracting them; the fixed rule was not leaving anything on the table.

`split_mutwt_tied_init`: revisits §F.1's regression (giving mt/wt separate projections cost
0.075 ens and doubled the seed spread) by initialising the two projections IDENTICAL per side
(copying one to the other, then perturbing each with independent noise, std 0.02) instead of
independently random, so the split architecture starts at approximately the shared-weight
leader's own computation and only diverges from there. **+0.375 ens** — recovers past the
leader's own score, and reaches the best `wc s|d` measured in this project (0.819). But the
per-seed mean (+0.261) and seed spread (0.229) are barely improved from the untied version
(+0.260, 0.237) — the ensemble recovery looks like averaging out continued per-run instability,
not fixing it. Tied initialisation fixes the SYMPTOM the earlier ablation measured (the
ensemble number) more than the CAUSE (mt and wt still drift into different spaces over
training); worth the second look, not yet worth trusting as a replacement for the shared
projection.

### G.6 The submitted recipe under the homology-cluster split, and a seq-depth-2 variant that loses on both splits

Two follow-ups on `kendall_soft` (the training-objective addition selected in §G.2):
does its generalisation gap under the harder homology-cluster split match the plain
leader's, and does giving the sequence branch its own extra pre-fuse layer
(`seq_mlp_depth=2`, §G.3's `arch_seq2_film` but now trained WITH the ranking-loss
addition) help once combined with it. Both checks run on the same 4-fold cluster split
used throughout this document (`data/cluster_folds.csv`), 3 seeds/fold:

| run | ens (cluster) | ens (standard split) | retention |
| --- | --- | --- | --- |
| `struct_film_chem` (plain MSE) | +0.272 | +0.369 | 74% |
| `struct_film_chem_kendall` (`kendall_soft` addition) | +0.258 | +0.391 | 66% |
| `+seq_mlp_depth=2` on top of `kendall_soft` | +0.215 | +0.354 | 61% |

The plain-MSE leader's cluster number reproduces its historical +0.272 exactly on the
current, decoupled architecture (`split_struct` refactor confirmed neutral). Against that,
`kendall_soft` retains less of its own standard-split gain under the cluster split (66%
vs. 74%) — a modest generalisation cost that comes with the ranking-loss addition,
disclosed honestly in [DETAILS.md](DETAILS.md#struct_film_chem) rather than left out.
Adding a second sequence-branch layer on top makes both numbers worse, not just the
cluster one — it loses on the standard split alone (+0.354 vs. `kendall_soft`'s +0.391,
consistent with §G.3's `arch_seq2_film` losing to the depth-1 leader there too) and
loses by a wider margin under the cluster split (+0.215, worst of the three, and the
lowest retention). Extra sequence-branch depth does not help this model on either split;
not pursued further.

---

## H. The antigen/antibody gap: an 11-variant sweep

`struct_film_chem`'s pooled Spearman (`scripts/mut_side_split.py`, matching Part I §2's original
methodology) splits sharply by which side carries the mutation: antibody ρ **+0.629** (539 rows,
35 complexes) against antigen ρ **+0.252** (320 rows, 26 complexes) — the same architecture,
same weights, same training run, reading two different distributions through the identical
`pair_ffn`. Eleven variants were tried against this gap specifically, each scored on the same
antigen-only split rather than the pooled metric, since the pooled `ens` number can improve
while antigen itself gets worse (and did, for several of these):

| variant | mechanism | antigen ρ | antibody ρ | ens |
| --- | --- | --- | --- | --- |
| `struct_film_chem` (baseline) | — | +0.252 | +0.629 | +0.369 |
| `mut_side_embed` | learned 3-way (ab/ag/both) category embedding added to the pooled sum | +0.208 | — | — |
| `mut_side_gate` | FiLM-style per-side gate on the pooled sum, `1+delta` (not sigmoid) | +0.169 | — | — |
| `proj_depth_ag` | independent depth-2 projection, antigen side only | +0.201 | — | — |
| `proj_depth_both` | independent depth-2 projection, both sides (matched control) | +0.141 | +0.570 | +0.353 |
| `side_tag_add` | per-side (ab=0/ag=1) identity added inside `pair_ffn`'s own computation | +0.165 | +0.556 | +0.374 |
| `side_tag_concat` | same, concatenated instead of added (wider `pair_ffn` input) | +0.142 | +0.649 | +0.349 |
| `oversample_ag2` | `WeightedRandomSampler`, antigen-side rows at 2× weight | +0.227 | +0.575 | +0.355 |
| `oversample_ag3` | same, 3× weight | +0.186 | +0.483 | +0.332 |
| `projag_oversample` | `proj_depth_ag` + 2× oversampling | +0.214 | +0.585 | +0.358 |
| `sidetagadd_oversample` | `side_tag_add` + 2× oversampling | **+0.258** | +0.598 | +0.372 |
| `sep_struct_pca` (see below) | per-side structure PCA instead of one pooled fit | +0.211 / **+0.264**\* | +0.549 / +0.566\* | +0.365 |
| `stack_sidetag_oversample_pca` | `side_tag_add` + 2× oversampling + `sep_struct_pca`, 5 seeds | +0.232 | +0.535 | +0.350 |

\* `sep_struct_pca` was first measured at 3 seeds (+0.211); a 5-seed re-measurement of the
identical config (`baseline_newpca`) landed at +0.264 — a 0.05 swing on the SAME configuration,
which is the antigen split's own seed noise (n=320 rows, 26 complexes) rather than a real
effect, and a concrete illustration of why this document scores an antigen-only split at more
than one seed count before trusting it.

**10 of 11 variants made antigen prediction worse**, most by more than the antigen split's own
seed-to-seed spread. The one exception, `sidetagadd_oversample`, gained **+0.006** over
baseline — and its antibody score fell by **−0.031**, about five times the size of the antigen
gain, in the opposite direction. Every variant that moved antigen moved antibody too, mostly by
more: the mechanisms tried all act on the SAME shared `pair_ffn` weights antigen and antibody
both pass through, so a change that helps one side's harder distribution generally taxes the
other side's easier one.

**A twelfth check settles whether that one exception was real**: stacking `sidetagadd_oversample`
with the `sep_struct_pca` fix, at 5 seeds, does not compound the gain — it reverses it. Antigen
falls to +0.232 (below baseline AND below `sidetagadd_oversample` alone), antibody falls to
+0.535 (its worst value in the whole sweep), and `ens` falls to +0.350 (worst in the sweep).
Combined with `sep_struct_pca`'s own 3-seed-vs-5-seed disagreement (+0.211 vs +0.264 on the
identical config) this closes the question: **the one apparent win in this sweep does not
survive a second look, and none of the twelve variants is adopted.** The antigen/antibody gap
is real, reproducible, and — on every mechanism tried here (side-identity signal, extra
capacity, oversampling, and a real structure-PCA inconsistency fix, alone or stacked) — not
closeable without giving up antibody performance at a worse exchange rate than any antigen gain
was worth, when the antigen gain survived scrutiny at all.

`sep_struct_pca` is worth separating from the other ten: it is not a speculative architecture
addition but a genuine inconsistency fix. `fold_pca` fit ONE sequence PCA per side (antibody and
antigen bases fit separately, correctly, because the two are different distributions) but ONE
POOLED structure PCA shared across both sides — the exact mistake the sequence side's own
fitting code argued against. Fitting the structure PCA per side too (same code path, same
`_regression_check.py`-style before/after verification) is the more consistent, more honest
default regardless of its antigen effect. Measured effect on the primary metric: **neutral to
slightly negative** (+0.365 ens vs. baseline's +0.369, both at their respective seed counts,
well inside the ~0.09 seed spread this metric carries) — implemented and verified, not adopted
as the new default, kept available (no CLI flag currently gates it; reverting requires reverting
`fold_pca`'s struct-fitting loop in `_perturb_v2_colab.py`).

---

## I. A verified structure-input limitation for the reverse-mutation augmentation

The reverse-mutation augmentation (`DS.__getitem__`'s `swap` branch, `_perturb_v2_colab.py`)
swaps wild-type/mutant sequences, BLOSUM rows, and negates the label — but `struct_ab`/
`struct_ag` was read from `Cache.mpnn(complex_key)`, a per-COMPLEX cache built once from the
wild-type structure, **identically regardless of swap**. Every mutation of a complex, forward or
reversed, saw the same structural input: a per-complex constant blind to which mutation was in
play (`docs/FUTURE_WORK.md` item 9b/6).

**Built and wired** (not merely proposed): a per-ROW ProteinMPNN structural-embedding cache
(`experiments/protattba_repro/extract_mutant_struct.py`), reusing `encoder_h_V`'s exact
extraction logic (`src/features/mpnn_repr.py`) pointed at each row's own FoldX `BuildModel`
mutant structure (`data_mutants/<row_safe_name>.pdb`, 940/940 extracted, 81s on a T4) instead of
the wild-type complex PDB. `Cache.mpnn_mutant(row_id)` and a `--mutant-struct` flag route a
swapped row's `struct_ab`/`struct_ag` through this cache instead of the wild-type one; the
forward direction is provably untouched (`_check_mutant_struct.py`, a direct before/after
comparison on real rows rather than a synthetic batch, since this is a data-loading change, not
a model change).

**Verified, before training anything, that this cannot show a per-mutation effect.** ProteinMPNN's
`encoder_h_V` is computed from backbone coordinates alone (N, CA, C, O — no side chain, no
sequence identity; `src/features/mpnn_repr.py`'s own docstring: *"h_V comes from backbone
geometry alone... it cannot distinguish two substitutions at the same position"*). FoldX's
`BuildModel`, run the way it was for this project's mutant-structure extraction (one run, no
explicit backbone relaxation), repacks side-chain rotamers only — the backbone it outputs is
**bitwise identical** across every mutation of a complex. Checked directly, not assumed: parsing
raw backbone coordinates for every row, **0 of 887 within-complex row pairs differ at all**
(`np.array_equal`, max diff exactly 0.0, across the full dataset). The two facts compound: the
one feature this fix touches is mathematically incapable of reflecting the one thing FoldX
changed. This is a different situation from the tabular-geometry analogue
(`src/features/geometry_mutant.py`'s rSASA/contacts), which DOES depend on side-chain atoms and
genuinely differs by mutation — that fix is real; this one, for `struct_ab`/`struct_ag`
specifically, is not.

A real fix would need the backbone itself to move between mutations — either FoldX run with
explicit backbone flexibility/relaxation enabled, or a different mutant-structure predictor
entirely (e.g. a full-atom, backbone-flexible model). Neither is in scope here.

**Trained anyway, 3 seeds/fold, matched control (`--mutant-struct` on vs. off, otherwise
identical config)** — the training comparison this project's own discipline calls for even when
a null result is expected, not assumed from the backbone check alone:

| run | ens | per-cx | spread | neg | wc s\|d | antigen ρ | antibody ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mutant_struct_off` (control) | +0.365 | +0.293 | 0.079 | 5 | 0.778 | +0.211 | +0.549 |
| `mutant_struct_on` | +0.365 | +0.293 | 0.079 | 5 | 0.778 | +0.211 | +0.549 |

**Every metric matches to the last reported digit.** Not "close" — identical, confirming the
backbone-invariance check directly: the tiny residual difference between a FoldX-repaired
structure and the original wild-type PDB (RepairPDB's one-time fixup, ~0.002 max per-feature
difference, itself unrelated to any specific mutation) has no measurable effect on this
architecture's predictions once trained. `--mutant-struct` is left in the codebase, off by
default, as verified-inert groundwork for a future backbone-flexible structure source rather
than reverted — the wiring, cache, and regression check are correct and reusable; only the
current FoldX extraction's rigid backbone makes today's result a null one.
