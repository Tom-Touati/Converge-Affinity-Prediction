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
> apparent win. Where families disagree, the later one is the later measurement and says so
> explicitly.

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
