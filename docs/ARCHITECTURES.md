# Every architecture tried, what it tested, and what it showed

`docs/ARCHITECTURE.md` is the design rationale written *before* the ladder ran — what 750
training rows can support, the pipeline, the parameter budget. This is the record of what was
actually built and measured afterwards.

**45 neural configurations completed at least one full 5-fold run**, plus the forest ladder.
Every number is per-complex Pearson on the frozen by-complex split, scored against one common
truth by `scripts/report_runs.py`. Seeds are averaged where more than one exists; the count is
in the table.

> **Read every gap against the seed spread.** Measured at **0.028 to 0.117** depending on
> configuration (ERROR_ANALYSIS §9). Most rows in this document are separated by less than
> that. The families tell a story; individual orderings inside a family mostly do not.

---

## Summary — the whole ladder in one view

| family | best | params | per-cx r | what the family established |
| --- | --- | --- | --- | --- |
| **A.** full attention model (v2/v3/v4) | `full` | 810,886 | +0.200 | capacity is not the constraint |
| **B.** pooled-delta MLPs | `mlp_delta_chem_reg` | 85,376 | +0.266 | chem features carry the gain |
| **C.** site-token fusion ladder | **`cat128_reg2_l1` ← the model** | 36,353 | +0.293 | fusion *mechanism* is not the constraint |
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
| **`cat128_reg2_l1` ← the model** | 36,353 | +0.293 | plain concatenation, one-layer head, **separate projection per modality** |
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
