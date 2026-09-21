# AbSci HER2 dataset — profile and verdict

Assessment of [AbSciBio/unlocking-de-novo-antibody-design](https://github.com/AbSciBio/unlocking-de-novo-antibody-design),
the experimental release accompanying Shanehsazzadeh et al., *Unlocking de novo antibody design
with generative artificial intelligence* (Absci;
[bioRxiv 2023.01.08.523187](https://www.biorxiv.org/content/10.1101/2023.01.08.523187v1)).

Every number below was computed by downloading the three CSVs and profiling them against our
pipeline's requirements. Licence is **Clear BSD** — permissive, no barrier to use.

**Two different papers are involved, and §9 covers the second one.** The dataset comes from
Absci. [Nature Communications 15:7785 (s41467-024-51563-8)](https://www.nature.com/articles/s41467-024-51563-8)
is a *different* paper — **GearBind**, Cai, Zhang, Wang, Zhong et al., *Pretrainable geometric
graph neural network for antibody affinity maturation* — which **consumes** this dataset as an
independent held-out test set. That paper is read and analysed in §9, and it independently
confirms most of the profile below.

## Verdict up front

| Use | Verdict |
| --- | --- |
| **Held-out generalisation test set** | **Strong — the best available, and it is what the authors intended** |
| **Extra training data for ΔΔG regression** | **No.** Same complex as one of our test complexes; wrong mutation regime; adds ~zero stabilising examples |
| **Censored / ranking supervision** | **Promising** — 1,097 labelled non-binders, but only once a ranking objective exists |

## 1. What the release contains

| File | Rows | Cols | Content |
| --- | --- | --- | --- |
| `spr-controls.csv` | **1,855** | 5 | HCDR1, HCDR2, HCDR3, `KD (nM)` (758 non-null), `Binder` (bool) |
| `zero-shot-binders.csv` | **422** | 7 | HCDR3, `KD (nM)`, −log KD, edit distances to trastuzumab / SAbDab / OAS |
| `functionality-developability-cross_reactivity-data.csv` | **13** | 48 | Lead candidates: Fab/mAb KD, cELISA, ADCC, cross-species, DSF, SEC, AC-SINS |

One antigen (**human HER2**), one parent antibody (**trastuzumab**, HCDR3 `SRWGGDGFYAMDY`,
KD 1.9 nM), one reference structure (**PDB 1N8Z**). No PDB files are shipped.

The 13-row file is a developability panel, not modelling data. The analysis below covers the
other two.

## 2. These are loop redesigns, not point mutations

This is the first and largest incompatibility. Our pipeline's mutation grammar
(`src/structures.py:MUT_RE`) expresses a row as substitutions at numbered positions — it cannot
represent an insertion or deletion at all.

| | zero-shot-binders | spr-controls |
| --- | --- | --- |
| HCDR3 length range (parent = 13) | 11–15 | 9–17 |
| Same length as parent | 201 (48%) | 1,266 (68%) |
| Median edit distance to trastuzumab | **8** | **8** |
| 1–2 substitutions | 1 (0.2%) | 22 (1.2%) |
| 5–7 substitutions | 152 (36.0%) | 586 (31.6%) |
| 8+ substitutions | 264 (62.6%) | 1,216 (65.6%) |
| **Pure substitutions, no indel** | **174 (41.2%)** | **1,139 (61.4%)** |
| …and ≤7 substitutions (SKEMPI's max) | **59** | **442** |

So of 1,855 SPR controls, **442 (23.8%)** could be written as a SKEMPI-style mutation string.
The rest either contain indels or exceed seven substitutions.

And even those 442 are a different object from what we train on. SKEMPI's AB/AG subset is 70.5%
single-point and 53.5% alanine scans — small, local perturbations of a native interface. These
are **de novo HCDR3 loops at a median of 8 substitutions**, which is closer to "a different
antibody" than "a mutant". A model trained on one has little reason to transfer to the other, in
either direction.

## 3. ΔΔG is derivable — but the distribution is the opposite of what we need

KD is absolute, not a change on mutation. But the parent is known, so
ΔΔG = RT·ln(KD_variant / KD_trastuzumab) with KD_parent = 1.9 nM is well defined and on exactly
our scale.

| | n | mean | sd | range | stabilising (< −0.5) | destabilising (> +0.5) |
| --- | --- | --- | --- | --- | --- | --- |
| zero-shot-binders | 422 | +1.67 | 0.75 | [−0.42, +3.94] | **0 (0.0%)** | 400 (94.8%) |
| spr-controls | 758 | +1.78 | 0.89 | [−0.72, +4.26] | **3 (0.4%)** | 713 (94.1%) |
| **our SKEMPI AB/AG** | **997** | **+1.03** | **1.82** | [−4.91, +7.94] | **130 (13.0%)** | 544 (54.6%) |

The three stabilising rows match the paper's own claim that three designs bound tighter than
trastuzumab.

**This kills the main reason we would have wanted it.** Our documented gap
([`PLAN.md`](PLAN.md), Limitations) is that only **130 rows are clearly stabilising**, and
affinity *improvement* is what antibody engineering actually wants predicted. AbSci adds
**three**. It is more skewed towards destabilising than our own data, not less.

There is a second, structural problem. Every row shares one parent, one antigen and one assay, so
they all carry **one identical per-complex offset**. Under our headline metric — per-complex
Spearman — 1,855 rows from one complex count as **one complex**, alongside the 27 we already
have. It would add 2× our row count and 1/27th of our metric's resolution, while doubling the
weight of a single epitope in training.

## 4. The censored labels are the genuinely interesting part

| | count | share |
| --- | --- | --- |
| Binder, with measured KD | 758 | 40.9% |
| **Non-binder, labelled, no KD** | **1,097** | **59.1%** |

Those 1,097 have the same structure as the 80 `n.b.` rows we currently discard from SKEMPI: a
real, informative label with no number attached. Under a ranking or censored-regression
objective a non-binder is a valid constraint — *ranked below every measured binder in the same
system* — and 1,097 of them is more censored data than the whole of our current dataset.

That is a real asset. It is also gated on machinery we explicitly deferred: the pairwise ranking
rung was parked on 2026-09-20 with a narrow trigger — *"build it only when we want the
non-binders and censored rows back."* This dataset is a reason to revisit that trigger, not a
reason to act now.

## 5. Leakage: this is the same complex as one of our test complexes

**PDB 1N8Z is in our dataset.** It is not a near neighbour — it is literally one of the 27
complexes our headline metric is computed on.

```
1N8Z_AB_C     14 rows     FOLD 3     cluster 3N85_A_LH
cluster members: 1BJ1_HL_VW  1CZ8_HL_VW  1N8Z_AB_C  1YY9_CD_A  3BDY_HL_V
                 3BE1_HL_A   3N85_A_LH   4KRL_A_B   4KRO_A_B   4KRP_A_B
rows in that cluster: 100
```

It is worse than complex-level overlap. Trastuzumab's HCDR3 maps to **chain B residues 97–109**
of 1N8Z, and **4 of our 14 SKEMPI 1N8Z rows mutate residues inside that exact loop**:

| Our mutation | ΔΔG | Inside HCDR3 (B97–109)? |
| --- | --- | --- |
| `DB102W` | **−0.71** | **yes** |
| `YB105F` | +0.85 | **yes** |
| `YB109V` | +0.23 | **yes** |
| `NA30S,HA91F,YA92W,TA94S,DB102W,YB105F,YB109V` | +0.05 | **yes** |
| `RB50A` | +2.98 | no (HCDR2) |
| `RB59A` | +1.80 | no |
| …8 others on chains A and C | | no |

`DB102W` at ΔΔG −0.71 is one of our scarce stabilising measurements, and it sits at position 102
— inside the loop AbSci redesigned 1,855 times.

**Consequence.** Adding this to training without removing `1N8Z_AB_C` *and its entire homology
cluster* (100 rows, fold 3) would contaminate a quarter of our evaluation. Our homology
clustering was built precisely to stop this, and it would catch it — but only if the AbSci rows
are assigned to the 1N8Z cluster when merged, which requires a deliberate step, not a default.

## 6. Structural compatibility

No structures ship with the dataset, but this is workable: we already parse 1N8Z, and HCDR3
locates cleanly at chain B 97–109 with an exact sequence match to the parent.

- **Same-length variants** (68% of spr-controls, 48% of zero-shot) could be threaded onto the
  1N8Z backbone position-for-position, and the full geometry + ProteinMPNN pipeline would run
  unchanged.
- **Indel variants** (32% / 52%) cannot. The backbone length changes, so there is no
  correspondence to thread onto — they would need a folded model (IgFold, ABodyBuilder) per
  variant.
- Either way, §2.2's ceiling still applies: threading gives every variant the *same* backbone, so
  the structure stream would be identical across all 1,855 designs.

## 7. Recommendations

1. **Use it as a held-out test set, which is what it is for.** Train on SKEMPI with the 1N8Z
   cluster **excluded**, then evaluate on the 442 substitution-only spr-controls. Different lab,
   different assay, different sequence regime, zero shared training signal — a genuinely strict
   generalisation test, and a far better one than another CV fold. This is the highest-value use
   and it costs one config change plus a loader.
2. **Do not merge it into training** for ΔΔG regression. It fails on three counts: wrong mutation
   regime (median 8 substitutions), no stabilising examples (3), and it collapses to a single
   complex under our headline metric — while overlapping a test complex.
3. **Revisit it when the ranking rung is built.** 1,097 labelled non-binders is the largest
   censored-data source available to us, and it is the strongest argument yet for the objective
   we deferred.
4. **If it is ever merged, re-run clustering jointly.** The AbSci rows must land in cluster
   `3N85_A_LH` with 1N8Z. `src/splits.py` will do this correctly given a sequence, but the merge
   has to happen before the split is frozen, not after.

## 8. What this does not solve

The dataset does not address the two things actually limiting us:

- **Stabilising examples.** 3 added, against a shortage of hundreds.
- **The §2.2 ceiling.** Every design shares one backbone, so the structure modality still cannot
  distinguish substitutions at a site. Only mutant side-chain modelling fixes that.

It is an excellent evaluation resource and a poor training resource, and the distinction is worth
stating plainly in the write-up.

---

## 9. GearBind — the paper that uses this dataset, and what it tells us

[Nature Communications 15:7785](https://www.nature.com/articles/s41467-024-51563-8), Cai, Zhang,
Wang, Zhong, Li, Zhong, Wu, Ying & Tang — *Pretrainable geometric graph neural network for
antibody affinity maturation*. GearBind trains on SKEMPI and uses the Absci HER2 data as an
**independent test set**, which is precisely the use §7 recommends.

### 9.1 It independently confirms this profile

> "This dataset contains high-quality binding affinity data, measured by surface plasmon
> resonance (SPR) on 419 HER2 binders with de novo designed CDR loops. The antibodies in the
> dataset are variants of Trastuzumab that have high edit distance (**7.6 on average**), making
> them **potentially challenging for ΔΔG_bind predictors trained on low-edit-distance data**."

They report mean edit distance 7.6; we measured median 8. And their stated concern is the same
one §2 reaches independently: this is a different mutation regime from the point mutations
SKEMPI-trained models see. It is the reason they use it as a test set rather than training data.

### 9.2 Their SKEMPI numbers, and why FoldX is the headline

Table 1 — split-by-complex fivefold CV on **full** SKEMPI v2.0, n = 5,729:

| Model | MAE ↓ | RMSE ↓ | Pearson ↑ | **Spearman ↑** |
| --- | --- | --- | --- | --- |
| FoldX | 1.364 | 2.027 | 0.491 | **0.526** |
| Flex-ddG | 1.236 | 1.849 | 0.497 | 0.484 |
| Bind-ddG | 1.255 | 1.759 | 0.581 | 0.443 |
| GearBind | 1.143 | 1.639 | 0.659 | 0.498 |
| GearBind+P *(pretrained)* | 1.115 | 1.611 | 0.676 | 0.525 |
| **Ensemble of all five** | **1.028** | **1.503** | **0.729** | **0.643** |

Three things fall out of this table that bear directly on our decisions.

**FoldX has the best Spearman of any individual model — 0.526, above the pretrained GNN's
0.525 and the plain GNN's 0.498.** It has the *worst* MAE and RMSE at the same time, so it ranks
well while being badly calibrated in absolute terms. The authors note the same asymmetry:
removing FoldX costs the ensemble more Spearman than removing anything else. For a project whose
headline metric is a rank correlation, that is a strong argument for the FoldX feature we have
twice recommended and not yet built.

**Geometric pretraining buys little.** GearBind → GearBind+P is +0.017 Pearson and +0.027
Spearman, from contrastive pretraining on mass-scale CATH structures. That is real but small,
and it is well inside our own harness's ±0.08 detectability floor. It is evidence against
expecting a pretraining stage to rescue a model at our sample size.

**The ensemble beats every member by a wide margin** — 0.643 Spearman against 0.526 for the best
single model. Our chem+geom+mpnn fusion is the same shape of result, arrived at independently.

*Comparability caveat:* their split is by complex on full SKEMPI; ours is by homology cluster on
the antibody–antigen subset only, and our headline averages Spearman within complexes rather than
pooling. Both differences make their numbers easier. These are not our numbers minus a constant.

### 9.3 They break the §2.2 ceiling the way we proposed — with a rotamer library

GearBind encodes **both** the wild-type and the mutant complex with a shared GNN, and the mutant
structure is built by **sampling side-chain torsion angles from a rotamer library**. The same
device powers their pretraining:

> "The model is trained to contrast between native structures and randomly mutated structures
> with **side-chain torsion angles sampled from a rotamer library**. Pretraining helps GearBind+P
> explore the energy landscape of native protein structures."

This is the rotamer-enumeration route proposed as option (b) for breaking §2.2, validated in a
Nature Communications paper and used for two jobs at once — generating mutant structures for
prediction, and generating decoys for self-supervision. It raises the priority of that option
relative to waiting on FoldX downloads.

### 9.4 Two more architectural confirmations

**The predictor is antisymmetric by construction.** ΔΔG is produced by "an antisymmetric
predictor given the GearNet-extracted representations of the two complexes" — the
ΔΔG(reverse) = −ΔΔG(forward) constraint is built into the architecture rather than tested after
the fact, which is how `PLAN.md` currently treats it.

**It is a graph network *with* attention, not an alternative to attention.** An earlier draft
of this section claimed GearBind supports "GNN over cross-attention". It does not, on two counts.
The paper runs no such comparison — its baselines are FoldX, Flex-ddG and Bind-ddG, none of which
isolates that axis. And GearBind itself is attentional:

> "Edge-level interactions are then captured by performing message passing on the line graph,
> **similar to a sparse version of AlphaFold's triangle attention**. Finally, after aggregating
> atom and edge representations for each residue, **a geometric graph attention layer** is applied
> to pass messages between residues."

That is the *hybrid* the `ARCHITECTURE.md` note lists as a middle option — attention computed only
over edges the graph admits, rather than over all pairs. The paper supports the hybrid. It says
nothing about pure message passing versus full cross-attention.

### 9.4b The ablation is the useful part, and it is not flattering to our design

| Ablation | Effect on SpearmanR |
| --- | --- |
| Multi-relational interface graph → **plain KNN graph** | **−23%** |
| Remove side-chain atoms from the graph | −15% |
| Full GearBind → simple RGCN (plain message passing) | −9% |

The ordering is the finding: **graph construction dominates the architecture built on it.**
Replacing typed, multi-relational edges with a KNN graph costs 23% Spearman; removing the
attention and line-graph machinery entirely costs 9%.

`ARCHITECTURE.md` Step 5 currently specifies a **top-48 nearest-neighbour** set — a KNN graph with
one edge type. By GearBind's measurement that is the weaker construction, and it costs more than
any of the fusion mechanisms we have been comparing.

The fix is cheap and it points at something already established here. §2.1 showed the binding
signal is the with-partner / without-partner *contrast*, and an earlier note observed that this is
naturally an **edge attribute**. Typed edges are how that gets expressed: at minimum
**same-chain vs cross-chain**, and plausibly also sequential-adjacency vs spatial-proximity, and a
side-chain vs backbone distinction given the −15% above. None of that costs parameters — it is
graph construction, not architecture.

The −15% for dropping side-chain atoms is a third independent vote for breaking the §2.2 ceiling:
GearBind's graph is full-atom, ours is not.

### 9.5 Their label-noise observation matches ours

> "When |ΔΔG_bind| is small, predictions from all methods have very low correlation with
> experimental ΔΔG_bind values, hinting either the **noises in data** or a deficiency of current
> tools in modeling weaker, more intricate interactions."

On the large-effect subset they report Pearson **0.707 against FoldX's 0.411**; on small-effect
rows, every method degrades. That is the same conclusion our GBT-vs-random-forest gap reached
from the other direction — near the measurement floor, model capacity stops being the binding
constraint.

### 9.6 What changes for us

| | Before reading GearBind | After |
| --- | --- | --- |
| FoldX features | recommended twice, not built | **stronger** — best individual Spearman in their table |
| Rotamer enumeration for mutant structures | proposed as the no-dependency option | **validated** — it is what GearBind does |
| Structural pretraining | argued down as low value for us | **partly corrected** — it works, but buys only +0.027 Spearman |
| GNN over cross-attention | argued from parameter counts | **not tested by this paper** — GearBind is itself attentional over graph edges, i.e. the hybrid |
| Neighbourhood as top-48 KNN | the Step 5 plan | **weakened** — KNN costs 23% Spearman against a typed multi-relational graph |
| Full-atom vs backbone graph | not considered | **add it** — dropping side chains costs 15% |
| Antisymmetry | a probe in `PLAN.md` | consider making it **architectural** |
| This dataset as a test set | our recommendation | **exactly what they do** |

One correction to an earlier position. Contrastive pretraining was argued down on the grounds
that aligning sequence and structure maximises *shared* information while fusion needs
*complementary* information. GearBind's pretraining is a different objective — native versus
rotamer-perturbed **decoy discrimination**, entirely within the structure modality — so that
objection does not apply to it. The argument against sequence–structure *alignment* stands; the
blanket scepticism about structural self-supervision does not.
