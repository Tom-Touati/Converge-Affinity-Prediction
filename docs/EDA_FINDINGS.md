# EDA findings — SKEMPI 2.0 antibody/antigen subset

Produced by `notebooks/01_eda.ipynb`. Every number below is reproduced by that notebook and was
independently re-derived with separate code before being written here.

## The usable set is smaller than the subset

| Step | Rows | Complexes |
| --- | --- | --- |
| AB/AG rows in SKEMPI 2.0 | 1,211 | 55 |
| After dropping rows with no parsable mutant or wild-type affinity | 1,131 | 54 |
| Unique (complex, mutation) pairs after deduplication | **997** | 54 |

Eighty rows carry no usable affinity and one complex disappears entirely with them. A further 256
rows are repeat measurements of 122 mutation-complex pairs. **The honest sample size is 997, not
1,211** — an 18% reduction from the headline figure, and the number that should appear in the
write-up.

## Structure verification passes completely

All 2,109 mutated positions across all 1,131 rows resolve in their PDB file and the stated
wild-type residue matches what the structure contains. Zero mismatches. This was confirmed twice
with different parsers (CA-only and all-atom).

This removes a risk the plan had budgeted for: no rows are lost to structure, so the structural
and sequence modalities see exactly the same training set and the fusion comparison is clean.

## Protein names cannot be trusted to identify the antibody

Matching antibody vocabulary against the `Protein 1` / `Protein 2` columns leaves five complexes
ambiguous. Scoring each chain group for conserved immunoglobulin framework motifs read straight
from the sequence resolves four of them, and exposes something more important: in `2BDN_HL_A` the
name-based and structure-based assignments **flatly contradict** each other. `Protein 1` is MCP-1,
the antigen, yet it corresponds to chains H and L. So the ordering of `Protein 1` / `Protein 2`
does not reliably follow the chain groups in `#Pdb`.

Consequence: chain roles get assigned from structure, never from the name columns. Anything built
on the assumption that Protein 1 is the first chain group would silently mislabel this complex.

The one complex that stays unresolved is `1DVF_AB_CD`, and correctly so — it is the anti-idiotype
pair D1.3 against E5.2, antibody binding antibody, with no antigen at all. Its 46 rows are a
genuine category of their own and should be flagged rather than forced into a side.

With roles assigned from structure, mutations divide as 655 on the antibody, 387 on the antigen,
43 spanning both, and 46 in the antibody-vs-antibody complex. The sides behave differently — mean
ΔΔG is +0.71 on the antibody side against +1.16 on the antigen side — which makes this a slice
worth adding to the error analysis. It was not in the original plan.

## The label carries real signal, and the interface gradient proves it

Mean ΔΔG is +1.06 with SD 1.80; 56% of rows are destabilising beyond +0.5, 12.6% stabilising
beyond −0.5, and 31.4% sit within ±0.5. Fifty-seven rows are exactly zero, which is worth a
suspicious glance later — it likely encodes "no change reported" rather than a measured zero.

Sorting single-point mutations by SKEMPI's own interface annotation gives a clean monotone
gradient: interface core +1.61, support +1.38, interior +0.74, rim +0.42, surface +0.08. That is
exactly what binding physics predicts, and it is the cheapest available confirmation that the
labels are not noise. It also sets the expectation for the structural modality: if structure is
doing its job, it should help most on core and support mutations.

## It is largely an alanine scan

70.5% of rows are single-point mutations, and 53.5% of those substitute alanine. Alanine
substitutions average +1.28 against +0.86 for everything else. A model can therefore score
respectably by learning "alanine scan means destabilising" without learning any binding physics,
which is why the headline metric gets reported separately on the 426 alanine singles and the 371
others.

## The noise ceiling, and why to quote the conservative one

Two estimates disagree, and the difference matters:

| Estimate | Noise SD | Max per-complex r | Max global r |
| --- | --- | --- | --- |
| Our duplicate pairs (95.1% agree within 1 kcal/mol) | 0.31 | 0.97 | 0.99 |
| SKEMPI paper (84% of 1,741 repeats within 1 kcal/mol) | 0.50 | **0.91** | 0.96 |

Our duplicates look cleaner than the paper's because they are frequently the same mutation
re-reported by the same group, whereas the paper's figure comes from independent groups. The
conservative number is the one to quote: **per-complex Pearson is capped near 0.91**, against a
global cap of 0.96. The headline metric has a materially lower ceiling than the global one, and
that should be stated before anyone reads a per-complex 0.6 as a weak result.

Within-complex ΔΔG spread (median SD 1.19) is much tighter than global spread (SD 1.80), which is
the whole reason the two ceilings differ — and the reason a global correlation flatters a model
that has only learned which complex it is looking at.

## Hazards, quantified

| Hazard | Rows | Share |
| --- | --- | --- |
| Temperature marked "assumed" | 540 | 47.7% |
| Duplicate (complex, mutation) rows | 256 | 22.6% |
| Non-native reference state (wild-type is an affinity-matured Fab or another mutant) | 167 | 14.8% |
| Curator flags a near-identical partner complex | 151 | 13.4% |
| Mutant affinity given as a bound | 45 | 4.0% |
| Wild-type affinity given as a bound | 41 | 3.6% |

Nearly half the subset has an assumed temperature, which is higher than the plan assumed and means
the ΔΔG scale is approximate for those rows. Assay method is also worth watching: the `SP` group
(n=44) averages +4.00 with SD 2.57, far outside every other method, and ELISA averages +1.48
against SPR's +0.85.

## Clustering works, and it changes the split plan

The 54 complexes collapse into **26 clusters**. The largest is unambiguous confirmation that a
random split would have leaked badly: thirteen complexes, 282 rows, 25% of the subset, every one
of them an antibody against hen egg-white lysozyme — D1.3, HyHEL-63, HyHEL-10, HyHEL-5 and D44.1.
A per-mutation random split would have put near-identical complexes on both sides of the line.

Two consequences for the plan:

**A balanced 5-fold split is not achievable.** That single lysozyme cluster is 25% of the data and
cannot be divided, so it becomes an entire fold on its own while the other four hold about 18.7%
each — a 1.33× imbalance, and a fold with no cluster diversity inside it. Options are four folds
instead of five, leave-one-cluster-out, or accepting the imbalance and reporting per-fold results
rather than a single average. This needs a decision before `src/splits.py` is written.

**SKEMPI's own grouping reaches outside the subset.** `Hold_out_proteins` explicitly names six
partner complexes, and four of them — 1GC1, 1UUZ, 3LB6, 3MZW — are classified outside AB/AG, so no
link can be formed within this subset. If we later add other SKEMPI subsets as extra data, those
links must be honoured or we reintroduce exactly the leakage we are trying to prevent.

After clustering, maximum antigen sequence identity between any test complex and any training
complex ranges from 0.22 to 0.69 across folds. The 0.69 cases show the 0.90 `difflib` threshold
used here is too permissive; MMseqs2 at 30% identity in `src/splits.py` will merge more and should
push that number down.

## What changes in the plan

1. Quote 997, not 1,211, as the sample size.
2. Drop the structure-verification risk — it passes completely.
3. Assign chain roles from structure, never from the `Protein 1` / `Protein 2` ordering.
4. Add antibody-side vs antigen-side as an error-analysis slice.
5. Flag `1DVF_AB_CD` (46 rows) as antibody-vs-antibody, not antibody-vs-antigen.
6. Decide between 4 folds, 5 unbalanced folds, and leave-one-cluster-out before writing
   `src/splits.py`.
7. Carry the SKEMPI cross-subset homology links forward into any future data addition.
8. Quote the conservative per-complex ceiling of 0.91.

## Next step

Repo skeleton with a dummy mean predictor running end to end through `make`, then `evaluate.py`
written against that dummy, then `src/splits.py` producing the frozen `folds.csv` with MMseqs2
replacing the `difflib` stand-in used here.
