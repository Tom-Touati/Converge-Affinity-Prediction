# Prior work — what the field does with SKEMPI, and what it changes here

Notes from reading the methods and datasets that benchmark on SKEMPI 2.0. The purpose is
calibration: what protocol counts as "the SKEMPI benchmark", what scores are actually achievable,
and which of our planned decisions the literature already validates or contradicts.

## The benchmark protocol is standardised — we should match it

RDE-Network and DiffAffinity/SidechainDiff both use the same setup, and it has become the de facto
protocol:

- **Three-fold cross-validation, split by structure**, each fold holding complexes that appear in
  no other fold.
- Metrics: overall Pearson, Spearman, RMSE, MAE, AUROC (on the sign of ΔΔG), **plus average
  per-structure Pearson and per-structure Spearman**, computed by grouping mutations by structure
  and **discarding groups with fewer than 10 mutations**.

Two things follow. First, the per-complex metric we picked as the headline is the field's own
convention, so that choice needs no special defence. Second, we should adopt the full metric suite
and the ≥10-per-group rule, or our numbers will not be comparable to anything published.

The important difference: their split is by *structure*, ours is by *homology cluster*. Ours is
strictly harder — it also separates the thirteen lysozyme complexes that a structure-level split
would happily spread across folds. **Our numbers will be lower than published ones, and the
write-up must say so up front**, or the submission reads as underperformance rather than a
stricter evaluation.

## Reference numbers, so we know what "good" looks like

Full SKEMPI, structure-level three-fold CV:

| Method | Overall Pearson | Overall Spearman |
| --- | --- | --- |
| DiffAffinity | 0.669 | 0.556 |
| RDE-Network | 0.632 | 0.527 |
| DDGPred | 0.630 | — |

RMSE lands around 1.54 and MAE around 1.09 kcal/mol for the best of these.

Per-structure correlations are far lower than overall ones, which is the same effect our own
ceiling calculation predicted. The precomputed-FoldX repository reports FoldX at **per-structure
Spearman 0.398 [0.276, 0.519]** on a thirteen-complex subset against CATH-ddG's 0.446, with pooled
single-point Spearman 0.437 across 4,081 rows. On antibody-specific test sets, abCAN reports
Pearson 0.731, against MutaBind2 at 0.627 and GeoPPI at 0.562.

**Calibration:** a per-complex Spearman around 0.4–0.5 on an antibody-only, homology-clustered
split would be a respectable result, not a weak one. Against our computed ceiling of 0.91 that is
the honest gap to discuss in the error analysis.

## The leakage estimate has a published magnitude

This is the most useful single number found. In the *Nature Computational Science* study of data
volume and diversity for antibody–antigen ΔΔG:

- Random train/test splits reached Pearson ≈ **0.87**.
- Imposing sequence-identity cutoffs so that no information leaks between train and test dropped
  performance by **63% on average**, and 90% CDR-identity cutoffs "degraded performance
  dramatically, revealing overtraining on the few hundred experimental data points available".

So the random-versus-clustered gap in our plan is not a footnote — on this data it is the majority
of the apparent performance. We will report that gap as our leakage estimate and now have an
external expectation to compare it against.

## Data volume is the bottleneck, and it is quantified

The same study ran learning curves from 580 to 450,000 mutations. Test correlation **only began to
plateau at around 90,000 mutations**, reaching 0.85. The authors' conclusion is blunt: "the major
challenge with experimental ΔΔG prediction lies in data availability rather than model
architectures", and current experimental sets — AB-Bind's 645 mutations, their own SKEMPI-derived
antibody benchmark of 608 — are short by orders of magnitude.

Our usable set is 997 unique complex-mutation pairs. That sits in the same regime as their 608.

**This changes priorities.** In the bottleneck map, data volume moves from "likely constraint" to
"almost certainly the binding constraint", and the learning-curve diagnostic should run early
rather than being deferred to the error-analysis phase. It also argues against spending the model
budget on fusion sophistication: the literature says architecture is not what is limiting us.

Diversity matters alongside quantity — minimum-substitution-type-diversity datasets showed 60%
lower standard deviation ratios than maximum-diversity ones. Given our subset is 53% alanine
substitutions, that is a direct warning about our own composition.

## A better version of our pretraining idea

Our deferred plan was to pretrain on real antibodies paired with random unrelated antigens as
confident non-binders. The literature has a stronger, already-validated version: **pretrain on
FoldX-generated synthetic ΔΔG**. The *Nature Computational Science* work built nearly a million
FoldX mutations and reached Pearson 0.89 under 90% CDR plus 70% antigen identity cutoffs, holding
0.89 even at 70%/70%, and stayed robust to injected Gaussian noise up to scale 5.

FoldX pseudo-labels beat random negatives on three counts: the signal is graded rather than
binary, it is structure-aware, and it is empirically shown to transfer under strict identity
cutoffs.

Better still, **we do not have to run FoldX**. The `skempi-foldx` repository ships precomputed
FoldX binding ΔΔG for essentially all of SKEMPI — 4,340 of 4,343 single-point entries and 1,767 of
1,850 multi-point — with twelve energy terms per mutation. That makes two things nearly free:

1. **A physics baseline rung** in the ladder, at a known difficulty (per-structure Spearman ≈ 0.40).
2. **A FoldX feature** the learned head can correct, which is the residual-correction framing that
   CATH-ddG and similar methods use to beat raw FoldX.

One caveat from that repository worth carrying: a mutation's FoldX ΔΔG depends on the whole
mutation list it was computed within — median spread 0.067 kcal/mol but up to 11.03. FoldX values
for multi-point mutations are therefore not decomposable into their singles, which interacts
directly with our additivity probe.

## Independent confirmation that these models do not generalise

Two 2025 papers land on the same verdict from different directions.

**AbAgym** curated roughly 324,000 non-redundant DMS mutations across 67–68 antibody–antigen
complexes, 36,541 of them at the interface, and benchmarked six computational methods on the
interface subset. The best result across all six was **Spearman 0.28 and ROC-AUC 0.72**.
Energy-based FoldX performed best and only marginally above baseline; evolutionary models did
poorly; and the authors conclude the machine-learning models "have been overtrained and do not
generalize outside their training set". Most pointedly, **relative solvent accessibility on its own
was a competitive baseline**.

**AbDesign** reaches the same place — its title states that a database of antibody point mutants
with associated structures "reveals poor generalization of binding predictions from machine
learning models".

Two consequences for us. Our ladder's trivial rungs are not strawmen: burial and substitution
chemistry are genuinely competitive with published deep models on held-out antibody data, so
beating them is a real result and failing to beat them is the expected outcome worth reporting
honestly. And the limitations section of the submission now has external support rather than
speculation.

## The antisymmetry probe has an off-the-shelf benchmark

DDAffinity evaluates anti-symmetry on **Ssym**, a set with balanced direct and inverse variants,
and its M1707 multi-point set explicitly includes reverse mutations. Separately, `DDGb_bias`
(Briefings in Bioinformatics, 2023) quantifies systematic biases across eight ΔΔG predictors.

Our antisymmetry check can therefore cite an established evaluation instead of inventing one.

## Candidate extra data, ranked

| Source | Size | Verdict |
| --- | --- | --- |
| FoldX precomputed on SKEMPI (`skempi-foldx`) | ~6,100 entries, 12 energy terms | **Use now** — free feature and free baseline |
| AB-Bind | 645 antibody mutations | Same label type, small, easy merge — re-cluster jointly |
| AbAgym | ~324k DMS mutations, 36.5k interface | Pretraining only: DMS scores are not ΔΔG and assay scales differ; non-commercial licence |
| Other SKEMPI subsets (Pr/PI, TCR/pMHC) | ~2,100 rows | Cheap, but honour the cross-subset homology links found in the EDA |

## What changes in the plan

1. Adopt the field's full metric suite and the ≥10-mutations-per-group rule for per-structure
   averages, keeping per-complex Spearman as the headline.
2. State the split difference loudly when comparing to published numbers, and expect roughly
   0.4–0.5 per-complex Spearman rather than 0.67.
3. Add FoldX as an explicit ladder rung — first as a standalone physics baseline, then as a
   feature the learned head corrects.
4. Add relative solvent accessibility alone as a baseline, since it is reported as competitive.
5. Move the learning-curve diagnostic early; data volume is very likely the binding constraint.
6. Replace random-negative pretraining with FoldX pseudo-label pretraining as the primary
   data-volume remedy, with AbAgym DMS as a secondary option.
7. Use Ssym for the antisymmetry probe.
8. Note that FoldX multi-point values are not decomposable, when running the additivity probe.

## Sources

- [SKEMPI 2.0 (Bioinformatics 2019)](https://academic.oup.com/bioinformatics/article/35/3/462/5055583)
- [Investigating the volume and diversity of data needed for generalizable antibody–antigen ΔΔG prediction (Nature Computational Science 2025)](https://www.nature.com/articles/s43588-025-00823-8)
- [SidechainDiff / DiffAffinity (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/99088dffd5eab0babebcda4bc58bbcea-Paper-Conference.pdf)
- [Rotamer Density Estimator / RDE-Network (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.02.28.530137v2.full)
- [DDAffinity (Bioinformatics 2024)](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i418/7700870)
- [abCAN (Briefings in Bioinformatics 2025)](https://academic.oup.com/bib/article/26/5/bbaf464/8251565)
- [skempi-foldx — precomputed FoldX ΔΔG for SKEMPI 2.0](https://github.com/cchin29/skempi-foldx)
- [AbAgym (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.07.15.664862v1.full) and [its repository](https://github.com/3BioCompBio/Abagym)
- [AbDesign (mAbs 2025)](https://pubmed.ncbi.nlm.nih.gov/41058476/)
- [DDGb_bias — quantification of biases in ΔΔG predictors](https://github.com/3BioCompBio/DDGb_bias)
