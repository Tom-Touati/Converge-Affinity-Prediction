# Converge ΔΔG assignment — basic plan

Antibody–antigen ΔΔG prediction on SKEMPI 2.0. Emphasis: time discipline, evaluation, error analysis, clean clustered splits.

## Principles

The split and the evaluation harness get built and frozen before any real model exists. Every later number is then comparable, and every gain can be attributed to one change.

- **One moving part per experiment.** Same folds, same seeds, same metrics; change one thing and log the delta.
- **Climb a ladder.** Each rung must beat the one below on the frozen harness, or we stop and ask why before climbing further.
- **Name the bottleneck before spending on it.** Data quantity, label noise, representation, fusion and optimisation each have a cheap diagnostic (see Moving parts). We run the diagnostic first.
- **Ablation is the justification.** The assignment grades reasons. "Fusion adds +X per-complex Spearman over the best single modality on unseen clusters, CI [a, b]" is a reason. "It is multimodal" is not.
- **Negative results stay in.** A rung that fails to help is reported, with the hypothesis it killed.

## Time budget

Roughly 40 working hours, with about 40% on data, splits and evaluation, 25% on error analysis and write-up, and only 35% on models. The submission deadline is an open question; scale the hours, keep the proportions.

| Phase | Hours | Exit criterion | Stop rule |
| --- | --- | --- | --- |
| 0. Repo skeleton, env, prompt log | 2 | `make data` and `make eval` run end to end on a dummy predictor | None; do not skip |
| 1. Data audit and cleaning | 5 | One parquet table, one audit notebook, every dropped row has a logged reason | Unparseable rows get dropped and counted, not rescued |
| 2. Clustered splits | 5 | Frozen fold file committed; leakage checks pass | If clusters are too unbalanced for 5 folds, fall back to leave-one-cluster-out |
| 3. Eval harness and trivial baselines | 4 | Report generator prints all metrics with CIs for mean and substitution-matrix baselines | None |
| 4. Feature caching (PLM, structure) | 5 | Embeddings and structure features cached on disk, keyed by row id | If a structure model fights the install for over 1.5 h, swap it for the next option |
| 5. Model ladder, rungs 2 to 6 | 12 | Each rung has a harness report and a one-line verdict | Max 2 h per rung, 4 h for the rung-6 attention module, before writing down the result as is |
| 6. Error analysis | 6 | Slice tables, residual plots, 10 hand-inspected worst cases | None; this is the graded part |
| 7. README, results, next steps | 5 | A stranger can reproduce the headline table from the README | Freeze code 5 h before submission |

No hyperparameter search beyond a small fixed grid chosen in advance. With this little data, search mostly fits the validation folds.

Log wall-clock time per phase and per training run from day one. The assignment asks for hardware and runtime, and the log also shows where time really went.

## Data

The antibody–antigen subset is 1,211 rows over 55 complexes, and three complexes hold 29% of it. Counts below come from a first pass over `skempi_v2.csv`, filtering `Hold_out_type` for `AB/AG`.

| Fact | Value | Consequence |
| --- | --- | --- |
| Rows / complexes | 1,211 / 55 | Tiny. Frozen encoders and small heads only |
| Rows per complex | median 10, max 122 (3HFM, 1MHP, 2B2X each 115+) | Global metrics are dominated by three complexes; report per-complex metrics too |
| Single-point mutations | 830 (69%) | Start with single-point only, add multi-point as a separate rung |
| Missing mutant affinity | 80 rows (non-binders, bounds) | Drop for regression, keep a flag; usable later as censored or classification labels |
| Repeated (complex, mutation) pairs | 123 groups, 258 rows | Aggregate to one row. Their spread is a free estimate of label noise, which caps achievable correlation |
| Temperature marked "assumed" | 604 rows | Parse the number, default 298 K, note it |
| ΔΔG distribution | mean +1.06, SD 1.80, range −4.9 to +7.9 kcal/mol | Skewed to destabilising; 41% above +1, 12% below −0.5 |
| Assay method | SPR 554, ELISA 204, FL 115, KinExA 98 | Slice errors by method; ELISA is noisier |

Label: ΔΔG = RT·ln(Kd_mut / Kd_wt) in kcal/mol, positive means weaker binding. We regress ΔΔG and derive the three classes from predictions by fixed thresholds, so one model serves both framings.

Cleaning steps, each logged with a row count:

1. Filter to AB/AG, parse temperature, compute ΔΔG, drop rows with missing affinity.
2. Aggregate duplicates by median; store n and SD per group.
3. Map `Mutation(s)_cleaned` onto chain sequences from the PDB files, and assert the wild-type residue matches at every position. Mismatches are dropped and listed.
4. Label each chain as heavy, light or antigen from the `#Pdb` chain field, and verify with ANARCI numbering.

Three hazards the SKEMPI 2.0 paper raises, each checked against this subset:

- **Alanine dominance.** 53.5% of the 797 single-point AB/AG mutations are X→Ala (mean ΔΔG +1.28 vs +0.86 for the rest). A model can score respectably by learning "alanine scan means destabilising", so alanine becomes a standing slice in the error analysis.
- **Non-native reference state.** About 140 rows carry curator notes that the "wild-type" is itself an affinity-matured Fab or another mutant from the same paper (89 rows "the affinity matured fab ... taken as wild-type", 31 "wild-type is affinity matured AR2", 20 "this mutant is taken as wild-type"). These are effectively reverse or mutant-to-mutant measurements. Flag them, keep them, and report the headline metric with and without.
- **Inequalities and non-binders.** 45 rows give the mutant affinity as a bound. The paper notes the author's choice between "non-binding" and "weaker than X" is arbitrary, which is the reason we drop rather than impute them.

## Known label defects, found after the EDA

Three problems with the labels themselves, all quantified against the 997-row modelling table.
The first is a correctness issue in the current pipeline, not a hypothetical.

| Defect | Rows | What is happening now | What it should be |
| --- | --- | --- | --- |
| **Censored affinities read as point labels** | **86** (45 mutant, 41 wild-type) | SKEMPI parses `">1e-6"` into a bare number and `src/data.py` trusts it, so an inequality is trained on as an exact measurement | Carry an `is_censored` flag and either drop, or handle as an inequality under a ranking objective |
| Qualitative non-binders discarded | 80 | All are literal `n.b.` / `n.b`. Dropped entirely for having no parsable affinity, losing the strongest destabilising evidence in the set | Usable as ranking constraints: a non-binder sits below every measured mutation in its complex. One complex is *entirely* non-binders and still contributes no within-complex signal |
| Exact zeros that are probably "not reported" | 37 | Treated as measured zeros; 22 of the 37 sit in a single complex (`2JEL_LH_P`) | Flag. A 60% concentration in one complex is a reporting convention, not 37 independent measurements of precisely 0.000 |

The censored-label issue affects 8.6% of the dataset and should be fixed before the encoders are
trained against these labels, since every rung inherits it.

## The label distribution, and what "imbalance" means here

`reports/figures/ddG_distribution.png`, regenerated by `python -m src.analysis labels`. Three
criteria for "destabilising" give three different answers on the same 997 rows:

| Criterion | Stabilising | Neutral | Destabilising |
| --- | --- | --- | --- |
| Sign of ddG | 25.5% | 3.7% (exactly 0) | 70.8% |
| Class edges +/-0.5 | 13.0% | 32.4% | 54.6% |
| Restricted to abs(ddG) > 1 (490 rows) | 14.1% | -- | 85.9% |

The third is the floor quoted for `sign_acc_big` and is only meaningful for that metric. Quote
the first as the dataset property.

The skew is physically real -- random substitutions at a binding interface mostly disrupt
binding -- and the test folds carry it too, so it is not a sampling artefact to reweight away.
It is only partly the alanine scan: X->Ala singles are 77.6% destabilising against 67.8% for
other singles. **The actionable shortage is not the ratio but the absolute count: 130 rows are
clearly stabilising, and affinity improvement is what antibody engineering actually wants
predicted.**

Interface location sets the expectation for every augmentation idea, because it is also the
test distribution (single-point rows):

| Location | n | mean ddG | sd | % neutral (abs <= 0.5) |
| --- | --- | --- | --- | --- |
| COR core | 295 | +1.55 | 1.72 | 23.7% |
| SUP support | 120 | +1.33 | 1.98 | 19.2% |
| INT interior | 34 | +0.79 | 1.42 | 61.8% |
| RIM rim | 168 | +0.40 | 0.94 | 53.0% |
| SUR surface | 79 | +0.09 | 0.37 | 86.1% |

83.8% of single-point rows are interface proper (COR + SUP + RIM). Surface mutations average
+0.09 with sd 0.37, below the measurement noise -- distal mutations really are neutral, which
is the empirical licence for the distal-neutral augmentation below, and also the reason it is
not the lever the model most needs.

## Clean clustered splits

The primary split is grouped 5-fold cross-validation over clusters of related complexes, so no test complex has a relative in training. A random per-mutation split is also run, but only to measure how much it inflates the score.

Leakage sources to close:

- **Same complex** in train and test. The model learns the complex's mean ΔΔG and hotspot positions.
- **Near-identical complexes**: the same antibody against antigen variants, or engineered variants of one antibody (SKEMPI's own Notes column flags "HyHEL-10 and HyHEL-63 are very similar" on 145 rows, 12% of the subset).
- **Same antigen**, different antibody. Shared epitope residues leak.
- **Forward and reverse pairs**, if we add antisymmetry augmentation later. Pairs must stay in one fold.
- **Duplicates**, already merged in cleaning.

Clustering method:

1. **Primary: SKEMPI's own homology groups.** The paper deems two interactions homologous if they share a binding partner, or have homologous partners (>50 similarity score, ≥30% identity), and share ≥70% of interface residues. It states these groups exist "for avoiding overfitting when developing models or for validation and estimating generalisation error". Using the dataset authors' own definition is the most defensible default and the easiest to justify in review.
2. **Supplement with sequence clustering** where SKEMPI leaves complexes ungrouped: MMseqs2 `easy-cluster` on antigen sequences at 30% identity, and on antibody variable regions at about 90%, since shared framework means a low threshold merges every antibody into one cluster.
3. **Honour the curator notes.** The HyHEL-10 / HyHEL-63 warning above is a manual link regardless of what the automatic clustering returns.
4. Clusters are the connected components of the union of these links.

Fold construction: greedy assignment of clusters to 5 folds, balancing row count first and ΔΔG mean second. The three 115+ row complexes must land in different folds. Output is a committed `folds.csv` (row id, cluster id, fold) that nothing downstream may regenerate.

Inside each training fold, model selection and early stopping use an inner grouped split on the same clusters. The outer test fold is never touched for any decision.

Sanity checks, run as tests:

- No cluster id appears in two folds.
- Maximum sequence identity between any test chain and any train chain, per fold, is reported in the README.
- Fold sizes and ΔΔG histograms per fold are plotted.
- The gap between random-split and cluster-split scores is reported as the leakage estimate.

## Evaluation protocol

One harness computes every number, from the same predictions file, for every model. It is built in phase 3 against trivial baselines and never changed afterwards, so a later score gain cannot come from a scoring change.

Headline metric: **per-complex Spearman**, computed within each complex then averaged over complexes, weighted equally. This is what matters for a real "which mutation is least harmful" decision and it is not gamed by the three large complexes. Global Spearman and Pearson are reported alongside, and the gap between them is itself a diagnostic.

Reported for every model:

| Metric | Why |
| --- | --- |
| Per-complex Spearman (mean, and per-complex table) | Primary; ranking within a target |
| Global Spearman / Pearson | Comparability with SKEMPI literature |
| RMSE and MAE (kcal/mol) | Absolute calibration, in physical units |
| 3-class macro-F1 and confusion matrix | The classification framing; thresholds fixed on train only |
| Sign accuracy on abs(ΔΔG) > 1 | Does it at least get stabilising vs destabilising right |

Confidence intervals: the score's spread comes mostly from which clusters land in the test fold, so we bootstrap **over complexes** (resample complexes with replacement, 1,000 times) and report 95% intervals. A per-mutation bootstrap would look tighter and lie. Paired bootstrap on the same resamples gives Δ between two models with a CI, which is how one rung is declared better than another.

Ceiling and floor, both plotted on every chart:

- **Floor**: predict the training-set mean ΔΔG. Any model below this is broken.
- **Noise ceiling**: from the paper's replicate statistics. 84% of 1,741 independently repeated ΔΔG measurements agree within 1 kcal/mol, implying a measurement SD near 0.5 kcal/mol. Against this subset's global spread (SD 1.80) that caps Pearson r around 0.96, but within a single complex the spread is only SD 1.19, capping per-complex r near 0.91, and lower for the tighter complexes. The headline metric therefore has a materially lower ceiling than the global one. Worth stating before anyone reads a per-complex 0.6 as weak. Our own duplicate groups give a second, subset-specific estimate to cross-check this.

Determinism: fixed seeds, sorted inputs, cached features hashed by content. The report is regenerated by one command and the headline table is copied verbatim into the README.

## Model ladder

Each rung is one experiment on the frozen harness. A rung is kept only if it beats the rung below by a paired-bootstrap Δ whose CI clears zero. All encoders are frozen; only small heads train, because 1,211 rows cannot fine-tune a protein language model without overfitting.

| Rung | Model | Tests the hypothesis | Cost |
| --- | --- | --- | --- |
| 0 | Predict train mean | Floor | none |
| 1 | Substitution features only (BLOSUM62 score, hydrophobicity Δ, volume Δ, charge Δ, is-proline, is-glycine) + gradient-boosted trees | How far does mutation chemistry alone get us, no protein context | minutes |
| 2 | Sequence PLM: ESM-2 (8M then 35M), embedding difference wt vs mut at the mutated position + mean-pool, into a ridge/MLP head | Does a sequence model beat raw chemistry | 1 GPU-hour |
| 3 | Structure only: ESM-IF1 or AntiFold inverse-folding log-likelihood ratio at the mutated position, plus rSASA and interface distance | Does structure alone beat sequence | 1–2 GPU-hr |
| 4 | Late fusion: concatenate rung-2 and rung-3 features, one head | Does combining modalities help at all, and by how much. **The control that rung 6 must beat** | +minutes |
| 5 | Fusion + explicit interface features (contacts across the antibody–antigen interface, Δ contacts on mutation) | Is the interface the missing signal | +1 GPU-hr |
| 6 | **Cross-attention fusion**: the mutation queries the interface residues around it, distance-biased, frozen encoders | Does an architecture that models *which* residues the mutation interacts with beat flat concatenation | +3-4 h |
| 6a | Uniform-attention control: mean-pool the same residue set | Was the gain the residue filter rather than attention | +minutes |
| 6b | Distance-bias-only control: attention weights from geometry, no content | Was the gain the physical prior rather than learned content | +minutes |

Mutation representation, decided up front: the difference vector between wild-type and mutant embeddings at the mutated residue, concatenated with the position's structural context. Difference vectors are the standard, cheap and strong choice for ΔΔG; whole-sequence pooling alone washes out a single-residue change. Multi-point mutations sum the per-position difference vectors as a first approximation. The paper's double-mutant cycles show this is only half true, 345 additive against 421 context-dependent, so the approximation gets tested rather than assumed.

Head: gradient-boosted trees or a 2-layer MLP with dropout for rungs 1 to 5; the attention module below for rung 6. With this sample size a linear or tree head on good features is genuinely hard to beat, so rungs 1 to 5 stay deliberately simple and rung 6 has to earn its parameters against them rather than replacing them.

Antisymmetry: for every forward mutation the reverse should give −ΔΔG. We test this as an evaluation probe on rung 2 onward, and only if it fails badly do we add reverse mutations as augmentation (deferred below).

## Cross-attention fusion (rung 6)

Concatenation is a weak model of what binding actually is. A mutation's effect on ΔΔG depends on
*which residues it contacts across the interface*, and a concatenated feature vector has thrown
that structure away before the head ever sees it. Rung 6 puts it back: the mutation is a query,
the residues around it are keys and values, and the model learns what to attend to.

**This is the component the evidence says matters least, and we build it anyway — with the
experiment designed so a negative result is still a result.** `PRIOR_WORK.md` finds data volume,
not architecture, is the binding constraint at n≈1,000, and `BACKBONE_COMPARISON.md` finds the
training objective swings scores far more than the encoder or the fusion. The assignment grades
multimodal fusion explicitly, and "we concatenated two vectors" is a thin answer to that. So the
question is put properly and answered with evidence either way.

### Architecture

| Component | Content |
| --- | --- |
| **Query** (1 token) | The mutation: ESM-2 embedding difference (wild type vs mutant) at the mutated position, concatenated with the rung-1 chemistry vector, projected to d = 64 |
| **Keys / values** (N ≈ 20–60 tokens) | Context residues: every residue with a heavy atom within 10 Å of the mutated residue, on both chains, with all across-interface partner residues forced in. Each token is [frozen per-residue ESM-2 embedding ‖ inverse-folding per-residue log-probability vector ‖ geometry: distance to the mutated residue, rSASA bound, rSASA unbound, ΔrSASA, is-partner-chain], projected to d = 64 |
| **Attention** | One multi-head layer, 4 heads, d = 64. Logits carry a learned monotone **distance bias**, so the module initialises near "attend to what is close" and learns deviations from it |
| **Head** | concat[query, attended context] → 64 → 32 → 1, dropout 0.2 |

Encoders stay frozen. Only the projections, the attention layer and the head train — on the order
of 50k parameters, which is reported next to every result.

### Why this is defensible at 997 rows

The honest risk is overfitting, so three of the four design decisions exist to manage it:

1. **Attention runs over 20–60 pre-filtered residues, not the ~600 in the complex.** The physical
   prior — binding effects are local to the interface — does the hard part of the problem. The
   attention layer only reweights inside a set that is already almost all relevant.
2. **The distance bias means the model starts from physics.** An untrained module behaves roughly
   like distance-weighted pooling, which is already a sensible predictor, and training moves it
   away from that only where the data supports it.
3. **Frozen encoders keep the trainable parameter count comparable to the MLP head in rung 4**,
   so the comparison against concatenation is about the *mechanism*, not about capacity.
4. Early stopping uses an **inner split grouped by cluster**, never a random one. We measured why:
   a constant-per-fold predictor already scores global ρ −0.36 on this data, so a random inner
   split would leak homology straight back in.

Ten seeds, ensembled, with seed variance reported. At this sample size a single run's number is
not a result.

### How we will know whether it worked

Rung 6 is kept only if it beats **rung 4** by a paired-bootstrap Δ whose CI clears zero — and only
if it also survives the two controls, which is where most of the honesty lives:

- **6a, uniform attention.** Replace the learned weights with a plain mean over the same filtered
  residue set. If this matches rung 6, the gain came from choosing the right residues, not from
  attention, and the right conclusion is that a better *feature set* beat a better architecture.
- **6b, distance-bias only.** Attention weights computed from geometry alone, with the content
  term switched off. If this matches rung 6, the gain came from the physical prior we injected,
  not from anything the model learned.

The training objective is held fixed across rungs 4, 5, 6, 6a and 6b. Given how much the objective
moves scores, varying it and the fusion mechanism together would make the comparison
uninterpretable.

**Stop rule: 4 hours.** If rung 6 does not clear rung 4, we report that, with the parameter count
and the seed variance, and the write-up argues from our own evidence that flat concatenation is
sufficient at this sample size. That is a defensible finding, and a more useful one than a fragile
win.

### The side benefit: attention weights are an explanation

Every prediction comes with a distribution over the residues it attended to. For the ten
hand-inspected worst cases in the error analysis, we can show *what the model looked at* and check
it against the known hotspot residues for that complex. A model that attends to the wrong side of
the interface is diagnosable in a way a gradient-boosted tree on pooled features is not.

## Error analysis

This is the graded core, so it runs on every rung, not just the last. The same predictions file feeds a fixed set of slice tables and plots.

Slices, each a table of per-complex Spearman and RMSE:

- By complex, sorted worst first, next to that complex's row count and ΔΔG variance. Reveals whether errors track small-sample or hard-target.
- By mutation location (`iMutation_Location(s)`: COR / RIM / SUP / SUR / INT). Interface-core mutations should be the ones structure helps most; if not, fusion is not doing its job.
- Single vs multi-point.
- By assay method (SPR vs ELISA vs the rest), to separate model error from label noise.
- By abs(ΔΔG) magnitude, and stabilising vs destabilising. Most methods are blind to stabilising mutations; we quantify that blindness.
- By mutation chemistry: to/from proline and glycine, charge reversals, large-to-small.

Two probes the paper opens up, both cheap and both interview-ready:

- **Alanine vs non-alanine.** Report the headline metric separately on the 426 X→Ala singles and the 371 others. If performance lives entirely in the alanine half, the model has learned a scanning artefact rather than binding physics.
- **Additivity.** 71 of the 109 double mutants here decompose into both constituent singles, giving usable double-mutant cycles. Comparing predicted against observed epistasis on those tells us directly whether summed difference vectors are an adequate multi-point representation, and gives a principled reason to either keep the sum or move to a joint encoding.

Diagnostic plots:

- Predicted vs true scatter, colored by complex, with the y=x line and the noise ceiling band.
- Residual vs Δ contacts at the interface, and vs rSASA. A trend means an unused structural signal.
- Residual vs training rows for the complex, to see if the problem is data volume.
- Calibration of the 3-class probabilities.

Hand inspection: the 10 worst residuals, read individually against their PDB structure. For each, a one-line note on the likely cause (buried vs surface, conformational change, likely label error, forward/reverse asymmetry). This is what an interviewer will push on, so the notes name a mechanism, not "hard case".

Every slice result routes back to the Moving parts map: it either implicates the encoder, the fusion, the labels, or the data volume, and that verdict decides the next rung.

## Moving parts and bottleneck map

The whole point of the ladder and the slices is to attribute a disappointing score to exactly one of these, and to have a cheap test for each before spending on a fix.

| Moving part | Cheap diagnostic | Signal it is the bottleneck | Fix (mostly deferred) |
| --- | --- | --- | --- |
| Label noise | Duplicate-measurement spread → noise ceiling | Model correlation near ceiling; ELISA slice much worse | Nothing to fix; report the ceiling |
| Data volume | Residual vs rows-per-complex; learning curve on subsampled folds | Error falls steeply as rows rise; small complexes carry the loss | Add data / pretraining (deferred) |
| Split leakage | Random vs cluster-split gap | Large gap | Trust cluster split; the random number was the lie |
| Sequence encoder | Rung 2 vs rung 1 Δ | Rung 2 barely beats chemistry | Bigger / antibody-specific PLM (AbLang2) |
| Structure encoder | Rung 3 vs rung 2; residual vs interface features | Structure adds nothing where it should (COR mutations) | Better structure model, real interface geometry |
| Fusion | Rung 4 vs max(rung 2, rung 3); then rung 6 vs rung 4 | Fusion ≤ best single modality, or cross-attention ≤ concatenation | Rung 6 is the fix, and rungs 6a/6b say whether it was the mechanism or the features |
| Optimisation / head | Train vs test gap; floor comparison | Overfitting, or not beating the mean | Stronger regularisation, simpler head |

The rule: we do not touch a moving part until its diagnostic says it is the binding constraint. Early on the likely constraints are label noise and data volume, not fusion sophistication, which is exactly why fusion upgrades are deferred and the ladder starts at a substitution-matrix baseline.

## Deferred: fusion, augmentation, more data

Each of these is written into the "next steps" section of the submission with the evidence that would trigger it. None is built until its trigger fires, so we always know why we added complexity.

| Idea | Trigger | Note |
| --- | --- | --- |
| Reverse-mutation augmentation (enforce ΔΔG(reverse) = −ΔΔG) | Antisymmetry probe fails badly | Cheap, and doubles data; keep pairs in one fold |
| Cheap-negative pretraining: real antibodies vs random unrelated antigens as confident non-binders, then fine-tune on SKEMPI | Learning curve shows data volume is the constraint | Strong if volume is the bottleneck, but validate the non-binder assumption |
| More labeled data (AB-Bind, other SKEMPI subsets, deep-mutational-scan sets) | Data-volume diagnostic positive; noise ceiling not yet reached | Watch for distribution shift and re-cluster jointly |
| Antibody-specific encoders (AbLang2, AntiFold, IgFold) | Generic PLM/structure encoder is the weak rung | Drop-in swaps behind the cached-feature interface |
| Ensemble / uncertainty | Only for the final report | Bootstrap ensemble gives per-prediction error bars, useful for a "which mutation to trust" story |

| Pairwise within-complex ranking loss | Per-complex Spearman stalls while RMSE improves, or the censored/non-binder rows are wanted | Not augmentation -- 19,834 pairs from 997 rows is 19.9x the examples but zero new information. Real merits: it optimises the headline metric directly, cancels the per-complex offset exactly (a constant-per-fold predictor scores global rho -0.36, so the shortcut is live), and is invariant to per-complex rescaling, which matters when 47.7% of temperatures are assumed. Costs: the top 3 complexes hold 43.2% of pairs against 22.8% of rows, so pairs need 1/C(n,2) weighting; near-ties are noise, so filter to abs(delta) > 1, which keeps 10,703 pairs (54%); no kcal/mol scale, so pair it with the regression head rather than replacing it. **Efficiency argument tested and not supported** -- see the centering result below. The data-recovery argument (80 non-binders, 86 censored rows) is untouched by that test and still stands |
| Reverse-mutation augmentation, revisited | Antisymmetry probe fails, or the stabilising slice stays at chance | The strongest lever for the *actual* deficiency: 544 destabilising rows (> +0.5) reversed give 544 synthetic clearly-stabilising examples against the 130 real ones, a 5x increase in the class the model is blind to. Known weakness: no mutant structure exists, so the wild-type backbone is reused |
| Distal-neutral augmentation (mutate far from the interface, label ~0) | Only as a regulariser or a probe, not as bulk training data | The labelling assumption is validated by our own data: SUR mutations average +0.09 with sd 0.37, below measurement noise, and 86.1% are neutral. But it targets the wrong gap -- it grows the neutral class (already 32.4%) and adds nothing stabilising -- and 83.8% of the test set is interface proper, so it trains a region the test set barely contains. It is also largely redundant with the rSASA / interface-distance features, which let the model learn "distal implies zero" from the 79 real SUR rows. Strongest forms: a penalty for predicting non-zero at distal positions, and a test-time probe that checks the interface prior was learned at all |
| FoldX pseudo-label pretraining | Learning curve shows data volume is the constraint | Strictly more informative than distal-neutral for the same purpose: a graded, structure-aware signal across the whole label range rather than a point mass at zero, and `skempi-foldx` ships precomputed values so FoldX never has to run |

### Negative result: within-complex target centering does not help

The cheapest test of the "ranking fixes the offset problem" hypothesis, run before building any
pair machinery. Train on `ddG - mean(ddG | complex)` using training-fold means only, predict the
deviation, rank within complex as usual. This captures the whole efficiency argument for a
ranking loss while changing exactly one thing and keeping the gradient-boosted head, so no
head-architecture change confounds it.

`python -m src.train --model gbt --features chem --center --name rung1c_chem_gbt_centered`

| Metric | Uncentered | Centered | Paired delta | 95% CI | Clears zero |
| --- | --- | --- | --- | --- | --- |
| **Per-complex Spearman (n>=10)** | 0.180 | 0.167 | **-0.015** | [-0.078, +0.058] | no |
| Per-complex Spearman (all) | 0.111 | 0.128 | +0.013 | [-0.072, +0.107] | no |
| Global Spearman | 0.207 | 0.105 | -0.103 | [-0.226, -0.002] | yes |

The global-Spearman drop is the **positive control**: centering removed the between-complex
component exactly as designed, so this is a real null and not a broken experiment. On the
headline metric 13 of 27 complexes improved, median delta -0.009 -- a coin flip, with individual
complexes swinging between -0.35 and +0.44.

What it rules out: that squared error wasting 38% of its gradient on an unlearnable per-complex
offset is what limits our within-complex ranking. It is not, at least for a tree head on
chemistry features.

What it does not rule out, stated honestly: the harness's minimum detectable effect for a
genuinely different model is about +0.10 (power analysis, `reports/`), so an effect of +0.05
would have been invisible either way. The result is consistent with anything from -0.08 to
+0.06. What tilts it negative is that the point estimate is negative -- a real mechanism should
at least have produced a positive one.

What survives: a ranking loss is invariant to per-complex *monotone rescaling*, not just
additive offsets, and it is the only route to using the 80 non-binders and 86 censored rows.
Those arguments are independent of this test.

The architecture is built so each of these is a swap behind a stable interface: encoders write cached feature files keyed by row id, and the fusion head reads features by name. Changing an encoder or adding a modality does not touch the split, the harness, or the error-analysis code.

## Repo and reproducibility

The grade weights code quality and reproducibility explicitly, so the layout is a deliverable, not an afterthought.

```
converge_bind/
  data/            raw skempi_v2.csv, SKEMPI2_PDBs.tgz (git-ignored), folds.csv (committed)
  notebooks/       01_eda.ipynb
  src/
    data.py        clean, parse dG, dedup, PDB map -> one parquet
    splits.py      clustering + fold assignment, deterministic
    features/      esm.py, structure.py, chem.py  (each writes a cached feature file)
    model.py       heads: mean, trees, mlp; fusion by feature concat
    evaluate.py    THE harness: one predictions file -> all metrics + CIs
    analysis.py    slice tables and plots
  configs/         one yaml per rung
  reports/         generated tables and figures
  README.md
  AI_PROMPTS.md    prompt history, appended as we go
  Makefile         data / splits / features / train / eval / report
  environment.yml
```

Rules:

- Every result comes from `make report`; no numbers are hand-copied except into the README, from the generated table.
- Features are cached to disk keyed by content hash, so encoder runs are paid once. This is what keeps the 40-hour budget real.
- One config file per rung; a run records its config, git SHA, seed, and wall-clock time into `reports/`.
- `AI_PROMPTS.md` is updated in the same commit as the code the prompt produced, not reconstructed at the end.
- README states hardware (which GPU, or CPU-only) and approximate runtime per phase, taken from the time log.

First three actions when we start building: create the repo skeleton with a dummy mean-predictor passing through the whole `make` chain, write `evaluate.py` against that dummy, and commit `folds.csv`. Only then does the first real feature get computed.
