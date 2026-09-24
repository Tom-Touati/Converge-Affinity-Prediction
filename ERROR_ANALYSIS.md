# Error analysis

Part I is the error analysis of the random forest, which is the model.
Part II is the error analysis of the *evaluation*.
Part III is bias from imbalance -- labels, complexes and mutation types -- and is
where the aggregate metric is shown to conceal the failures that matter most.

> **Superseded in places — see [HANDOFF.md](docs/HANDOFF.md).** Numbers here predate two
> corrections: the per-complex threshold moved from 10 mutations to 5 (the model to beat
> reads 0.388, not 0.498, on the same predictions), and residual learning was removed.
> Any figure produced by a net trained on `ddG - forest_prediction` had the forest added
> back at inference and is not that network's own score.


# Part I — errors of the model

Model: `rungN0_chem_geom_mpnn_rf` — chemistry + interface geometry + ProteinMPNN into a random
forest, per-complex Spearman **0.475 [0.393, 0.552]**. Regenerate everything here with:

```bash
python -m src.error_analysis --run rungN0_chem_geom_mpnn_rf
```

Tables land in `reports/<run>/error_analysis/`, figure in `diagnostics.png`.

---

## How imbalance is handled, and why it needed handling

Three imbalances in this dataset can each manufacture a finding that is not there.

| Imbalance | Scale | What it would fake |
| --- | --- | --- |
| Complexes | 54 complexes, median 10 rows, max 87; top 3 hold 23%; 27 fall below the 10-row rule | A category that is really one complex |
| Label | 70.8% destabilising by sign; only 130 rows clearly stabilising | "High RMSE" that is really "high variance" |
| Categories | Interface location COR 295 → INT 34; assay SPR 475 → single digits | Three-decimal precision on 34 rows |

Every slice therefore carries four defences:

* **`n_cx`** — how many complexes contributed, not just how many rows.
* **`top_cx`** — the share from the single largest complex. Above 0.6 the row is labelled
  `one-complex`, because the "finding" is that complex wearing a category name.
* **`sd_true` beside `rmse`** and a **`skill`** column (`1 − MSE/Var` within the slice). Skill of
  0 means no better than that slice's own mean, so a high-variance slice cannot look bad merely
  for being high-variance.
* **Confidence intervals bootstrapped over complexes *within* the slice**, never over rows. Row
  bootstrapping on a one-complex slice returns a tight interval around an artefact.

Both aggregations are reported everywhere: **micro** (pool rows — "error on a random
measurement") and **macro** (per complex, then average — "error on a random target", matching the
headline metric). Where they disagree, the disagreement is the finding.

**This caught four would-be results.** `ab_vs_ab` looks respectable at ρ 0.51 — it is 100% one
complex (1DVF). Assay method `SP` looks excellent at ρ 0.76 — 85% one complex. `CSPRIA`, `IASP`,
`BI` and `IAFL` are each a single complex. And the non-native-reference slice (120 rows, ρ −0.07)
spans only 4 complexes with a CI of [−0.12, +0.93], which is not a measurement. Without `top_cx`
all five would have been reported as findings.

---

## 1. The dominant failure is compression towards the mean

Predictions span roughly 0 to +4.4 kcal/mol. The truth spans −4.9 to +7.9. The model refuses to
commit to extremes in either direction, and the bias table makes it exact:

| True effect | n | bias = mean(pred − true) | 95% CI | ρ within slice |
| --- | --- | --- | --- | --- |
| stabilising (< −0.5) | 130 | **+2.26** | [+1.91, +2.79] | −0.16 [−0.43, +0.05] |
| neutral (±0.5) | 323 | +0.54 | [+0.43, +0.66] | 0.12 [−0.01, +0.25] |
| destabilising (> +0.5) | 544 | −1.00 | [−1.34, −0.59] | 0.32 [+0.13, +0.46] |

By magnitude the same pattern: +0.54 on |ΔΔG| ≤ 0.5, +0.61 on 0.5–1, +0.08 on 1–2, and
**−1.47 [−2.01, −0.53]** on |ΔΔG| > 2.

**On stabilising mutations the model is worse than useless.** It over-predicts by +2.26 kcal/mol
on average — it says "weaker binding" when the truth is "tighter binding" — and its within-slice
rank correlation is *negative*. Affinity improvement is exactly what antibody engineering wants
predicted, and this is where the model has no signal at all.

This is the label imbalance expressed through a squared-error objective. With 70.8% of rows
destabilising and a training mean of +1.03, hedging towards the mean is the loss-minimising
strategy, and the forest does it. It is a property of the objective and the class balance, not a
bug in the features.

## 2. The antibody side is where the model is blind

| Side mutated | n | complexes | top_cx | ρ | 95% CI |
| --- | --- | --- | --- | --- | --- |
| **antigen** | 332 | 27 | 0.13 | **0.62** | [+0.51, +0.68] |
| **antibody** | 584 | 38 | 0.12 | **0.15** | [−0.11, +0.44] |

Both slices are broad and un-dominated, and the intervals barely overlap. This is the most
trustworthy comparison in the analysis, and the most awkward: **two thirds of the data is on the
antibody side, and that is the side the model cannot rank.**

The mechanism is the same one that killed the sequence encoder. Antibody binding residues sit in
CDR loops, which are hypervariable by design — evolution deliberately does not conserve them, so
neither evolutionary statistics nor inverse-folding likelihoods have much to say about them.
Antigen epitope residues sit in an ordinarily-evolved fold where both signals behave normally.

It also reframes what the 0.475 headline means: the model is substantially an *antigen-side*
predictor, and antibody engineering is the antibody-side problem.

## 3. Multi-point failure is a non-additivity failure, and now it is measured

| | n | complexes | ρ | 95% CI |
| --- | --- | --- | --- | --- |
| single-point | 696 | 50 | 0.52 | [+0.40, +0.62] |
| multi-point | 301 | 30 | 0.14 | [−0.16, +0.51] |

The double-mutant cycles say why. Seventy 2-point mutations have both constituent singles
measured in the same complex:

* Observed epistasis: mean −0.87, **sd 1.60**, range [−4.67, +2.19]
* Predicted epistasis: mean −1.09, **sd 0.53**
* Correlation: Spearman **+0.435** (p < 0.001)
* **44% of cycles have |epistasis| > 1 kcal/mol**

So the model gets the *direction* of epistasis partly right and its *magnitude* badly wrong — it
predicts with a third of the true spread. And additivity is violated in nearly half of cycles, so
summing per-position features is not an adequate multi-point representation. `PLAN.md` flagged
this as an assumption to test rather than assume; the test says the assumption fails.

## 4. Errors do not track per-complex sample size

Spearman between a complex's row count and its Spearman score: **+0.04 (p = 0.85)**. Flat.

That is worth stating because the intuitive reading — "small complexes score badly, so we need
more data per target" — is not what the data shows. Complexes with 11 rows score anywhere from
0.07 to 0.93; the 87-row complex scores 0.54. At this scale *per-complex* volume is not the
limiting factor, which shifts the data argument from "more mutations per complex" to "more
complexes, and more diversity across them".

## 5. Interface location behaves as binding physics predicts

| Location (single-point) | n | ρ | 95% CI | sd of true ΔΔG |
| --- | --- | --- | --- | --- |
| SUP support | 120 | 0.51 | [+0.30, +0.64] | 1.97 |
| COR core | 295 | 0.44 | [+0.32, +0.54] | 1.72 |
| RIM rim | 168 | 0.32 | [+0.14, +0.47] | 0.94 |
| SUR surface | 79 | −0.01 | [−0.26, +0.31] | 0.37 |
| INT interior | 34 | −0.28 | [−0.71, +0.23] | 1.40 |

The model is strongest exactly where binding is decided — support and core — and degrades
outward through rim to surface. **The surface result is correct behaviour, not failure:** those
mutations have an SD of 0.37 kcal/mol, below the ~0.5 measurement noise floor, so there is
genuinely nothing there to rank. INT is not interpretable at n=34.

**That ordering alone does not establish the claim, because label spread runs in the same order**
(SUP 1.98, COR 1.72, RIM 0.94, SUR 0.37; Spearman between spread and ρ across the five locations
is +0.70). Ranking is mechanically easier when the things being ranked are further apart. The test
that does establish it is an ablation — the same rows scored by a model with *no* structural
features at all:

| Location | n | chemistry only | full model | **gain from structure** |
| --- | --- | --- | --- | --- |
| SUP support | 120 | +0.166 | +0.511 | **+0.346** |
| COR core | 295 | +0.121 | +0.435 | **+0.314** |
| RIM rim | 168 | +0.137 | +0.319 | +0.182 |
| SUR surface | 79 | −0.080 | −0.005 | +0.075 |
| INT interior | 34 | −0.172 | −0.280 | −0.109 |

The chemistry-only column is **flat** — 0.166, 0.121, 0.137 across slices whose label spreads
differ by a factor of two. If variance drove the gradient, it would appear there too. It does not.
The gradient appears only once structural features are added, and it appears where the partner
buries the residue. SUP versus COR is *not* established — their intervals overlap heavily — only
"interface yes, surface no".

## 6. The alanine artefact is real but does not account for performance

| | n | ρ | 95% CI |
| --- | --- | --- | --- |
| X→Ala singles | 379 | 0.59 | [+0.48, +0.67] |
| other singles | 317 | 0.43 | [+0.21, +0.58] |

Alanine substitutions do score better, but the intervals overlap substantially, so the model has
**not** simply learned "alanine scan means destabilising". `PLAN.md` set this up as the test of
whether performance was a scanning artefact. It is not.

## 7. The ten worst residuals

Two clean clusters, and both are the compression problem at its extremes.

**Five of ten are 3HFM double-alanine hotspot mutations.** True ΔΔG +7.2 to +7.9; predicted +1.5
to +2.4. These are the classic HyHEL-10 / lysozyme hotspots where removing two contacting side
chains destroys binding almost completely. The model has never seen an effect that large in
training and will not extrapolate to one.

**Three of ten are 3BE1 / 3BDY 13- and 14-point humanisation variants.** True ΔΔG −2.8 to −3.2
(i.e. *improved* binding); predicted about +3.0 — wrong by six kcal/mol and wrong in sign. These
are affinity-maturation designs where many coordinated substitutions jointly improve binding,
which is simultaneously the multi-point failure, the non-additivity failure and the
stabilising-blindness failure in one row.

The remaining two: `2VIS_AB_C|IC89T` (true −4.91, predicted +1.34) sits in a complex with a
**single** row, so nothing about it could be learned; and `2JEL_LH_P|TP34N` (true +6.82, predicted
+0.90) is another extreme hotspot.

---

## What this implies for the next model

Ranked by what the evidence supports, not by what is most interesting to build.

1. **Correction: compression is a symptom, not the binding constraint.** An earlier version of
   this document put a rank-based objective first, on the grounds that shrinkage was the dominant
   error. That was wrong. Pearson (+0.376) and Spearman (+0.377) are *identical* on this model,
   and Spearman is scale-invariant — if the model were merely squashing the truth monotonically,
   the two would diverge. They do not, so compression costs RMSE and costs the headline metric
   nothing. Two further measurements agree: target centering, a related objective change, scored
   0.167 against 0.180; and deeper trees widened prediction spread from 0.747 to 0.768 while
   leaving bias on stabilising rows at +2.27, so the shrinkage is not over-averaging either. The
   limitation is informational, not one of scale or objective.

   **Reverse-mutation augmentation is the intervention that does move it.** Training folds
   doubled with reversed rows and flipped labels: predictions below zero 59 → 192 against a true
   254, bias on stabilising rows +2.28 → +1.67, their rank correlation −0.19 → +0.10, balanced
   sign accuracy 0.512 → 0.681. It costs per-complex ρ (0.487 → 0.416), so it is a choice about
   what the model is *for* rather than a free improvement — a model that flags 192 of 254 affinity
   improvements is more useful for antibody engineering than one that flags 59, even at a lower
   ranking score.
2. **The antibody side is the real gap.** ρ 0.15 against 0.62 on the antigen side. Antibody-
   specific representations — the *sequence* side, where AbBiBench measures CurrAb at +0.074 —
   target this directly. Note this cuts against AntiFold, which is antibody-specific on the
   structure side and measures −0.097.
3. **Stop summing per-position features for multi-point mutations.** 44% of cycles violate
   additivity and the model under-predicts epistasis spread threefold. A joint encoding of the
   mutation set, rather than a sum, is now justified by measurement.
4. **Per-complex data volume is not the constraint.** The learning-curve argument still stands at
   the dataset level, but "more rows for the complexes we have" is specifically not supported.
5. **Cross-attention remains unjustified by this analysis.** Nothing here says the model fails
   because it cannot see *which* residues a mutation contacts. It fails because it will not
   predict extreme values, cannot read antibody CDR loops, and cannot compose multiple mutations.
   Attention addresses none of those three.

---

# Part II — what the neural ladder revealed about the evaluation

Part I analyses the errors of the random forest. This part analyses the errors of the
*measurement*, and it is the more consequential of the two: several conclusions this project
had already drawn turned out to be smaller than the noise they were measured against.

Every number regenerates with `make report`. The dataset is 940 rows over 53 complexes, of
which **32 carry the five rows needed for a within-complex correlation**. Median 9 rows per
complex, maximum 87, and the three largest complexes hold 24% of the data.

## 8. Pooled Pearson mostly measures complex identity

Predicting nothing but each complex's own mean ΔΔG — a model with no access to the mutation
at all — scores:

| | per-complex r | pooled r | RMSE | concordance |
| --- | --- | --- | --- | --- |
| **complex mean only** | **+0.000** | **+0.672** | **1.144** | **0.00** |
| best network here | +0.282 | +0.383 | 1.523 | 0.66 |
| random forest | +0.397 | +0.509 | 1.335 | 0.74 |

It beats every model in this repository on pooled Pearson *and* on RMSE, while being useless
for the question a designer actually asks. Between-complex variance dominates both: complexes
differ in mean ΔΔG far more than mutations within a complex differ from one another, so a
model that merely identifies the complex scores well.

The consequences are not cosmetic.

- **Pooled Pearson and RMSE cannot be headline metrics here.** Every table in this repository
  leads with per-complex correlation, and `report_runs.py` prints this floor underneath every
  comparison so a reader can see what the pooled column is worth.
- **Any feature that identifies the complex is a leak in disguise.** The wild-type
  binding-site pool is identical for every mutation of a complex and is exactly such a
  feature. Turning it off (`use_site_pool=False`) is a one-line ablation, and it *helped*.
- Published ΔΔG results that lead with pooled correlation on SKEMPI are, to an unknown
  degree, reporting this quantity.

## 9. The noise floor is larger than most of the effects

Re-running an identical configuration under a different seed moves per-complex Pearson by
more than the architectural differences the ladder was built to measure.

| configuration | seeds | per-complex r | spread |
| --- | --- | --- | --- |
| no-PCA grouped, reg2 | 3 | +0.254 / +0.137 / +0.189 | **0.117** |
| no-PCA grouped, baseline | 2 | +0.274 / +0.199 | 0.075 |
| gated fusion, reg2 | 3 | +0.271 / +0.216 / +0.217 | 0.055 |
| cross-attention reversed, no PCA | 3 | +0.194 / +0.154 / +0.200 | 0.046 |
| gated fusion, baseline | 2 | +0.282 / +0.261 | 0.020 |

Within a *single fold* it is starker: on fold 1, three seeds of one configuration scored
pooled r of 0.050, 0.425 and 0.195 — a range of 0.376.

Set against that, the entire architectural ladder:

| change | Δ per-complex r |
| --- | --- |
| drop the fold-local PCA for a learned block-diagonal reduction | +0.025 (2 seeds) |
| cross-attention, structure attending to sequence, vs no attention | −0.030 (3 seeds) |
| gated fusion vs a plain MLP on the same features | +0.006 (2 seeds) |
| one MLP head layer instead of two | +0.011 (partial) |
| heavier regularisation (dropout 0.35, noise 0.25, wd 0.10) | −0.037 to −0.043 (3 seeds) |

**Every one of these sits inside the seed spread of at least one configuration being
compared.** The rule this imposes: no architectural claim here rests on fewer than three
seeds. Two claims that were made from partial folds — that dropping the PCA gained +0.062,
and that heavier regularisation halved the seed spread — reversed when the final fold
arrived. Both are recorded in `AI_PROMPTS.md` rather than quietly dropped.

## 10. Fusion mechanism is not the binding constraint

Six ways of combining sequence and structure were built and measured on the frozen split.
The control that matters — the same model with the fusion deleted — was missing until late,
and it changes the reading of the whole family.

| fusion | per-complex r | vs the no-structure control (+0.212) |
| --- | --- | --- |
| gated fusion, vector gates over both modalities | +0.282 | +0.070 |
| FiLM on the structure delta | +0.227 | +0.015 |
| cross-attention, structure → sequence | +0.204 | −0.008 |
| cross-attention, sequence → structure, weighted residual | +0.199 | −0.013 |
| cross-attention, separate MPNN projection | +0.192 | −0.020 |

Four of five attention variants score at or below a model with the attention removed. A
learned interaction between the modalities does not pay for itself. The one mechanism that
looks ahead, gated fusion, is +0.070 on two seeds whose own spread is 0.020, and it loses
0.037 the moment its regularisation is changed.

Against all of this, **the largest single gain in the ladder came from features, not
architecture**: adding 26 substitution-chemistry columns to the same network moved it from
+0.191 to +0.266.

## 11. The networks ride homology; the forest does not

Two splits over the same 940 rows. `frozen5` withholds whole complexes; `cluster` withholds
whole homology clusters, which is the harder and more honest generalisation test.

| model | by complex | by homology cluster | retained | seeds |
| --- | --- | --- | --- | --- |
| random forest, 49 handcrafted columns | +0.418 ρ | **+0.225** | **54%** | 3 |
| random forest, pooled ESM + ProteinMPNN | +0.362 ρ | +0.186 | 51% | 3 |
| our network | +0.246 ρ | +0.084 | **34%** | 1 |

The cluster split is harder for everything — the forest gives up nearly half — but the
network gives up two thirds, and 11 of its complexes finish with a *negative* within-complex
correlation.

Two cautions on this table. The forest rows are three-seed means; the network row is a single
seed, so its 34% is the least well estimated number here. And an earlier version of this
section reported the forest retaining 76%, from one seed — three seeds put it at 54%, which
narrows the gap considerably without closing it.
The collapse is therefore a property of the model rather than of the split: the
embedding-based network was relying on homology that the cluster split withholds, and the
handcrafted columns were not.

This is the strongest single argument in the repository for preferring the forest, and it is
why "use a larger encoder" does not appear in the next steps.

## 12. It is not a bookkeeping bug — the alignment is verified

Before accepting a ceiling near +0.27 for the networks, four ways the mutation could be in
the wrong place were checked. Each would look identical from a loss curve. Run with
`make align`.

| check | result |
| --- | --- |
| residue at the index equals the mutation string's WT residue | **0 mismatches across 1,726 sites in all 940 rows** |
| WT and MT sequences differ only at the recorded sites, same length | **0 violations** — rules out SEQRES-vs-ATOM and chain mix-up |
| ‖δ‖ = ‖t_mt − t_wt‖ peaked at the mutated residues | mutated sites occupy the **top k on 97% of sides** |
| magnitude of that peak | median **22×** the median residue, minimum 2.4× |

The peak test is judged on top-k rather than on rank 0 deliberately: with k mutations on one
side only one *can* be rank 0, so a rank-0 threshold marks every multi-point row a failure by
arithmetic and reports "suspect" on data that is clean — which is what the first version of
this check did.

The ceiling is the model and the task, not the pipeline.

## 13. What limits performance, ranked

1. **Data volume.** 752 training rows per fold, 32 complexes with enough rows to score. The
   seed spread in §9 is what that looks like from the inside. No regularisation setting
   removed it: raising dropout 75%, noise 150% and weight decay tenfold cost 0.037–0.043 on
   two different architectures and reduced variance on neither.
2. **Label noise.** Within-complex label standard deviation has a median of 1.104 kcal/mol,
   and repeated (complex, mutation) measurements in SKEMPI disagree at a scale that caps any
   achievable correlation.
3. **Between-complex variance swamping within-complex signal**, which is §8 — it makes the
   obvious metric the wrong one and rewards the wrong features.
4. **Representation, last.** Six probes, three protein language models, two structure
   encoders, and the sequence arm still does not clear zero. Twenty-six columns of
   substitution chemistry outperform 1280-dimensional embeddings of the same mutation.

**Capacity is not on this list.** Shrinking the model 64→32 cost 0.035; growing the head from
32k to 82k parameters gained 0.032; removing an entire head layer — 16,512 parameters, 31% of
the model — cost nothing measurable. A model that is insensitive to a third of its parameters
is not capacity-limited.

---

# Part III — bias from imbalance: labels, complexes, mutations

The dataset is unbalanced in three independent ways, and each hides a different failure. The
question in each section is not whether the imbalance exists but whether **error depends on
it**. Regenerate with `python scripts/bias_analysis.py`.

## 14. Label imbalance — the model cannot rank the class design cares about

Error by true class, `E0a_rf_handcrafted`, 940 rows:

| true class | n | mean true | mean pred | bias | MAE | within-class r |
| --- | --- | --- | --- | --- | --- | --- |
| stabilising | 126 | −1.429 | **+0.280** | **+1.709** | 1.710 | **0.007** |
| neutral | 315 | +0.046 | +0.465 | +0.419 | 0.558 | 0.093 |
| destabilising | 499 | +2.161 | +1.194 | −0.968 | 1.279 | 0.406 |

Two things here are worse than the headline +0.381 suggests.

**The average stabilising mutation is predicted to be destabilising.** True mean −1.429,
predicted mean +0.280 — the sign is wrong, not just the magnitude.

**Within stabilising mutations the model has no ranking ability at all: r = 0.007.** All of
the +0.381 comes from the destabilising class (r 0.406) and from separating destabilising
from the rest. For affinity maturation, where the task is to rank candidate *improving*
mutations against each other, this model is a coin flip.

The mechanism is compression:

```
predicted sd    0.897   vs true sd    1.793   (50%)
predicted range 6.68    vs true range 12.85   (52%)
```

The forest predicts into half the label's dynamic range. That is ordinary regression toward
the mean under a squared loss with 752 training rows — and it is exactly why cutting at the
**label** edges (−0.5, +0.5) starves the minority class: the predictions never reach out that
far. §"the defect and the fix" in the README shows quantile calibration recovering stabilising
recall 0.06 → 0.29 at zero cost to ranking, which treats the symptom. The absence of *ranking*
signal inside the class (r = 0.007) is not fixable by calibration and is the real limitation.

## 15. Complex imbalance — 40% of complexes are invisible to the headline metric

```
rows per complex: median 9, min 1, max 87; the top 3 hold 24% of all rows
32 of 53 complexes have the >=5 rows needed to score a within-complex correlation
21 do not, and never enter the headline number
```

That last line matters for how every table in this repository should be read: **per-complex
correlation is computed over 60% of the complexes**, systematically the larger ones.

Binned by size:

| complexes | count | rows | mean r | mean MAE | mean label spread |
| --- | --- | --- | --- | --- | --- |
| small | 11 | 98 | **0.210** | 1.207 | 0.998 |
| medium | 10 | 227 | 0.499 | 0.849 | 1.161 |
| large | 11 | 564 | 0.496 | 1.048 | 1.281 |

Small complexes score less than half as well. But the cause is not sample size:

```
corr(rows per complex,  per-complex r) = +0.109
corr(label spread,      per-complex r) = +0.225
```

**It is twice as much about how much variation a complex has to rank as about how many rows
it has.** The five worst complexes are mostly low-spread ones:

| complex | n | r | label spread |
| --- | --- | --- | --- |
| 3G6D_LH_A | 7 | −0.857 | 0.783 |
| 2NY7_HL_G | 11 | −0.433 | 0.416 |
| 1AHW_AB_C | 11 | −0.204 | 1.104 |
| 2B2X_HL_A | 70 | −0.179 | 1.038 |
| 1MLC_AB_E | 24 | −0.164 | 1.062 |

A complex whose mutations all land within ~0.4 kcal/mol of each other is being asked to be
ranked at a resolution finer than the assay's own reproducibility (within-complex label sd
has a median of 1.104). A negative correlation there is close to meaningless, yet it enters
the mean with equal weight — and 2B2X, at 70 rows, is not a small-sample artifact.

**Implication for the metric.** An unweighted mean over per-complex correlations treats a
7-row complex with 0.78 spread the same as an 87-row one. Weighting by rows, or excluding
complexes whose label spread is below the measurement noise, would both be defensible; the
current number does neither and should be read with that in mind.

## 16. Mutation imbalance — the model is good at alanine scanning

**X→A is 46% of all rows.** SKEMPI is dominated by alanine scans, and performance follows:

| mutation | n | mean true | bias | MAE | r |
| --- | --- | --- | --- | --- | --- |
| X→A | 429 | +1.336 | −0.106 | 1.038 | **0.630** |
| everything else | 511 | +0.665 | −0.176 | 1.143 | **0.363** |

The model is **nearly twice as good on the over-represented mutation type**. Reported
performance is therefore substantially a statement about alanine scanning, and a design
campaign proposing non-alanine substitutions should expect roughly the "everything else" row.

By wild-type residue, the largest groups:

| wt | n | mean true | mean pred | bias | r |
| --- | --- | --- | --- | --- | --- |
| Y | 121 | +2.015 | +1.726 | −0.289 | 0.546 |
| N | 104 | +0.320 | +0.677 | +0.358 | 0.568 |
| D | 87 | +0.999 | +0.945 | −0.055 | **0.735** |
| **V** | **84** | **+1.485** | **+0.094** | **−1.392** | **−0.284** |
| T | 75 | +0.435 | +0.448 | +0.012 | 0.289 |
| W | 55 | +1.879 | +1.629 | −0.250 | 0.514 |

**Valine is a systematic failure.** 84 rows, mutations that genuinely cost +1.49 kcal/mol on
average, predicted at +0.09 — under-predicted by **1.4 kcal/mol** — and ranked *backwards*
(r = −0.284). This is the single largest residue-level defect in the model and it is not
visible anywhere in the aggregate metrics. Valine is small, β-branched and buried; the
features may be describing its environment as tolerant when removing it actually collapses
packing. Worth a targeted look before any deployment.

Two mutant-side classes are also inverted on smaller samples: proline (n=19, r −0.130) and
cysteine (n=13, r −0.336) — both residues with backbone or disulfide effects that a per-residue
feature set is unlikely to capture.

## 17. Multi-point mutations — calibrated but imprecise

| | n | mean true | bias | MAE | r |
| --- | --- | --- | --- | --- | --- |
| single-point | 668 | +0.964 | −0.047 | **0.889** | 0.541 |
| multi-point | 272 | +0.988 | −0.382 | **1.601** | 0.530 |

Multi-point rows are **80% worse on MAE** while holding almost the same correlation. So the
model still *orders* them about as well; it cannot get their *magnitude* right, and it
under-predicts them by 0.38 on average. That is the additivity assumption showing: ΔΔG is not
additive in the number of mutations (corr(k, ΔΔG) = −0.151 overall, −0.292 among multi-point
rows), and a feature set built per-residue has no way to represent epistasis.

## 18. What this changes

1. **Report per-class and per-mutation-type performance, not just the aggregate.** The
   headline +0.381 conceals r = 0.007 on stabilising mutations and r = −0.284 on valine.
2. **Calibrate before thresholding** — free, and recovers most of the class balance.
3. **The per-complex mean is computed over 32 of 53 complexes** and weights a 7-row complex
   like an 87-row one. Row-weighting, or a minimum label-spread filter, would both be more
   honest; both should be stated rather than silently chosen.
4. **Do not generalise from alanine.** Half the data is X→A and performance on it is twice
   as good.
5. **Valine and multi-point rows are the two concrete targets** for the next modelling
   iteration, ahead of any architecture change.

## 19. Which descriptors predict the error — systematically

§14–§17 found failures by hand. This is the same question asked of every available descriptor
against every model at once, so that a defect belonging to the *task* can be told from one
belonging to an architecture. Spearman throughout; regenerate with
`python scripts/error_drivers.py`.

### Signed error — where the models are biased

Correlation between a descriptor and `y_pred − y_true`. Negative means the model
under-predicts as the descriptor grows.

| descriptor | forest | gated | concat | nopca |
| --- | --- | --- | --- | --- |
| **true ΔΔG** | **−0.792** | **−0.775** | **−0.842** | **−0.705** |
| **deviation from the complex's own mean** | −0.592 | −0.576 | −0.599 | −0.573 |
| \|ΔΔG\| | −0.477 | −0.521 | −0.560 | −0.508 |
| the complex's mean ΔΔG | −0.459 | −0.487 | −0.547 | −0.416 |
| **rows the complex has** | −0.329 | −0.314 | −0.366 | −0.310 |
| interface contacts | −0.116 | −0.323 | −0.349 | −0.292 |
| the complex's label spread | −0.116 | −0.307 | −0.289 | −0.165 |
| substitutions to alanine | +0.070 | +0.210 | +0.180 | +0.224 |

Three things, and they are the same for every architecture:

**Regression to the mean is the dominant error structure, at ρ ≈ −0.8.** Not a tendency — the
single largest relationship in the data. Every model shrinks toward the training mean, and
that one fact explains §14's class bias, §16's valine under-prediction and the compressed
output range together.

**It persists *within* a complex** (ρ ≈ −0.58 against deviation from the complex mean). That
is the quantity per-complex correlation is built from, so the shrinkage is not just a
between-complex offset a per-complex metric would cancel — it is eating the signal the metric
measures.

**Rows in larger complexes are systematically under-predicted** (ρ ≈ −0.33, all four models).
This is a representation bias in the plain sense: how often a complex appears in training
changes the direction of the error on it, not merely its size.

Alanine substitutions are *over*-predicted (+0.18 to +0.22 on the networks) — the opposite
sign to everything else, consistent with §16's finding that X→A is the over-represented type.

### Absolute error — where the models are imprecise

| descriptor | forest | gated | concat | nopca |
| --- | --- | --- | --- | --- |
| \|ΔΔG\| | 0.535 | 0.489 | 0.586 | 0.446 |
| deviation from the complex mean | 0.188 | 0.326 | 0.315 | 0.327 |
| interface contacts | 0.354 | 0.209 | 0.317 | 0.230 |
| ΔrSASA on mutation | 0.360 | 0.173 | 0.283 | 0.187 |
| distance to the partner chain | −0.272 | −0.230 | −0.301 | −0.195 |
| BLOSUM of the substitution | −0.274 | −0.209 | −0.200 | −0.174 |
| number of mutations | 0.349 | 0.116 | 0.189 | 0.124 |

Large effects, buried and highly-contacting sites, and radical substitutions (low BLOSUM) are
harder for everything. The forest is markedly more sensitive to the geometric descriptors and
to multi-point rows than the networks are, which is what one would expect from a model whose
features *are* the geometry.

### The models do not fail on the same rows

Spearman between per-row \|error\|:

```
         forest  gated  concat  nopca
forest    1.000  0.464   0.542  0.422
gated     0.464  1.000   0.866  0.611
concat    0.542  0.866   1.000  0.630
nopca     0.422  0.611   0.630  1.000
```

The two networks in the same family agree at 0.866 — near-duplicates, which is why the fusion
ladder in Part II went nowhere. But **the forest agrees with the networks only 0.42–0.54**, so
they are carrying partly independent information.

Testing that directly, z-scoring each model within complex and blending:

| | per-complex r |
| --- | --- |
| forest alone | +0.397 |
| `l1_gated` alone | +0.300 |
| `st64_nopca_grouped` alone | +0.274 |
| forest + `l1_gated`, 25 % net | **+0.418** |
| forest + `st64_nopca_grouped`, 25 % net | **+0.424** |
| forest + `st64_nopca_grouped`, 50 % net | +0.414 |

**Four of four blends beat the forest**, by up to +0.027. Read that as a demonstration that
complementary signal exists rather than as a validated model: the blend weight was not chosen
on held-out data, and +0.027 sits just inside the forest's own 0.032 seed spread. The
robustness across two different networks and two weights is what makes it worth pursuing —
and it is a cheaper, better-evidenced next step than any architecture in Part II.
