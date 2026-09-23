# Error analysis

> **Superseded in places — see [HANDOFF.md](docs/HANDOFF.md).** Numbers here predate two
> corrections: the per-complex threshold moved from 10 mutations to 5 (the model to beat
> reads 0.388, not 0.498, on the same predictions), and residual learning was removed.
> Any figure produced by a net trained on `ddG - forest_prediction` had the forest added
> back at inference and is not that network's own score.


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
