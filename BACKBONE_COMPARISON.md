# Frozen backbone comparison

What the recent literature actually measures about frozen protein encoders on antibody–antigen
ΔΔG, and what it changes about the encoders this project picked.

Sources are the items logged in [`LITERATURE_SCAN_LOG.md`](LITERATURE_SCAN_LOG.md). Six were read
in full (AbBiBench, CIR-DDG, de Kanter & Greiff, ProtAttBA, AbRank, BindPred); the rest are marked
unread at the bottom and nothing here rests on them.

## The table that matters most

CIR-DDG, Table 1. Six backbones scored on **SKEMPI antibody–antigen interface single-point
mutations**, complex-level five-fold CV over 343 complexes. This is the closest published setup
to ours, and it is the only head-to-head of zero-shot inverse folding against supervised models
on exactly our cohort.

| Backbone | Type | Pearson r | **Spearman ρ** | RMSE | AUROC |
| --- | --- | --- | --- | --- | --- |
| ESM-IF | zero-shot inverse folding | 0.048 | **0.119** | 1.645 | 0.640 |
| ProteinMPNN | zero-shot inverse folding | 0.181 | **0.172** | 1.617 | 0.639 |
| DDAffinity | supervised on SKEMPI | 0.502 | 0.345 | 1.423 | 0.671 |
| RDE-Network | supervised on SKEMPI | 0.522 | 0.362 | 1.403 | 0.689 |
| DiffAffinity | supervised on SKEMPI | 0.555 | 0.439 | 1.369 | 0.737 |
| Vanilla Pythia-PPI | supervised | 0.595 | 0.470 | 1.322 | 0.753 |

Two things follow.

**Zero-shot inverse-folding likelihood is weak on antibody–antigen interfaces.** ESM-IF at ρ =
0.119 and ProteinMPNN at 0.172 are far below the supervised models, and ProteinMPNN beats ESM-IF
on every metric. Our own rung 1 — substitution chemistry into gradient-boosted trees, no protein
context at all — scores global ρ 0.207 on a *stricter* homology-clustered split. The comparison
is not exact (different cohort size, and a pooled ρ against our pooled ρ), but it is close enough
to say that a frozen inverse-folding LLR is not obviously worth more than BLOSUM plus volume.

**The supervised numbers are not a frozen-backbone result.** RDE-Network, DiffAffinity and
DDAffinity are trained on SKEMPI ΔΔG. They are the right target to aim at, not an encoder we can
drop in.

## Question 1: does the choice of frozen sequence encoder matter?

No. This is the most consistent finding across the set.

**ProtAttBA** holds one cross-attention head fixed and swaps four *frozen* PLMs behind it
(explicitly "Masked Language Model (frozen)"):

| Frozen backbone | AB645 PCC / ρ | S1131 PCC / ρ | AB1101 PCC / ρ |
| --- | --- | --- | --- |
| ESM-2 | 0.47 / 0.48 | 0.84 / 0.75 | 0.65 / 0.63 |
| ESM-1b | 0.47 / 0.47 | 0.82 / 0.70 | 0.64 / 0.63 |
| Ankh | 0.46 / 0.47 | 0.84 / 0.76 | 0.69 / 0.66 |
| ProtBert | 0.47 / 0.49 | 0.81 / 0.71 | 0.65 / 0.62 |

Four different pretraining corpora, architectures and parameter counts land within **0.03 PCC**.
S1131 is 1,131 SKEMPI interface single-point mutations — our own dataset's scale and source.

**BindPred** mean-pools the final layer of ESM-2 650M versus MINT (ESM-2 650M extended with
cross-chain attention) into gradient-boosted trees: PCC **0.860 vs 0.863** on random-split
five-fold CV over 11,919 complexes. On the SKEMPI subset specifically, plain ESM2 *beats* MINT.

**AbBiBench** finds ProGen2 (6.4B) and ProtGPT2 (738M) "fail to achieve competitive accuracy" —
scale does not rescue a sequence-only autoregressive model on this task.

> **Consequence for us.** The planned four-point ESM-2 capacity ladder (8M → 35M → 150M → 650M)
> is measuring a lever the literature has already shown to be flat. Two points are enough to
> confirm saturation on our own data; the other two buy a prettier curve, not a decision.

## Question 2: does structure beat sequence?

Contested, and the answer flips with the protocol — which is itself the useful finding.

- **AbBiBench** (15 frozen models, zero-shot, log-likelihood of the *mutant–antigen complex*
  against measured affinity, 14 Ab–Ag systems): inverse-folding models take the highest average
  Spearman and the highest 5-fold precision@10. The stated reason is scope — they "encode the
  entire Ab–Ag complex as a single global representation", where models using *local* structure
  tokens (SaProt, ProSST, ESM-3) do not. SaProt is the best non-inverse-folding model.
- **CIR-DDG** (table above): those same inverse-folding models are the *weakest* of six on the
  SKEMPI Ab–Ag interface cohort.
- **BindPred**: adding explicit structural energy terms on top of embeddings barely moves
  anything — PyRosetta-only 0.783 vs embeddings-only 0.860, combined "minimal"; BindCraft-only
  0.810 vs embeddings 0.870, combined **+0.008**. Conclusion: "incorporating explicit energy terms
  did not materially improve predictive accuracy." Caveat: BindPred predicts *absolute* log₁₀Kd,
  not ΔΔG of mutation, at 12× our data scale — a different and easier task.

The reconciliation is that AbBiBench ranks *whole complexes and designed variants*, where global
structural plausibility dominates, while CIR-DDG ranks *point mutations within one complex*, which
is our task. Structure helps most where the question is "is this a plausible binder"; it helps
least where the question is "which of these 40 point mutants binds tightest".

## Question 3: is ESM-IF1 reading binding, or stability?

Stability. This is the mechanism behind Question 2, and it is the single most actionable result.

**de Kanter & Greiff** benchmarked nine models (ESM-IF1, ProteinMPNN, AbMPNN, AntiFold,
ThermoMPNN, RaSP, KORPM, Rosetta Flex, ESM2) against 7,185 AlphaSeq mutations across four
VHH–antigen complexes:

- ESM-IF1 (single-chain mode) predicted the **control** epitope's ΔKD *as well as* the primary
  epitope's: **r = 0.64 control vs r = 0.59 primary**. A model reading the interface could not do
  that.
- On protein-*interaction* mutants specifically, ESM-IF1 and ThermoMPNN had consistently lower
  correlation (**p = 0.005**).
- ESM-IF1 degrades on double mutants: **r = 0.44 vs 0.59** for singles (p = 0.03).

Their conclusion: these models "can already capture *protein-quality* effects with reasonable
accuracy... however, they can only predict little of the *protein-interaction* changes."

> **Consequence for us.** Their ESM-IF1 was run in **single-chain** mode. Multichain conditioning
> on the full complex — which is exactly why we chose ESM-IF1 — is the untested variable. That
> makes the with/without-partner-chain ablation already in our plan a *primary* experiment rather
> than a nice-to-have, and it gives it a published control: if scoring the isolated antibody chain
> correlates with ΔΔG as well as scoring the complex does, our structure modality is a stability
> predictor wearing a binding hat.

## Question 4: are antibody-specific encoders better?

A coin flip, and fine-tuning can destroy the signal. From AbBiBench, both measured on the same
protocol across all datasets:

| Model | Base | Δ Spearman | Δ precision@10 |
| --- | --- | --- | --- |
| CurrAb (Ab-finetuned ESM-2) | ESM-2 | **+0.074** | +0.067 |
| AntiFold (Ab-finetuned ESM-IF1) | ESM-IF1 | **−0.097** | −0.078 |

The authors attribute AntiFold's drop to "potential catastrophic forgetting". They also flag the
protocol dependence themselves: AntiFold reports 0.427 on the 1mlc dataset when scored over CDR
regions with light- and heavy-chain variants, whereas AbBiBench excludes light-chain mutants and
scores the whole complex. So this is evidence against AntiFold *in a complex-scoring protocol*,
not a flat verdict.

## Question 5: does the objective matter more than the backbone?

Yes, by a wide margin. **AbRank**, AUC across splits with increasing distribution shift:

| Model | Unrelated: Balanced / Hard Ab / Hard Ag | Local perturbation: Balanced / Hard Ab / Hard Ag |
| --- | --- | --- |
| WALLE-Affinity (ranking) | 0.866 / 0.763 / 0.746 | 0.671 / 0.581 / 0.637 |
| WALLE-Affinity (regression) | 0.833 / 0.753 / 0.612 | 0.685 / 0.552 / 0.610 |
| ESM-2 + AntiBERTy (ranking) | 0.761 / 0.719 / 0.758 | 0.595 / 0.516 / 0.532 |
| ESM-2 + AntiBERTy (regression) | 0.836 / 0.740 / 0.814 | 0.572 / 0.467 / 0.478 |
| MINT (ranking) | 0.775 / 0.741 / 0.688 | 0.625 / 0.538 / 0.497 |
| MINT (regression) | 0.456 / 0.429 / 0.428 | 0.446 / 0.398 / 0.488 |
| AF3 + FoldX | 0.689 | 0.585 |
| Boltz1 + FoldX | 0.504 | 0.517 |
| AF3 + ANTIPASTI | 0.459 | 0.597 |
| Boltz1 + PBEE | 0.362 | 0.480 |

**MINT swings from 0.43 to 0.78 AUC on the loss function alone**, with the backbone unchanged.
Ranking wins on *every* Local Perturbation split for every model — and local perturbation, point
mutations within a complex, is our task. On the Unrelated Complex benchmark regression sometimes
wins, which is the opposite regime.

This is stronger support for the pairwise-ranking reformulation than the argument I made from
first principles earlier: the effect size is larger than any backbone difference in this document.

## What this changes

| Decision | Current | Evidence | Recommendation |
| --- | --- | --- | --- |
| Structure backbone | ESM-IF1 primary, ProteinMPNN fallback | CIR-DDG: ESM-IF 0.119 < ProteinMPNN 0.172 on our exact cohort. de Kanter: ESM-IF1 reads stability | **Swap the roles.** ProteinMPNN primary — better on the one head-to-head, and a trivial install on this machine. ESM-IF1 becomes the comparison arm, not the fallback |
| Sequence ladder | 8M / 35M / 150M / 650M | ProtAttBA: four PLM families within 0.03 PCC. BindPred: ESM2 vs MINT 0.003 | **Trim to 35M + 650M.** Two points confirm saturation; four buy a curve the literature already drew |
| Partner-chain ablation | planned as a probe | de Kanter's control-epitope design | **Promote to a primary experiment.** It is the stability-vs-binding test, and it decides whether our structure modality is real |
| Stability control | not planned | de Kanter, p = 0.005 | **Add.** Score the isolated antibody chain and a pure stability predictor; report how much of the structure signal survives |
| AntiFold | deferred encoder swap | AbBiBench −0.097, catastrophic forgetting | Keep deferred. Record the evidence so we do not spend time on it |
| MINT | not considered | BindPred, AbRank | Cheap addition to the sequence arm — an interaction-aware ESM-2 with cross-chain attention. Note SKEMPI is the subset where plain ESM2 beat it |
| Ranking loss | deferred augmentation idea | AbRank: 0.32 AUC swing on MINT; ranking wins every local-perturbation split | **Promote.** Larger effect than any backbone choice measured here |

What does **not** change: encoders stay frozen, the split stays homology-clustered, and the
headline metric stays per-complex Spearman. Nothing in these papers argues against any of those,
and CIR-DDG's complex-level split and BindPred's random split are both weaker than ours, so their
absolute numbers should be read as optimistic relative to what we will report.

## Sources

Read in full:

- [AbBiBench (arXiv 2506.04235)](https://arxiv.org/abs/2506.04235) — 15 frozen models, zero-shot complex log-likelihood, 14 Ab–Ag systems
- [CIR-DDG (arXiv 2608.27530)](https://arxiv.org/abs/2608.27530) — six backbones + residual adapter, SKEMPI Ab–Ag, complex-level 5-fold
- [de Kanter & Greiff (bioRxiv 2026.03.24.713863)](https://www.biorxiv.org/content/10.64898/2026.03.24.713863v1) — 7,185 AlphaSeq VHH mutations, stability vs interaction
- [ProtAttBA (arXiv 2505.20301)](https://arxiv.org/abs/2505.20301) — frozen PLM swap behind a fixed head, AB645 / S1131 / AB1101
- [AbRank / WALLE-Affinity (arXiv 2506.17857)](https://arxiv.org/abs/2506.17857) — 380k assays, ranking vs regression under distribution shift
- [BindPred (Bioinformatics btag309)](https://academic.oup.com/bioinformatics/article/42/6/btag309/8678529) — ESM-2 vs MINT + GBT, PPB-Affinity 11,919 complexes

Logged but **not yet read**; nothing above depends on them:

- USP-ddG (bioRxiv 2025.11.09.687124) — MoE fusion of inverse folding + FoldX + geometry, CATH-superfamily split
- AbAffinity chain-aware PLM (bioRxiv 2026.06.19.733375) — frozen ESM-2 + gated cross-attention
- Multi-scale fusion on AbAgym (bioRxiv 2026.06.09.730151) — reports that only multi-scale fusion beats baseline under LOCO
- DSSA-PPI (PMC12758018) — disentangled cross-attention, +0.03–0.04 PCC over concatenation on S1131
- AbLWR (arXiv 2604.11272) — partially read: uses IgFold (512-d) for antibody CDRs and ESM-2 (1280-d) for antigen, an asymmetric backbone assignment worth noting
- Light-DDG / SKEMPI-Aug (arXiv 2502.06913), ANDD (Nature Sci Data 2026) — data resources, not backbone evidence

USP-ddG and the two fusion papers are the ones that would most change the picture, since they bear
on whether fusion beats the best single modality — the core question of rung 4.
