# The multimodal model — `cat128_reg2_l1` (superseded)

> **This is no longer the submitted model.** `cat128_reg2_l1` was superseded by
> **`struct_film_chem`** (+0.369 against +0.293) — specified in
> [DETAILS.md](DETAILS.md#struct_film_chem), summarised in README §1. A later sweep of
> mutation-position features tried to improve on it and did not, once a baseline confound in
> that sweep was caught and corrected (`ARCHITECTURES.md` §Family F) — `struct_film_chem` is
> still the submitted model. **This document is left intact and is still accurate for the
> model it describes**, which remains the reference point for most of `JUSTIFICATIONS.md` and
> Parts I–III of `ERROR_ANALYSIS.md`. Read it as the full specification of a previous
> submission, not the current one.

One architecture, specified completely. `docs/ARCHITECTURES.md` records the 45 configurations
measured to arrive at it.

**`cat128_reg2_l1` — ESM-2 and ProteinMPNN each given their own projection, read at the mutated
residues, concatenated with substitution chemistry into a single-hidden-layer head. 36,353
trainable parameters, both encoders frozen.**

**It is the simplest fusion in the family, and it won.** Forty-five configurations were
measured — cross-attention in five variants, antibody↔antigen attention across the interface,
FiLM, gated fusion, a two-tower model, a learned block-diagonal reduction replacing the PCA,
binding-area pooling, an ordinal head. Not one of them beats plain concatenation outside the
seed spread, and several lose to a model with the fusion **deleted**. The largest model built
in this project has 810,886 parameters and scores +0.200 against this model's +0.293.

That is the result, not an apology for it. `ERROR_ANALYSIS.md §25` collects why: there is no
mutant structure, so the structure term is identical for every mutation of a complex and
attention has little to attend to; ProteinMPNN's signal is local and washes out when pooled;
regression to the mean dominates the error of *every* architecture equally; and capacity is
free to add and free to remove. On 752 training rows per fold, the mechanism of fusion is not
what separates models.

> **Why not the gated model.** An earlier draft submitted `l1_gated`, which scores +0.300
> against this model's +0.293. That model turned out to share **one** `Linear(128 → 64)`
> between ESM-2 and ProteinMPNN — a bug, not a design: its config set `mpnn_proj=64` and the
> model silently ignored it, because `gated_fusion` was missing from the condition that builds
> the structure projection. Two encoders with unrelated output spaces were being forced through
> one map. The +0.007 difference is well inside both models' seed spreads, so nothing
> measurable is given up by preferring the architecture that is coherent. The bug is fixed in
> `model_simple.py`; re-running gated fusion with a real structure projection is the obvious
> follow-up and is **not** claimed here.

---

## Inputs

Both encoders are **frozen**. Nothing in them trains.

| tensor | shape | source |
| --- | --- | --- |
| `seq_{ab,ag}_{wt,mt}` | (B, L, 128) | ESM-2 650M per residue (1280) → fold-local PCA-128, fitted per side on training folds only |
| `struct_{ab,ag}` | (B, L, 128) | ProteinMPNN `encoder_h_V` per residue (128) → PCA-128 |
| `site_{ab,ag}` | (B, L) | 1 at mutated residues |
| `mask_{ab,ag}` | (B, L) | the crop |
| `chem` | (B, 26) | substitution chemistry, standardised on training folds only |

**The crop**: interface (<12 Å) ∪ site neighbourhood (<10 Å) ∪ mutated ±2, median ~85 residues.

**The mutant is re-embedded, not edited.** The mutant sequence is the wild-type with the
substitution applied, passed through ESM again. Editing a latent vector would assume the
encoder is linear in the substitution; it is not.

**Why ProteinMPNN for the structure side.** Its encoder is message passing over a
k-nearest-neighbour graph — each residue described by distances and relative orientations to
its spatial neighbours. A residue's embedding is therefore **a description of the pocket around
it**, which is what a binding-site mutation is a question about. Being an inverse-folding
model, it emits the conditional we want directly. Rotation- and translation-invariant by
construction, CPU-only, ~7 minutes for all 54 complexes.

**There is no mutant structure.** ProteinMPNN sees the wild-type backbone only — the sharpest
limitation of this design, and the reason the structure term is identical for every mutation of
a given complex.

## Forward pass

**1 — Shared perturbation** (training only). Gaussian noise at 0.25 × per-feature sd; feature
dropout at 0.35 with the mask drawn **once per width and reused across branches**, so it
becomes `mask·(mt − wt)` rather than corrupting the difference.

**2 — One projection per modality.** `Linear(128 → 64)`, no bias, and there are **two of
them**: one for sequence, one for structure. The sequence map is shared between the wild-type
and mutant branches — that sharing *is* justified, because the subtraction in step 3 only means
something if both sides land in one space. Sharing *across modalities* would not be: ESM-2 and
ProteinMPNN are never subtracted from one another, only concatenated, so there is nothing for a
common space to buy.

**3 — The sequence edit.**
```
tok_wt = Σ_sides site_mean(proj_seq(px(seq_side_wt)), site_side)      (B, 64)
tok_mt = Σ_sides site_mean(proj_seq(px(seq_side_mt)), site_side)      (B, 64)
z_seq  = LayerNorm(tok_mt − tok_wt)                                   (B, 64)
```
`site_mean` averages over mutated residues only. A multi-point row averages its k tokens rather
than being dropped — 29 % of the data, and 7 complexes have no single-point row at all. A side
carrying no mutation contributes exactly zeros, so the sum selects the mutated side without the
model being told which it is.

**4 — The structure context**, through its own projection:
```
z_str = LayerNorm( Σ_sides site_mean(proj_str(px(struct_side)), site_side) )   (B, 64)
```

**5 — Fusion by concatenation.**
```
z = [ z_seq(64) ; z_str(64) ; chem(26) ]     (B, 154)
```

**6 — Head.** `Linear(154 → 128) → GELU → Dropout(0.35) → Linear(128 → 1)`. One hidden layer;
the second was removed after measuring that it cost nothing (ARCHITECTURES §C4).

Each block carries its **own** LayerNorm, all `elementwise_affine=False` so they cost zero
parameters. Measured, not assumed: the pools arrive ~7× larger than the edit (2.396 vs 0.327).
Normalised jointly, the pools set the scale and the edit — the only part that varies between
mutations of one complex — is crushed.

## Parameters

```
sequence projection   Linear(128 -> 64), no bias     8,192   shared WT/MUT, NOT across modalities
structure projection  Linear(128 -> 64), no bias     8,192
head                  Linear(154 -> 128)            19,840
head                  Linear(128 -> 1)                 129
LayerNorms            affine=False                       0
                                                   -------
                                                    36,353
```

## Training

| | |
| --- | --- |
| loss | MSE on ΔΔG clipped to ±4 kcal/mol |
| optimiser | AdamW, lr 3e-4, weight decay 0.10 |
| batch | 32 |
| gradient clip | 5 (turning it off costs 0.028 — measured, ARCHITECTURES §C4) |
| early stopping | on **per-complex Spearman**, patience 10 |
| validation split | whole complexes until 20 % of training **rows** are reached |
| augmentation | reverse-mutation, p = 0.5, training only |
| input noise | Gaussian, 0.25 × per-feature sd, training only |
| feature dropout | 0.35, mask drawn once per width, shared across branches |
| head dropout | 0.35 |
| protocol | 5 folds by complex × 3 seeds |

**Early stopping is on per-complex Spearman, not on loss.** Selecting on loss selects for
predicting complex means, which is the failure mode the whole evaluation exists to avoid.

**The validation split counts rows, not complexes**, because complexes range from 2 to 87 rows.

**Three perturbations, and they behave differently in the difference the model consumes.**

- **Reverse-mutation** is exact label information: ΔΔG is antisymmetric, so swapping negates
  it. It moves the training label mean from **+0.720 to −0.031**, removing the incentive to
  guess "destabilising" on a set that is 70 % destabilising.
- **Feature dropout** uses one mask per width reused across branches, so it becomes
  `mask·(mt − wt)` — it zeroes features *of* the edit rather than corrupting it.
- **Gaussian input noise** misbehaves: drawn fresh on each branch, it does **not** cancel in
  the difference — it compounds by √2 and reaches **0.8× the edit's own sd** (edit sd 0.128;
  0.25 × 0.282 × √2 = 0.0998). Measured by `audit_augmentation.py`. Sharing the draw is a
  two-line change and is the top untried lever.

## Results

Per-complex Pearson, frozen by-complex split, one common truth:

| | ensemble | per seed | spread |
| --- | --- | --- | --- |
| **`cat128_reg2_l1`** | **+0.293** | +0.274 / +0.234 / +0.210 | 0.063 |
| gated fusion *(shared-projection bug)* | +0.300 | +0.264 / +0.249 / +0.235 | 0.028 |
| **no-fusion control** | **+0.212** | — | — |
| random forest baseline | +0.381 | +0.379 | 0.032 |

Three-class: balanced accuracy **0.439**, stabilising recall **0.19** — three times the
forest's 0.06.

## What can and cannot be claimed

**Fusion is worth something: +0.081 over the no-fusion control**, roughly 2.5× the seed spread.
The control is the identical network with the structure branch deleted, which is what makes
that number mean anything.

**It does not beat the random forest**, +0.293 against +0.381 — a gap that does clear the
noise. The forest is the better ranker; this model finds three times as many stabilising
mutations (0.19 against 0.06).

**It is not distinguishable from gated fusion**, and the gated number came from a model with an
unintended shared projection. Neither the +0.007 nor the tighter seed spread should be read as
a property of gating until it is re-run correctly.

**The seed spread is 0.063**, wider than the gated model's 0.028. That is the honest cost of
this choice, and it sits inside the range of every other configuration in the project.

## Why this one

From `docs/ARCHITECTURES.md`:

- **Cross-attention was measured five ways; four score at or below the no-fusion control**, at
  twice the parameters. Antibody↔antigen attention was measured as the v2/v3/v4 family at
  +0.112 to +0.200, also below.
- **FiLM reaches +0.227**, above the control but below concatenation.
- **Capacity is not the constraint** — an 810,886-parameter model scores +0.200.
- Concatenation and gating are the two cheapest mechanisms and the only two that clear the
  control convincingly. Between them, concatenation is the one whose weights are arranged
  defensibly.
