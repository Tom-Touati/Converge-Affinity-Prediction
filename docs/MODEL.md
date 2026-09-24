# The multimodal model

One architecture, specified completely. `docs/ARCHITECTURES.md` records the 45 configurations
that were measured to arrive at it; this is the one that is submitted.

**`l1_gated` — gated fusion of ESM-2 and ProteinMPNN at the mutated residues, 48,001 trainable
parameters.**

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
substitution applied and then passed through ESM again. Editing a latent vector instead would
assume the encoder is linear in the substitution, which it is not.

**Why ProteinMPNN for the structure side.** Its encoder is a message-passing network over a
k-nearest-neighbour graph — each residue represented by distances and relative orientations to
its spatial neighbours. A residue's embedding is therefore **a description of the pocket around
it**, which is precisely what a binding-site mutation is a question about: what the substituted
side chain can do in the space it occupies. Being an inverse-folding model, its output is
already the conditional we want (how compatible is this residue with this pocket) rather than a
generic embedding. It is also rotation- and translation-invariant by construction, and cheap —
CPU-only, ~7 minutes for all 54 complexes, frozen.

**There is no mutant structure.** ProteinMPNN sees the wild-type backbone only — the single
sharpest limitation of this design, and the reason the structure term is identical for every
mutation of a given complex.

## Forward pass

**1 — Shared perturbation** (training only). Gaussian noise at 0.25 × per-feature sd; feature
dropout at 0.35 with the mask drawn **once per width and reused across branches**, so it
becomes `mask·(mt − wt)` rather than corrupting the difference.

**2 — One shared projection.** `Linear(128 → 64)`, no bias, applied to **sequence and
structure, wild-type and mutant alike**. One map means the subtraction in step 3 happens in a
common learned space, and it is why this model is smaller than the concatenation variant
despite doing more.

**3 — The sequence edit.**
```
tok_wt = Σ_sides site_mean(proj(px(seq_side_wt)), site_side)      (B, 64)
tok_mt = Σ_sides site_mean(proj(px(seq_side_mt)), site_side)      (B, 64)
z_seq  = LayerNorm(tok_mt − tok_wt)                               (B, 64)
```
`site_mean` averages over mutated residues only. A multi-point row averages its k tokens rather
than being dropped — 29 % of the data, and 7 complexes have no single-point row at all. A side
carrying no mutation contributes exactly zeros, so the sum selects the mutated side without the
model being told which it is.

**4 — The structure context.**
```
z_str = LayerNorm( Σ_sides site_mean(proj(px(struct_side)), site_side) )   (B, 64)
```

**5 — Gated fusion.** The mechanism this architecture is named for:
```
g = sigmoid( W [z_seq ; z_str ; chem] )            W: 154 → 128
z = [ g[:64] ⊙ z_seq ; g[64:] ⊙ z_str ; chem ]     (B, 154)
```
Every one of the 128 feature channels gets its own gate, and the gate is computed from **both
modalities and the chemistry together**. So the model decides per row and per channel how much
sequence and how much structure to admit, conditioned on what the mutation is — rather than
mixing them at a fixed ratio the way concatenation does.

**6 — Head.** `Linear(154 → 128) → GELU → Dropout(0.35) → Linear(128 → 1)`. One hidden layer;
the second was removed after measuring that it cost nothing (§C4 of ARCHITECTURES).

Each block carries its **own** LayerNorm, all `elementwise_affine=False` so they cost zero
parameters. This is deliberate and was measured: the pools arrive ~7× larger than the edit
(2.396 vs 0.327). Normalised jointly, the pools set the scale and the edit — the only part that
varies between mutations of the same complex — is compressed toward nothing.

## Parameters

```
shared projection   Linear(128 -> 64), no bias        8,192   sequence AND structure
fusion gate         Linear(154 -> 128)               19,840
head                Linear(154 -> 128)               19,840
head                Linear(128 -> 1)                    129
LayerNorms          affine=False                          0
                                                    -------
                                                     48,001
```

## Training

| | |
| --- | --- |
| loss | MSE on ΔΔG clipped to ±4 kcal/mol |
| optimiser | AdamW, lr 3e-4, weight decay 0.10 |
| batch | 32 |
| gradient clip | 10 (above the measured norm of 5.0–5.3, so it rarely binds) |
| early stopping | on **per-complex Spearman**, patience 10 |
| validation split | whole complexes until 20 % of training **rows** are reached |
| augmentation | reverse-mutation, p = 0.5, training only |
| input noise | Gaussian, 0.25 × per-feature sd, training only |
| feature dropout | 0.35, mask drawn once per width and shared across branches |
| head dropout | 0.35 |
| protocol | 5 folds by complex × 3 seeds |

**Early stopping is on per-complex Spearman, not on loss.** Selecting on loss selects for
predicting complex means, which is the failure mode the whole evaluation is built to avoid.

**The validation split counts rows, not complexes.** Complexes range from 2 to 87 rows, so
taking a fixed fraction of *complexes* gives wildly variable validation sizes.

**Three perturbations, and they behave differently in the difference the model consumes.**

- **Reverse-mutation** is exact label information, not a heuristic: ΔΔG is antisymmetric, so
  swapping wild-type and mutant negates it. It moves the training label mean from **+0.720 to
  −0.031**, removing the incentive to guess "destabilising" on a set that is 70 % destabilising.
- **Feature dropout** uses one mask per width, reused across branches, so it becomes
  `mask·(mt − wt)` — it zeroes features *of* the edit rather than corrupting it. 35 % dropped,
  survivors scaled 1.54×, a median of 83 of 128 features surviving.
- **Gaussian input noise** is the one that misbehaves. It is drawn **fresh on each branch**, so
  it does **not** cancel in the difference — it compounds by √2 and reaches **0.8× the edit's
  own standard deviation** at this setting (edit sd 0.128; noise 0.25 × 0.282 × √2 = 0.0998).
  At the heaviest setting tried it would exceed the signal. The code comment claimed both were
  shared across branches; that was true of the mask and false of the noise.

Measured by `experiments/protattba_repro/audit_augmentation.py`, which runs on the local caches
and needs no GPU. This is the most likely reason heavier regularisation kept *costing* accuracy
(−0.037 and −0.043 on two architectures) without reducing seed variance, and sharing the noise
draw is listed first among the untried levers.

## Results

Per-complex Pearson, frozen by-complex split, scored on one common truth:

| | ensemble | per seed | spread |
| --- | --- | --- | --- |
| **`l1_gated`** | **+0.300** | +0.264 / +0.249 / +0.235 | **0.028** |
| plain concatenation | +0.293 | +0.274 / +0.234 / +0.210 | 0.063 |
| **no-fusion control** | **+0.212** | — | — |
| random forest, 49 handcrafted columns | +0.381 | +0.379 | 0.032 |

Three-class performance: balanced accuracy **0.463**, stabilising recall **0.25** — four times
the forest's 0.06.

## What can and cannot be claimed

**It beats the no-fusion control by +0.088**, which is roughly three times the seed spread.
Fusing the two modalities is worth something, and this is the measurement that says so — the
control is the same network with the structure branch deleted.

**It does not beat plain concatenation.** +0.300 against +0.293 is +0.007, far inside the
noise. Gating is presented as the architecture because it is a genuine fusion mechanism, is no
worse, and is **markedly more stable** — spread 0.028 against 0.063, with its worst seed
(+0.235) above concatenation's worst (+0.210). Claiming gating is *better* would not survive
the seeds.

**It does not beat the random forest**, +0.300 against +0.381 — about 2.5× the spread, and one
of the few gaps in this project that clears the noise. The forest is the stronger model for
ranking; this is the stronger model for finding stabilising mutations.

**The gates are not inspected.** No checkpoints were saved, so the obvious question — *what
does the model actually gate on* — cannot be answered from the artifacts. Saving weights and
reading the gate distribution per row is the first thing to do with this architecture, and
would turn the fusion mechanism from a score into an explanation.

## Why this one

The short version of `docs/ARCHITECTURES.md`:

- **Cross-attention was measured five ways and four of them score at or below the no-fusion
  control**, at twice the parameters. Antibody↔antigen attention was also measured, as the
  v2/v3/v4 family, at +0.112 to +0.200.
- **FiLM reaches +0.227**, above the control but below gating and concatenation.
- **Capacity is not the constraint** — an 810,886-parameter model scores +0.200.
- Gating and concatenation are the two cheapest mechanisms and the only two that clear the
  control convincingly. Between them, gating is the more stable.
