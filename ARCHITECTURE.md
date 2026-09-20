# Architecture: encoders, pipeline, and what 750 training rows can actually support

This document is the design spec for the learned models. It exists because the fusion question
deserves an explicit answer rather than an assertion, and because the honest answer involves
saying which parts of an attractive architecture we cannot afford.

Everything here is sized against one number: **723–757 training rows per fold**. Not 997 — that
is the whole dataset. Under grouped 4-fold CV the model never sees more than 757 examples at
once.

---

## 1. Where we actually are

Measured on the frozen split, 1,000-resample paired bootstrap over complexes.

| Rung | Features | Per-complex ρ | 95% CI |
| --- | --- | --- | --- |
| 1 | chemistry (21) | 0.180 | [0.108, 0.255] |
| 2a | ESM-2 35M scalars (5) | −0.066 | [−0.135, +0.011] |
| 2b | ESM-2 35M 480-dim + ridge | 0.074 | [−0.020, 0.175] |
| 3 | interface geometry (7) | 0.245 | [0.107, 0.349] |
| 3b | ProteinMPNN (5) | 0.204 | [0.139, 0.269] |
| 4 | chemistry + geometry (28) | 0.349 | [0.226, 0.452] |
| 5 | chemistry + geometry + ESM-2 (33) | 0.333 | [0.236, 0.424] |
| 6 | chemistry + geometry + ProteinMPNN (33), GBT | 0.408 | [0.307, 0.494] |
| **N0** | **same features, random forest** | **0.475** | **[0.393, 0.552]** |

Two facts to carry forward:

* **Structure works, sequence does not.** Hand-computed geometry (0.245) and ProteinMPNN (0.204)
  both beat chemistry alone; ESM-2 35M is at or below zero and *subtracts* 0.016 when added to
  chem+geom. Any design that spends parameters symmetrically on the two modalities is
  misallocating them.

  The one that didn't work was the protein language model. ESM-2 scored at or below zero and
  actively made the combination worse. Its notion of "surprising" is driven by evolutionary
  conservation, and antibody binding sites are the one part of a protein that evolution
  deliberately doesn't conserve.

  This is not an inference from the score alone — it is visible in the features. Mean
  masked-marginal log-likelihood ratio, grouped by SKEMPI's interface annotation:

  | | INT | SUR | SUP | COR | RIM |
  | --- | --- | --- | --- | --- | --- |
  | ESM-2 35M, mean LLR | −1.165 | −0.922 | −0.575 | **−0.449** | **−0.350** |

  The ordering is backwards for binding. ESM-2 finds mutations at buried *interior* positions
  most surprising, which is exactly right for folding stability, while interface **core and rim**
  — the positions that actually govern binding — surprise it least. It has learned burial-driven
  conservation, which is a folding signal.

  ProteinMPNN, asked the same way, orders by contact with the partner instead: its
  partner-attributable term `llr_delta` runs COR −0.486, RIM −0.301, SUP −0.098, INT −0.021,
  SUR +0.015 — concentrated at the interface and *zero* where nothing is touching. Two encoders,
  the same dataset, opposite orderings; only one of them is answering the question we asked.

  Consequence for the design: do not spend a parameter budget on a sequence encoder whose
  inductive bias points away from the target. If a sequence arm is kept at all, it should be
  there to supply residue identity and local context to the fusion, not to contribute a
  likelihood score of its own.
* **The best model has no learned encoder in its head at all.** 33 scalar features into a random
  forest. **0.475 is the number a neural model has to beat**, not 0.408.
* **Label noise looks like the binding constraint.** The forest beats the boosted trees on
  identical features by +0.069 [+0.002, +0.144] and is far steadier across folds. Bagging beating
  boosting is what noisy targets look like, and it argues against spending the remaining budget on
  capacity.

## 2. The three measurements that constrain the design

### 2.1 The structure signal is a *contrast*, not a representation

| ProteinMPNN feature | global ρ vs ΔΔG | sd |
| --- | --- | --- |
| `llr_complex` — log P(mut) − log P(wt), conditioned on the whole complex | −0.033 | 3.50 |
| `llr_alone` — same, conditioned on the mutated chain only | +0.099 | 3.44 |
| **`llr_delta` = `llr_complex` − `llr_alone`** | **−0.266** | 1.27 |

Neither absolute likelihood predicts binding. Their difference does, and beats chemistry. The
absolute value reads "does this residue fit this fold"; the difference reads "does this residue
fit this fold *because of the partner*". Only the second is binding.

**Design consequence.** The model must be built around the with-partner / without-partner
contrast. A fusion architecture that consumes complex-conditioned representations and hopes to
discover the contrast internally is being asked to learn, from 750 examples, something we can
hand it for free.

### 2.2 The mutant has no structure

We have wild-type backbones only. For every mutation the geometry is the *wild-type* geometry.
Measured earlier: 256 of 696 single-point rows (36.8%) sit at a site that is also mutated to a
different amino acid elsewhere in the data; `3HFM_HL_Y` position Y101 carries **nine** different
substitutions. For all nine, every geometric feature is numerically identical.

So "model the local geometry of the mutant" is not a thing the data supports. The mutant differs
from the wild type in exactly two places:

1. the amino-acid identity token at the mutated position, and
2. the PLM embeddings of the mutant *sequence*.

**Design consequence.** The structure stream is a **site** descriptor; the sequence stream is the
only **substitution** descriptor. Within a site, the entire structural contribution to ranking
nine substitutions is one 20-way softmax from the inverse-folding decoder. This is a hard ceiling,
not a modelling choice, and it is why the sequence modality failing is more serious than it looks.

### 2.3 The neighbourhood is small; the antigen is not

Sampled over 120 rows, residues with a heavy atom within 10 Å of the mutated residue:

| | value |
| --- | --- |
| median neighbourhood size | 46 residues |
| p10 / p90 / max | 28 / 61 / 69 |
| median fraction on the antigen side | 0.34 |
| rows with **no** antigen residue within 10 Å | 8% |
| antigen chain length (median / min / max) | 200 / 68 / 1488 |

**Design consequence.** A top-k of 48 captures the neighbourhood (and matches ProteinMPNN's own
`k_neighbors=48`, so the two representations stay consistent). Encoding the *whole* antigen is
out of reach — a 1,488-residue key set on 750 training rows is not a model, it is a memoriser.
The epitope is the tractable object, and it is already inside the 10 Å neighbourhood.

---

## 3. The pipeline, raw data to encoder input

This is the part that is usually hand-waved. Every step below is code that exists or is specified
to the level where writing it is mechanical.

### Step 0 — one row of SKEMPI

```
#Pdb                 3HFM_HL_Y
Mutation(s)_cleaned  YH33A
Affinity_wt_parsed   3.30e-10        Affinity_mut_parsed  8.90e-06
Temperature          298
```

`src/data.py` turns this into a modelling row with `row_id = "3HFM_HL_Y|YH33A"` and
`ddG = RT·ln(Kd_mut/Kd_wt) = +6.04` kcal/mol. 1,211 raw rows become 997 unique
(complex, mutation) pairs.

### Step 1 — parse the structure

`src/structures.py::parse_pdb` reads the *cleaned* PDB that SKEMPI distributes (not the RCSB
entry — see the data dictionary, trap 1) and produces per chain:

| field | shape | content |
| --- | --- | --- |
| `seq` | string | one-letter sequence in file order |
| `keys` | list | `(resnum, icode)` per position |
| `index` | dict | `(resnum, icode) → offset` — **the bridge from SKEMPI numbering to array index** |
| `backbone` | (L, 4, 3) | N, CA, C, O coordinates |
| `heavy` | list of (n, 3) | all heavy-atom coordinates per residue |

For 3HFM: chain H 215 aa, chain L 214 aa, chain Y 129 aa.

### Step 2 — locate the mutation

`parse_mutations("YH33A")` → `Mutation(wt='Y', chain='H', resnum=33, icode='', mut='A')`.
Then `chains['H'].index[(33, '')] → 32`.

That single lookup is what connects the label to both encoders. It is verified on all 2,109
mutated positions with zero mismatches, and `apply_mutations` asserts the wild-type residue
before editing, so an off-by-one cannot pass silently.

### Step 3 — sequence encoder input

Two strings per mutation:

```
wild type  ...TCSVTGDSITSD[Y]WSWIRKFPGNRL...
mutant     ...TCSVTGDSITSD[A]WSWIRKFPGNRL...
```

* wild type = `chains['H'].seq`
* mutant = `apply_mutations(structure, muts)['H']`
* they differ in exactly one character, at offset 32
* ESM-2 prepends `<cls>`, so the **token index is 33**

We run ESM-2 on both and keep, per residue in the neighbourhood, the final-layer hidden state
from *each* pass. The per-residue pair (wild-type state, mutant state) is what lets the model see
the substitution rather than just the site.

**Cost:** one forward pass per mutation per sequence version. ~997 mutant passes plus one
wild-type pass per chain. Measured at ~16 min on CPU for 35M.

**Known limitation:** ESM-2 is single-chain. For 3HFM it never sees chain Y — the antigen this
antibody binds. That is not a detail; it is the reason the structure encoder is necessary.

### Step 4 — structure encoder input

ProteinMPNN takes `(L, 4, 3)` backbone coordinates — N, CA, C, O — plus chain ids and residue
indices, for **all chains at once**. For 3HFM that is 558 residues across H, L and Y in one pass,
so the interface is visible to it in a way it never is to ESM-2.

Its encoder is structure-only: a k-nearest-neighbour graph (k=48) with edges encoded as
RBF-expanded distances between N, CA, C, O and a virtual Cβ, plus relative orientation and
sequence separation. Output is a per-residue hidden state (128-dim) that is a pure function of
geometry.

Its decoder additionally consumes sequence, and that is the only route by which a mutation's
*identity* reaches the structure model: feed the wild-type sequence as context, read the 20-way
distribution at the mutated position.

**We run it twice per complex:**

| pass | chains supplied | yields |
| --- | --- | --- |
| complex | all chains | `h_complex` per residue, `llr_complex` |
| alone | mutated chain only | `h_alone` per residue, `llr_alone` |

The difference of the two is the interface-specific signal established in §2.1. Both passes are
**per complex, not per mutation** — the backbone never changes — so this is 54 × 2 = 108 passes
total, cached once.

### Step 5 — assemble the neighbourhood

For each row, take residues with any heavy atom within 10 Å of the mutated residue's heavy atoms,
keep the nearest 48, and build one token per residue:

| slot | source | dim |
| --- | --- | --- |
| wild-type PLM state | ESM-2, wild-type pass | 480 → reduced |
| mutant PLM state | ESM-2, mutant pass | 480 → reduced |
| structure state, complex | ProteinMPNN encoder, complex pass | 128 → reduced |
| structure state, alone | ProteinMPNN encoder, chain-alone pass | 128 → reduced |
| geometry scalars | `src/features/geometry.py` | ~8 |
| distance to mutated residue | RBF-expanded, 16 Gaussians 2–22 Å | 16 |
| **amino-acid identity** | learned embedding, 20 entries | d |
| **chain role** (antibody-H / antibody-L / antigen) | learned embedding, 3 entries | d |
| **interface region** (COR/SUP/RIM/INT/SUR) | learned embedding, 5 entries | d |

The three learned embeddings are **added** to the projected vector, exactly as positional
embeddings are added in a transformer. They cost 20d + 3d + 5d parameters — negligible — and they
are how the model knows an antigen residue is an antigen residue.

**This is where the antigen enters.** Not as a separate encoded object, but as the ~34% of
neighbourhood tokens carrying the antigen chain-role embedding. §5 explains why that is a
deliberate compromise rather than a shortcut.

### Step 6 — the query

One token, built from the mutation itself:

* wild-type amino-acid embedding + mutant amino-acid embedding
* the 21 chemistry features from `src/features/chem.py`
* the wild-type and mutant PLM states at the mutated position, and their difference
* `llr_complex`, `llr_alone`, `llr_delta` from ProteinMPNN

---

## 4. Baseline: random forest

Added alongside the existing gradient-boosted trees, on the identical feature blocks and the
identical folds.

Not a formality. GBT and RF have opposite error profiles — boosting reduces bias by fitting
residuals sequentially and can chase label noise, bagging reduces variance by averaging
decorrelated trees and is markedly more robust to it. With a measurement noise floor near 0.5
kcal/mol, the gap between them is itself diagnostic: if RF matches or beats GBT, label noise is
limiting us, not model capacity.

Fixed configuration, no search: 500 trees, `max_features='sqrt'`, `min_samples_leaf=5`,
`max_depth=None`. Reported with the same paired bootstrap as every other rung.

---

## 5. The neural model, and the parameter budget

### 5.1 What the proposed architecture would cost

Bidirectional cross-attention between sequence and geometry, then cross-attention to a separate
antigen encoding. At `d=64`:

| Component | Parameters |
| --- | --- |
| ESM-2 480 → 64 projection | 30,720 |
| ProteinMPNN 128 → 64 projection | 8,192 |
| geometry ~24 → 64 projection | 1,600 |
| learned embeddings (aa ×2, chain role, region) | 2,752 |
| cross-attention block: Q,K,V,O (4d²) + LN + FFN (2·d·2d) | 33,024 **each** |
| — sequence → geometry | 33,024 |
| — geometry → sequence | 33,024 |
| — → antigen encoding | 33,024 |
| output head 64 → 64 → 1 | 4,225 |
| **Total** | **≈ 146,600** |

Against **750 training rows**: roughly **195 parameters per training example**.

For scale, the model that currently leads — rung 6 — fits 33 features on 757 rows, i.e. **23 rows
per parameter**. The proposal inverts that ratio by more than three orders of magnitude.

### 5.2 Why this is not merely "risky"

Three independent lines of evidence, all from inside this repo:

1. **The published learning curve.** `PRIOR_WORK.md`: test correlation on antibody–antigen ΔΔG
   only begins to plateau around **90,000 mutations**. We have 997. The authors' conclusion is
   that data availability, not architecture, is the limit.
2. **Our own sequence modality is dead.** ESM-2 35M scores −0.066 alone and reduces chem+geom
   from 0.349 to 0.333. **Bidirectional attention spends half its parameters wiring together two
   streams, one of which has no measured signal.** That is the single worst available allocation
   of a scarce budget.
3. **The objective moves results more than the architecture.** `BACKBONE_COMPARISON.md`: AbRank
   swings MINT from 0.43 to 0.78 AUC on the loss function alone, while four different frozen PLMs
   land within 0.03 PCC of each other.

**Verdict: no, the data does not support the full architecture.** Not as a matter of taste — the
sequence stream it would spend half its capacity on currently contributes negative signal.

### 5.3 What the data does support

Collapse three attention stages into **one**, and let the token construction carry what the extra
stages would have carried.

```
query  = mutation token  (wt/mut embeddings + chemistry + PLM diff + llr triple)
keys   = 48 neighbourhood residues, each token already containing BOTH modalities
         (PLM states ‖ ProteinMPNN states ‖ geometry ‖ learned embeddings)
attn   = 1 multi-head block, 4 heads, logits biased by RBF distance
head   = concat[query, attended] → MLP → ΔΔG
```

The substitution is deliberate: **early fusion at the token level instead of bidirectional
attention between two streams.** Each residue token already holds its sequence state and its
geometry state side by side, so "sequence informs geometry" happens inside the token projection
— a linear map — rather than across a 33k-parameter attention block. At 750 rows a linear map is
what we can afford to estimate.

With `d=32` and ESM-2 reduced to 32 dims by PCA fitted on **training folds only**:

| Component | Parameters |
| --- | --- |
| token projection (PCA-reduced inputs ~90 → 32) | 2,912 |
| learned embeddings (aa ×2, chain role, region) | 1,536 |
| one cross-attention block at d=32 | 8,320 |
| head 32 → 32 → 1 | 1,089 |
| **Total** | **≈ 13,900** |

About **19 parameters per training example** — the same order as a modest MLP, and defensible.

**Note: a GNN is the lighter version of the same mechanism, and may be the better trade here.**

Cross-attention and message passing answer the same question — *which neighbours matter, and what
do they contribute* — but they differ in how the neighbour set is restricted:

| | Cross-attention | GNN (message passing) |
| --- | --- | --- |
| Which neighbours interact | all 48, weights *learned* | only those joined by an edge, edges *given* |
| Distance prior | a learned bias added to logits | enforced by construction |
| Params per layer/block at d=32 | 8d² + FFN ≈ **8,320** | ~3d² ≈ **3,100** |
| Inspectable per prediction | yes, α is a 48-vector summing to 1 | only via edge gates, if added |

Three arguments for the GNN at our size:

1. **The sparsity is free regularisation.** Attention must *learn* that a residue 9 Å away on the
   far side of the chain is irrelevant; a distance-cut graph never offers that edge. At 757 rows,
   a prior enforced by construction costs nothing to estimate and a learned one costs data.
2. **It matches the source representation.** ProteinMPNN *is* a k=48 graph network. Building our
   head as a GNN over the same neighbourhood keeps the geometry of the two consistent rather than
   re-deriving it under a different inductive bias.
3. **Edge features are where our signal lives.** §2.1 showed the binding signal is a *contrast*
   between with-partner and without-partner. In a GNN that is naturally an **edge** attribute —
   same-side versus cross-side, distance, buried surface — and messages can be typed by it.
   Attention has to reconstruct the same thing from node features plus a chain-role embedding.

Against it: attention weights are directly readable per prediction, which matters for the error
analysis, and a GNN's messages are not. A middle option is an **attention-gated GNN** (GAT-style):
sparse edges from the distance cut, learned weights on the edges that survive. That keeps the
inspectability and the sparsity, at roughly 4d² ≈ 4,100 per layer.

Sequencing: if N1 passes, try the GNN before the attention block. It is cheaper, its prior is
stronger, and if it fails the attention version is very unlikely to succeed.

### 5.4 The ladder

Each step is kept only if it beats the one below by a paired-bootstrap Δ whose CI clears zero.
The objective is held fixed throughout, since it moves results more than any of these changes.

| Step | What | Params | Tests |
| --- | --- | --- | --- |
| **N0** ✓ | Random forest on rung-6 features | — | Label noise or capacity — **done: 0.475 [0.393, 0.552]**, +0.069 over GBT |
| **N1** | MLP on rung-6 features | ~5k | Does *any* neural head beat trees on 33 scalars |
| **N2** | Single cross-attention over the 48-residue neighbourhood | ~14k | Does attending to local structure beat pooled scalars |
| **N3** | Add the with-partner / without-partner contrast to every token | +0 | Does the §2.1 contrast generalise from the LLR to the hidden states |
| **N4** | Second cross-attention to a separately encoded epitope | +8k | Is the neighbourhood too small a view of the antigen |
| **N5** | Bidirectional sequence ↔ geometry | +16k | Only if a sequence encoder ever shows signal |

**N1 is not a formality.** If a plain MLP cannot beat gradient-boosted trees on the same 33
features, attention over 48 residues will not either, and the ladder stops there with a reportable
finding.

**N5 is gated on the sequence modality working.** With ESM-2 35M at −0.066 the gate is currently
shut. The ESM-2 650M run is what opens it, if anything does.

### 5.5 Regularisation, because this is the whole game at this size

* Frozen encoders. Only projections, embeddings, attention and head train.
* Dropout 0.2 on attention weights and the head.
* Weight decay 1e-2, AdamW, lr 3e-4.
* Early stopping on an **inner split grouped by cluster**, never random — a constant-per-fold
  predictor already scores global ρ −0.36 here, so a random inner split leaks homology straight
  back in.
* **10 seeds, ensembled, with seed variance reported.** At this sample size a single run's number
  is not a result. If seed spread exceeds the gap to rung 6, the comparison is not resolvable and
  we say so.

---

## 6. What we are giving up, and why

| Wanted | Delivered | Why the gap |
| --- | --- | --- |
| Bidirectional sequence ↔ geometry cross-attention | Early fusion inside the token projection | 66k parameters on 750 rows, half of them wiring a modality measuring −0.066 |
| Cross-attention to the antigen | Antigen residues inside the 48-token neighbourhood, tagged by a learned chain-role embedding | Antigen chains run to 1,488 residues; the epitope is the tractable object and it is already in the neighbourhood. Promoted to N4 if N2 clears |
| Local geometry of the *mutant* | Wild-type geometry only | No mutant structures exist. Predicting them adds a second error source on top of a 750-row problem |
| Local sequence, mutant and wild type | Both, as paired per-residue tokens | This one we can afford — it is two forward passes, not parameters |
| Learned embeddings for categoricals | Yes, added to token vectors | ~1.5k parameters total. Cheap and clearly right |

The compromise is consistent: **keep everything that costs forward passes or feature engineering,
cut everything that costs trainable parameters on a modality that has not yet earned them.**

## 7. The honest framing for the write-up

If N2 beats rung 6, we have shown that modelling *which* residues a mutation contacts beats
pooling them into scalars, on a strict homology split, at n≈1,000 — a real result.

If it does not, we report the parameter counts, the seed variance and the learning-curve evidence,
and conclude that at this sample size the binding constraint is data, not architecture. That
conclusion is directly supported by the literature and by our own ESM-2 result, and it is a more
useful submission than a fragile win.

Either way the fusion question gets an answer with a confidence interval attached, which is what
the assignment is actually asking for.
