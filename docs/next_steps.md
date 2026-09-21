# Next steps, ranked by expected value

Each item names the evidence from this run that motivates it. Numbers are from
`results/summary.md` and `results/error_analysis.md` unless stated.

---

### 1. Extract per-residue ESM-2 and ProteinMPNN for the 940 antibody-antigen rows

**Why now:** this is the single blocker on E1–E5. Every cached block under `data/features/` is
pooled or scalar, one row per mutation, so there is nothing for a cross-attention head to
attend over. The plan assumed these caches existed; they do not.

**Evidence:** `E0b_rf_pooled_esm` reaches pooled Pearson 0.252 against `E0a_rf_handcrafted`'s
0.489 with 3,853 columns against 49. Pooling a 1280-dimensional per-residue field down to one
vector per mutation is the obvious suspect, and it cannot be tested without the per-residue
form. The prerequisite is already done: `experiments/protattba_repro/cache/project_sequences.parquet`
holds the four sequences per row, validated (940 rows, 884 distinct sequences, 335,261
residues, 34 longer than ESM2's 1022 positions — and rotary extrapolation was measured to work
to at least 1492 tokens, so no windowing is needed).

**Cost:** ~3 min on a T4 for ESM-2 650M, by measurement (1.1 min for S1131's 129k tokens;
this is 2.6x the work). ProteinMPNN per-residue needs a new extractor because
`src/features/mpnn_repr.py` pools; rule 9 says add rather than modify.

---

### 2. Fix the magnitude bias before adding any architecture

**Why now:** the model's error is dominated by a systematic, measured shrinkage, not by
missing modality.

**Evidence** (`E0a`, out-of-fold):

| \|ddG\| bin | n | bias | rmse / sd |
|---|---|---|---|
| ≤0.5 | 315 | **+0.42** | **2.94** |
| 0.5–1 | 178 | +0.72 | 1.27 |
| 1–2 | 210 | +0.97 | 1.09 |
| >2 | 237 | **−1.39** | 1.02 |

Small effects are over-predicted by 0.42 and large ones under-predicted by 1.39, and
`rmse/sd ≥ 1` in every bin means the model beats that bin's own mean nowhere. A quantile or
two-part loss, or simply training on rank plus a calibrated scale, addresses this directly.
Note that adding modality did **not** help here: `E0c` has `rmse/sd` ≈ 0.97–1.0 across the
chain slices, i.e. no within-slice skill at all.

---

### 3. Treat the antibody side as the hard case explicitly

**Why now:** it is the case the project exists to serve, and it is the weak one.

**Evidence** (`E0a`): antibody-side mutations, 539 rows over 35 complexes, micro Spearman
**0.292** and `rmse/sd` **0.954**; antigen-side, 320 rows, micro Spearman **0.603** and
`rmse/sd` 0.848. `E0c` makes the antibody side worse still (micro 0.123). Affinity maturation
is an antibody-side problem, so a headline driven by antigen-side rows is flattering.

**Concretely:** report the antibody-side slice next to the headline in every future run, and
consider a CDR-aware feature or an antibody-specific PLM channel (`abplm.parquet` and
`currab150M.parquet` are already cached and unexplored on the frozen split).

---

### 4. Benchmark on AB645 and AB1101, not S1131

**Why now:** S1131 cannot support any antibody claim, and the sets that can are already in the
checkout.

**Evidence:** classified by SKEMPI's own `Hold_out_type`, S1131 is 990 rows Pr/PI and **0 rows
AB/AG**, with zero PDB overlap with our 53 complexes. AB645 and AB1101 carry real antibody
chain columns and share 19 of 25 and 20 of 28 complexes with us.

**Caution, and it is the point:** that overlap makes their published numbers a *poor* external
comparison for us — 499 of AB645's 645 rows and 656 of AB1101's 1100 sit in complexes we also
model, and the canonical duplicate count is 202 and 284 exact `(pdb, mutation)` matches. Any
training on them must keep the frozen cluster split or it leaks.

---

### 5. Re-run the whole ladder on the cluster-grouped split before believing any of it

**Why now:** tonight's table is on the plan's 5-fold **by-complex** split, which is the looser
protocol.

**Evidence:** the project measured that 42 of 54 complexes gain a TM > 0.8 training twin under
complex grouping against 0 of 54 under cluster grouping, and the forest's per-complex Spearman
moves 0.239 (cluster) → 0.354 (complex) on the same rows. Tonight's `E0a` per-complex Spearman
is 0.418 on 5-fold by-complex. None of these are comparable to each other, and only the
cluster number is the project's headline.

---

### 6. Add a paired bootstrap before writing any "X beats Y"

**Why now:** the E0 gaps are large enough to survive it, but the E1–E5 gaps probably will not,
and the project has already been burned once by an unpaired comparison.

**Evidence:** seed spread on `E0a` is ±0.001 pooled Pearson and ±0.020 per-complex Spearman —
so per-complex differences below ~0.04 are seed noise. `evaluate.paired_bootstrap(a, b,
n_boot=2000)` exists and is unused. `HANDOFF.md` §9 already lists this as open.

---

### 7. Decide the censored-affinity treatment, because it is worth more than any architecture

**Why now:** it is a larger effect than anything on tonight's ladder.

**Evidence:** dropping 57 censored rows (6% of the data) moved the forest from 0.388 to 0.239
per-complex Spearman. That is 0.149 from 6% of rows — more than the entire gap between the
handcrafted and multimodal baselines. Censored regression (a one-sided loss) or a ranking
constraint uses the row as the bound it actually is.

---

### 8. Finish the ProtAttBA S1131 reproduction, then stop

**Why now:** three of ten folds are in and they track the paper closely; the rest is
confirmation, not discovery.

**Evidence:**

| fold | ours PCC | theirs PCC |
|---|---|---|
| 0 | 0.7826 | 0.7673 |
| 1 | 0.8629 | 0.8638 |
| 2 | 0.8138 | 0.7780 |

**Caveat that limits how much the finished number is worth:** their protocol selects the best
epoch on the *test* fold (`WrapperDataset` assigns the test indices to `val_dataset` and both
loaders return it), so 0.84 is best-of-120-epochs on the reported data. The `honest` protocol
in `run_cv.py` measures the same folds with selection moved off them and has not been run. That
number, not 0.84, is the one a fused model would have to beat.
