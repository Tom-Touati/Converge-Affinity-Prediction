# Perturbation model for antibody–antigen ΔΔG: implementation, ablations, error analysis

You are working autonomously. Nobody will answer questions. When something is ambiguous, pick the simplest defensible option, log it in `docs/decisions.md`, and continue. Never stall.

Assumed to exist already (inspect, don't assume paths; write `docs/STATUS.md` with what you find in ≤20 min): a working ProtAttBA reproduction, cached per-residue ESM embeddings for WT and mutant sequences of both chains, cached per-residue ProteinMPNN encoder embeddings (128-d) for WT complexes, WT PDB files, and the SKEMPI 2.0 antibody–antigen subset. Everything new goes under `src/perturb/`, `configs/`, `scripts/`, `results/`, `docs/`. Never modify cached features or the ProtAttBA reproduction.

---

## 1. The model

Two branches with identical weights: **ITW** (in-the-wild, wild type) and **MUT**. The mutation enters as a localized edit; ITW is the null edit. The head reads the difference.

**Inputs**
- ESM per-residue embeddings (raw width of the cached model), reduced in step 0.
- ProteinMPNN encoder per-residue embeddings of the WT complex, 128-d, shared by both branches, reduced in step 0.
- Cα coordinates from the WT PDB → all-vs-all Cα distances, cached per complex. Used for the crop, the mutation neighbourhood and the attention bias.
- BLOSUM62 rows of the WT and mutant residue at each mutated site (20 + 20). ITW uses the pair (a, a).

**Token sets**
- Antibody token set = heavy chain ⊕ light chain (concatenated). Antigen token set = all antigen chains concatenated. Each token carries `chain_type ∈ {heavy, light, antigen}`, `chain_id`, and its original per-chain residue index (PDB numbering order, not the index in the concatenated array).
- Crop, computed from the WT structure and identical for both branches (assert equality):
  interface residues (both sides, Cα–Cα < `r_iface` = 12 Å to any residue of the partner)
  ∪ mutation-site neighbourhood (both sides, Cα < `r_site` = 10 Å to any mutated residue)
  ∪ mutated residues ± 2 in sequence.
  Both radii are config values. Log the crop-size distribution per fold. Typical size 40–120 tokens; a non-interface mutation yields two disconnected regions, which is intended.
- Variable length is handled by padding to the longest in the batch and masking everywhere (attention, pooling, FiLM statistics). Bucket batches by crop length.

**Per branch, per residue i in the crop** (widths fixed; do not tune)
0. PCA each modality to 128, fit on the training fold's tokens only (unsupervised), cached per fold; same PCA for WT and mutant ESM. Then a learned per-feature scale and shift on each modality (2 × 128 × 2 = 512 params). Then a low-rank projection 128→16→64 per modality (no nonlinearity between the two factors): `seq_i ∈ R^64`, `t_i ∈ R^64`. Add the learned chain-type embedding `E_chain[chain_type_i]` (3×64) to `seq_i`.
1. Sequence delta `δ_i = seq_i(mut) − seq_i(wt)`. ITW: δ = 0. Flag `delta_scope ∈ {site, all}`, default `site` (δ zeroed off the mutated positions).
2. Structure refinement where δ ≠ 0: `t'_i = (1 + γ(δ_i)) ⊙ t_i + β(δ_i)`, `γ, β: 64→64`, initialised near zero. Where δ = 0, `t'_i = t_i` (ITW never touches structure).
3. Fuse: `h_i = GELU(W_f [seq_i ; t'_i] + b_f)`, `W_f: 128→64`.
4. BLOSUM pair (flag `use_blosum`, default on): `Linear(40→64)` of `[blosum_a; blosum_b]`, added to `h_i` at mutated positions only. ITW uses the pair (a, a).
5. Cross-chain attention, one layer, 2 heads × 32 = 64, **one set of W_Q, W_K, W_V, W_O shared by both directions**, pre-LN, residual, distance bias `b(bin(d_ij))` on the logits (16 bins to 20 Å + "beyond", shared across heads; flag `dist_bias`, default on). Both updates computed from the pre-update tensors, then applied:
   `H_ab ← H_ab + Attn(LN(H_ab), LN(H_ag))`, `H_ag ← H_ag + Attn(LN(H_ag), LN(H_ab))`.
   RoPE on the original per-chain residue index. No FFN.
6. Masked mean pooling over each token set → 64 per set → concat antibody ⊕ antigen → `f ∈ R^128`.

**Parameter budget** (trainable; backbones frozen and cached)

| component | shape | params |
|---|---|---|
| learned diagonal, both modalities | 2 × 128 × 2 | 512 |
| low-rank projections | 2 × (128→16→64) | 6,304 |
| chain-type embedding | 3 × 64 | 192 |
| FiLM δ_seq → structure | 64→64 × 2 | 8,320 |
| fuse concat → 64 | 128→64 | 8,256 |
| BLOSUM injection | 40→64 | 2,624 |
| cross-attention, shared | Q,K,V,O at 64 | 16,640 |
| distance bias | 17 bins | 17 |
| LayerNorm | 64 | 128 |
| head | 128→64→1 | 8,321 |
| **total** | | **≈ 51k** |

**Head**: `ŷ = MLP(f_MUT − f_ITW)`, 128→64→1, dropout 0.2.

**Training**: AdamW, lr 3e-4, weight decay 1e-2, dropout 0.2, batch 32, ≤60 epochs, early stopping on validation Pearson (10% of training complexes, grouped by PDB), seeds 0/1/2. Reverse-mutation augmentation on by default (swap WT/mutant, reverse BLOSUM pair, negate label; crop unchanged). No hyperparameter search.

**Required unit tests** (`tests/`): (a) with no mutation, ŷ == 0 exactly; (b) ITW branch output is identical with and without step 2; (c) ITW and MUT receive identical crop, mask, chain ids and residue indices for every sample; (d) distance bias reproduces contact-map ordering on a toy complex; (e) pooling is invariant to padding length (same output for a sample padded to 64 vs. 256); (f) residue-index alignment: for every SKEMPI row, the WT residue in the mutation string matches the residue at that index in both the ESM input sequence and the PDB-derived residue list; drop and log mismatches.

---

## 2. Protocol (fixed before any training)

- Splits: `data/splits/skempi_abag_5fold_by_complex.json`, 5 folds grouped by PDB ID, seed 0. Never regenerated. Secondary split if `mmseqs` is available: 5 folds grouped by 30% antigen sequence identity clusters — run only for the full model and the two reference rows.
- Metrics per (experiment, fold, seed): RMSE, MAE, Pearson, Spearman, R², 3-class accuracy and macro-F1 with bins |ΔΔG| < 0.5 / 0.5–2.0 / > 2.0 kcal/mol derived from the regression output. Also per-complex Pearson (mean over complexes with ≥ 5 mutations).
- `results/results.csv`, one row per (exp, fold, seed), columns: `exp, split, fold, seed, n_train, n_test, rmse, mae, pearson, spearman, r2, acc3, f1_macro3, per_complex_pearson, params, train_minutes, git_sha`. `scripts/summarize.py` regenerates `results/summary.md` (mean ± std over folds, seeds pooled) after every experiment.
- Out-of-fold predictions for every experiment saved to `results/oof/<exp>.csv` with columns `pdb, chain, mutation, ddg_true, ddg_pred, fold, seed`. The error analysis reads only these files.
- Each experiment ≤ 45 min wall time over all folds and seeds; if projected over, cut to 40 epochs, then to 1 seed, and log it. Failures go to `results/failures.md`; move on. Commit after each experiment.
- Append this file to `docs/ai_prompts/`.

---

## 3. Ablations

Run in this order. Each is the full model with exactly one change unless stated. Stop at the section boundary if time runs out; sections are ordered by importance.

**A. Reference rows** (needed to interpret everything below)
- `ref_rf`: random forest on [mean-pooled ESM per chain (WT, mut), ESM delta at site, mean-pooled MPNN per chain, MPNN at site, BLOSUM pair, ESM log-odds at site].
- `ref_protattba_seq`: the ProtAttBA reproduction, sequence only, on the frozen splits.
- `full`: the model in §1.

**B. Fusion mechanism** (the assignment's main criterion)
- `no_delta_film`: step 2 off. Mutation enters only via seq delta in the concat and BLOSUM.
- `struct_film_on`: add a second FiLM, structure → sequence (`γ₂, β₂: 64→64`, +8.3k), before the concat. Tests whether the second direction of modulation matters.
- `film_as_concat`: step 2 replaced by concatenating δ_i to t_i and a linear 128→64. Same information, no modulation.
- `fuse_linear`: step 3 without the GELU.
- `no_structure`: MPNN replaced by zeros (step 2 becomes identity + bias). Sequence-only perturbation model — the honest "does structure help" row.
- `shuffled_structure`: MPNN tokens permuted within each chain. Tests content vs. parameter count.
- `late_fusion`: no FiLM; two ProtAttBA towers (sequence tower, structure tower with MPNN tokens), pooled vectors concatenated before the head.

**C. Mutation encoding**
- `no_blosum`: step 4 off.
- `blosum_only`: seq delta zeroed; mutation carried by BLOSUM pair alone.
- `learned_aa_table`: BLOSUM rows replaced by a learned 20×20 embedding table.
- `delta_all`: `delta_scope=all`.
- `full_mutant_tokens`: no delta; MUT branch simply uses mutant ESM tokens everywhere (the siamese formulation). Direct test of perturbation vs. siamese.

**D. Branch structure and readout**
- `single_branch`: no ITW branch; head on `f_MUT`.
- `concat_head`: head on `[f_MUT; f_ITW]` instead of the difference.
- `no_reverse_aug`: augmentation off.

**E. Tokens and crop**
- `full_length_tokens`: no crop; all residues of all chains. The memorization-surface comparison.
- `iface_8A` / `iface_16A`: `r_iface` = 8 / 16 Å.
- `no_site_neighbourhood`: crop = interface only (non-interface mutations keep just the ±2 window).
- `site_only`: crop = mutation neighbourhood only, no interface set.
- `no_chain_embedding`: chain-type embedding removed.
- `array_index_rope`: RoPE on the index in the concatenated cropped array instead of the original residue index.
- `no_learned_diag`: step 0 without the per-feature scale/shift.
- `dense_proj`: step 0 low-rank replaced by dense 128→64 (+8.3k per modality).
- `rank_8` / `rank_32`: low-rank bottleneck at 8 / 32.

**F. Attention**
- `no_dist_bias`: distance bias off.
- `dist_bias_only_no_attn`: attention weights replaced by softmax of the distance bias alone (no Q·K). Tests whether learned attention adds anything beyond geometry.
- `pooled_hadamard`: attention removed; pooled `f_ab ⊙ f_ag` as the interaction (0 params).
- `pairwise_hadamard`: attention replaced by `H_ab ⊙ (W_d H_ag)` and the reverse, with `W_d` the row-normalised contact weights exp(−d²/2σ²), σ = 6 Å (0 params). Cross-attention minus the learned alignment.
- `heads_1x64`: 1 head.
- `separate_attn_dirs`: unshared weights per direction (+16.6k).
- `with_ffn`: add a 64→128→64 FFN after attention (+16.6k).
- `d128`: everything at width 128, 2 heads × 64 (≈ 4× params) — the "we scaled up" row.
- `no_cross_chain`: cross-chain attention removed; self-attention within each chain instead.

**G. Backbone and reduction** (only if A–F finished with ≥ 2 h left)
- `pca_64`, `pca_256`, `random_proj_128` (random Gaussian instead of PCA).
- `esm_layer_mid`: ESM embeddings from a middle layer instead of the last (recompute cache for the affected fold only if cheap; otherwise skip and log).

For each ablation write one sentence in `results/summary.md` under the table: what changed and whether the effect exceeds the fold spread of `full`.

---

## 4. Error analysis

`scripts/error_analysis.py` reads `results/oof/*.csv` and the SKEMPI metadata and writes `results/error_analysis.md` plus figures in `results/plots/`. Run on `full` and on `ref_protattba_seq`; every section below compares the two so the writeup can say what structure fixed and what it didn't.

**4.1 Global fit**
- Predicted vs. true scatter, coloured by fold, with y=x and the fitted line; report slope (regression-to-the-mean check).
- Residual histogram and residual vs. |ΔΔG_true|. Report Pearson separately for |ΔΔG| < 1, 1–2, > 2.
- Calibration of the 3-class bins: confusion matrix, and the ΔΔG ranges where classes are confused.
- Seed variance: std of predictions across seeds per sample; list the 20 highest-variance rows.

**4.2 Where the model fails, by biology**
Stratify residuals (mean |error| and Pearson, with n) by:
- Mutation chain: antibody vs. antigen; within antibody, CDR vs. framework (use a Kabat/Chothia annotation if available, else a hydrophobic-loop heuristic and say so).
- Interface distance: mutated residue's min Cα distance to the partner chain, bins < 5 / 5–8 / 8–12 / > 12 Å. Expect the model to be worse far from the interface; report whether it is.
- Substitution type: to-Ala, to/from Gly, to/from Pro, charge gain/loss/flip, size change (|Δvolume| terciles), hydrophobic ↔ polar. Which categories violate the fixed-backbone assumption most?
- Single vs. multi-point mutations, and for multi-point, by number of sites. Report crop size per bucket; if multi-point crops are systematically larger, flag it as a confound.
- Crop topology: mutations whose neighbourhood overlaps the interface set vs. disconnected (non-interface) — the latter tests whether the site can still "see" the interface through the distance-biased attention.
- Burial: relative solvent accessibility of the WT residue (DSSP or Shrake–Rupley via Biopython), buried / intermediate / exposed.
- Complex-level: per-complex Pearson sorted with n; name the 5 worst complexes and the 5 best; check whether the worst share a property (antigen type, resolution, number of mutations, ΔΔG range).
- Label range: complexes whose ΔΔG values span < 1 kcal/mol vs. more — per-complex correlation is undefined-ish on the former; report it honestly.

**4.3 What the model actually uses**
- Chain-type usage: attention mass from the mutated residue to heavy vs. light vs. antigen tokens, split by which chain was mutated.
- Attention sanity: for 30 random test mutations, Spearman correlation between the cross-chain attention weights from the mutated residue and the true inter-chain contact vector (1/d_ij). Report mean ± std for `full` and `no_dist_bias`. Plot 4 examples: attention row vs. contact row.
- Distance-bias curve: plot the learned `b(bin)` across folds. It should be monotone decreasing; say whether it is.
- FiLM magnitude: distribution of ‖γ₂(t'_i)‖ over interface vs. non-interface residues. If structure isn't modulating interface residues more, say so.
- Delta magnitude vs. error: is ‖δ_site‖ (ESM's opinion of the substitution) predictive of error? Scatter and correlation.
- Ablation-delta per sample: for `full` vs. `no_structure`, which samples improved most and which got worse? Table of the top 15 each with their biology labels from 4.2.

**4.4 Sanity and leakage checks**
- Fold-level: per-fold Pearson spread; flag any fold > 1.5× the others' spread.
- Reverse-mutation consistency on test rows: predict the reverse mutation and report mean |ŷ(fwd) + ŷ(rev)|. Should be small even though the constraint isn't hard.
- Near-duplicate check: for test rows whose complex has a ≥ 90% identical complex in training (chain-level identity), report performance separately.
- Null model: predict per-complex training mean; report its Pearson/RMSE so the reader sees what "0.3 Pearson" means on this data.

**4.5 Output**
`results/error_analysis.md` with: a 6-line executive summary at the top (the three biggest failure categories, what structure fixed, what it didn't, what the attention is doing), then the sections above with tables and linked plots. End with `docs/next_steps.md`: 5–8 proposals ranked by expected value, each tied to a specific finding above (e.g. "Pro/Gly and large→small at buried interface sites have 2× the error → repack side chains with FoldX and use ΔMPNN as an extra edit").

---

## 5. Morning deliverables

- `results/summary.md` — reference rows, full model, all ablations, one line of interpretation per ablation.
- `results/error_analysis.md` + `results/plots/`.
- `docs/decisions.md`, `docs/STATUS.md` (what ran, what failed, wall time, GPU), `docs/next_steps.md`.
- `configs/` with one YAML per experiment; single entry point `python -m src.perturb.run --config configs/<exp>.yaml --fold 0 --seed 0`. Every number in the tables must be reproducible from a config and the frozen splits.
