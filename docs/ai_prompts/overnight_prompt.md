# Overnight run: multimodal fusion experiments for SKEMPI ΔΔG

You are working autonomously for ~8 hours. Nobody will answer questions. When something is ambiguous, pick the simplest defensible option, write the decision down in `docs/decisions.md`, and keep going. Never stall waiting for input.

## Goal

Extend our local ProtAttBA reproduction into a sequence + structure multimodal model for antibody–antigen ΔΔG on SKEMPI 2.0, run a fixed ladder of fusion experiments on identical splits, and leave behind a results table, plots, and an error-analysis report that I can read in the morning. Primary criterion: every comparison must be apples-to-apples. A clean negative result is worth more than a messy positive one.

## Current state (inspect first, do not assume paths)

- ProtAttBA reproduction runs locally (sequence-only, ESM per-residue features, dual cross-chain attention, attention pooling, MLP head).
- Cached per-residue ESM features for all sequences (WT and mutant, both chains).
- Cached per-residue ProteinMPNN encoder features for the WT complexes.
- Random-forest baseline on SKEMPI (hand-crafted / pooled features).
- Datasets available: SKEMPI 2.0 antibody–antigen subset (the benchmark), plus AB645, S1131, AB1101 from the ProtAttBA repo.

Start by spending ≤20 minutes reading the repo. Write `docs/STATUS.md` with: repo layout, where features are cached and their shapes, how the ProtAttBA repro is invoked, what the RF baseline uses. Then proceed.

## Hard rules

1. **Frozen evaluation protocol before any modeling.** Create `data/splits/skempi_abag_5fold_by_complex.json`: 5 folds, grouped by PDB ID, seed 0. Every experiment uses these folds. Never regenerate them.
2. **No leakage from auxiliary datasets.** S1131 is derived from SKEMPI and AB-bind overlaps with it. For each fold, any row from any dataset whose PDB ID is in that fold's test set is excluded from training. Deduplicate exact (PDB, chain, mutation) matches across datasets before anything else and log counts.
3. **Weighted sampling.** When training on the union, use a weighted sampler so that ~70% of drawn examples come from SKEMPI. Make the ratio a config flag.
4. **One results file.** Append one row per (experiment, fold, seed) to `results/results.csv` with columns: `exp, fold, seed, n_train, n_test, rmse, pearson, spearman, acc3, f1_macro3, params, train_minutes, git_sha`. `scripts/summarize.py` regenerates `results/summary.md` (mean ± std over folds per experiment) — run it after every experiment.
5. **3-class labels** for `acc3`/`f1_macro3`: bin ΔΔG by |ΔΔG| < 0.5 (low), 0.5–2.0 (medium), > 2.0 kcal/mol (high). Derived from the regression output — do not train separate classifiers.
6. **Fixed training recipe** across all neural experiments unless an experiment says otherwise: AdamW, lr 3e-4 for new modules, weight decay 1e-2, dropout 0.2, batch 32, max 60 epochs, early stopping on fold-validation Pearson (carve 10% of each fold's train complexes as val, grouped by PDB), 3 seeds (0,1,2). Model width d=64, 1 attention layer, 4 heads. No hyperparameter search.
7. **Time budget.** Each experiment ≤ 45 min wall time across all folds and seeds. If an experiment is projected to exceed it, reduce to 40 epochs, then to 1 seed, and note it in `docs/decisions.md`. Do not drop folds.
8. **Failures don't stop the run.** Wrap each experiment; on exception, write the traceback to `results/failures.md` and continue to the next.
9. **Never modify** the cached feature files, the original ProtAttBA reproduction, or the RF baseline code. Add new modules under `src/fusion/`. Do not rename or move existing files.
10. **Commit after each experiment**: `git commit -am "exp: <name> — <one-line result>"`.
11. Append this file verbatim to `docs/ai_prompts/overnight_prompt.md` (the assignment requires AI prompt history).

## Experiments, in priority order

Run in this order. Do not skip ahead. If short on time, E7–E8 are optional.

### E0 — Baselines on the frozen splits
- `E0a_rf_handcrafted`: existing RF baseline, re-run on the frozen folds.
- `E0b_rf_pooled_esm`: RF on mean-pooled ESM per chain, WT and mutant, concatenated, plus the ESM delta at the mutation site (mut − wt).
- `E0c_rf_pooled_esm_mpnn`: same as E0b plus mean-pooled ProteinMPNN per chain and MPNN at the mutation site. This is the "multimodal baseline" everything else must beat.
- `E0d_protattba_seq`: ProtAttBA reproduction (sequence only), fixed recipe, frozen folds.

### E1 — Early fusion by concatenation
Per-residue token = Linear([ESM_i ; MPNN_i]) → d. MPNN features come from the WT complex and are shared by the WT and mutant branches (mutant branch gets WT structure; the only difference between branches is the sequence embedding). Rest of the pipeline is ProtAttBA unchanged.

### E2 — FiLM fusion (structure modulates sequence)
token_i = proj(ESM_i) ⊙ (1 + γ(MPNN_i)) + β(MPNN_i), with γ, β linear maps 256 → d. Replaces ProtAttBA's kernel-1 conv gating. Also run the reverse direction once (sequence modulates structure) as `E2b`.

### E3 — Distance-biased cross-chain attention
Parse WT PDBs (Biopython) to Cα coordinates; build the inter-chain antibody–antigen distance matrix once per complex and cache to `data/cache/dist/<pdb>.npy`. In the cross-attention, add a learned bias `b(bin(d_ij))` to the logits, with 16 distance bins up to 20 Å plus one "beyond" bin, shared across heads. Build on E2 (FiLM tokens). If residue numbering between features and PDB cannot be aligned for a complex, drop that complex from E3 only, log it, and report how many were dropped.

### E4 — Interface-restricted pooling + mutation-site token
Build on E3. Pool only over residues within 10 Å of the partner chain, plus the mutated residue(s), plus a learned per-residue flag (1 for mutated positions, 0 otherwise) added to the token before attention. Ablate: `E4a` interface pooling only, `E4b` mutation flag only, `E4c` both.

### E5 — Antisymmetric head
Build on E4c. Replace concat(WT, MT) → MLP with `ŷ = g(MT, WT) − g(WT, MT)` where g is the existing MLP on concat. This enforces ΔΔG(A→B) = −ΔΔG(B→A) exactly. Also add the reverse-mutation training augmentation (swap branches, negate label) as `E5b`.

### E6 — Multi-dataset training
Take the best of E1–E5 by mean fold Pearson. Train on SKEMPI ∪ AB645 ∪ S1131 ∪ AB1101 with leakage exclusion (rule 2) and weighted sampling (rule 3) at 70/30. Also run 50/50 and 90/10 once each. Evaluate only on SKEMPI test folds.

### E6b — Paper-protocol benchmarks on all three ProtAttBA datasets
So the numbers are directly comparable to ProtAttBA Table 1, evaluate on each of the three benchmarks under the paper's own protocol: 10-fold CV on AB645 and S1131, 5-fold CV on AB1101, RMSE / R² / Pearson / Spearman, mean ± std. Save fold assignments to `data/splits/protattba_<dataset>_kfold.json` (seed 0, grouped by PDB ID) and never regenerate them. Run for: `E0d_protattba_seq` (our reproduction, should land near the paper), `E0c_rf_pooled_esm_mpnn`, and the best model from E1–E5. Write to a separate `results/benchmarks.csv` with a `dataset` column and add a "vs. paper" section to `results/summary.md` that puts the published ProtAttBA-ESM2 numbers next to ours. If the paper's fold assignments are in their repo, use those instead and note it.

### E7 — Modality ablations (optional)
On the best architecture: ESM only (zero MPNN), MPNN only (zero ESM), and shuffled MPNN (structure features permuted across residues within a chain — tests whether structure content matters or just the extra parameters).

### E8 — Mutation representation (optional)
On the best architecture: feed only the WT sequence to both branches and inject the mutation as an ESM delta vector at the mutated position (mut − wt embedding), instead of running the full mutant sequence. Cheaper and tests whether the model needs the full mutant context.

## Error analysis (run after E6, on the best model's out-of-fold predictions)

Write `scripts/error_analysis.py` producing `results/error_analysis.md` and plots under `results/plots/`:
- Predicted vs true scatter (all folds pooled), colored by fold.
- Residual vs |ΔΔG| — do we regress to the mean on large effects?
- Per-complex Pearson, sorted, with n per complex. Name the 5 worst complexes.
- Error by mutation category: to-alanine vs other, charge change vs none, interface vs non-interface (10 Å), single vs multi-point.
- Error by chain: antibody-side vs antigen-side mutations.
- Attention sanity check for E3+: for 10 random test mutations, correlation between the cross-attention weights from the mutated residue and the true inter-chain contact vector. Report the mean.
- Calibration of the 3-class bins: confusion matrix.

## Morning deliverables

- `results/summary.md` — the SKEMPI results table plus the three-benchmark "vs. paper" table.
- `results/error_analysis.md` + plots.
- `docs/decisions.md` — every judgment call you made, one line each, with the reason.
- `docs/STATUS.md` — updated with what ran, what failed, total wall time, GPU used.
- `docs/next_steps.md` — 5–8 concrete proposals ranked by expected value, each with the evidence from tonight's results that motivates it.

Keep code readable: type hints, docstrings on public functions, a single `configs/` directory with one YAML per experiment, and one entry point `python -m src.fusion.run --config configs/E2_film.yaml --fold 0 --seed 0`. Every experiment must be reproducible from its config and the frozen splits alone.
