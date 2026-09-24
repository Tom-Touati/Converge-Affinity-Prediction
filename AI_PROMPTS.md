# AI prompt history

The assignment's AI policy is "highly encouraged", and asks for the prompt history. This file is
appended in the same commit as the code a prompt produced, not reconstructed afterwards.

Tooling: Claude Code (Opus 5) in the Claude desktop app, working directly in this repository.

What it converged on: **`cat128_reg2_l1`** — see [docs/MODEL.md](docs/MODEL.md). The errors and retractions listed at the end of this file are part of the record on purpose; several of them are the reason the submitted model is the simplest one in the family rather than the most elaborate.

---

## Session 1 — 2026-09-18 — planning and background

Earlier prompts in this session produced `PLAN.md`, `EDA_FINDINGS.md`, `PRIOR_WORK.md`,
`notebooks/01_eda.ipynb` and `daily_scan_prompt.md`. They are summarised rather than quoted
because they predate this log; the artefacts themselves are the record.

## Session 2 — 2026-09-18 — repo skeleton, split, harness, rung 1

> read the files in the directory

Read the assignment PDF, the three planning documents and the EDA notebook, and summarised the
project state and open decisions. Identified that no code existed yet, the directory was not a
git repository, and the fold-count decision was still open.

> yeah sure, then push it as a new repo to my git

Approval to build the phase-0 skeleton and publish it.

> I think we want to start by choosing our frozen backbones, and extracting their outputs for
> our dataset.

Redirected the order of work: encoder choice and feature extraction first, rather than the
skeleton alone. Prompted a hardware check, which found no GPU — that finding reframed the
backbone choice around install friction and RAM rather than compute.

**Decision prompt put back to the user** (structure backbone): ESM-IF1 with ProteinMPNN fallback
/ ProteinMPNN only / SaProt / AntiFold. **Chosen: ESM-IF1 with ProteinMPNN as fallback**, on the
grounds that ESM-IF1 conditions on the whole multi-chain complex and is the field standard, with
its fragile dependency tree timeboxed against a fallback that exposes the same interface.

**Decision prompt** (sequence backbone): full ESM-2 capacity ladder / 650M only / 35M only /
ESM-C 300M. **Chosen: the full ladder, 8M→35M→150M→650M**, because caching all four costs about
one CPU-hour and turns an arbitrary size choice into a reportable ablation.

> we will have a gpu down the line.

Changed the code rather than the choice: every extractor takes `--device auto|cpu|cuda`, and
feature caches are keyed by content so a later GPU run is a no-op on anything already computed.

> once you get the data, try a first tree-boosted solution to get a feel on any issues.

Produced rung 0 and rung 1 on the frozen split. This surfaced three issues that changed the
harness:

1. A constant-per-fold predictor scores global Spearman −0.357, purely from between-fold label
   shift under grouped CV. Documented in `evaluate.py` as the empirical justification for the
   per-complex headline metric.
2. Sign accuracy on |ΔΔG| > 1 has a majority-class floor of 0.859, which rung 1 fails to beat.
   Added `sign_acc_big_base` and `sign_acc_big_balanced` so the floor is always reported next to
   the score.
3. Half the complexes (27 of 54) fall below the field's ≥10-mutations rule, so the headline
   metric is computed on a minority of complexes. Both variants are now reported.

### Notable corrections made during the session

- The EDA's `difflib`-based sequence identity was replaced with local Smith-Waterman (BLOSUM62,
  identity over the shorter sequence). This closed the open risk in `EDA_FINDINGS.md`: clusters
  went 26 → 17 and worst test–train antigen identity went 0.69 → 0.21, without needing MMseqs2.
- torch 2.14 failed to import with `WinError 1114` on `c10.dll`. Diagnosed as an MSVC runtime
  mismatch (machine has 14.28, torch ≥2.5 needs ≥14.40) and pinned torch to 2.4.1 rather than
  pushing a system-level installer. Recorded in `requirements.txt`.
- A failing test turned out to be a bug in the test's own helper (uneven complex sizes), not in
  the harness. Fixed the helper.

---

## Session, 2026-09-21 — GPU runs, and removing residual learning

The longest session of the project. Full technical state is in [HANDOFF.md](docs/HANDOFF.md); this
records how the work was directed and where the assistant was wrong.

### What was asked for, in order

Move training to a GPU; add reverse-mutation augmentation; run the AbRank-style pairwise loss;
deeper error analysis for both models; then a sequence of corrections that reshaped the results
— change the ≥10-mutation rule to 5, add a per-cluster metric, add a distance metric usable
below n=5, add an intermediate split, and finally **stop using residual methods entirely and
compare the network to the random forest on its own**.

### Corrections the assistant had to make to its own claims

These are recorded because the pattern matters more than any single number: on this dataset a
plausible mechanism attached to a small measurement is usually wrong.

- **"The net beats the forest, 0.506 vs 0.498."** False. The net scored 830 rows and the forest
  997. On matched rows the forest led, 0.518 to 0.506. Sweep rows now carry the forest's score
  on exactly the rows the net scored.
- **"Capacity reduction is the dominant lever."** False. `d=32` looked best as a single lever,
  but combining the regularisers at full width matched it (0.506 vs 0.502) — the width
  reduction contributed nothing.
- **"Validation peaks at step 40 then decays."** That was noise on a coarse trace. On the loss,
  at finer resolution, there is no decay in that window.
- **"Tough regularisation is the fix."** It suppressed memorisation (train ρ 0.912 → 0.654)
  without improving generalisation (validation ρ flat near 0.23 in every configuration).
- **"`no_scalars` is the best configuration."** A one-fold, one-seed probe artefact: 0.499 on
  the probe, 0.378 on the real sweep — *below* the control.
- **"ESM-2 650M costs 0.026."** Overstated: the paired bootstrap CI was [−0.065, +0.013] and
  does not clear zero. It is a tie, not a demonstrated harm.
- **"The intrinsic-tier reversal is explained by cluster size."** Not supported —
  corr(ρ, log cluster rows) = −0.148. The tiers differ in *composition* instead, and no
  mechanism is established.

### Where the user caught what the assistant did not

- Reading the live loss curves: *"it doesn't look like the overfit is after 1 epoch"* — an
  epoch had become 135–205 steps, so early stopping could not evaluate before step 146 and was
  checkpointing a memorised model. The fix (step-level evaluation) roughly doubled the score.
- *"Why does validation look like 0.2 correlation when we are talking about 0.5?"* — this is
  what exposed residual learning as the thing carrying every reported number.
- *"Correlation can be misleading"* — prompted adding MAE, median error and rmse/sd, which
  showed destabilising mutations scoring ρ +0.439 while being worse than a constant.
- *"Why do we need to extract? We should have everything."* — ~20 minutes of GPU time per run
  was being spent rebuilding feature files that already existed locally.

### Assistant errors in its own tooling

Worth recording separately, because they cost more time than the modelling did: gating on exit
codes from a CLI that always returns 0; `tail -f` silently doing nothing on `/mnt/c`; `grep`
block-buffering off a TTY; emitting bare `NaN` in JSON and verifying it with `curl`, which does
not parse the body; and grouping training curves by epoch when evaluation had moved to steps.

### Closing the session: censored affinities

Tom's final instruction was to mark the censored affinities as an issue to resolve, with the
current policy being to drop them. Implemented in `src/data.py` with a `--keep-censored`
escape hatch for reproducing earlier numbers.

The cost turned out to be much larger than the 6% of rows suggested, and it is worth recording
because the instinct "6% of rows, so a small effect" was wrong:

    997 rows, 54 complexes  ->  940 rows, 53 complexes
    forest, cluster split    0.388 -> 0.239
    forest, complex split    0.413 -> 0.354
    concordance              0.735 -> 0.668

Censored rows average +2.09 kcal/mol against +0.97 for the rest — they are large,
one-directional effects and therefore the *easiest* rows in the set to rank. Six percent of
the data was carrying 0.149 of the headline.

Two follow-on decisions, both deliberate:

- **The frozen split was regenerated rather than the test relaxed.** `test_splits.py` asserts
  folds.csv and dataset.parquet hold the same row_ids, and it failed — correctly, because the
  dataset had changed underneath a frozen split. Relaxing it to a subset check would have
  disarmed the guard; regenerating is the honest response to a deliberate data change.
  `row_id` is `#Pdb|mutations`, content-based, so cached features survived untouched.
- **Dropping is documented as a placeholder, not a fix.** It discards real evidence at exactly
  the strongly-destabilising end the model is worst at. Censored regression or a ranking
  constraint is the correct treatment.

---

## Session, 2026-09-21 (later) — reproducing ProtAttBA as an external baseline

Separate branch, `baseline/protattba-repro`, under `experiments/protattba_repro/`. Nothing
under `src/`, `configs/`, `data/` or the existing `Makefile` targets was modified.

### What was asked for

> Task: reproduce ProtAttBA as an isolated external baseline, on a new branch. Do not touch
> any existing project code (src/, configs/, data/, evaluate.py, etc.) — this is a
> validation exercise, not a new ladder rung.
>
> [...] Reproduce their S1131 result (S1131 = 1,131 single-point mutations from SKEMPI, Xiong
> et al. 2017 curation — the same underlying source as this project's own dataset), ESM2
> embedding variant, 10-fold cross-validation, from their Table 1: PCC 0.84 ± 0.05, Spearman
> ρ 0.75 ± 0.06, RMSE 1.31 ± 0.09 kcal/mol. [...]
>
> Try to run their code as-is, unmodified, on their shipped S1131 data first. This is the
> priority path — a faithful run of the original code is much more likely to land on their
> numbers than a reimplementation from the paper's text, since papers routinely omit details
> (init, exact preprocessing, minor architectural choices) that matter.
>
> Time-box debugging to reach the target numbers at 2 hours past having something training
> end-to-end. [...] A documented near-miss with a clear reason is a fine outcome; an
> open-ended tuning loop is not.

The full prompt also specified the fallback reimplementation spec, the deliverables, and the
instruction not to merge to main.

Two mid-session redirections:

> go through the repo at C:\Users\tomto\workspaces\converge_bind , start with handoff.md
> we already have the esm2 features extracted, connection to launch training in colab, error
> analysis and evaluation. copy those as well and perform the exact error analysis with the
> dashboard. then make sure you copy the results

This caught that the branch had been cut from `main` and was missing 30 commits of
infrastructure — the current evaluation harness, the error analysis, the dashboard, the Colab
runner, and the censored-affinity drop that moved the dataset from 997 rows to 940.

> the forest works as written. we want to check if the external repo logic can be run on this
> infrastructure and reach the same results

Refocused the work: the point is not to re-validate our own forest but to establish that an
external repository's logic runs on this infrastructure and lands on its published number.

### What it produced

- `experiments/protattba_repro/` — the reproduction: metrics, the published-number check, the
  frozen-encoder assumption tests, the embedding cache, the CV runner, the comparison, and a
  Colab runner following `scripts/colab_run.sh`'s conventions.
- The branch brought up to the infrastructure branch's file contents.

### Findings worth keeping

- **Their Table 1 is internally consistent and their split is recoverable.** The repo commits
  the full per-example output of one 10-fold run. Scoring it reproduces the published
  0.84/0.75/1.31 exactly, and `KFold(10, shuffle=True, random_state=3407)` reproduces their
  fold assignment for all 1131 rows. So the target needed no guessing, and the metric
  definition turned out to be a per-fold mean with a *population* std.
- **Model selection in their code runs on the test fold.** `WrapperDataset` assigns
  `test_idxes` to `self.val_dataset`, and `get_val_loader`/`get_test_loader` return the same
  object, so `EarlyStopping` and `ModelCheckpoint` monitor `val_pearson_corr` on the fold that
  is then reported. The published number is best-of-120-epochs on the test fold. Reproduced
  deliberately, with an `honest` protocol alongside it on identical folds.
- **Their attention mask uses `1e-10` instead of `-inf`.** Padded keys keep softmax mass, so a
  prediction depends on how wide its batch happened to be padded. This is why batch size 12
  was held fixed rather than reduced to fit the local GPU.
- **The ESM2 checkpoint the paper never names is 650M**, pinned independently by
  `MODEL_LOCATE="./model/esm2_650m"` and by `HIDDEN_SIZE=1280`, which is 650M's hidden size
  and no other ESM2 size's.
- **Running their file literally is out of reach here, for an arithmetic reason rather than a
  debugging one.** It calls ESM2-650M four times per example per step with the backbone
  frozen — about 4 CPU-hours per epoch and ~1700 hours for the 10-fold run. The encoder was
  cached instead, after measuring the three properties that make that an identity, and the
  cache took 159 minutes on this CPU against about 3 on a T4.
- **Colab has moved to python 3.13 and ProtAttBA's 2024 pins have no wheels for it.** Their
  `requirments.txt` cannot be installed there at all. Only what the S1131 path imports was
  installed, and the resulting library drift was cross-checked by re-embedding three sequences
  on the VM and comparing against the cache built under their own pins.
- **121 of the 123 run directories in `reports/` predate the censored-affinity drop.** Only
  `ov_both_geom` and `rf_complex` report n=940. HANDOFF flags this; the count makes it
  concrete.

### Assistant errors and corrections during the session

- Wrote a padding-invariance test that drew its four sequences from consecutive csv rows, which
  were all the same PDB and the same length — it padded 191 to 193 and proved nearly nothing.
  Rewritten to span 4.7x in length.
- Consumed `get_K_fold_with_test_generator` lazily inside the fold loop. Its validation split
  is drawn from the global numpy RNG, so fold k's split would have depended on how much
  randomness training folds 0..k-1 burned. Made eager.
- Assumed the 2 GB GTX 1050 would hold the run with gradient checkpointing. It does not; after
  the CUDA context and the Windows display, PyTorch gets about 900 MB and it OOMs at batch 12
  regardless.

---

## Session, 2026-09-23/24 — the perturbation fusion ladder, and what it is measuring

The longest session so far, and the one that changed what we believe. It set out to find a
fusion architecture that beats the forest and instead established that almost none of the
differences the ladder had been reading were larger than the noise.

> clip large effects at +-4, and the error analysis tab doesnt seem to respond to a change in
> run selection

> i cant seem to understand anything from this dashboard. transfer it to something like
> tensorboard or other opensource visualization

> i want to try a simpler model. pca to 128. layer norm, linear layer. delta seq -> film
> structure, mean site, concat modalities -> MLP

> we have to normalize the subtracted data and the non subtracted data separately

> can we cross attend with 32 per head, 2 heads, after the 128>64? what is the param count

> lets create one version where the modalities are reversed, with structure attending to
> sequence

> try one without the pca, just group each 128 to 16 in the net

> the no-pca is more important. what is the architecture you used to train antiberty data?

> check if The mutant embeddings are wrong. Off-by-one from insertion codes, a chain mix-up,
> or the substitution applied to the SEQRES sequence instead of the ATOM-derived one. [...]
> If ‖δ‖ isn't sharply peaked at the site, the alignment is broken and everything downstream
> is noise.

> we have to try the most promising models and add more intense regularization

> lets try also pca to 128 / then concat modalities with mlp head / with strong regularization

> run regularisation experiments only. you can use 2 concurrent on same session

> can we remove an mlp head layer?

### What it produced

- `src/perturb/model_simple.py` grown into four model families with a shared config:
  `PerturbSimple`, `PerturbMLP`, `PerturbTwoTower` and `PerturbSiteToken`, the last carrying
  cross-attention in both directions, FiLM, gated fusion, plain concatenation, and a
  block-diagonal `GroupedReduce` that replaces the PCA.
- `scripts/report_runs.py` — the results table, every run on one common truth, with the
  complex-mean floor printed underneath it.
- `src/perturb/check_alignment.py` — the four-part mutation alignment check.
- `experiments/protattba_repro/reconcile_partial.py` and fold-level resume in `bring_up.sh`,
  without which nothing survived a reclaimed session.
- `scripts/to_tensorboard.py` and `scripts/tb_writer.py` — a one-way mirror of `reports/` into
  TensorBoard, writing event protos directly because torch stopped importing on this machine.

### Findings worth keeping

- **Pooled Pearson mostly measures complex identity.** Predicting each complex's own mean
  scores **+0.672 pooled and +0.000 per complex**, beating every model here on the first and
  being useless for design on the second. Every table since reports per-complex first.
- **The seed spread is larger than almost every effect the ladder was reading.** Per-complex
  Pearson moved 0.075 between two seeds of the same configuration; within a single fold,
  pooled Pearson ranged 0.049 to 0.425 across three seeds. The attention family, FiLM, gated
  fusion versus the plain MLP, and PCA versus no PCA are all separated by less than that.
- **Cross-attention over ProteinMPNN is worth nothing.** Against `st64_noattn` -- the control
  with the attention deleted, which the ladder had been missing -- four of five variants score
  at or below it. Removing the PCA did not rescue it: three seeds of the reversed direction
  average +0.182 against +0.237 for the same model with no attention at all.
- **Heavier regularisation does not help.** Gated fusion on identical folds: +0.272 at the
  baseline settings against +0.235 with dropout 0.35, noise 0.25, feature-dropout 0.35 and
  weight decay 0.10. It costs accuracy and does not reduce the seed spread.
- **The networks ride homology; the forest does not.** Under the homology-cluster split the
  forest keeps 76% of its per-complex rho (0.361 -> 0.275) while our networks keep 34%
  (0.246 -> 0.084). The forest on that split had never been run before this session.
- **The embeddings are not misaligned.** Over all 940 rows and 1,726 mutated sites, zero
  residue-letter mismatches and no difference between the WT and MT strings outside the
  recorded sites; ‖δ‖ puts the mutated sites in the top k of the whole sequence on 97% of
  sides, at a median of 22x the median residue. The ~0.27 ceiling is the model and the task.
- **AntiBERTy loses to ESM-2 on the antibody side** by 0.046 on a same-session control, in a
  model where the antigen stays ESM-2 and one shared projection serves both.

### Assistant errors and corrections during the session

- **Claimed dropping the PCA was the session's biggest win, at +0.062 over its control and
  "three times the seed noise".** The seed replicate came back at +0.199 against the first
  seed's +0.274, so the gain is +0.025 against a 0.075 spread and is not established. The
  "most fold-stable model we have" claim went with it -- that was one seed.
- **Claimed heavy regularisation halved the seed spread**, from four folds. The fifth fold
  reversed it: 0.055 against 0.020. Two conclusions drawn from partial folds, both wrong, and
  after the second the rule became to report only complete runs.
- **Called cross-attention "ahead on every fold"** after reading pooled Pearson over four
  folds. Per-complex over five it was -0.012.
- **Said high gradient clipping "effectively caps the learning rate"**, which is the SGD
  intuition. Under AdamW a uniform rescale largely cancels in m/√v; the real effect is that
  the clipping is intermittent.
- **Asserted a FiLM "starved gradient" mechanism from input scale.** Measured on real data the
  ratio was 1.3x, which cannot explain a 40x difference. Retracted; the real asymmetry was
  zero-initialised gamma/beta against a default-initialised branch.
- **Reported "16 collectors running"** when that was 16 bash processes, several of them
  subshells of the same collector.
- Wrote the alignment check to judge peakedness on rank 0, which marks every multi-point row a
  failure by arithmetic -- one site of k can be rank 0 -- and printed "the alignment is
  suspect" on data that is clean. Fixed to judge on top-k.
- Killed the collector during a cleanup and launched four configurations with nothing
  collecting them; all four were lost to a reclaim.
