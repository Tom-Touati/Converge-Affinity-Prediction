# AI prompt history

The assignment's AI policy is "highly encouraged", and asks for the prompt history. This file is
appended in the same commit as the code a prompt produced, not reconstructed afterwards.

Tooling: Claude Code (Opus 5) in the Claude desktop app, working directly in this repository.

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

The longest session of the project. Full technical state is in [HANDOFF.md](HANDOFF.md); this
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
