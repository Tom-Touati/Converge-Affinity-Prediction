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
