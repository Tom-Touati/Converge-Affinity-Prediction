# Antibody–antigen ΔΔG prediction on SKEMPI 2.0

Predicting the change in antibody–antigen binding free energy on mutation, from sequence and
structure, on the antibody–antigen subset of SKEMPI 2.0.

> **Status: in progress.** The data pipeline, the frozen homology split, the evaluation harness
> and the first two rungs of the model ladder are built and tested. The sequence and structure
> encoders are chosen and their extractors are next. Numbers below are real and reproducible;
> the multimodal model is not built yet.

## Headline results so far

Grouped 4-fold cross-validation over homology clusters. Every number comes from
`src/evaluate.py` on the frozen split — no number in this table was typed by hand.

| Rung | Model | Per-complex ρ (n≥10) | 95% CI | Global ρ | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | train mean (floor) | n/a — constant | — | −0.357 † | 1.906 | 1.428 |
| 1 | substitution chemistry + GBT | **0.180** | [0.108, 0.255] | 0.207 | 1.874 | 1.368 |

† Not a typo, and not a model. A predictor that emits one constant *per fold* still scores
|global ρ| = 0.36, because each fold is excluded from its own training mean and the folds differ
sharply in mean ΔΔG (+1.89 to +0.34). Global correlations on a grouped split therefore carry a
between-fold component that has nothing to do with the model. This is the empirical reason the
headline metric is per-complex, and it is measured here rather than argued.

**Context for 0.180.** Published methods reach ~0.55 global Spearman on *full* SKEMPI with a
structure-level split. Ours is an antibody-only, homology-clustered split, which is strictly
harder — the 2025 AbAgym benchmark found the best of six published methods reached Spearman 0.28
on held-out antibody data, with relative solvent accessibility alone competitive. Measurement
noise caps per-complex Pearson near **0.91**. So 0.180 from pure substitution chemistry with no
protein context is a sane floor to climb from, not a failure.

## Getting the data

Two files go in `data/`, both git-ignored:

- `skempi_v2.csv` — from <https://life.bsc.es/pid/skempi2/>
- `SKEMPI2_PDBs.tgz` — the cleaned structures from the same page (unpacked automatically on
  first run)

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

Python 3.10. See `requirements.txt` for why torch is pinned to 2.4.1 — it is an MSVC-runtime
constraint on the development machine, not a code dependency, and can be lifted elsewhere.

## Running

`make data && make splits && make ladder`, or directly if `make` is unavailable (it is not
present on the development machine, so these are the commands actually used):

```bash
python -m src.data                          # 1,211 raw rows -> 997 modelling rows
python -m src.splits -k 4                   # homology clusters -> frozen data/folds.csv
python -m src.train --model mean --features chem --name rung0_mean
python -m src.train --model gbt  --features chem --name rung1_chem_gbt
python -m pytest tests -q                   # split integrity + harness contract
```

Each run writes `reports/<name>/` containing `predictions.csv`, `per_complex.csv`,
`metrics.json`, and a `run.json` recording the config, git SHA, seed and wall-clock time.

## Key decisions, and why

**The honest sample size is 997, not 1,211.** Eighty rows record the mutant affinity as literal
`n.b.` — a qualitative non-binder with no number to parse — and a further 256 rows are repeat
measurements of 122 (complex, mutation) pairs, aggregated by median. The spread within those
repeats is retained: it is the subset's own label-noise estimate. Bounded affinities (`<` / `>`)
are a *separate* and currently unhandled case: SKEMPI parses them into bare numbers, so 86 of
them survive into the table as if they were exact. See Limitations.

**The split is by homology cluster, not by structure.** The field's convention (RDE-Network,
DiffAffinity) splits by structure. That is not enough here — thirteen of the 54 complexes are
different antibodies against hen egg-white lysozyme, a quarter of all rows. Clustering by
SKEMPI's own hold-out annotation, local Smith-Waterman identity (antigens at 30%, antibodies at
90% since shared framework puts unrelated antibodies near 70–80%), and the curator's explicit
"very similar" notes collapses 54 complexes into **17 clusters**. Worst test–train antigen
identity across the split boundary is **0.21**. Our numbers will be lower than published ones,
and that is the evaluation being stricter, not the model being worse.

*Fold 0 is a single cluster* — the 14 lysozyme binders, mean ΔΔG +1.89 against +0.34–1.22
elsewhere. It has no internal cluster diversity and a shifted label distribution, so per-fold
results are always reported alongside the average.

**Chain roles are read from structure, never from the `Protein 1` / `Protein 2` columns.** Those
columns contradict the chain groups in `#Pdb` for `2BDN_HL_A`, where Protein 1 is the antigen yet
maps to chains H and L. Roles come from conserved immunoglobulin framework motifs scored directly
off the sequence. `1DVF_AB_CD` correctly stays unresolved — it is the anti-idiotype pair D1.3
against E5.2, antibody bound to antibody, with no antigen at all; its 38 rows are flagged as
their own category.

**The headline metric is mean per-complex Spearman**, over complexes with ≥10 mutations (the
field's rule, for comparability). It is what matters for the real decision — *which mutation
hurts binding least, for this target* — and it is not gamed by the three complexes holding 29% of
the rows. Confidence intervals bootstrap over **complexes**, not rows; a per-row bootstrap would
look tighter and would lie.

**Two metric floors are reported next to the metrics they undermine.** 86% of rows with |ΔΔG| > 1
are destabilising, so "always predict weaker binding" scores 0.859 sign accuracy. Rung 1 scores
0.841 — *below* that floor — and 0.586 when balanced across direction. The model is close to
blind to stabilising mutations, which the raw number would have hidden.

**Encoders stay frozen; only small heads train.** 997 rows cannot fine-tune a protein language
model without overfitting it. Gradient boosting uses a fixed, pre-chosen configuration with
early stopping deliberately *off*: sklearn's built-in early stopping carves out a random
validation split, which would put near-identical complexes on both sides of it and reintroduce
exactly the homology the outer split exists to prevent.

## Chosen encoders

| Modality | Model | Why |
| --- | --- | --- |
| Sequence | ESM-2, capacity ladder 8M → 35M → 150M → 650M | All four cached; the capacity ablation ("does a bigger PLM help on 997 rows, or does it saturate?") is itself a result, and costs about one CPU-hour once |
| Structure | ESM-IF1, with ProteinMPNN as fallback | ESM-IF1 conditions on the whole multi-chain complex, so it can see the interface. Its install (torch-geometric/scatter/sparse) is timeboxed; ProteinMPNN exposes the same per-position log-likelihood interface with no fragile dependencies |
| Geometry | rSASA, interface contacts, Δ contacts | Built regardless — reported as competitive with published deep models on held-out antibody data |

Deferred as drop-in swaps behind the cached-feature interface: AbLang2, AntiFold, SaProt.

## Repo layout

```
src/
  paths.py       repo layout; the only module that knows where things live
  util.py        device resolution (--device auto|cpu|cuda), content hashing, timing
  structures.py  PDB parsing, Ig-motif chain-role assignment, mutation application
  data.py        raw SKEMPI -> measurements.parquet + dataset.parquet, every drop counted
  splits.py      homology clustering -> frozen folds.csv, plus the leakage report
  evaluate.py    THE harness: one predictions table -> every metric, with bootstrap CIs
  model.py       heads: mean, GBT, ridge, MLP
  train.py       one rung, one command, one reports/ directory
  analysis.py    error-analysis and data-description figures
  features/chem.py   rung 1: substitution chemistry only
tests/           split integrity and the harness contract
notebooks/       00_data_dictionary.ipynb  (generated by _build_00_data_dictionary.py)
                 01_eda.ipynb
data/folds.csv   the frozen split, committed on purpose
```

`data/folds.csv` is committed and nothing regenerates it automatically. `make splits` exists to
rebuild it deliberately; it is not a dependency of any model target, so no run can silently
re-split the data underneath itself.

## Hardware and runtime

Development machine: **Intel Core i7-8650U, 4 cores @ 1.9 GHz, 8 GB RAM, no GPU**, Windows 10.
All scripts take `--device auto|cpu|cuda` and are cached by content, so a later GPU run is a
no-op on anything already computed.

| Step | Wall clock |
| --- | --- |
| `src.data` (parse, verify 2,109 positions, dedup) | ~65 s |
| `src.splits` (1,431 pairwise local alignments ×2 sides + leakage report) | ~28 s |
| `src.train` rung 1, incl. 400-sample bootstrap | ~84 s |
| `pytest tests` | ~9 s |

## Analysis and background

- [`notebooks/00_data_dictionary.ipynb`](notebooks/00_data_dictionary.ipynb) — column-by-column
  orientation to the raw file: meaning, literature use, value set and a plot for all 29 columns,
  plus the two ways of reading SKEMPI that fail silently
- [`PLAN.md`](PLAN.md) — time budget, model ladder, bottleneck map, deferred ideas with triggers
- [`EDA_FINDINGS.md`](EDA_FINDINGS.md) — what the data audit found and what it changed
- [`PRIOR_WORK.md`](PRIOR_WORK.md) — literature calibration: what "good" looks like here
- [`AI_PROMPTS.md`](AI_PROMPTS.md) — prompt history

## Limitations and known issues

Open defects, quantified, with the fix each one needs. Detail and decisions in
[`PLAN.md`](PLAN.md).

- **86 censored affinities are trained on as exact measurements** (45 mutant, 41 wild-type).
  SKEMPI parses `">1e-6"` into a bare number and `src/data.py` currently trusts it. That is 8.6%
  of the dataset, and it should be fixed before the encoders are trained against these labels.
- **80 qualitative non-binders (`n.b.`) are discarded**, losing the strongest destabilising
  evidence in the set. Recoverable as ranking constraints rather than point labels.
- **37 rows are exactly 0.000, and 22 of them sit in one complex** (`2JEL_LH_P`) -- a reporting
  convention rather than 37 independent measurements of zero.
- **Only 130 rows are clearly stabilising** (ddG < -0.5). Affinity *improvement* is what
  antibody engineering wants predicted, and it is the class with the least evidence. Rung 1
  already scores 0.586 balanced sign accuracy, which is near chance on direction.
- **Half the complexes fall below the field's >=10-mutations rule**, so the headline metric is
  computed on 27 of 54 complexes. Both variants are reported.

## What is next

1. Extract and cache the ESM-2 ladder and the inverse-folding log-likelihood ratios, plus a
   with/without-partner-chain ablation that isolates whether the structure model actually uses
   the interface.
2. Rungs 2–5: sequence alone, structure alone, late fusion, fusion + explicit interface features.
   Each kept only if a paired-bootstrap delta over the rung below has a CI clearing zero.
3. Error analysis: alanine vs non-alanine, antibody-side vs antigen-side, interface location,
   additivity on the double-mutant cycles, and ten hand-inspected worst residuals.
4. The learning curve, early — the literature's verdict is that data volume, not architecture, is
   the binding constraint at this sample size.
