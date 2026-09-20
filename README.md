# Antibody–antigen ΔΔG prediction on SKEMPI 2.0

Predicting the change in antibody–antigen binding free energy on mutation, from sequence and
structure, on the antibody–antigen subset of SKEMPI 2.0.

> **Status: in progress.** The data pipeline, the frozen homology split, the evaluation harness
> and ten rungs are built and tested. Both structure arms — hand-computed interface geometry and
> frozen ProteinMPNN inverse folding — contribute signal that clears zero on a paired bootstrap;
> the sequence arm at ESM-2 35M does not, and the reason is diagnosed below rather than guessed.
> Still open: the ESM-2 650M capacity check, the learned fusion ladder specified in
> [`ARCHITECTURE.md`](ARCHITECTURE.md), and the error analysis. Every number below is regenerated
> by `src/evaluate.py`; none was typed by hand.

## Headline results so far

Grouped 4-fold cross-validation over homology clusters. Every number comes from
`src/evaluate.py` on the frozen split — no number in this table was typed by hand.

| Rung | Model | Per-complex ρ (n≥10) | 95% CI | Global ρ | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | train mean (floor) | n/a — constant | — | −0.357 † | 1.906 | 1.428 |
| 1 | substitution chemistry + GBT | 0.180 | [0.108, 0.255] | 0.207 | 1.874 | 1.368 |
| 2a | ESM-2 35M scalars + GBT | −0.066 | [−0.135, +0.011] | −0.056 | 1.994 | — |
| 2b | ESM-2 35M 480-dim + ridge | 0.074 | [−0.020, 0.175] | −0.026 | 2.195 | — |
| 3 | interface geometry + GBT | 0.245 | [0.107, 0.349] | 0.300 | 1.799 | 1.264 |
| 3b | ProteinMPNN inverse folding + GBT | 0.204 | [0.139, 0.269] | 0.238 | 1.915 | — |
| 4 | chemistry + geometry + GBT | 0.349 | [0.226, 0.452] | 0.436 | 1.663 | 1.156 |
| 5 | chemistry + geometry + ESM-2 + GBT | 0.333 | [0.236, 0.424] | 0.442 | 1.652 | — |
| 6 | chemistry + geometry + ProteinMPNN + GBT | 0.408 | [0.307, 0.494] | 0.439 | 1.682 | — |
| **N0** | **chemistry + geometry + ProteinMPNN + random forest** | **0.475** | **[0.393, 0.552]** | 0.377 | 1.691 | **1.194** |

**Fusion is justified, and here is the justification.** Paired bootstrap over complexes, 1,000
resamples, same folds:

| Comparison | Δ per-complex ρ | 95% CI | Clears zero |
| --- | --- | --- | --- |
| chem → geom | +0.063 | [−0.102, +0.210] | no |
| chem → chem+geom | **+0.167** | [+0.032, +0.284] | **yes** |
| geom → chem+geom | **+0.104** | [+0.039, +0.176] | **yes** |
| chem+geom → + ProteinMPNN | **+0.059** | [+0.005, +0.123] | **yes** |
| GBT → random forest, identical features | **+0.069** | [+0.002, +0.144] | **yes** |
| rung 4 → rung N0 (both changes) | **+0.128** | [+0.063, +0.202] | **yes** |

Combining the modalities beats *both* of them individually on unseen homology clusters. Neither
single modality is provably better than the other (row 1 spans zero), which is what makes the
gain real complementarity rather than one block carrying the other. ProteinMPNN then adds a
further +0.059 on top of hand-computed geometry, so the learned structure encoder is not
redundant with rSASA and contact counts.

**The random forest result is a diagnostic, not just a better number.** Boosting and bagging fail
in opposite directions: gradient boosting fits residuals sequentially, reducing bias and letting
it chase label noise, while a forest averages decorrelated trees, reducing variance and tolerating
noisy targets. The forest wins on identical features (+0.069) and is markedly steadier across
folds — 0.444 / 0.461 / 0.479 / 0.498 against the boosted model's wider spread. A
variance-reducing model beating a bias-reducing one is the signature of *label noise* as the
binding constraint rather than insufficient capacity, which is evidence against reaching for a
larger architecture. Note the lower bound of +0.002 is thin: this is "better, at the edge of what
27 counted complexes can resolve", not a comfortable win.

**The sequence modality currently contributes nothing.** ESM-2 35M scores at or below zero alone,
and adding it to chem+geom moves the headline metric 0.349 → 0.333. The reason is visible in the
features rather than guessed: mean masked-marginal log-likelihood ratio by interface location runs
INT −1.165, SUR −0.922, SUP −0.575, COR −0.449, RIM −0.350. That ordering is *backwards* for
binding — ESM-2 finds buried interior positions most surprising, which is correct for folding
stability, while interface core and rim, the positions that govern binding, surprise it least. For
antibodies the mechanism is plain: the binding residues sit in CDR loops that evolution
deliberately does not conserve, so sequence likelihood has little to say about them. This matches
de Kanter & Greiff's finding that ESM-IF1 reads protein quality rather than interaction
([`BACKBONE_COMPARISON.md`](BACKBONE_COMPARISON.md)). Whether this is a capacity limit or
something fundamental is the open question the 650M run answers.

† Not a typo, and not a model. A predictor that emits one constant *per fold* still scores
|global ρ| = 0.36, because each fold is excluded from its own training mean and the folds differ
sharply in mean ΔΔG (+1.89 to +0.34). Global correlations on a grouped split therefore carry a
between-fold component that has nothing to do with the model. This is the empirical reason the
headline metric is per-complex, and it is measured here rather than argued.

**Context for 0.475.** Published methods reach ~0.55 global Spearman on *full* SKEMPI with a
structure-level split. Ours is an antibody-only, homology-clustered split, which is strictly
harder — the 2025 AbAgym benchmark found the best of six published methods reached Spearman 0.28
on held-out antibody data, with relative solvent accessibility alone competitive, and CIR-DDG
measures zero-shot ESM-IF at 0.119 and ProteinMPNN at 0.172 on exactly this cohort. Measurement
noise caps per-complex Pearson near **0.91**. So 0.475 from substitution chemistry, hand-computed
interface geometry and a frozen inverse-folding encoder into a random forest sits at the top of
the 0.4–0.5 band the literature calls respectable here — and it is the number any learned fusion
model has to beat.

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

ProteinMPNN is vendored rather than pip-installable, and its weights are git-ignored, so the
structure arm needs one more step:

```bash
git clone https://github.com/dauparas/ProteinMPNN third_party/ProteinMPNN
```

`src/features/proteinmpnn.py` loads `vanilla_model_weights/v_48_020.pt` from there. Without it
every rung except 3b, 6 and N0 still runs.

Python 3.10. See `requirements.txt` for why torch is pinned to 2.4.1 — it is an MSVC-runtime
constraint on the development machine, not a code dependency, and can be lifted elsewhere.

## Running

`make data && make splits && make ladder`, or directly if `make` is unavailable (it is not
present on the development machine, so these are the commands actually used):

```bash
python -m src.data                       # 1,211 raw rows -> 997 modelling rows
python -m src.splits -k 4                # homology clusters -> frozen data/folds.csv
python -m src.features.geometry          # rSASA, burial, contacts   (~5.5 min, no model)
python -m src.features.esm2 --model esm2_t12_35M_UR50D --device auto   # (~16 min on CPU)
python -m src.features.proteinmpnn --device auto        # inverse-folding LLRs (needs third_party/)
python -m src.analysis labels            # the ddG distribution figure
python -m pytest tests -q                # split integrity + harness contract
```

Then the ladder. Each writes its own `reports/<name>/`:

```bash
python -m src.train --model mean --features chem                  --name rung0_mean
python -m src.train --model gbt  --features chem                  --name rung1_chem_gbt
python -m src.train --model gbt  --features esm2_t12_35M@scalars  --name rung2a_esm35M_scalars_gbt
python -m src.train --model ridge --features esm2_t12_35M         --name rung2b_esm35M_ridge
python -m src.train --model gbt  --features geom                  --name rung3_geom_gbt
python -m src.train --model gbt  --features mpnn                  --name rung3b_mpnn_gbt
python -m src.train --model gbt  --features chem,geom             --name rung4_chem_geom_gbt
python -m src.train --model gbt  --features chem,geom,esm2_t12_35M@scalars --name rung5_chem_geom_esm_gbt
python -m src.train --model gbt  --features chem,geom,mpnn        --name rung6_chem_geom_mpnn_gbt
python -m src.train --model rf   --features chem,geom,mpnn        --name rungN0_chem_geom_mpnn_rf
```

`make ladder` runs all ten in order; `make help` lists every target.

A `@scalars` suffix on a block keeps its summary columns and drops the wide embedding vector: a
480-dimensional difference is hopeless for a tree head on ~750 training rows but fine for ridge,
so the two heads read different views of one cache. `--center` trains on the within-complex
deviation instead of the raw label (tested; see PLAN.md).

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
| Sequence | ESM-2 35M (cached); 650M pending | **Measured at −0.066 alone and −0.016 in combination.** Kept in the repo as a negative result with a diagnosis, not as a component |
| Structure | **ProteinMPNN** (built, primary) | Chosen over ESM-IF1 on evidence: CIR-DDG measures zero-shot ProteinMPNN at Spearman 0.172 against ESM-IF's 0.119 on exactly this cohort, and it is pure PyTorch with no torch-geometric dependency tree. Adds +0.059 over geometry alone |
| Geometry | rSASA bound/unbound, burial, interface contacts | Strongest single block at 0.245, and the literature reports rSASA alone competitive with published deep models on held-out antibody data |

Deferred as drop-in swaps behind the cached-feature interface: AbLang2, ESM-IF1, SaProt. AntiFold
is deferred *with a reason* — AbBiBench measures it at −0.097 Spearman against its own ESM-IF1
base, and it cannot score the 37.6% of our rows whose mutation is on the antigen side.

**Fusion mechanism.** Feature-concatenation rungs are numbered 0–6; the *learned* fusion ladder
is numbered **N0–N5** to keep the two apart. Concatenation into a tree head (rung 6 / N0) is the
control that any attention model must beat, and that control now sits at **0.475**.

The target architecture is a single cross-attention block: the mutation is a query, the nearest
48 residues within 10 Å are keys and values, and attention logits carry a distance bias so the
module starts near "attend to what is close". Encoders stay frozen; roughly 14k parameters train.
Measured neighbourhood size is median 46 residues (p90 61), of which a median 34% are antigen-side
— which is how the antigen enters, via a learned chain-role embedding rather than a separate
encoder.

[`ARCHITECTURE.md`](ARCHITECTURE.md) carries the full specification: the pipeline from a SKEMPI
row to encoder tensors, the parameter budget that rules out bidirectional cross-attention at 750
training rows, and the three measurements that shape the design.

## Repo layout

```
src/
  paths.py       repo layout; the only module that knows where things live
  util.py        device resolution (--device auto|cpu|cuda), content hashing, timing
  structures.py  PDB parsing, Ig-motif chain-role assignment, mutation application
  data.py        raw SKEMPI -> measurements.parquet + dataset.parquet, every drop counted
  splits.py      homology clustering -> frozen folds.csv, plus the leakage report
  evaluate.py    THE harness: one predictions table -> every metric, with bootstrap CIs
  model.py       heads: mean, GBT, random forest, ridge, MLP
  train.py       one rung, one command, one reports/ directory
  analysis.py    error-analysis and data-description figures
  features/
    chem.py      substitution chemistry: BLOSUM, volume, charge, hydropathy deltas
    geometry.py  rSASA bound/unbound, burial by partner, interface contacts (Shrake-Rupley)
    esm2.py      frozen ESM-2: masked- and wt-marginal LLRs, embedding difference at the site
    proteinmpnn.py  frozen ProteinMPNN: LLR conditioned on the complex, on the chain alone,
                    and their difference -- the partner-attributable term
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
| `src.features.geometry` (Shrake-Rupley over 54 complexes) | ~5.5 min |
| `src.features.esm2` 35M (997 rows, masked + wt marginals) | ~16 min (0.95 s/row) |
| `src.features.esm2` 650M, projected from the 35M rate | ~4–6 h, not yet run |
| `src.train` one rung, incl. 400-sample bootstrap | ~90 s |
| `evaluate.paired_bootstrap`, 1,000 resamples | ~3 min per pair |
| `pytest tests` | ~9 s |

## Analysis and background

- [`notebooks/00_data_dictionary.ipynb`](notebooks/00_data_dictionary.ipynb) — column-by-column
  orientation to the raw file: meaning, literature use, value set and a plot for all 29 columns,
  plus the two ways of reading SKEMPI that fail silently
- [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) — where the model fails and why, with every
  slice guarded against the dataset's three imbalances
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — encoder pipeline, parameter budget, and what 750
  training rows can support
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
2. Rungs 2–6: sequence alone, structure alone, late fusion by concatenation, fusion + explicit
   interface features, then **cross-attention fusion** — the mutation queries the interface
   residues around it, with distance-biased attention over frozen encoders. Concatenation
   (rung 4) is the control it must beat, and two further controls (uniform attention,
   distance-bias only) separate "attention helped" from "we fed it interface geometry".
   Each kept only if a paired-bootstrap delta over the rung below has a CI clearing zero.
3. Error analysis: alanine vs non-alanine, antibody-side vs antigen-side, interface location,
   additivity on the double-mutant cycles, and ten hand-inspected worst residuals.
4. The learning curve, early — the literature's verdict is that data volume, not architecture, is
   the binding constraint at this sample size.
