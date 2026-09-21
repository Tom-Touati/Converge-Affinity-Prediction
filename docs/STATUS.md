# STATUS — repo state before the overnight fusion run

Written at the start of the run, from inspection rather than from the plan's description of
the repo. Four of the plan's stated preconditions do not hold; they are called out below
because they change what E1–E5 need.

Updated at the end with what ran. See `docs/decisions.md` for every judgment call.

---

## 1. Repo layout

```
src/
  data.py          SKEMPI 2.0 -> data/processed/dataset.parquet (AB/AG subset only)
  splits.py        homology clusters -> data/folds.csv (frozen, committed)
  evaluate.py      THE metric harness; per-complex Spearman is the headline
  error_analysis.py  slice tables guarded against the dataset's imbalances
  train.py         one entry point for the tree/linear ladder
  model.py         MODELS registry (mean, ridge, gbt, rf, ...)
  structures.py    PDB parse, role assignment, apply_mutations
  tmscore.py       TM-align tiering, cached at data/tm_matrix.csv
  features/        chem, geometry, geom_rev, esm2, hf_plm, antibody_plm,
                   proteinmpnn, mpnn_repr
  fusion_net.py fusion_v2.py fusion_v3.py rank_fusion.py   existing neural work
scripts/
  colab_run.sh colab_job.py     the GPU runner (WSL + Colab CLI)
  dashboard/                    live view; serve.py polls a running VM
experiments/protattba_repro/    the ProtAttBA reproduction (this branch's other half)
data/
  processed/dataset.parquet     940 rows x 27 cols, the modelling table
  folds.csv folds_complex.csv   frozen splits (cluster / complex grouped)
  features/*.parquet            11 cached feature blocks, keyed by row_id
  PDBs/                         690 pdb files
reports/                        123 run directories (see the warning in §6)
```

## 2. The benchmark dataset

`data/processed/dataset.parquet`, built by `src/data.py` from `data/skempi_v2.csv` by
selecting `Hold_out_type` containing `AB/AG`:

| | |
|---|---|
| rows | 940 |
| complexes (`#Pdb`) | 53 |
| distinct pdb ids | 53 |
| columns | 27, no sequence columns — sequences are read from `data/PDBs/` |
| label | `ddG` in kcal/mol, from the two affinities and T |
| censored affinities | dropped (72 measurements, 57 rows, one whole complex) |

Key columns for fusion work: `row_id` (`#Pdb|mutations`, content-based), `#Pdb`, `pdb`,
`ab_chains`, `ag_chains`, `mutations`, `n_mut`, `mut_side`, `is_alanine`, `location`, `ddG`.

`ab_chains` and `ag_chains` are **empty strings for 1DVF** (38 rows), an anti-idiotope
antibody–antibody pair where `assign_roles` correctly returns a tie. Every existing consumer
handles it with `r["ag_chains"] or r["side2"]`; new code must too.

## 3. Cached features — what exists, and what the plan assumed

All under `data/features/`, indexed by `row_id`, **997 rows** (they predate the censored-row
drop to 940; `row_id` is content-based so the 940 current rows are a subset).

| file | cols | what it is |
|---|---|---|
| `chem.parquet` | 21 | substitution chemistry, BLOSUM, volume/charge deltas |
| `geom.parquet` | 7 | rSASA, burial, contact counts |
| `geomrev.parquet` | 16 | reversible geometry |
| `mpnn.parquet` | 5 | ProteinMPNN log-likelihood ratios |
| `mpnnrep.parquet` | 384 | **pooled** ProteinMPNN representation |
| `esm2_t12_35M.parquet` | 485 | ESM-2 35M scalars + pooled difference |
| `esm35M_pair.parquet` | 1453 | pair view, 35M |
| `esm150M.parquet` | 1933 | 150M |
| `currab150M.parquet` | 653 | antibody-specific PLM, 150M |
| `abplm.parquet` | 526 | antibody PLM scalars |
| `esm650M_pair.parquet` | 3853 | ESM-2 650M pair view, pooled |

**Precondition mismatch 1 — there are no per-residue caches for this dataset.** Every block
above is pooled or scalar, one row per mutation. The plan's E1–E5 all require per-residue
`[L, d]` tensors for ESM and ProteinMPNN. The only per-residue cache in the repo is
`experiments/protattba_repro/cache/esm2_650m_embeddings.npy`, and that is for **S1131**, not
for this dataset.

**Precondition mismatch 2 — S1131 is not an antibody dataset and shares no complexes with
this one.** Classified by SKEMPI's own `Hold_out_type`: 990 rows Pr/PI, the rest unannotated,
**0 rows AB/AG**, and zero PDB overlap with our 53 complexes. It cannot be used as auxiliary
antibody training data (plan rule 2 / E6). AB645 and AB1101 *are* antibody data, and overlap
us heavily: 19 of 25 and 20 of 28 of their complexes are among our 53.

**Precondition mismatch 3 — the ProtAttBA reproduction is sequence-only on S1131, not on this
dataset.** It runs on a Colab T4 (the local 2 GB GTX 1050 cannot hold it at their batch size).
Running it on *our* rows needs the four-sequence table
(`experiments/protattba_repro/cache/project_sequences.parquet`, built and validated: 940 rows,
884 distinct sequences, 335,261 residues) plus a per-residue extraction that has not been run.

**Precondition mismatch 4 — AB-bind is not in this repo.** `source_data/AB-bind/` exists in the
upstream ProtAttBA checkout only, and no pipeline here consumes it.

## 4. How the existing baselines are invoked

Tree / linear ladder, one entry point:

```bash
python -m src.train --model rf --features chem,geom,geomrev,mpnn \
       --name <run> --grouping cluster --n-boot 1000
```

`--features` is a comma-separated list of blocks resolved by `src/train._feature_block`;
a `@scalars` suffix drops the wide embedding columns. Writes
`reports/<run>/{predictions.csv,metrics.json,per_complex.csv,run.json}`.
`predictions.csv` columns: `row_id, complex, cluster, fold, y_true, y_pred`.

Metrics come from `src/evaluate.metrics`, which requires `row_id, complex, y_true, y_pred`.
Headline is **mean per-complex Spearman with `MIN_GROUP=5`**; confidence intervals bootstrap
over complexes, not rows.

Error analysis: `python -m src.error_analysis --run <run>`.

## 5. Frozen splits already in the repo

- `data/folds.csv` — 4 folds, **grouped by homology cluster** (Smith-Waterman on antigen
  chains, committed artefact). This is the project's headline protocol and is stricter than
  grouping by complex: under it 0 of 54 complexes gain a TM > 0.8 training twin, against 42 of
  54 under complex grouping.
- `data/folds_complex.csv` — 4 folds grouped by complex, secondary.

The plan's rule 1 asks for a new 5-fold by-complex split. It is created at
`data/splits/skempi_abag_5fold_by_complex.json` and the two files above are **not touched**.
Because it is by-complex rather than by-cluster it is the *looser* protocol, so numbers on it
are not comparable to the project's headline figures. Both are reported.

## 6. Warning about `reports/`

Of the 123 run directories, **only two** (`ov_both_geom`, `rf_complex`) were produced on the
current 940-row dataset. The other 121 report `n=997` and predate the censored-affinity drop,
so no net-vs-forest comparison can be read off them. Re-verified at the start of this run: the
forest reproduces bit-for-bit (per-complex Spearman +0.239, max |Δy_pred| 1.3e-15 against
`ov_both_geom`), and `pytest tests -q` passes 12 tests.

## 7. Hardware

| | |
|---|---|
| local CPU | Intel i7-8650U, 4 cores, 7.9 GB RAM |
| local GPU | GTX 1050, 2 GB, driver 461.40 / CUDA 11.2 |
| local GPU verdict | runs cu118 via minor-version compatibility, but PyTorch gets ~900 MB after the CUDA context and display; OOMs on the ProtAttBA head at batch 12 even with attention recomputed in backward |
| remote | Colab T4 16 GB via `scripts/colab_run.sh` conventions |
| measured | ESM2-650M per-residue extraction: 1.1 min on T4 vs 159 min on this CPU |

Neural experiments therefore have to run on Colab. Tree baselines run locally.

---

## 8. What ran tonight

**Completed.**

| step | outcome |
|---|---|
| frozen 5-fold split | `data/splits/skempi_abag_5fold_by_complex.json`, 940 rows, 53 complexes, 188 test rows per fold |
| cross-dataset dedup + leakage | AB645 202 and AB1101 284 exact `(pdb, mutation)` duplicates; per-fold usable counts in the json |
| `E0a_rf_handcrafted` | pooled Pearson 0.489 ± 0.001, per-complex Spearman 0.418 ± 0.020 |
| `E0b_rf_pooled_esm` | 0.252 ± 0.015 / 0.273 ± 0.011 |
| `E0c_rf_pooled_esm_mpnn` | 0.303 ± 0.015 / 0.366 ± 0.022 |
| `E0e_mean` (floor) | pooled Pearson −0.274; a constant is not a zero baseline here |
| error analysis | `results/error_analysis.md` for E0a, `results/error_analysis_E0c.md` for E0c, 6 plots |
| summary | `results/summary.md`, regenerable with `python scripts/summarize.py` |
| entry point | `python -m src.fusion.run --config configs/<exp>.yaml` |

Headline: **the plan's "multimodal baseline everything else must beat" (E0c) is beaten by 49
handcrafted columns, by 0.19 pooled Pearson.** Pooled ProteinMPNN does add signal over pooled
ESM alone (0.252 → 0.303), so the structure channel is not empty; it is the pooling that
costs. This reproduces on the frozen split what `HANDOFF.md` §6 recorded as dilution.

**Did not run: E1–E8, and E0d.** All of them need per-residue ESM and ProteinMPNN tensors for
these 940 rows, and no such cache exists — every block under `data/features/` is pooled or
scalar (§3, precondition mismatch 1). `src/fusion/run.py` raises with that explanation rather
than substituting a stand-in representation. The prerequisite table is built and validated;
see `docs/next_steps.md` item 1, which is ~3 min of T4 time.

**Also did not run: E6 multi-dataset training as specified.** Its auxiliary pool was to be
SKEMPI ∪ AB645 ∪ S1131 ∪ AB1101, but S1131 contains no antibody-antigen complexes at all and
shares no PDB with this dataset, so it is not usable auxiliary antibody data (§3, mismatch 2).
AB645 and AB1101 are, and the leakage exclusion for them is computed and stored.

**In flight on Colab: the ProtAttBA S1131 reproduction.** 3 of 10 folds complete, tracking the
paper closely (PCC 0.7826 / 0.8629 / 0.8138 against their 0.7673 / 0.8638 / 0.7780). Two
Colab sessions were reclaimed mid-run, so the driver now works one fold per remote call and
resumes from whatever is on local disk.

**Wall time and hardware.** Everything in the table above ran on the local CPU (i7-8650U, 4
cores) in well under the plan's 45-minute-per-experiment budget: the three forests take about
0.5 min per seed. The neural work is on a Colab T4 because the local GTX 1050's 2 GB cannot
hold the ProtAttBA head at their batch size.

**Failures file.** `results/failures.md` was not created, because nothing raised.
