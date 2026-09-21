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

<!-- RUN LOG -->
*Filled in as experiments complete. See `results/summary.md` for numbers and
`results/failures.md` for anything that raised.*
