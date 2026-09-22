# Decisions — overnight fusion run

One line each, with the reason. Plan rule: when something is ambiguous, pick the simplest
defensible option, write it down, keep going.

## Protocol and splits

- **D1. Built the plan's 5-fold by-complex split as a new file and left `data/folds.csv`
  alone.** The repo's frozen split is 4-fold grouped by *homology cluster*, which is strictly
  harder: the project measured that 42 of 54 complexes gain a TM > 0.8 training twin under
  complex grouping against 0 of 54 under cluster grouping. Overwriting it would have destroyed
  the project's headline protocol. Both are reported and the summary says explicitly that the
  5-fold numbers are looser and not comparable.
- **D2. Shuffled complex order with seed 0 before `GroupKFold`.** `GroupKFold` is
  deterministic and ignores a seed, so the plan's "seed 0" would otherwise have been
  decorative. Shuffling first makes the seed meaningful and recorded in the json.
- **D3. Refuse to regenerate the frozen split without `--force`.** Rule 1 says never
  regenerate; a flag makes a deliberate rebuild possible and an accidental one impossible.

## Cross-dataset leakage (rule 2)

- **D4. Canonicalise mutations to `chain:wt:pos:mut` tuples rather than comparing raw
  strings.** My first implementation sorted the raw strings and reported **zero** duplicates
  across datasets, which was vacuous: this repo writes `DC101A` and AB645 writes `C:D101A` for
  the same 1DQJ mutation. After the fix, AB645 has 202 and AB1101 has 284 exact
  `(pdb, mutation)` duplicates of our rows. A leakage check that cannot match is worse than none.
- **D5. Split AB1101's separator-free multi-mutations on the field boundary, not with
  `findall`.** A position may carry a PDB insertion code (`100A`) that is indistinguishable
  from the mutant letter, so `findall` read `C:K138AC:D139A` as position `138A`, mutant `C`.
  Splitting immediately before each `<chain>:` removes the ambiguity. Verified: 0 unparsed
  rows across all four datasets.
- **D6. Use the PDB-numbered mutation column, not S1131's `mutation_clean`.** The two
  disagree (`A:C171A` against `CA182A`) because one is a sequence index; only the PDB-numbered
  form matches this repo's convention.
- **D7. S1131 is excluded from auxiliary training for the antibody task.** Classified by
  SKEMPI's own `Hold_out_type`, S1131 is 990 rows Pr/PI and **0 rows AB/AG**, with zero PDB
  overlap with our 53 complexes. The plan assumed it was "derived from SKEMPI" and therefore a
  leakage risk; it is neither a leakage risk nor useful antibody data. E6 uses AB645 and
  AB1101, which *are* antibody data and *do* overlap us heavily.

## Metrics

- **D8. Implemented the plan's 3-class bins separately from the repo's `CLASS_EDGES`.** The
  plan bins `|ddG|` (<0.5 / 0.5-2 / >2) and measures effect size; the repo bins signed `ddG`
  and measures direction. They are not comparable, so both exist under different names and
  `src/evaluate.py` is untouched.
- **D9. Report pooled metrics *and* the project's per-complex Spearman on every row.** Under
  grouped CV a constant-per-fold predictor scores pooled Pearson **-0.274** on this split
  (measured, `E0e_mean`), so pooled correlations carry a between-fold component that is not
  model skill. Reporting only one of the two would mislead.
- **D10. Added `E0e_mean`, a floor the plan did not ask for.** Without it the table has no
  zero, and on this data the floor is negative rather than zero.

## Baselines

- **D11. E0a re-implements the *invocation* of the existing baseline, not the model.** Rule 9
  forbids touching `src/train.py`, and that script uses its own fold file. `src/fusion` calls
  the same `src.model.MODELS` registry with the same feature blocks on the frozen split, so
  the estimator is literally the project's.
- **D12. E0b uses whole-sequence pooled ESM rather than per-chain pooling.** The cache stores
  a pooled vector per branch because `src/features/esm2.py` deliberately embeds only the
  mutated chain -- ESM-2 is single-chain and concatenating a complex invents a covalent bond.
  Per-chain pooling would need a fresh 650M extraction, which is a GPU job; it is folded into
  the per-residue extraction E1-E5 need anyway.
- **D13. Ran 3 seeds on the tree baselines too.** The plan only asks for 3 seeds on neural
  experiments, but forests are seeded and it is nearly free, so the table shows how much of
  any gap is seed noise. On E0a it is ±0.001 pooled Pearson.
- **D14. "params" for a forest is total node count.** There is no parameter count for a tree
  ensemble; node count is the closest honest analogue and is labelled as such.

## Remote execution

- **D15. Neural experiments run on Colab, not locally.** The local GTX 1050 has 2 GB and
  PyTorch gets ~900 MB of it after the CUDA context and the Windows display; the ProtAttBA
  head OOMs at their batch size of 12 even with the attention blocks recomputed in backward.
  Reducing the batch is **not** neutral, because their key mask is filled with `1e-10` instead
  of `-inf`, so padded keys keep softmax mass and a prediction depends on its batch's padding
  width.
- **D16. ProtAttBA's pinned requirements are not installed on Colab.** Their environment is
  python 3.10; Colab ships python 3.13 and their 2024 pins have no wheels for it, so
  `pip install -r requirments.txt` fails outright. Only the packages the S1131 code path
  imports are installed, and the resulting transformers 4.40 → 5.16 drift is cross-checked by
  re-embedding three sequences and comparing against the cache built under their own pins
  (agreement ~1e-10 on distribution moments).
- **D17. Work is chunked one fold per remote call.** `colab exec` lost its connection 35
  minutes into a 10-fold call and the session was later reclaimed twice, wiping `/content`. At
  ~25 s an epoch a protocol is 3-5 hours, longer than a session can be relied on, so fold
  predictions are written and pulled down individually and a rerun resumes from what is
  already on disk.
- **D18. Replaced `LitModel.load_from_checkpoint` with an explicit rebuild-and-load.** Under
  Lightning 2.6 with an argparse `Namespace` in hparams it never returned; an 8-epoch probe
  trained in 262 s then sat 18 minutes with the checkpoint already written. The replacement
  constructs the module and loads the saved `state_dict`, raising on any key mismatch.

## Scope calls

- **D19. Per-residue ESM and ProteinMPNN caches for *this* dataset do not exist and are a
  prerequisite for E1-E5.** Every cached block is pooled or scalar, one row per mutation. The
  plan listed these as already available. The four-sequence table they need is built and
  validated (940 rows, 884 distinct sequences, 335,261 residues); the extraction itself is a
  GPU job queued behind the S1131 reproduction.
- **D20. AB-bind is not in this repo** and no pipeline consumes it, so it is out of the
  leakage accounting. It exists only inside the gitignored upstream ProtAttBA checkout.

## Integrating main's modules (asked for explicitly)

- **D21. Bridged the two report formats instead of changing either.** `src/error_analysis.py`
  and the dashboard read `reports/<run>/predictions.csv`; the fusion ladder writes
  `results/predictions/<exp>_seed<n>.csv` because the plan specifies its own schema.
  `src/fusion/export.py` publishes one into the other, so both of main's tools work on a
  fusion run unmodified. The bridge validates itself: `E0a_rf_handcrafted__cluster`, scored
  through the project's own `src/evaluate.metrics`, gives per-complex Spearman **+0.239** —
  the project's documented figure to three decimals.
- **D22. `cluster` labels come from the project's split, `fold` from the run.** A homology
  cluster is a property of the complex, computed once with Smith-Waterman over antigen chains,
  and is independent of how folds were later drawn. So a run scored on the 5-fold by-complex
  split still gets true cluster labels and the dashboard's per-cluster panel stays meaningful,
  while the fold column keeps describing the split the model was actually scored on.
- **D23. Fixed a real bug in `src/error_analysis.py` rather than working around it.** Its
  `if __name__ == "__main__": main()` sat *above* `def figures`, so `main()` ran before
  `figures` was bound and every invocation ended in `NameError` — after printing all the
  tables, so the crash looked cosmetic while in fact no figure had ever been written. Moving
  the guard to the end of the file is the whole change; no logic was touched. This is the one
  exception to rule 9, and it was necessary to make the module usable at all.
- **D24. Made the split strategy a flag on the fusion ladder, not a fork of it.**
  `--grouping frozen5|cluster|complex|random` selects between the plan's 5-fold by-complex
  split and the project's own `src/splits.py` groupings, and the grouping is recorded on every
  results row. A fold number means nothing without the grouping that produced it. Runs on a
  non-default grouping are suffixed `__<grouping>` so the two can never be averaged together
  by accident.
- **D25. Per-residue ProteinMPNN reuses `mpnn_repr.encoder_h_V` verbatim.** That function
  already computes the `(L, 128)` field the fusion head needs and then reduces it to 384
  pooled numbers; `src/fusion/mpnn_per_residue.py` writes it out unreduced instead. No second
  implementation of the encoder exists to drift. Cached per **complex** rather than per
  mutation, because `h_V` is computed from backbone geometry with no sequence input, so it is
  identical for the wild type and the mutant — which is exactly what the plan's E1 assumes.
  Stored fp16 (18.2 MB for all 53 complexes, 159 s on CPU) and verified against
  `mpnnrep.parquet`: max difference 0.00097 over 80 single-point rows, i.e. fp16 precision, so
  the residue indexing agrees with the existing cache.

## Extraction, PCA, sampling, dashboard scoping (asked for 2026-09-22)

- **D26. Repaired two defects in ProtAttBA's released CSVs rather than dropping the rows.**
  S1131 stores the PDB id `1E96` as `1.00E+96` in 2 rows (Excel read it as scientific
  notation), and 87 AB645/AB1101 rows use `HM_1KTZ`-style ids for homology models, which a
  naive `split("_")[0]` turns into `HM`. Both are repaired in
  `src/fusion/benchmark_data.normalise_pdb_id`, taking structure coverage from 111/112,
  24/25 and 27/28 to complete.
- **D27. Fall back to H/L chain naming when `Partners` cannot be honoured.** Five
  AB645/AB1101 complexes are annotated with SKEMPI chain letters (`AB_E` for 1MLC) while the
  shipped AB-bind structure uses `E, H, L`. The fallback treats H and L as the antibody and
  the rest as the antigen, fires only when the annotation fails, and only when the structure
  actually has an H or L chain, so it cannot override a valid annotation. It recovered the
  last 10 of 173 benchmark complexes.
- **D28. PCA is refit inside every fold, on training rows only.** Fitting once on all rows
  would choose components from the test complexes' variance structure, which under a grouped
  split is exactly the homology the split withholds. Standardisation is fit the same way,
  because ESM dimensions differ in scale by more than an order of magnitude.
- **D29. PCA-128 does not rescue the pooled sequence features — it hurts.** 128 components
  capture 93.8% of variance, and on the cluster split pooled Pearson falls from 0.188 to
  0.094 (per-complex rho 0.186 to 0.091); on the by-complex split 0.303 to 0.255. So the E0
  gap is not the tree's feature sampling being diluted. The likely mechanism is that PCA
  maximises variance, and in a whole-sequence pooled vector the dominant variance is *which
  complex this is*, not what the single mutated residue did. That is an argument for
  per-residue features, not for a better projection.
- **D30. Sampling weight is a per-row odds ratio, and the resulting share is reported next to
  it.** Our dataset at 2x per row gives it 39.5% of draws over the full union and 41.5% after
  fold 0's leakage exclusion, because the share also depends on pool sizes. `describe`
  prints both so the configured ratio is never mistaken for the batch composition.
- **D31. Leakage exclusion drops rows rather than zero-weighting them.** A zero-weight row is
  still in the training table and can be counted, logged or used by anything that does not
  consult the sampler.
- **D32. The dashboard defaults to our dataset and says so on screen.** `/api/runs` now
  returns `{run, dataset}`, the selector lists only `skempi_abag` runs unless "other datasets"
  is ticked, and a badge reads either "our antibody-antigen SKEMPI" or a red warning naming
  the other dataset. ProtAttBA's benchmarks are a different problem -- S1131 has no
  antibody-antigen complexes at all -- so a number from one of them must never be read as ours
  just because it was on screen.

## E6b, the benchmark comparison

- **D33. ProtAttBA's csvs contain replicate rows with conflicting labels, and that broke my
  own join.** `(dataset, pdb, mutation)` is not unique: 16 rows over 8 ids, each a repeat
  measurement with a *different* ddG -- 1N8Z `B:Y105F` appears as -0.05 and 0.82, a 0.87
  kcal/mol spread. A `.loc[row_ids]` with duplicate labels silently returns more rows than it
  was given, so the geometry features shifted against their labels and the row count moved
  from 2465 to 2489. Caught by the count, not by any error. `row_id` now carries a positional
  suffix, and `run_benchmarks` asserts uniqueness and length before and after the join.
  Consequence worth stating separately: under their row-wise CV a replicate can sit in
  training while its twin is in test, which is direct leakage of a near-identical measurement.
- **D34. The misalignment was masking a real effect.** With it, adding interface geometry
  changed nothing (S1131 0.747 -> 0.740). Corrected, every number rose: S1131 0.770, AB1101
  0.717, AB645 0.458. The earlier "geometry adds nothing" reading was an artefact of my bug.
- **D35. A third of AB1101 is not antibody data.** 379 of 1100 rows (34%) and 81 of 645 AB645
  rows have no antibody heavy chain at all, because the complex is not an antibody-antigen
  pair: cyclophilin A with HIV-1 capsid (1AK4), beta-lactamase with BLIP (1JTG), TGF-beta with
  its receptor (1KTZ), CheY with CheA (1FFW), and 1T83 alone for 246 rows. Their `cat_seq`
  returns just the light chain for those, and ours matches, so nothing is corrupted -- but a
  benchmark named AB1101 being a third non-antibody changes how its number should be read.
- **D36. I raised a false alarm on "nan" in sequences and withdrew it.** A substring check
  flagged 188 rows; "NAN" is Asn-Ala-Asn and occurs in real sequences. Checking every
  character against the amino-acid alphabet gives zero non-residue characters and zero
  sequences equal to "nan" across all four columns.

## Perturbation model

- **D37. The head is bias-free, so a null edit gives exactly zero.** With no mutation both
  branches compute identical vectors and the difference is the zero vector; a bias would turn
  that into a learned constant, which is wrong on its face for a no-op edit. This is what
  makes required test (a) an equality rather than a tolerance.
- **D38. Masked attention uses -inf, not ProtAttBA's 1e-10.** Their mask lets padded keys keep
  softmax mass, so a prediction depends on its batch's padding width. That is reproduced
  faithfully in the reproduction and deliberately not copied here: every ablation would
  otherwise depend on batch composition.
- **D39. The distance-bin off-by-one was caught by a test, not by a failure.** Taking
  `linspace(0, 20, 17)[1:-1]` leaves 15 boundaries, caps the bin index at 15, and silently
  merges "beyond 20 A" into the 18.75-20 A bin -- leaving the bias no parameter for
  non-contacts at all, which is the one thing that bin exists for. Nothing raised.
- **D40. The alignment audit (test d) found zero mismatches over all 940 rows.** Every
  mutation's wild-type residue matches both the PDB residue list and the concatenated ESM
  input at the same index, so nothing had to be dropped.
- **D41. Training runs on Colab, not locally.** A CPU epoch did not finish in ten minutes; a
  T4 epoch takes 24 s. The same applied to ESM extraction: 3 min on the GPU against 2.2 hours
  locally. I initially ran the extraction on the CPU "because the GPU was busy", which was the
  wrong trade and was corrected.
- **D42. The VM is bootstrapped from scratch rather than kept warm.** Sessions were reclaimed
  three times, wiping /content each time. The bundle uploaded is 41 MB (resolved rows,
  ProteinMPNN cache, distance cache, model and scripts) and the 0.86 GB ESM cache is
  re-extracted on the GPU in under three minutes, which beats pushing it over the wire. Rows
  are shipped with their mutation sites already resolved, so the VM needs no PDB files and no
  structure parser.
- **D43. Results are polled off the VM while it trains.** The exec client times out long
  before a 5-fold run finishes, but the remote process survives it and `colab download` works
  against a BUSY kernel. Polling is therefore both the progress view and the thing that makes
  results survive a reclaimed session.

## Parameter-economy redesign (model_v2)

- **D44. The head stays bias-free, which is why the budget lands at 51,089 rather than the
  spec's 51,314.** The gaps are all bias terms I dropped: the head (required for a null edit
  to give exactly zero), the attention Q/K/V/O, and the low-rank ``down`` factor. The one
  addition is a second LayerNorm, because pre-LN cross-attention needs separate norms for the
  query and key/value streams. 15.9x fewer parameters than the first model.
- **D45. Heavy and light chains are told apart by length, and that is a heuristic.** The
  longer chain of the antibody pair is taken as heavy. A Kabat or Chothia annotation would be
  correct; none is available in this repo. Logged here rather than buried, because the
  chain-type embedding is only as good as this call.
- **D46. The crop keeps 13% of the tokens.** Median 85 tokens against 608 uncropped, with 91%
  of rows inside the spec's expected 40-120 band and a maximum of 138. Per-fold medians run
  80-92, so no fold is systematically cropping differently.
- **D47. 6% of rows have a disconnected crop** -- the mutation's neighbourhood does not touch
  the interface set. That is intended, not a bug: whether the site can still reach the
  interface through the distance-biased attention is exactly what the error analysis asks.
  It is also a confound to watch, since those rows have two separate regions rather than one.
- **D48. Both branches read the crop from one batch, and the test asserts it on the tensors.**
  Test (c) spies on what ``encode`` actually receives rather than checking the code path, so
  a future change that had MUT re-derive its own crop would fail rather than pass quietly.
