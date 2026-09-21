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
