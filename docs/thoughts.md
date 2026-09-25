# Raw working notes — first pass, kept deliberately

> **Unedited notes from the start of the project, before anything was built.** They are kept
> because most of what is in them was eventually done, and the record of what was guessed up
> front versus what the measurements actually said is worth more than a tidy summary. Where
> each idea landed:
>
> | note | outcome |
> | --- | --- |
> | "5 fold based on homology clusters, very tough standard" | Built. `data/cluster_folds.csv`, ERROR_ANALYSIS §22. The networks lose ~26–38% there; it is the honest protocol. |
> | "generalization check with other dataset" | Planned and specified: AbSci HER-2, FUTURE_WORK §16, profiled in ABSCI_DATASET.md. Not yet run. |
> | "proteinmpnn for structure / esm for sequence" | Both used, frozen. ESM-IF1 added later as an alternative encoder and ties ProteinMPNN. |
> | "test-retest error ... distribution of differences between pairs" | Measured. Label noise 0.240 kcal/mol, implying a ceiling of 0.979. ERROR_ANALYSIS §23. |
> | "difficulty based on same-complex samples, and similar points in train set" | Measured. 75% of test rows have a near-identical training twin; only 19.8% are genuinely hard. ERROR_ANALYSIS §20–21. |
> | "unbalanced labels ... multi-point vs single ... antibody vs antigen side" | All three became Part III of the error analysis. |
> | "classification into bins over regression because it neutralizes measurement error" | Tried (ordinal head). Did not beat regression; the bins are reported alongside instead. |
> | "cross attention between ag to ab" | Built and measured many times. It is the **worst** fusion mechanism tested — bottom two of eleven in ARCHITECTURES Family E. |
> | "rotating the 3d structure" as augmentation | Not done. Both encoders are rotation-invariant by construction, so it would be a no-op. |
> | "a model that infers structure after mutation, use as feature" | Not done. Still the single clearest gap — FUTURE_WORK §9 (FoldX). |
> | "I still need to understand every column" | EDA_FINDINGS.md. |

---

we also have an imbalance in replaced amino acids. what can we do about that?
in the reasoning doc we need to write down that we choosing classification into bins over regression because it neutralizes the measurement error. we want to test the test-retest error also in the eda. (distribution of differences between pairs of the same unique complex mutation pair). another augmentation can be rotating the 3d structure? also use a model that infers structure after mutation? and then use it as feature.
we still need a base model comparison, clear literature, feature output from each model to be stored. how many of the proteins and antigens did the model actually previously see?
I still need to understand every column.
we have a problem with under representation.

encode the ab binding site sequence and the ag too. then run cross attention between ag to ab, for both mutated and not.
maybe we can encode before and after

evaluation:
difficulty based on same-complex samples, and similar points in train set
underrepresented labels
train vs validation then test loss

methodology:
5 fold based on homology clusters, very tough standard
generalization check with other dataset.

more data:

models:
proteinmnn for structure based on literature.
esm for sequence
fold x to reverse measurements.
issues:
unbalanced labels (stabalizing vs neutral vs distabalizing)
extreme values in positive ddg (destabalizing)
multi-point vs single point
antibody side mutations vs antigen side mutations