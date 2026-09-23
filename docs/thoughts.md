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