PYTHON ?= python
K      ?= 4
BOOT   ?= 1000

.PHONY: help data splits rung0 rung1 ladder report test clean clean-features

help:
	@echo "make data      - clean SKEMPI into data/processed/*.parquet"
	@echo "make splits    - build homology clusters and freeze data/folds.csv (K=$(K))"
	@echo "make rung0     - floor: predict the training mean"
	@echo "make rung1     - substitution chemistry + gradient-boosted trees"
	@echo "make ladder    - every rung built so far, in order"
	@echo "make test      - split-integrity and harness tests"
	@echo "make clean     - remove generated reports (keeps the frozen split)"

data:
	$(PYTHON) -m src.data

# The split is a committed artefact. This target regenerates it deliberately; it is not a
# dependency of anything, so no model run can silently re-split the data underneath itself.
splits:
	$(PYTHON) -m src.splits -k $(K)

rung0:
	$(PYTHON) -m src.train --model mean --features chem --name rung0_mean --n-boot $(BOOT)

rung1:
	$(PYTHON) -m src.train --model gbt --features chem --name rung1_chem_gbt --n-boot $(BOOT)

ladder: rung0 rung1

test:
	$(PYTHON) -m pytest tests -q

clean:
	rm -rf reports/*/

clean-features:
	rm -rf data/features/*.parquet
