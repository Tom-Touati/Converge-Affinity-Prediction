PYTHON ?= python
K      ?= 4
BOOT   ?= 1000
DEVICE ?= auto
ESM    ?= esm2_t12_35M_UR50D

.PHONY: help data splits features geom esm mpnn errors \
        rung0 rung1 rung2a rung2b rung3 rung3b rung4 rung5 rung6 rungN0 \
        ladder figures test clean clean-features third-party

help:
	@echo "Setup"
	@echo "  make data          - clean SKEMPI into data/processed/*.parquet"
	@echo "  make splits        - homology clusters -> frozen data/folds.csv (K=$(K))"
	@echo "  make third-party   - how to fetch the ProteinMPNN checkout (prints instructions)"
	@echo ""
	@echo "Feature extraction (cached by row_id; each is paid once)"
	@echo "  make geom          - rSASA, burial, contacts          (~5.5 min, no model)"
	@echo "  make esm           - ESM-2 embeddings, ESM=$(ESM)      (~16 min CPU for 35M)"
	@echo "  make mpnn          - ProteinMPNN inverse-folding LLRs  (needs third_party/)"
	@echo "  make features      - all three of the above"
	@echo ""
	@echo "Ladder (each writes its own reports/<name>/)"
	@echo "  make rung0  - floor: predict the training mean"
	@echo "  make rung1  - substitution chemistry + GBT"
	@echo "  make rung2a - ESM-2 scalars + GBT        rung2b - ESM-2 480-dim + ridge"
	@echo "  make rung3  - interface geometry + GBT   rung3b - ProteinMPNN + GBT"
	@echo "  make rung4  - chemistry + geometry + GBT"
	@echo "  make rung5  - chemistry + geometry + ESM-2 + GBT"
	@echo "  make rung6  - chemistry + geometry + ProteinMPNN + GBT"
	@echo "  make rungN0 - chemistry + geometry + ProteinMPNN + random forest  (best)"
	@echo "  make ladder - every rung above, in order"
	@echo ""
	@echo "  make figures       - regenerate reports/figures/"
	@echo "  make errors        - slice tables, probes and diagnostics for the best model"
	@echo "  make test          - split-integrity and harness tests"
	@echo "  make clean         - remove generated reports (keeps the frozen split)"

data:
	$(PYTHON) -m src.data

# The split is a committed artefact. This target regenerates it deliberately; it is not a
# dependency of anything, so no model run can silently re-split the data underneath itself.
splits:
	$(PYTHON) -m src.splits -k $(K)

# --- feature extraction ------------------------------------------------------------------
geom:
	$(PYTHON) -m src.features.geometry

esm:
	$(PYTHON) -m src.features.esm2 --model $(ESM) --device $(DEVICE)

# Needs a ProteinMPNN checkout under third_party/ -- see `make third-party`.
mpnn:
	$(PYTHON) -m src.features.proteinmpnn --device $(DEVICE)

features: geom esm mpnn

third-party:
	@echo "ProteinMPNN is vendored, not pip-installable, and its weights are git-ignored:"
	@echo ""
	@echo "  git clone https://github.com/dauparas/ProteinMPNN third_party/ProteinMPNN"
	@echo ""
	@echo "src/features/proteinmpnn.py loads vanilla_model_weights/v_48_020.pt from there."

# --- the ladder --------------------------------------------------------------------------
rung0:
	$(PYTHON) -m src.train --model mean  --features chem                 --name rung0_mean                  --n-boot $(BOOT)
rung1:
	$(PYTHON) -m src.train --model gbt   --features chem                 --name rung1_chem_gbt              --n-boot $(BOOT)
rung2a:
	$(PYTHON) -m src.train --model gbt   --features esm2_t12_35M@scalars --name rung2a_esm35M_scalars_gbt   --n-boot $(BOOT)
rung2b:
	$(PYTHON) -m src.train --model ridge --features esm2_t12_35M         --name rung2b_esm35M_ridge         --n-boot $(BOOT)
rung3:
	$(PYTHON) -m src.train --model gbt   --features geom                 --name rung3_geom_gbt              --n-boot $(BOOT)
rung3b:
	$(PYTHON) -m src.train --model gbt   --features mpnn                 --name rung3b_mpnn_gbt             --n-boot $(BOOT)
rung4:
	$(PYTHON) -m src.train --model gbt   --features chem,geom            --name rung4_chem_geom_gbt         --n-boot $(BOOT)
rung5:
	$(PYTHON) -m src.train --model gbt   --features chem,geom,esm2_t12_35M@scalars --name rung5_chem_geom_esm_gbt --n-boot $(BOOT)
rung6:
	$(PYTHON) -m src.train --model gbt   --features chem,geom,mpnn       --name rung6_chem_geom_mpnn_gbt    --n-boot $(BOOT)
rungN0:
	$(PYTHON) -m src.train --model rf    --features chem,geom,mpnn       --name rungN0_chem_geom_mpnn_rf    --n-boot $(BOOT)

ladder: rung0 rung1 rung2a rung2b rung3 rung3b rung4 rung5 rung6 rungN0

figures:
	$(PYTHON) -m src.analysis labels

errors:
	$(PYTHON) -m src.error_analysis --run rungN0_chem_geom_mpnn_rf

test:
	$(PYTHON) -m pytest tests -q

clean:
	rm -rf reports/*/

clean-features:
	rm -rf data/features/*.parquet
