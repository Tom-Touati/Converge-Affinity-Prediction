"""The single entry point the plan asks for: one experiment, from its config.

    python -m src.fusion.run --config configs/E0c_rf_pooled_esm_mpnn.yaml
    python -m src.fusion.run --config configs/E0a_rf_handcrafted.yaml --fold 0 --seed 0

Every experiment is reproducible from its config plus the frozen split alone; nothing else is
read. ``kind`` in the config selects the dispatcher.

**Only ``kind: tree`` is implemented.** The neural kinds (E1 concat, E2 FiLM, E3
distance-biased attention, E4 interface pooling, E5 antisymmetric head) are specified in
``docs/ai_prompts/overnight_prompt.md`` and blocked on a prerequisite that does not exist in
this repo: per-residue ESM and ProteinMPNN tensors for these 940 rows. Every cached block under
``data/features/`` is pooled or scalar, one row per mutation. Rather than emit a plausible
number from a stand-in representation, ``run.py`` raises and points at what is missing. See
``docs/decisions.md`` D19 and ``docs/next_steps.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.fusion import run_baselines

TREE_KINDS = {"tree"}
NEURAL_KINDS = {"concat", "film", "dist_attn", "interface", "antisym"}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for key in ("exp", "kind", "dataset", "split"):
        if key not in cfg:
            raise SystemExit(f"{path} is missing required key {key!r}")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fold", type=int, default=None,
                    help="restrict to one fold (default: every fold in the frozen split)")
    ap.add_argument("--seed", type=int, default=None,
                    help="restrict to one seed (default: the config's seeds)")
    cli = ap.parse_args()

    cfg = load_config(cli.config)
    kind, exp = cfg["kind"], cfg["exp"]
    seeds = (cli.seed,) if cli.seed is not None else tuple(cfg.get("seeds", (0, 1, 2)))

    if kind in TREE_KINDS:
        if cli.fold is not None:
            # The tree runner fits every fold in one pass; a single fold would need a
            # different loop and no experiment calls for it, so this is refused rather than
            # silently ignored.
            raise SystemExit(
                "--fold is not supported for kind: tree; the runner fits all folds in one "
                "pass. Drop the flag, or use --seed to narrow."
            )
        run_baselines.run_one(exp, cfg["feature_set"], cfg["model"],
                              bool(cfg.get("drop_vectors", False)), seeds=seeds)
        return

    if kind in NEURAL_KINDS:
        raise SystemExit(
            f"kind: {kind} ({exp}) is specified but not implemented.\n"
            "It needs per-residue ESM and ProteinMPNN tensors for the 940 antibody-antigen "
            "rows, and no such cache exists in this repo -- every block under data/features/ "
            "is pooled or scalar, one row per mutation.\n"
            "The four-sequence table it would be built from is ready at "
            "experiments/protattba_repro/cache/project_sequences.parquet (940 rows, 884 "
            "distinct sequences, 335,261 residues).\n"
            "See docs/decisions.md D19 and docs/next_steps.md item 1."
        )

    raise SystemExit(f"unknown kind {kind!r}; have {sorted(TREE_KINDS | NEURAL_KINDS)}")


if __name__ == "__main__":
    main()
