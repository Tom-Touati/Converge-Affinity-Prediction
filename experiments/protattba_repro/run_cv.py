"""Run ProtAttBA's 10-fold cross-validation on S1131, using their own code.

Every component that decides the number comes from the upstream checkout, imported rather than
copied, so there is no opportunity for a quiet edit:

===========================  ==========================================================
their module                  what it decides here
===========================  ==========================================================
``utils.common``              how S1131.csv becomes four sequence lists
``utils.data_split``          the fold assignment
``dataset.WrapperDataset``    tokenisation, padding, batching, shuffling
``model.SeqBindModel``        the whole architecture
``model_module.rope_attn``    the rotary cross-attention block
``litmodel.LitModel``         loss, metrics, optimiser
===========================  ==========================================================

The one substitution is ``model.EsmModel = CachedEsmEncoder``, which serves the frozen
encoder's output from ``extract_embeddings.py``'s cache instead of recomputing it. With
``--freeze_backbone`` that is an identity, and ``check_encoder_assumptions.py`` measures the
three properties it relies on. It is not an optimisation for taste: their file as written needs
about 1700 CPU-hours for this run on this hardware, against about 5 GPU-hours with the cache.

Hyperparameters are ``cross_validation/scripts/bash_cross-validation.sh`` verbatim, plus
``src_s1131/trainer.py``'s defaults for what the script leaves unset. The shipped script names
AB1101; the README says to use it "with different args" for the other sets, and src_s1131 and
src_ab1101 carry identical defaults, so these are the S1131 values too. Deviations are listed
in DEVIATIONS below and echoed at the top of every run.

Two protocols, selected with ``--protocol``:

``upstream``
    What their code does. ``WrapperDataset`` is constructed as
    ``get_dataset(train_idxes, test_idxes)`` and assigns its second return value to
    ``self.val_dataset``; ``get_val_loader`` and ``get_test_loader`` then both return that same
    object. So the ``val_pearson_corr`` that drives ``EarlyStopping`` and ``ModelCheckpoint`` is
    computed on the test fold, the best epoch is chosen by test-fold correlation, and
    ``trainer.test`` re-scores that same fold. This is the configuration that produced the
    published 0.84, and reproducing it is the task.

``honest``
    Identical in every other respect, but the monitored split is carved out of the training
    rows with their *own* unused helper ``get_K_fold_with_test_generator`` (train_ratio=0.875),
    which reuses the same ``KFold`` call and therefore leaves the ten test folds row-for-row
    identical. Reports the same metrics on the same rows with model selection no longer
    touching them, which is the number this project would have to beat.

Run: ``python run_cv.py --protocol upstream`` then ``--protocol honest``
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from argparse import Namespace

# Set before torch initialises CUDA, exactly as their trainer.py does at import time.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from metrics import S1131_CSV, SEED, format_summary, score_by_fold, summarise
from paths_local import ESM2_DIR, RESULTS, add_upstream_to_path

add_upstream_to_path()

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402

import model as upstream_model  # noqa: E402  their src_s1131/model.py
from cached_encoder import CachedEsmEncoder  # noqa: E402

#: The only change to their code path. ``SeqBindModel.__init__`` resolves the name ``EsmModel``
#: in this module's globals at call time, so rebinding it here is enough and their file on disk
#: is never touched.
upstream_model.EsmModel = CachedEsmEncoder

import model_module.rope_attn as rope_attn  # noqa: E402

from dataset import WrapperDataset  # noqa: E402
from litmodel import LitModel  # noqa: E402
from utils.common import get_s1131_data  # noqa: E402
from utils.data_split import (  # noqa: E402
    get_K_fold_generator,
    get_K_fold_with_test_generator,
)


def enable_gradient_checkpointing() -> None:
    """Recompute the four cross-attention blocks in backward instead of storing them.

    Needed only because this box has a 2 GB GTX 1050. Their batch of 12 at S1131's longest
    antibody chain (965 residues) stores, per attention block, a [12, 967, 1280] tensor for
    each of the layer norms, the query projection, the rotary-rotated query, the attention
    output and three more feed-forward steps -- about 60 MB each, four blocks deep, on top of
    a 124 MB head with 372 MB of AdamW state. It ran out at 917 MB allocated with 28 MB
    requested.

    This is a compute-for-memory trade and nothing else: ``checkpoint`` re-runs the same
    forward under the same RNG state (``preserve_rng_state`` defaults to True, so the
    attention dropout draws the identical mask), so gradients and therefore the trained model
    are unchanged. Batch size stays at their 12, which matters -- see the ``1e-10`` key mask
    note in cached_encoder.py, where a prediction genuinely depends on its batch's padding
    width.

    Patched on the class rather than by wrapping the module, so the module tree and the
    state_dict keys stay exactly as their code builds them and ModelCheckpoint /
    load_from_checkpoint keep working.
    """
    original = rope_attn.MutilHeadSelfAttn.forward

    def forward(self, q, k, v, mask=None):
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                original, self, q, k, v, mask, use_reentrant=False
            )
        return original(self, q, k, v, mask)

    rope_attn.MutilHeadSelfAttn.forward = forward

#: bash_cross-validation.sh, plus src_s1131/trainer.py defaults for what it does not set.
UPSTREAM_ARGS = dict(
    seed=SEED,                      # SEED=3407
    max_epochs=120,                 # MAX_EPOCHS=120
    accumulate_grad_batches=1,      # ACC_BATCH=1
    lr=3e-5,                        # LR=3e-5
    patience=20,                    # PATIENCE=20
    batch_size=12,                  # BATCH_SIZE=12
    hidden_size=1280,               # HIDDEN_SIZE=1280
    num_heads=4,                    # NUM_HEADS=4
    dropout=0.1,                    # DROPOUT=0.1
    loss="mse",                     # loss="mse"
    freeze_backbone=True,           # --freeze_backbone
    out_dim=1,                      # trainer.py default
    n_fold=10,                      # trainer.py default
    monitor="val_pearson_corr",     # trainer.py default
    gradient_clip_val=5.0,          # trainer.py default
    gradient_clip_algorithm="value",  # trainer.py default
    strategy="auto",                # STRATEGY="auto"
    num_nodes=1,                    # NUM_NODES=1
    devices=1,                      # DEVICES=1
)

DEVIATIONS = [
    "frozen ESM2-650M served from a precomputed cache instead of recomputed in the loop "
    "(identity under --freeze_backbone; see check_encoder_assumptions.py)",
    "DataLoader num_workers=0 on Windows instead of their 4 (spawn cost); their 4 is used "
    "on Linux. The sampler runs in the parent process either way, so batch composition and "
    "iteration order are unchanged.",
]


#: Where to write per-epoch history. The project's dashboard (`scripts/dashboard/serve.py`)
#: polls a *running* VM with `colab download`, which works while the kernel is BUSY, and looks
#: for `<REMOTE>/rank_fusion_sweep.csv` plus `<REMOTE>/<name>/history.csv` where REMOTE is
#: `/content/converge_bind/reports`. Writing into that layout means the existing dashboard
#: shows this run live with no change to it -- `serve.py --session pab`.
HISTORY_ROOT = Path(os.environ.get("PROTATTBA_HISTORY_DIR",
                                   "/content/converge_bind/reports"
                                   if os.path.isdir("/content") else str(RESULTS / "history")))
SWEEP_CSV = HISTORY_ROOT / "rank_fusion_sweep.csv"


class HistoryCallback(pl.Callback):
    """Append one row per validation epoch, in the dashboard's history.csv schema.

    Exists because the first remote attempt was invisible: with `enable_progress_bar=False`
    and `logger=False` there was no way to tell a slow epoch from a hung process, and a
    35-minute silence had to be diagnosed by guessing. This makes the run observable while it
    runs, and records seconds-per-epoch so the wall clock can be explained rather than
    estimated.
    """

    COLUMNS = ("epoch", "step", "steps", "loss", "val_rho", "val_pearson", "val_rmse",
               "train_rho", "seconds", "fold", "n_train", "n_val", "name", "protocol")

    def __init__(self, name: str, protocol: str, fold: int, n_train: int, n_val: int):
        self.path = HISTORY_ROOT / name / "history.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = dict(name=name, protocol=protocol, fold=fold,
                         n_train=n_train, n_val=n_val)
        self.t0 = time.time()
        if not self.path.exists():
            self.path.write_text(",".join(self.COLUMNS) + "\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = {k: float(v) for k, v in trainer.logged_metrics.items()
             if hasattr(v, "item") or isinstance(v, (int, float))}
        row = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "steps": trainer.num_training_batches,
            "loss": m.get("loss_epoch", m.get("loss", "")),
            "val_rho": m.get("val_spearman_corr", ""),
            "val_pearson": m.get("val_pearson_corr", ""),
            "val_rmse": m.get("val_mse", "") ** 0.5 if m.get("val_mse") else "",
            "train_rho": "",          # their LitModel does not compute a training correlation
            "seconds": round(time.time() - self.t0, 1),
            **self.meta,
        }
        with open(self.path, "a") as f:
            f.write(",".join(str(row[c]) for c in self.COLUMNS) + "\n")


def write_sweep_row(name: str, protocol: str, folds_done: int, note: str = "") -> None:
    """Keep one row per run in the file the dashboard reads first, so the run is listed."""
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = {}
    if SWEEP_CSV.exists():
        try:
            for r in pd.read_csv(SWEEP_CSV).to_dict("records"):
                rows[r.get("name")] = r
        except Exception:
            rows = {}
    rows[name] = {"name": name, "protocol": protocol, "folds_done": folds_done,
                  "resid": False, "min_grp": "n/a", "dataset": "S1131", "note": note}
    pd.DataFrame(list(rows.values())).to_csv(SWEEP_CSV, index=False)


def build_args(protocol: str, device: str, max_epochs: int | None) -> Namespace:
    args = Namespace(**UPSTREAM_ARGS)
    args.model_locate = str(ESM2_DIR)
    args.data_path = str(S1131_CSV)
    args.data_name = "S1131"
    # Their trainer.py default is 4. Windows pays a process-spawn cost per epoch that makes
    # that a loss, but on Linux it is both faster *and* their actual setting, so it is used
    # wherever it works. Either way the sampler runs in the parent process, so batch
    # composition and iteration order are identical -- this cannot move the result.
    #
    # It matters more than it looks: the collator calls the tokenizer four times per batch
    # over sequences up to 965 residues, and with a single worker that CPU work does not
    # overlap the GPU at all. A T4 epoch measured 30 s at roughly 1% utilisation.
    args.num_workers = 0 if os.name == "nt" else 4
    args.accelerator = device
    args.protocol = protocol
    if max_epochs is not None:
        args.max_epochs = max_epochs
    return args


def restore_best(args: Namespace, ckpt_path: str) -> LitModel:
    """Rebuild the model and load the best epoch's weights.

    Their trainer.py does ``LitModel.load_from_checkpoint(best_model_path)``. That classmethod
    reconstructs the module from the checkpoint's stored hyperparameters, and under Lightning
    2.6 with an argparse ``Namespace`` in ``hparams`` it did not return -- an 8-epoch probe
    trained in 262 s and then sat for 18 minutes with the checkpoint already written.

    This does the same two things explicitly: construct ``LitModel(args)`` from the arguments
    we already hold, and load the saved ``state_dict`` strictly. Same weights, same module, no
    dependence on how a given Lightning version round-trips hyperparameters. ``strict=True`` is
    the guard -- if the module tree and the checkpoint ever stopped matching, this raises
    rather than silently evaluating a partly-initialised head.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lit = LitModel(args)
    missing, unexpected = lit.load_state_dict(blob["state_dict"], strict=False)
    # The cached encoder holds no parameters, so nothing of it appears in the state dict.
    missing = [k for k in missing if not k.startswith("model.encoder.")]
    unexpected = [k for k in unexpected if not k.startswith("model.encoder.")]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    return lit


@torch.no_grad()
def predict(lit: LitModel, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Collect per-example predictions.

    Their ``test_step`` only logs aggregate metrics, so the per-example predictions they ship
    must have come from a script not in the repo. This walks the same test loader -- their
    object, ``shuffle=False``, their batch size -- so each example is padded exactly as their
    ``trainer.test`` would pad it. That matters because of the ``1e-10`` key mask noted in
    cached_encoder.py: a prediction is not independent of its batch's padding width.
    """
    lit = lit.to(device).eval()
    preds, trues = [], []
    for batch in loader:
        out = lit.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"].to(device),
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"].to(device),
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"].to(device),
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"].to(device),
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"].to(device),
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"].to(device),
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"].to(device),
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"].to(device),
        )
        preds.append(out.float().cpu().numpy())
        trues.append(batch["labels"].numpy())
    return np.concatenate(preds), np.concatenate(trues)


def folds_for(protocol: str, n_rows: int, args: Namespace) -> list[tuple]:
    """(train_idx, monitor_idx, test_idx) per fold, materialised before any training starts.

    Eager on purpose. ``get_K_fold_with_test_generator`` draws its validation split with
    ``np.random.choice``, i.e. from the *global* numpy RNG, so consuming it lazily inside the
    fold loop would make fold k's validation split depend on how much randomness folds 0..k-1
    happened to burn during training. Draining the generator first makes the splits a function
    of the seed alone, so a rerun -- or a run of only folds 3-5 -- gets the same splits.
    """
    rows = np.arange(n_rows)
    upstream = [(tr, te) for tr, te in get_K_fold_generator(rows, n_fold=args.n_fold,
                                                            seed=args.seed)]
    if protocol == "upstream":
        # monitor == test: this is the leak, reproduced deliberately
        return [(tr, te, te) for tr, te in upstream]

    np.random.seed(args.seed)
    out = []
    gen = get_K_fold_with_test_generator(rows, n_fold=args.n_fold, seed=args.seed)
    for fold, (train_idx, val_idx, test_idx) in enumerate(gen):
        # Their helper reuses the same KFold call, so the test folds must be identical to the
        # upstream protocol's. Asserted rather than assumed, because the honest number is only
        # comparable to the published one if it is measured on the very same rows.
        assert set(test_idx) == set(upstream[fold][1]), "honest protocol moved the test fold"
        assert not (set(train_idx) & set(val_idx)), "train/val overlap"
        assert not (set(train_idx) & set(test_idx)), "train/test overlap"
        assert not (set(val_idx) & set(test_idx)), "val/test overlap"
        out.append((train_idx, val_idx, test_idx))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="upstream", choices=["upstream", "honest"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--max-epochs", type=int, default=None, help="override for smoke tests")
    ap.add_argument("--folds", type=int, default=None, help="run only the first N folds")
    ap.add_argument("--tag", default=None, help="suffix for output filenames")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute attention blocks in backward; needed under ~4 GB VRAM")
    ap.add_argument("--fresh", action="store_true",
                    help="discard per-fold files already on disk instead of resuming")
    cli = ap.parse_args()

    if cli.grad_checkpoint:
        enable_gradient_checkpointing()

    args = build_args(cli.protocol, cli.device, cli.max_epochs)
    tag = cli.tag or cli.protocol
    device = torch.device("cuda" if cli.device == "gpu" and torch.cuda.is_available() else "cpu")

    print(f"protocol   {cli.protocol}")
    print(f"device     {device}")
    print(f"upstream   {UPSTREAM_ARGS}")
    print("deviations from bash_cross-validation.sh:")
    for d in DEVIATIONS:
        print(f"  - {d}")
    if cli.grad_checkpoint:
        print("  - cross-attention blocks recomputed in backward (memory only; batch size "
              "stays 12 and gradients are unchanged)")
    if cli.protocol == "upstream":
        print("\nNOTE: in this protocol EarlyStopping and ModelCheckpoint monitor the TEST fold,"
              "\n      because WrapperDataset assigns the test indices to self.val_dataset and"
              "\n      get_val_loader/get_test_loader return the same object. Reproduced on"
              "\n      purpose -- it is the configuration behind the published number.")

    seed_everything(args.seed)
    ab_wt, ag_wt, ab_mt, ag_mt, labels = get_s1131_data(args.data_path)
    n_rows = len(labels)
    assert n_rows == 1131, f"expected 1131 rows, got {n_rows}"

    ckpt_root = RESULTS / f"checkpoints_{tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)

    # One file per fold, written the moment the fold finishes, and folds already on disk are
    # skipped. A 10-fold run on a Colab VM is hours long and `colab exec` lost its connection
    # 35 minutes into the first attempt -- with results written only at the end, that lost
    # everything. HANDOFF.md §7 records the same lesson for the project's own runner.
    records, fold_log = [], []
    done = {}
    for p in sorted(RESULTS.glob(f"{tag}_fold*.csv")):
        k = int(p.stem.rsplit("fold", 1)[1])
        done[k] = pd.read_csv(p)
    if done and not cli.fresh:
        print(f"\nresuming: folds {sorted(done)} already on disk, "
              f"{sum(len(v) for v in done.values())} predictions")
    elif cli.fresh:
        for p in RESULTS.glob(f"{tag}_fold*.csv"):
            p.unlink()
        done = {}

    # Registered before the first fold, not after it. The dashboard pulls the sweep file first
    # and only fetches history for names it lists, so registering late leaves the run invisible
    # for exactly the stretch when watching it matters most.
    run_name = f"protattba_{cli.protocol}"
    write_sweep_row(run_name, cli.protocol, len(done), note="running")
    print(f"\nhistory -> {HISTORY_ROOT / run_name / 'history.csv'}")
    print(f"sweep   -> {SWEEP_CSV}")

    t_start = time.time()

    splits = folds_for(cli.protocol, n_rows, args)
    if cli.folds is not None:
        splits = splits[: cli.folds]

    for fold, (train_idx, mon_idx, test_idx) in enumerate(splits):
        if fold in done:
            records.extend(done[fold].to_dict("records"))
            one = score_by_fold(done[fold]).iloc[0]
            print(f"fold {fold}: reused from disk | test PCC {one.pcc:.4f} "
                  f"rho {one.rho:.4f} RMSE {one.rmse:.4f}", flush=True)
            continue
        t_fold = time.time()

        # Their WrapperDataset takes (train, test) and exposes train/val/test loaders where val
        # and test are the same object. Used twice so the monitored split and the reported split
        # can differ in the honest protocol, with their collator driving both.
        seqs = dict(wt_ab_seqs=ab_wt, mt_ab_seqs=ab_mt, wt_ag_seqs=ag_wt, mt_ag_seqs=ag_mt,
                    labels=labels)
        mon_pack = WrapperDataset(args=args, train_idxes=train_idx, test_idxes=mon_idx, **seqs)
        test_pack = WrapperDataset(args=args, train_idxes=train_idx, test_idxes=test_idx, **seqs)

        lit = LitModel(args)
        trainable = sum(p.numel() for p in lit.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lit.model.parameters())
        if fold == 0:
            print(f"\nhead parameters  trainable {trainable / 1e6:.2f} M of {total / 1e6:.2f} M "
                  f"(the frozen 652 M encoder is served from cache, so it is not a parameter here)")

        mode = "max" if "corr" in args.monitor else "min"
        ckpt = ModelCheckpoint(str(ckpt_root / str(fold)), monitor=args.monitor, mode=mode,
                               save_top_k=1, filename=f"fold-{fold}", verbose=False)
        early = EarlyStopping(monitor=args.monitor, mode=mode, patience=args.patience,
                              verbose=False)

        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            strategy=args.strategy,
            num_nodes=args.num_nodes,
            max_epochs=args.max_epochs,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            deterministic=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[ckpt, early,
                       HistoryCallback(f"protattba_{cli.protocol}", cli.protocol,
                                       fold, len(train_idx), len(mon_idx))],
        )
        trainer.fit(lit, mon_pack.get_train_loader(), mon_pack.get_val_loader())

        best = restore_best(args, ckpt.best_model_path)
        preds, trues = predict(best, test_pack.get_test_loader(), device)

        assert len(preds) == len(test_idx), f"{len(preds)} predictions for {len(test_idx)} rows"
        # get_test_loader is shuffle=False, so row k of the loader is test_idx[k].
        np.testing.assert_allclose(trues, np.asarray(labels)[test_idx], rtol=0, atol=1e-5)

        fold_rows = [{"fold": fold, "row": int(row), "y_pred": float(p), "y_true": float(t)}
                     for row, p, t in zip(test_idx, preds, trues)]
        records.extend(fold_rows)
        pd.DataFrame(fold_rows).to_csv(RESULTS / f"{tag}_fold{fold}.csv", index=False)

        mins = (time.time() - t_fold) / 60
        fold_log.append({
            "fold": fold, "n_train": len(train_idx), "n_monitor": len(mon_idx),
            "n_test": len(test_idx), "epochs_run": int(trainer.current_epoch),
            "best_monitor": float(ckpt.best_model_score), "minutes": round(mins, 2),
        })
        one = score_by_fold(pd.DataFrame([r for r in records if r["fold"] == fold])).iloc[0]
        print(f"fold {fold}: {trainer.current_epoch:3d} epochs, best {args.monitor} "
              f"{float(ckpt.best_model_score):.4f} | test PCC {one.pcc:.4f} rho {one.rho:.4f} "
              f"RMSE {one.rmse:.4f} | {mins:.1f} min", flush=True)

        write_sweep_row(f"protattba_{cli.protocol}", cli.protocol, fold + 1,
                        note=f"{trainer.current_epoch} epochs, {mins:.1f} min/fold")
        shutil.rmtree(ckpt_root / str(fold), ignore_errors=True)
        del lit, best, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wall = (time.time() - t_start) / 60
    preds_df = pd.DataFrame(records).sort_values(["fold", "row"]).reset_index(drop=True)
    per_fold = score_by_fold(preds_df)

    preds_df.to_csv(RESULTS / f"{tag}_predictions.csv", index=False)
    per_fold.to_csv(RESULTS / f"{tag}_per_fold.csv", index=False)
    summarise(per_fold).to_csv(RESULTS / f"{tag}_summary.csv", index=False)
    with open(RESULTS / f"{tag}_run.json", "w") as f:
        json.dump({"protocol": cli.protocol, "device": str(device),
                   "upstream_args": {k: str(v) for k, v in UPSTREAM_ARGS.items()},
                   "deviations": DEVIATIONS, "wall_clock_minutes": round(wall, 1),
                   "folds": fold_log}, f, indent=2)

    print(f"\n{per_fold.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    print(f"\n{cli.protocol}: {format_summary(per_fold)}")
    print(f"wall clock {wall:.1f} min over {len(per_fold)} folds")


if __name__ == "__main__":
    main()
