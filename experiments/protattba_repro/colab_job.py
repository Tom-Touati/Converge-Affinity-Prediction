#!/usr/bin/env python3
"""Run the ProtAttBA reproduction on a Colab GPU, one stage per invocation.

The local box cannot do this run. Their config trains at batch 12 over antibody chains up to
965 residues, and the head stores roughly 60 MB per tensor per cross-attention block; a 2 GB
GTX 1050 hands PyTorch about 900 MB once the CUDA context and the Windows display have taken
their share, and it dies at 822 MB allocated even with the attention blocks recomputed in
backward. Dropping the batch size would change the result -- see the ``1e-10`` key mask note in
cached_encoder.py, where a prediction genuinely depends on its batch's padding width -- so the
honest move is a bigger card rather than a smaller batch.

Mirrors ``scripts/colab_job.py``'s conventions, which exist because of specific failures
recorded in HANDOFF.md:

* ``colab exec`` returns 0 even when the remote script raises, so nothing can be gated on an
  exit code. Every stage prints ``STAGE_OK <name>`` and the runner greps for it.
* ``colab exec`` relays only what the kernel process itself writes, so ``subprocess`` output
  vanishes. ``sh()`` reads the pipe and re-prints it.
* Colab ships numpy 2 and a preinstalled jax that ``transformers`` imports transitively.
  ProtAttBA's ``requirments.txt`` pins ``numpy==1.26.4`` and ``scipy==1.10.1``, which downgrade
  it and break that import chain, so both pins are filtered out here. Their torch pin is
  dropped too: Colab's own CUDA build is already correct for the VM's driver.

Stages:

    bootstrap   clone ProtAttBA at its pinned SHA, install deps, fetch ESM2-650M   ~6 min
    extract     per-residue embeddings for S1131's 1021 distinct sequences, on GPU ~3 min
    upstream    10-fold CV in their configuration (monitor == test fold)
    honest      10-fold CV with the monitored split carved out of training rows
    collect     tar results/ for download

Usage on the VM: ``python colab_job.py <stage>``
"""
import os
import subprocess
import sys

#: Pinned so the reproduction is against a fixed tree, not whatever main holds later.
PROTATTBA = "https://github.com/code4luck/ProtAttBA.git"
PROTATTBA_SHA = "9adbf9899a7b3aa69da4ea00f5b3ca9e14bddbbd"

ROOT = "/content/protattba_repro"
UPSTREAM = f"{ROOT}/ProtAttBA"

#: ProtAttBA's requirments.txt cannot be honoured on Colab, and not for want of trying.
#: Their environment is python 3.10; Colab now ships **python 3.13**, and their 2024 pins
#: (protobuf 3.19.6, llvmlite 0.43.0, rjieba, progressbar2 ...) have no 3.13 wheels, so pip
#: falls back to building from source and dies with "Getting requirements to build wheel did
#: not run successfully". Installing the file is therefore not an option on this runtime.
#:
#: Instead we install only what the S1131 code path actually imports, and let Colab's own
#: stack provide the rest. Nothing in src_s1131, model_module or utils touches xgboost,
#: matplotlib, seaborn, llvmlite, rjieba, progressbar2, lxml or protobuf.
#:
#: The version drift this leaves is real and is cross-checked rather than waved away: the
#: `crosscheck` stage re-embeds three sequences here and compares them against the cache built
#: locally under their own pins (transformers 4.40.2 / torch 2.1.2 / python 3.10).
NEEDED = ("pytorch-lightning", "torchmetrics")


def sh(cmd, cwd=ROOT, check=True):
    """Run a shell command, relaying its output through Python's stdout.

    A plain subprocess.run inherits the kernel's descriptors, so a child's stdout goes
    somewhere the exec stream never shows. Reading the pipe and re-printing puts it back.
    """
    print(f"\n$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    if rc and check:
        # colab exec always exits 0, so a non-zero child is invisible in the exit status.
        print(f"!!! STAGE_FAILED ({rc}): {cmd}", flush=True)
        raise SystemExit(rc)
    return rc


def stage_bootstrap():
    sh("nvidia-smi || echo 'NO GPU -- runtime has no accelerator'", cwd="/content", check=False)
    os.makedirs(ROOT, exist_ok=True)

    if not os.path.isdir(UPSTREAM):
        sh(f"git clone {PROTATTBA} {UPSTREAM}", cwd="/content")
    sh(f"git -C {UPSTREAM} fetch --depth 50 origin && "
       f"git -C {UPSTREAM} checkout {PROTATTBA_SHA}", check=False)
    sh(f"git -C {UPSTREAM} rev-parse HEAD")

    print(f"\ninstalling only {list(NEEDED)}; see NEEDED for why their pins are not used",
          flush=True)
    sh(f"pip install -q {' '.join(NEEDED)}")

    # Their code path must actually import under Colab's much newer stack. transformers 5.x
    # is two majors past their 4.40.2, so EsmModel's presence is checked rather than assumed,
    # and the whole upstream import chain is exercised before any GPU time is spent.
    sh("python -c \""
       "import sys, torch, transformers, pytorch_lightning as pl, torchmetrics; "
       "print('python', sys.version.split()[0]); "
       "print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); "
       "print('transformers', transformers.__version__, '| lightning', pl.__version__, "
       "'| torchmetrics', torchmetrics.__version__); "
       "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
       "from transformers import EsmModel, AutoTokenizer; print('EsmModel import OK')\"")
    sh("python -c \""
       "import paths_local; paths_local.add_upstream_to_path(); "
       "import model, litmodel, dataset; "
       "from model_module.rope_attn import MutilHeadSelfAttn; "
       "from utils.common import get_s1131_data; "
       "from utils.data_split import get_K_fold_generator, get_K_fold_with_test_generator; "
       "print('upstream import chain OK:', model.__file__)\"")

    sh("python fetch_esm2.py")
    print("STAGE_OK bootstrap", flush=True)


def stage_verify():
    sh("python verify_published.py")
    sh("python check_encoder_assumptions.py")
    print("STAGE_OK verify", flush=True)


def stage_extract():
    if os.path.exists(f"{ROOT}/cache/esm2_650m_index.pkl"):
        print("embedding cache already present, skipping", flush=True)
    else:
        sh("python extract_embeddings.py --device cuda --batch-tokens 32768")
    print("STAGE_OK extract", flush=True)


#: Fingerprints of three sequences' embeddings, computed locally under ProtAttBA's own pins
#: (python 3.10, torch 2.1.2+cu118, transformers 4.40.2). `crosscheck` recomputes them on the
#: VM and compares. This is the guard on the one deviation that could silently change the
#: result: Colab's transformers is two majors newer, and if its ESM2 implementation differed
#: the whole reproduction would be measuring a different encoder.
CROSSCHECK = f"{ROOT}/crosscheck_local.json"


def stage_crosscheck():
    sh("python crosscheck_embeddings.py --reference crosscheck_local.json")
    print("STAGE_OK crosscheck", flush=True)


def _run(protocol):
    sh(f"python run_cv.py --protocol {protocol} --device gpu")
    print(f"STAGE_OK {protocol}", flush=True)


def stage_upstream():
    _run("upstream")


def stage_honest():
    _run("honest")


def stage_compare():
    sh("python compare.py")
    print("STAGE_OK compare", flush=True)


def stage_collect():
    # Never tar the 0.66 GB embedding cache or the 2.6 GB checkpoint; only the results.
    sh(f"tar czf /content/protattba_results.tgz -C {ROOT} results", check=False)
    sh("ls -la /content/protattba_results.tgz", cwd="/content")
    print("STAGE_OK collect", flush=True)


STAGES = {
    "bootstrap": stage_bootstrap, "verify": stage_verify, "crosscheck": stage_crosscheck,
    "extract": stage_extract, "upstream": stage_upstream, "honest": stage_honest,
    "compare": stage_compare, "collect": stage_collect,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "bootstrap"
    if name not in STAGES:
        print(f"!!! STAGE_FAILED: unknown stage {name!r}; have {sorted(STAGES)}", flush=True)
        raise SystemExit(2)
    STAGES[name]()
