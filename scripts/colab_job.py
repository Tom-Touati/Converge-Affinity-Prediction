#!/usr/bin/env python3
"""Run the converge_bind pipeline on a Colab VM, one stage per invocation.

Sent to the VM with `colab exec -s <session> -f scripts/colab_job.py`. The CLI reads this
file locally and executes it in the remote kernel, so it does not need to be uploaded and
the VM never needs credentials -- the repo is public.

Staged deliberately. `colab exec` streams output live but a single 60-minute call is one
disconnect away from losing everything, and the kernel keeps state between calls, so each
stage is independently re-runnable:

    bootstrap   clone, fetch SKEMPI, install deps          ~5 min
    extract     every feature incl. ESM-2 650M             ~30-60 min
    runs        the three sweeps that wanted a GPU         ~20 min
    esm650      does a bigger PLM rescue the sequence arm  ~5 min
    collect     tar the reports for download

Usage on the VM: python colab_job.py <stage>
"""
import os
import subprocess
import sys
import urllib.request

REPO = "https://github.com/Tom-Touati/Converge-Affinity-Prediction.git"
BRANCH = "label-defects-and-augmentation-options"
ROOT = "/content/converge_bind"

# Byte counts of the files the reported results were produced with. BSC serves SKEMPI 2.0
# openly, so this needs no registration -- but a silently truncated download would shift
# every number downstream, so the sizes are asserted rather than trusted.
DATA = {
    "data/skempi_v2.csv": (
        "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv", 1602208),
    "data/SKEMPI2_PDBs.tgz": (
        "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz", 30482090),
}


def sh(cmd, cwd=ROOT, check=True):
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=cwd).returncode
    if rc and check:
        sys.exit(f"!!! failed ({rc}): {cmd}")
    return rc


def stage_bootstrap():
    sh("nvidia-smi || echo 'NO GPU -- runtime has no accelerator'", cwd="/content", check=False)
    if not os.path.isdir(ROOT):
        sh(f"git clone --branch {BRANCH} {REPO} {ROOT}", cwd="/content")
    sh("git log --oneline -3")

    os.makedirs(f"{ROOT}/data", exist_ok=True)
    for rel, (url, expect) in DATA.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path) or os.path.getsize(path) != expect:
            print(f"downloading {rel} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        got = os.path.getsize(path)
        if got != expect:
            sys.exit(f"!!! {rel}: got {got} bytes, expected {expect}")
        print(f"ok  {rel}  {got:,} bytes", flush=True)

    # transformers is a lazy import in features/hf_plm.py; without it the ESM-2 steps fail.
    # Written to a file rather than piped via process substitution: subprocess(shell=True)
    # runs /bin/sh, which has no <(...).
    sh("grep -v '^torch==' requirements.txt > /tmp/req_colab.txt")
    sh("pip install -q -r /tmp/req_colab.txt 'transformers==4.46.3' antiberty 2>&1 | tail -5")
    sh("python -c \"import torch; print('torch', torch.__version__,"
       " '| cuda', torch.cuda.is_available())\"")


def stage_extract():
    # --system-site-packages so the venv sees the VM's CUDA torch; REQ_FILE strips the
    # torch==2.4.1 pin, which exists only for the Windows dev box's MSVC 14.28 runtime.
    sh("grep -v '^torch==' requirements.txt > /tmp/req_colab.txt")
    sh("REQ_FILE=/tmp/req_colab.txt VENV_ARGS=--system-site-packages "
       "bash scripts/setup_remote.sh --big")


def _py():
    return f"{ROOT}/.venv/bin/python" if os.path.exists(f"{ROOT}/.venv/bin/python") else "python"


def stage_runs():
    py = _py()
    # Each is allowed to fail without taking the others down -- a wedged sweep should not
    # cost the two that would have finished.
    sh(f"{py} -m src.rank_fusion --sweep --device cuda --max-steps 200 --seeds 5", check=False)
    sh(f"{py} -m src.fusion_v3 --sweep --seeds 5", check=False)
    sh(f"{py} -m src.compare_models", check=False)


def stage_esm650():
    # Six probes say the sequence arm adds nothing over structure+chemistry, and both
    # diagnosed mechanisms are about what the PLM represents, not about capacity. If 650M
    # also fails here, that is a finding rather than a compute limit. Baseline to beat: 0.487.
    sh(f"{_py()} -m src.train --model rf --features chem,geom,geomrev,mpnn,esm650M_pair "
       f"--name colab_rf_esm650M --n-boot 1000", check=False)


def stage_collect():
    sh("tar czf /content/reports.tgz reports data/features", check=False)
    sh("ls -la /content/reports.tgz", cwd="/content", check=False)


STAGES = {
    "bootstrap": stage_bootstrap, "extract": stage_extract, "runs": stage_runs,
    "esm650": stage_esm650, "collect": stage_collect,
}

if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else "bootstrap"
    if want not in STAGES:
        sys.exit(f"unknown stage {want!r}; pick one of {', '.join(STAGES)}")
    print(f"=== stage: {want} ===", flush=True)
    STAGES[want]()
    print(f"=== stage {want} done ===", flush=True)
