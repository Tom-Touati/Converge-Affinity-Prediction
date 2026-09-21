#!/usr/bin/env bash
# Bring a fresh GPU machine from nothing to a trained model.
#
# Everything here is idempotent: each step checks for its own output and skips if present, so the
# script can be re-run after a disconnect without repeating an hour of extraction.
#
#   bash scripts/setup_remote.sh            # full bootstrap, then the cheap extractions
#   bash scripts/setup_remote.sh --big      # also ESM-2 650M, which is what the GPU is for
#
# On a host that already has a working CUDA torch outside a venv (Colab, some images):
#   VENV_ARGS=--system-site-packages bash scripts/setup_remote.sh --big
# Otherwise the fresh venv cannot see it and a CPU wheel gets installed over it.
# Pair it with REQ_FILE to drop the torch pin so the existing build survives:
#   grep -v '^torch==' requirements.txt > /tmp/req.txt
#   REQ_FILE=/tmp/req.txt VENV_ARGS=--system-site-packages bash scripts/setup_remote.sh --big
# On Colab the venv cannot be built at all (ensurepip fails) and is unnecessary, since the
# system interpreter already has a CUDA torch. Skip it:
#   PYTHON_BIN=$(which python3) bash scripts/setup_remote.sh --big
#
# The two data files are NOT in git (they are large and SKEMPI asks you to register). Either
# download them from https://life.bsc.es/pid/skempi2/ into data/, or copy them from a machine
# that already has them:
#
#   scp data/skempi_v2.csv data/SKEMPI2_PDBs.tgz <host>:~/converge_bind/data/
set -euo pipefail

BIG=0
[[ "${1:-}" == "--big" ]] && BIG=1

cd "$(dirname "$0")/.."
ROOT=$(pwd)
echo "=== converge_bind bootstrap in $ROOT ==="

# ---------------------------------------------------------------- python environment
# PYTHON_BIN skips the venv entirely and uses the interpreter given. Needed on hosts where
# the venv is both broken and pointless: on Colab, `python3 -m venv --system-site-packages`
# fails outright in ensurepip, and the system interpreter already carries a CUDA torch, so
# building a venv at all was solving a problem that did not exist there.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
  echo "--- using $PY directly (no venv)"
else
  if [[ ! -d .venv ]]; then
    echo "--- creating .venv"
    python3 -m venv ${VENV_ARGS:-} .venv
  fi
  PY=.venv/bin/python
  [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe      # windows layout, just in case
fi

$PY -m pip install --quiet --upgrade pip
# REQ_FILE lets a caller substitute a filtered requirements file. On hosts that already
# ship a CUDA torch (Colab), the torch==2.4.1 pin -- a Windows-only MSVC constraint --
# has to be stripped or pip replaces the good build with an older one.
$PY -m pip install --quiet -r "${REQ_FILE:-requirements.txt}"

# The torch pin in requirements.txt exists only for this project's Windows development box
# (MSVC runtime 14.28). On a GPU machine take a current CUDA wheel instead.
if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "--- installing a CUDA build of torch (the CPU pin is a Windows-only constraint)"
  $PY -m pip install --quiet --upgrade torch --index-url https://download.pytorch.org/whl/cu121
fi
$PY -c "import torch; print('    torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---------------------------------------------------------------- third-party weights
if [[ ! -d third_party/ProteinMPNN ]]; then
  echo "--- cloning ProteinMPNN (vendored, not pip-installable)"
  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git third_party/ProteinMPNN
fi

# ---------------------------------------------------------------- data
for f in data/skempi_v2.csv data/SKEMPI2_PDBs.tgz; do
  if [[ ! -f "$f" ]]; then
    echo "!!! missing $f"
    echo "    Download from https://life.bsc.es/pid/skempi2/ or scp it from a machine that has it:"
    echo "      scp data/skempi_v2.csv data/SKEMPI2_PDBs.tgz <this-host>:$ROOT/data/"
    exit 1
  fi
done

# ---------------------------------------------------------------- pipeline
step () {  # step <output-file> <description> <command...>
  if [[ -e "$1" ]]; then
    echo "--- skip: $2 (already present)"
  else
    echo "--- $2"
    shift 2
    "$@"
  fi
}

step data/processed/dataset.parquet     "parsing SKEMPI"            $PY -m src.data
# folds.csv is committed on purpose; regenerate only if somehow absent
step data/folds.csv                     "freezing the split"        $PY -m src.splits -k 4
step data/features/geom.parquet         "interface geometry"        $PY -m src.features.geometry
step data/features/geomrev.parquet      "reversible geometry"       $PY -m src.features.geom_rev
step data/features/mpnn.parquet         "ProteinMPNN log-odds"      $PY -m src.features.proteinmpnn --device auto
step data/features/mpnnrep.parquet      "ProteinMPNN representations" $PY -m src.features.mpnn_repr --device auto
step data/features/esm35M_pair.parquet  "ESM-2 35M (wt/mut/diff)"   \
     $PY -m src.features.hf_plm --model facebook/esm2_t12_35M_UR50D --tag esm35M_pair --device auto

if [[ $BIG -eq 1 ]]; then
  # The open question in the sequence arm. 35M took 16 minutes on a laptop CPU; 650M is roughly
  # 20x that and has never been run. If a bigger PLM is going to rescue the sequence modality,
  # this is the run that shows it.
  step data/features/esm650M_pair.parquet "ESM-2 650M (wt/mut/diff)" \
       $PY -m src.features.hf_plm --model facebook/esm2_t33_650M_UR50D --tag esm650M_pair --device auto
fi

echo
echo "=== baseline check: the model to beat ==="
$PY -m src.train --model rf --features chem,geom,geomrev,mpnn --name ov_both_geom --n-boot 0
$PY -m pytest tests -q

echo
echo "=== ready. Suggested GPU runs ==="
echo "  $PY -m src.rank_fusion --sweep --device cuda --max-steps 200 --seeds 5"
echo "  $PY -m src.fusion_v3   --sweep --seeds 5"
echo "  $PY -m src.compare_models"
