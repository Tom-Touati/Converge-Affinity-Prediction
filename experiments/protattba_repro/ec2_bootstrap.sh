#!/usr/bin/env bash
# Bring a bare g4dn.xlarge up to "can train the perturbation ladder".
#
# The instances are plain Ubuntu 26.04, not a Deep Learning AMI: no NVIDIA driver, no CUDA,
# no pip, and the system Python is 3.14. Two of those matter.
#
#   * 3.14 is too new for the torch wheels this project needs. Colab set the same trap
#     earlier in this project at 3.13 and it cost hours, so a known-good 3.12 is installed
#     alongside rather than discovered to be missing halfway through an install.
#   * the driver needs the kernel headers for THIS kernel, and a fresh instance is often
#     still running unattended-upgrades, which holds the apt lock. Both are waited for
#     rather than assumed.
#
# Idempotent: every step checks before doing. Safe to re-run after a reboot.
#
#   bash ec2_bootstrap.sh 2>&1 | tee -a /home/ubuntu/bootstrap.log
set -uo pipefail
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "waiting for cloud-init and any apt lock to clear"
cloud-init status --wait >/dev/null 2>&1 || true
for i in $(seq 1 60); do
  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  sleep 10
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "installing the NVIDIA driver (this is the slow step)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
       linux-headers-$(uname -r) build-essential ubuntu-drivers-common >/dev/null
  # autoinstall picks the driver that matches this GPU and kernel; naming a version by hand
  # is how you end up with a driver that will not build against 7.0.0-aws
  sudo ubuntu-drivers install --gpgpu >/dev/null 2>&1 \
    || sudo ubuntu-drivers autoinstall >/dev/null 2>&1 \
    || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver-570-server >/dev/null
  say "driver installed; a reboot is required before it loads"
  touch /home/ubuntu/.needs_reboot
else
  say "driver already present: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  say "installing python3.12 and pip"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
       python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1 \
    || { sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
         sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
         sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
              python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1; }
fi

PY=$(command -v python3.12 || command -v python3.13 || command -v python3)
say "using $PY ($($PY -V 2>&1))"

if [ ! -d /home/ubuntu/venv ]; then
  say "creating the venv"
  $PY -m venv /home/ubuntu/venv 2>/dev/null \
    || { sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv >/dev/null
         $PY -m venv /home/ubuntu/venv; }
fi
V=/home/ubuntu/venv/bin
$V/python -m pip install -q --upgrade pip wheel >/dev/null 2>&1

if ! $V/python -c "import torch" 2>/dev/null; then
  say "installing torch (cu124) and the rest"
  $V/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124 \
    || $V/python -m pip install -q torch
  $V/python -m pip install -q numpy pandas scikit-learn pyarrow transformers joblib scipy
fi

say "python  : $($V/python -V 2>&1)"
say "torch   : $($V/python -c 'import torch;print(torch.__version__)' 2>&1)"
say "cuda ok : $($V/python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
mkdir -p /home/ubuntu/perturb/out
say "BOOTSTRAP DONE"
