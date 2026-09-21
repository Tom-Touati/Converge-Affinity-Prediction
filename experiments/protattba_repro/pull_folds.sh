#!/usr/bin/env bash
# Pull completed fold predictions off the VM while a run is still executing.
#
# colab download works while the kernel is BUSY, which is the documented capability the
# project's dashboard relies on. The driver only pulls between execs, so a long exec that
# computes several folds leaves them on a VM that can be reclaimed. This runs alongside and
# copies each fold down as soon as it appears, making the results durable independently of
# whatever the driver is doing.
#
#   bash pull_folds.sh upstream 180      # protocol, seconds between sweeps
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO=${1:-upstream}
EVERY=${2:-180}
SESSION=${SESSION:-pab}
mkdir -p "$HERE/results"
while true; do
  for k in 0 1 2 3 4 5 6 7 8 9; do
    dst="$HERE/results/${PROTO}_fold$k.csv"
    [[ -s "$dst" ]] && continue
    if timeout 120 colab download -s "$SESSION" \
         "/content/protattba_repro/results/${PROTO}_fold$k.csv" "$dst" >/dev/null 2>&1; then
      [[ -s "$dst" ]] && echo "[$(date -u +%H:%M:%S)] pulled ${PROTO}_fold$k.csv" || rm -f "$dst"
    else
      rm -f "$dst"        # a failed download must not leave a 0-byte file that looks done
    fi
  done
  n=$(ls "$HERE"/results/${PROTO}_fold*.csv 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%S)] $n/10 folds held locally"
  [[ $n -ge 10 ]] && { echo "all folds held"; break; }
  sleep "$EVERY"
done
