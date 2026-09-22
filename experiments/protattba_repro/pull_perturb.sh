#!/usr/bin/env bash
# Pull perturbation results off the VM while training runs.
#
# `colab download` works while the kernel is BUSY, which is what lets the dashboard show a
# live run. The exec CLIENT times out long before the remote process finishes, so polling is
# the only way to see progress -- and, more importantly, the only way results survive a
# session being reclaimed mid-run.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION=${SESSION:-pb}
EXP=${1:-full}
EVERY=${2:-180}
mkdir -p "$HERE/reports/perturb_$EXP" "$HERE/results/oof"
while true; do
  timeout 180 colab download -s "$SESSION" "/content/perturb/out/perturb_$EXP/history.csv" \
    "$HERE/reports/perturb_$EXP/history.csv" >/dev/null 2>&1
  timeout 180 colab download -s "$SESSION" "/content/perturb/out/${EXP}_results.csv" \
    "$HERE/reports/perturb_$EXP/${EXP}_results.csv" >/dev/null 2>&1
  timeout 300 colab download -s "$SESSION" "/content/perturb/out/${EXP}_oof.csv" \
    "$HERE/results/oof/${EXP}.csv" >/dev/null 2>&1
  timeout 120 colab download -s "$SESSION" "/content/perturb/out/rank_fusion_sweep.csv" \
    "$HERE/reports/rank_fusion_sweep.csv" >/dev/null 2>&1
  n=$(tail -n +2 "$HERE/reports/perturb_$EXP/${EXP}_results.csv" 2>/dev/null | wc -l)
  e=$(tail -n +2 "$HERE/reports/perturb_$EXP/history.csv" 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%S)] $EXP: $n (fold,seed) done, $e epochs logged"
  [[ $n -ge 15 ]] && { echo "all folds and seeds held"; break; }
  sleep "$EVERY"
done
