#!/usr/bin/env bash
# Pull the v3 ladder off the VM and refresh the TensorBoard mirror, on a loop.
#
# `colab download` works while the kernel is BUSY, which is what makes a live view possible
# at all: the exec client times out long before a run finishes, so polling is the only way
# to see progress and the only way results survive the session being reclaimed.
#
# TensorBoard runs with --reload_interval 20, so re-running the mirror is enough -- nothing
# needs restarting and an open page picks the new points up on its own.
#
# Run this from Git Bash, NOT from inside WSL. A loop started with nohup inside a
# `wsl -e bash -lc` invocation dies with it: when the last process in the distro exits, WSL
# shuts the distro down and takes the "detached" job with it. Here the loop is a Windows
# process and only the transfers step into WSL.
#
#   SESSION=pe bash experiments/protattba_repro/pull_ladder.sh [every_seconds]
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION=${SESSION:-pe}
EVERY=${1:-180}
PYBIN=${PYBIN:-$HERE/experiments/protattba_repro/.venv_protattba/Scripts/python.exe}
EXPS="v3_base v3_site v3_indep v3_reg v3_sitemean v3_noaug v3_site_reg"
# the same worktree, as WSL sees it
WSLROOT=$(wsl -e wslpath -a "$(cygpath -w "$HERE")" | tr -d '\r')

cd "$HERE"

grab() {   # grab <remote-path> <local-path>
  wsl -e bash -lc "cd '$WSLROOT' && timeout 240 colab download -s $SESSION '$1' '$2'" \
    >/dev/null 2>&1
}

while true; do
  got=""
  for e in $EXPS; do
    mkdir -p "reports/perturb_$e" results/oof
    grab "/content/perturb/out/${e}_results.csv" "reports/perturb_$e/${e}_results.csv"
    grab "/content/perturb/out/${e}_oof.csv"     "results/oof/${e}.csv"
    grab "/content/perturb/out/perturb_$e/history.csv" "reports/perturb_$e/history.csv"
    n=$(tail -n +2 "reports/perturb_$e/${e}_results.csv" 2>/dev/null | wc -l)
    n=${n// /}
    [ "${n:-0}" -gt 0 ] && got="$got $e:$n"
    # Publishing needs a COMPLETE out-of-fold table. A partial one would be scored as if
    # those were all the rows there are, and the per-complex number would be nonsense.
    if [ "${n:-0}" -ge 5 ]; then
      "$PYBIN" -m src.perturb.publish "$e" --name "perturb_$e" --model perturb_v3 \
        >/dev/null 2>&1
    fi
  done

  "$PYBIN" scripts/to_tensorboard.py v3_ perturb_v2_full E0a_rf_handcrafted_seed0 \
    >/dev/null 2>&1
  echo "[$(date -u +%H:%M:%S)] folds:$got"

  done_n=0
  for e in $EXPS; do
    n=$(tail -n +2 "reports/perturb_$e/${e}_results.csv" 2>/dev/null | wc -l)
    n=${n// /}
    [ "${n:-0}" -ge 5 ] && done_n=$((done_n + 1))
  done
  [ "$done_n" -ge 7 ] && { echo "ladder complete"; break; }
  sleep "$EVERY"
done
