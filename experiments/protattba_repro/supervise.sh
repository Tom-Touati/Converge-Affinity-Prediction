#!/usr/bin/env bash
# Keep one Colab session training a fixed queue, across reclaims, unattended.
#
# Sessions are reclaimed after roughly an hour of GPU. Every run in this project longer than
# that has therefore needed a human to notice, rebuild, re-upload and relaunch -- which is
# why the no-PCA run was lost twice at the same fold before fold-level resume existed. With
# resume in place the only thing still missing is somebody to press the button, and that is
# what this does.
#
# It does NOT decide anything. It rebuilds the same session name with the same queue and
# lets `_run_ladder.py` skip whatever already has its full complement of (fold, seed) rows.
# Progress accumulates in reports/ either way, so a supervisor that dies loses nothing that
# the collector has already pulled.
#
#   SESSION=np11 DRIVER=_pair_np11.py EXPS="a b c" \
#     bash experiments/protattba_repro/supervise.sh [check_seconds]
#
# Run from Git Bash, not inside WSL: a loop started with nohup inside `wsl -e bash -lc` dies
# when the distro shuts down behind it.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION=${SESSION:?set SESSION}
DRIVER=${DRIVER:?set DRIVER}
EXPS=${EXPS:?set EXPS}
EVERY=${1:-420}
WSLROOT=$(wsl -e wslpath -a "$(cygpath -w "$HERE")" | tr -d '\r')
cd "$HERE"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# "Alive" has to mean PROVISIONED, not merely responding. A fresh session answers exec
# immediately while holding none of the caches, and a supervisor that accepted that would see
# "alive but not training" forever, re-upload five code files every cycle, and never
# bootstrap. The rows table is the cheapest thing that exists only after bring_up has run.
alive() {
  cat > "$HERE/experiments/protattba_repro/_ping.py" <<'PY'
from pathlib import Path
print("ok" if (Path("/content/perturb") / "perturb_rows.parquet").exists() else "bare")
PY
  wsl -e bash -lc "cd '$WSLROOT' && timeout 150 colab exec -s $SESSION     -f experiments/protattba_repro/_ping.py 2>&1" | grep -q '^ok'
}

training() {  # is anything actually on the GPU?
  wsl -e bash -lc "cd '$WSLROOT' && timeout 150 colab exec -s $SESSION \
    -f experiments/protattba_repro/_ping_train.py 2>&1" | grep -q 'TRAINING yes'
}

# Whether a collector is watching THIS session. Counting pull_ladder processes cannot answer
# that: the session is passed by environment, so every collector looks identical on the
# command line, and with two supervisors running each would see the other's and conclude its
# own was fine. The collector writes a line per pass into its own log, so a log nobody has
# touched recently means nobody is pulling this session.
collector_up() {
  local log="/tmp/pull_${SESSION}.log"
  [ -f "$log" ] || { echo 0; return; }
  local now=$(date +%s)
  local mt=$(stat -c %Y "$log" 2>/dev/null || echo 0)
  [ $(( now - mt )) -lt 900 ] && echo 1 || echo 0
}

launch() {   # push the current code and start the queue
  for f in "src/perturb/model_simple.py:model_simple.py" \
           "src/perturb/model_v2.py:model_v2.py" \
           "experiments/protattba_repro/_run_ladder.py:_run_ladder.py" \
           "experiments/protattba_repro/_perturb_v2_colab.py:_perturb_v2_colab.py" \
           "experiments/protattba_repro/$DRIVER:_pair_driver.py"; do
    wsl -e bash -lc "cd '$WSLROOT' && timeout 300 colab upload -s $SESSION \
      '${f%%:*}' '/content/perturb/${f##*:}'" >/dev/null 2>&1 \
      || say "  upload FAILED ${f##*:}"
  done
  wsl -e bash -lc "cd '$WSLROOT' && timeout 300 colab exec -s $SESSION \
    -f experiments/protattba_repro/_launch_pair.py 2>&1" | grep -q launched \
    && say "  queue launched" || say "  launch FAILED"
}

# the probe the training() check runs, written once
cat > "$HERE/experiments/protattba_repro/_ping_train.py" <<'PY'
import subprocess
p = subprocess.run(["bash", "-lc", "pgrep -af '_perturb_v2_colab|_pair_driver'"],
                   capture_output=True, text=True).stdout
print("TRAINING", "yes" if p.strip() else "no")
PY

say "supervising $SESSION with $DRIVER, queue: $EXPS"
while true; do
  if ! alive; then
    say "$SESSION is gone -- rebuilding"
    EXPS="$EXPS" bash "$HERE/experiments/protattba_repro/bring_up.sh" "$SESSION" \
      > "/tmp/bringup_${SESSION}.log" 2>&1
    if grep -q "collector started" "/tmp/bringup_${SESSION}.log"; then
      say "  rebuilt; pushing the queue"
      launch
    else
      say "  rebuild FAILED -- see /tmp/bringup_${SESSION}.log; retrying next cycle"
    fi
  elif ! training; then
    # session alive but idle: either the queue finished or the driver died
    say "$SESSION alive but idle -- relaunching the queue"
    launch
  else
    n=""
    for e in $EXPS; do
      c=$(tail -n +2 "reports/perturb_$e/${e}_results.csv" 2>/dev/null | wc -l | tr -d ' ')
      n="$n $e:${c:-0}"
    done
    say "$SESSION training —$n"
  fi

  # bring_up kills collectors; make sure one is still watching this session
  if [ "$(collector_up)" = "0" ]; then
    say "  no collector -- starting one"
    EXPS="$EXPS" SESSION="$SESSION" nohup bash \
      "$HERE/experiments/protattba_repro/pull_ladder.sh" 150 \
      > "/tmp/pull_${SESSION}.log" 2>&1 &
  fi
  sleep "$EVERY"
done
