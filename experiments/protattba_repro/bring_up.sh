#!/usr/bin/env bash
# Stand up a fresh Colab VM and resume the ladder where the last one was reclaimed.
#
# Sessions get reclaimed roughly every 40 minutes of GPU time; this is the fifth rebuild.
# Doing it by hand each time is how the v4 configs got lost twice, so it lives here.
#
# Every finished result on disk is pushed back BEFORE the ladder starts. The runner skips
# any configuration that already has five folds, so a rebuild resumes rather than restarts
# and no GPU time is spent retraining what we already have.
#
# Run from Git Bash, not from inside WSL: a loop started with nohup inside `wsl -e bash -lc`
# dies when the distro shuts down behind it.
#
#   bash experiments/protattba_repro/bring_up.sh <session-name>
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
S=${1:?usage: bring_up.sh <session-name>}
PYBIN=${PYBIN:-$HERE/experiments/protattba_repro/.venv_protattba/Scripts/python.exe}
WSLROOT=$(wsl -e wslpath -a "$(cygpath -w "$HERE")" | tr -d '\r')
cd "$HERE"

wsl_() { wsl -e bash -lc "cd '$WSLROOT' && $*"; }
push() {
  if wsl_ "timeout 900 colab upload -s $S '$1' '/content/perturb/$2'" >/dev/null 2>&1
  then echo "  ok   $2"; else echo "  FAIL $2"; fi
}

echo "== creating session $S =="
# A session that never came up must stop the script. Without this it pushed 21 files into
# nothing, reported "launched" and started a collector against a session that did not
# exist -- every step after the failure "succeeded" while doing nothing at all.
if ! wsl -e bash -lc "timeout 500 colab new -s $S --gpu T4" 2>&1 | tail -2; then
  echo "FATAL: could not create session $S"; exit 1
fi
cat > /tmp/_mkdirs.py <<'PY'
import os; os.makedirs('/content/perturb/out', exist_ok=True); print('dirs made')
PY
probe=$(wsl_ "timeout 240 colab exec -s $S" < /tmp/_mkdirs.py 2>&1 | tail -3)
if ! echo "$probe" | grep -q "dirs made"; then
  echo "FATAL: session $S is not usable: $probe"; exit 1
fi
echo "  dirs made"

echo "== harness =="
for f in _perturb_bootstrap.py _esm_colab.py _perturb_v2_colab.py _run_ladder.py \
         _crops.tgz _mpnn_cache.tgz _dist_cache.tgz; do
  push "experiments/protattba_repro/$f" "$f"
done
push src/perturb/model_v2.py model_v2.py
push src/perturb/model_simple.py model_simple.py
push experiments/protattba_repro/cache/perturb_rows.parquet perturb_rows.parquet
push experiments/protattba_repro/cache/project_sequences.parquet project_sequences.parquet
[ -f data/features/chem_perturb.parquet ] && \
  push data/features/chem_perturb.parquet chem_perturb.parquet

echo "== results pushed back, so the runner resumes rather than restarts =="
# PARTIAL runs go back too, not just complete ones. The trainer already skips any
# (fold, seed) it finds in the results file, so pushing a one-fold table back turns a
# rebuild into a resume. Sessions last about 40 minutes of GPU and the ESM bootstrap
# eats 30 of them, so a five-fold run does NOT fit in one session -- without this the
# no-PCA run was lost twice at exactly the same fold.
n=0; part=0
for d in reports/perturb_*/; do
  e=$(basename "$d" | sed 's/^perturb_//')
  f="$d/${e}_results.csv"
  [ -f "$f" ] || continue
  k=$(tail -n +2 "$f" | wc -l | tr -d ' ')
  [ "${k:-0}" -ge 1 ] || continue
  # The two files are grabbed as separate transfers, so the OOF table can be a fold
  # AHEAD of the results table; resuming from that pair scores a fold twice.
  [ -f "results/oof/$e.csv" ] &&     "$PYBIN" experiments/protattba_repro/reconcile_partial.py       "$f" "results/oof/$e.csv" >/dev/null 2>&1
  push "$f" "out/${e}_results.csv"
  [ -f "results/oof/$e.csv" ] && push "results/oof/$e.csv" "out/${e}_oof.csv"
  if [ "$k" -ge 5 ]; then n=$((n + 1)); else part=$((part + 1)); echo "    partial $e: $k folds"; fi
done
echo "  $n complete, $part partial configurations pushed back"

echo "== bootstrap, then the ladder =="
wsl_ "timeout 300 colab exec -s $S" <<'PY' 2>&1 | tail -1
import subprocess
from pathlib import Path
R = Path('/content/perturb')
inner = (f'cd {R} && python _perturb_bootstrap.py > {R}/bootstrap.log 2>&1; '
         f'python _run_ladder.py > {R}/ladder.log 2>&1')
subprocess.Popen(['bash', '-lc',
                  f'nohup sh -c {chr(39)}{inner}{chr(39)} > {R}/driver.log 2>&1 &'],
                 start_new_session=True)
print('launched')
PY
# Start the collector HERE, not as a note for the operator to follow. A rebuild once
# launched four configurations with no poller running -- the collector had been killed
# separately -- and the session was reclaimed before anything was pulled, losing all four.
# Nothing should be able to train on that VM without something collecting it.
# `pkill -f` cannot see Windows processes from Git Bash: it matched nothing every time,
# silently, and each rebuild leaked another collector. Sixteen were found running at once,
# all polling `colab download` in a loop against sessions that no longer existed. Kill by
# PID through PowerShell, which can, and print the count so a failure is visible.
powershell -NoProfile -ExecutionPolicy Bypass -File \n  "$(cygpath -w "$HERE/experiments/protattba_repro/kill_collectors.ps1")"
sleep 1
SESSION=$S nohup bash "$HERE/experiments/protattba_repro/pull_ladder.sh" 180 \
  > /tmp/pull_$S.log 2>&1 &
echo
echo "collector started against $S (log: /tmp/pull_$S.log)"
