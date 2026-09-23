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
WSLROOT=$(wsl -e wslpath -a "$(cygpath -w "$HERE")" | tr -d '\r')
cd "$HERE"

wsl_() { wsl -e bash -lc "cd '$WSLROOT' && $*"; }
push() {
  if wsl_ "timeout 900 colab upload -s $S '$1' '/content/perturb/$2'" >/dev/null 2>&1
  then echo "  ok   $2"; else echo "  FAIL $2"; fi
}

echo "== creating session $S =="
wsl -e bash -lc "timeout 500 colab new -s $S --gpu T4" 2>&1 | tail -2
wsl_ "timeout 240 colab exec -s $S" <<'PY' 2>&1 | tail -1
import os; os.makedirs('/content/perturb/out', exist_ok=True); print('dirs made')
PY

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

echo "== finished results, so the runner skips them =="
n=0
for d in reports/perturb_*/; do
  e=$(basename "$d" | sed 's/^perturb_//')
  f="$d/${e}_results.csv"
  [ -f "$f" ] || continue
  [ "$(tail -n +2 "$f" | wc -l | tr -d ' ')" -ge 5 ] || continue
  push "$f" "out/${e}_results.csv"
  [ -f "results/oof/$e.csv" ] && push "results/oof/$e.csv" "out/${e}_oof.csv"
  n=$((n + 1))
done
echo "  $n finished configurations pushed back"

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
echo
echo "now: SESSION=$S bash experiments/protattba_repro/pull_ladder.sh 180 &"
