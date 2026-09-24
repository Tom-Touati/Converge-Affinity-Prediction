#!/usr/bin/env bash
# Pull EC2 training results into reports/ and refresh the TensorBoard mirror, on a loop.
#
# The instances are not reclaimed, so unlike the Colab collector this is not protecting
# against loss -- it exists so the runs are visible while they are still running, and so the
# scoring tools (which read reports/) work against live results.
#
# Publishing needs a COMPLETE out-of-fold table: a partial one would be scored as though
# those were all the rows, and the per-complex number would be nonsense. That is why a run
# is only published once its results table has every (fold, seed) row it was asked for.
#
#   IPS="1.2.3.4 5.6.7.8" EXPS="cls_film cls_xattn" \
#     bash experiments/protattba_repro/ec2_collect.sh [every_seconds]
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
KEY=${KEY:-C:/Users/tomto/.ssh/tom_aws.pem}
IPS=${IPS:?set IPS}
EXPS=${EXPS:?set EXPS}
EVERY=${1:-120}
ROWS=${ROWS:-10}          # folds x seeds a run needs before it is worth publishing
# The experiment keeps its name on the instance; only the LOCAL run is renamed. Colab already
# holds partial runs under the bare names, trained with a different patience on a machine that
# kept losing them, and silently merging the two sets would produce a run that is neither.
PREFIX=${PREFIX:-ec2_}
PYBIN=${PYBIN:-$HERE/../baseline-protattba-repro-82952e/experiments/protattba_repro/.venv_protattba/Scripts/python.exe}
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
cd "$HERE"

while true; do
  got=""
  for ip in $IPS; do
    for e in $EXPS; do
      L="${PREFIX}${e}"
      mkdir -p "reports/perturb_$L" results/oof
      scp -q -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        ubuntu@"$ip":/home/ubuntu/perturb/out/"${e}"_results.csv \
        "reports/perturb_$L/${L}_results.csv" 2>/dev/null
      scp -q -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        ubuntu@"$ip":/home/ubuntu/perturb/out/"${e}"_oof.csv \
        "results/oof/${L}.csv" 2>/dev/null
      scp -q -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        ubuntu@"$ip":/home/ubuntu/perturb/out/perturb_"${e}"/history.csv \
        "reports/perturb_$L/history.csv" 2>/dev/null
      n=$(tail -n +2 "reports/perturb_$L/${L}_results.csv" 2>/dev/null | wc -l | tr -d ' ')
      [ "${n:-0}" -gt 0 ] && got="$got $L:$n"
      if [ "${n:-0}" -ge "$ROWS" ]; then
        "$PYBIN" -m src.perturb.publish "$L" --name "perturb_$L" --model perturb_v3 \
          >/dev/null 2>&1
      fi
    done
  done

  TB=""; for e in $EXPS; do TB="$TB ${PREFIX}${e}"; done
  "$PYBIN" scripts/to_tensorboard.py $TB >/dev/null 2>&1
  # The running TensorBoard serves the OTHER worktree's runs/tb, so writing only here makes
  # these runs invisible. Copy them across rather than restarting a server the user is using.
  if [ -n "${SERVED_TB:-}" ]; then
    for d in runs/tb/perturb_${PREFIX}*; do
      [ -d "$d" ] && cp -r "$d" "$SERVED_TB/" 2>/dev/null
    done
  fi
  echo "[$(date -u +%H:%M:%S)] $got"
  sleep "$EVERY"
done
