#!/usr/bin/env bash
# Drive a Colab GPU session through a list of stages, from WSL.
#
#   bash scripts/colab_run.sh direct           # bootstrap, push features, run, collect
#   bash scripts/colab_run.sh ablate aug       # several stages in order
#
# Three things this exists to get right, each learned the hard way:
#
#  1. **Nothing here can be gated on an exit code.** `colab exec` returns 0 even when the
#     remote script raises, and `colab status` returns 0 for a session that does not exist.
#     Every check below reads output text instead. An earlier runner gated on $? and reported
#     four clean stages having done nothing at all.
#  2. **Download after every stage.** A VM was reclaimed once between the sweep finishing and
#     `collect` running, which lost the history and figures while keeping only what had been
#     printed to the log.
#  3. **Push the cached features instead of re-extracting them.** /content is wiped with the
#     session, so a fresh VM would otherwise spend ~20 minutes rebuilding parquet files that
#     already exist locally. setup_remote.sh skips any step whose output is present, so
#     uploading them first turns `extract` into a no-op.
set -uo pipefail

REPO=/mnt/c/Users/tomto/workspaces/converge_bind
LOG=$REPO/../colab_$(date -u +%H%M%S).log
OUT=$REPO/reports_colab
SESSION=${SESSION:-cb}
GPU=${GPU:-T4}
REMOTE=/content/converge_bind

# Only what the current models actually read. esm650M_pair is 19 MB and used by one
# experiment, so it is pushed only when PUSH_BIG=1.
FEATURES=(chem geom geomrev mpnn mpnnrep esm35M_pair)
[[ "${PUSH_BIG:-0}" == "1" ]] && FEATURES+=(esm650M_pair)

cd "$REPO" || exit 1
mkdir -p "$OUT"
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure () {
  timeout 90 colab sessions 2>&1 | grep -q "^\[$SESSION\]" && return 0
  say "allocating $GPU"
  timeout 300 colab new -s "$SESSION" --gpu "$GPU" >>"$LOG" 2>&1
  timeout 90 colab sessions 2>&1 | grep -q "^\[$SESSION\]"
}

stage () {  # stage <name> <timeout-seconds>
  ensure || { say "NO SESSION for $1"; return 1; }
  { printf 'import sys; sys.argv = ["colab_job.py", "%s"]\n' "$1"; cat scripts/colab_job.py; } \
      > "/tmp/job_$1.py"
  say "### $1"
  timeout "$2" colab exec -s "$SESSION" -f "/tmp/job_$1.py" >>"$LOG" 2>&1
  if tail -80 "$LOG" | grep -q "STAGE_OK $1"; then say "OK $1"; else say "FAILED $1"; fi
}

push_features () {
  ensure || return 1
  say "### push cached features (skips re-extraction)"
  timeout 120 colab exec -s "$SESSION" >>"$LOG" 2>&1 <<'PY'
import os
os.makedirs("/content/converge_bind/data/features", exist_ok=True)
print("feature dir ready")
PY
  for f in "${FEATURES[@]}"; do
    local src="data/features/$f.parquet"
    [[ -f "$src" ]] || { say "  (missing locally: $f)"; continue; }
    if timeout 600 colab upload -s "$SESSION" "$src" "$REMOTE/$src" >>"$LOG" 2>&1; then
      say "  pushed $f ($(du -h "$src" | cut -f1))"
    else
      say "  upload FAILED for $f -- extract will rebuild it"
    fi
  done
}

grab () {  # collect + download; never let a reclaimed VM cost the results
  stage collect 900
  local dest="$OUT/reports_$1.tgz"
  if timeout 900 colab download -s "$SESSION" /content/reports.tgz "$dest" >>"$LOG" 2>&1; then
    say "SAVED $(basename "$dest") ($(du -h "$dest" 2>/dev/null | cut -f1))"
  else
    say "download FAILED for $1"
  fi
}

: > "$LOG"
say "start $(git rev-parse --short HEAD)  stages: $*"
stage bootstrap 1200
push_features
stage extract 5400
grab extract
for st in "$@"; do
  stage "$st" 14400
  grab "$st"
done
say "stopping session"
timeout 300 colab stop -s "$SESSION" >>"$LOG" 2>&1 && say "stopped" || say "STOP FAILED - CHECK"
say "done -- log at $LOG"
