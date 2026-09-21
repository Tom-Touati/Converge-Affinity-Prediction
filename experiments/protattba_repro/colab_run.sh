#!/usr/bin/env bash
# Drive the ProtAttBA reproduction on a Colab GPU, from WSL.
#
#   bash experiments/protattba_repro/colab_run.sh                 # everything
#   bash experiments/protattba_repro/colab_run.sh upstream        # one stage
#
# Follows scripts/colab_run.sh rather than inventing a second runner, because the three rules
# it encodes were each learned by losing a run (see HANDOFF.md §7):
#
#  1. Nothing can be gated on an exit code. `colab exec` returns 0 when the remote script
#     raises, and `colab sessions`/`colab status` return 0 for a session that does not exist.
#     Every check below reads output text and greps for a STAGE_OK marker.
#  2. Download after every stage. A VM was reclaimed once between a sweep finishing and
#     `collect` running.
#  3. Upload what already exists rather than rebuilding it remotely -- except the 0.66 GB
#     embedding cache, which is faster to rebuild on a T4 (~3 min) than to upload.
#
# Unlike the project runner this does not clone our repo on the VM: this branch is local, so
# the eight harness files and the vendored S1131 csv are uploaded directly.
set -uo pipefail

REPO=/mnt/c/Users/tomto/workspaces/converge_bind/.claude/worktrees/baseline-protattba-repro-82952e
HERE=$REPO/experiments/protattba_repro
LOG=$HERE/colab_$(date -u +%H%M%S).log
SESSION=${SESSION:-pab}
GPU=${GPU:-T4}
REMOTE=/content/protattba_repro

# The harness. paths_local.py resolves everything relative to its own directory, so the same
# files run unchanged locally and on the VM.
FILES=(metrics.py paths_local.py cached_encoder.py extract_embeddings.py run_cv.py
       compare.py fetch_esm2.py check_encoder_assumptions.py verify_published.py
       crosscheck_embeddings.py crosscheck_local.json colab_job.py)
UPSTREAM_FILES=(upstream/S1131.csv upstream/S1131_published_predictions.csv)

cd "$HERE" || exit 1
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure () {
  timeout 90 colab sessions 2>&1 | grep -q "^\[$SESSION\]" && return 0
  say "allocating $GPU"
  timeout 600 colab new -s "$SESSION" --gpu "$GPU" >>"$LOG" 2>&1
  timeout 90 colab sessions 2>&1 | grep -q "^\[$SESSION\]"
}

push () {
  ensure || return 1
  say "### upload harness"
  timeout 120 colab exec -s "$SESSION" >>"$LOG" 2>&1 <<PY
import os
for d in ("$REMOTE", "$REMOTE/upstream", "$REMOTE/results"):
    os.makedirs(d, exist_ok=True)
print("dirs ready")
PY
  for f in "${FILES[@]}" "${UPSTREAM_FILES[@]}"; do
    if timeout 300 colab upload -s "$SESSION" "$f" "$REMOTE/$f" >>"$LOG" 2>&1; then
      say "  pushed $f"
    else
      say "  upload FAILED for $f"
    fi
  done
}

stage () {  # stage <name> <timeout-seconds>
  ensure || { say "NO SESSION for $1"; return 1; }
  { printf 'import sys, os\nsys.argv = ["colab_job.py", "%s"]\nos.chdir("%s")\n' "$1" "$REMOTE"
    cat colab_job.py; } > "/tmp/pab_$1.py"
  say "### $1"
  timeout "$2" colab exec -s "$SESSION" -f "/tmp/pab_$1.py" >>"$LOG" 2>&1
  if tail -200 "$LOG" | grep -q "STAGE_OK $1"; then say "OK $1"; return 0; fi
  say "FAILED $1  (grep the log: $LOG)"
  return 1
}

grab () {
  stage collect 900
  if timeout 900 colab download -s "$SESSION" /content/protattba_results.tgz \
       "$HERE/results_colab_$1.tgz" >>"$LOG" 2>&1; then
    say "SAVED results_colab_$1.tgz ($(du -h "$HERE/results_colab_$1.tgz" 2>/dev/null | cut -f1))"
    tar xzf "$HERE/results_colab_$1.tgz" -C "$HERE" && say "unpacked into results/"
  else
    say "download FAILED for $1"
  fi
}

: > "$LOG"
STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(upstream honest)

say "start  session=$SESSION gpu=$GPU  stages: ${STAGES[*]}"
push
stage bootstrap 2400 || { say "bootstrap failed, stopping"; exit 1; }
stage verify 900
# Colab's python 3.13 forces a much newer transformers than their pin. Prove the encoder is
# unchanged before spending GPU hours on a run that would otherwise be unattributable.
stage crosscheck 900 || {
  if grep -q "EXCEEDS TOLERANCE" "$LOG"; then
    say "ENCODER DIFFERS across stacks -- the remote run would measure a different encoder"
  else
    say "crosscheck stage errored (not an encoder mismatch); see $LOG"
  fi
  exit 1
}
stage extract 3600 || { say "extract failed, stopping"; exit 1; }
# The run stages are hours long and `colab exec` dropped its connection 35 minutes into the
# first attempt ("RuntimeError: Connection was lost."), losing the whole fold. run_cv.py now
# writes one file per fold and skips folds already on disk, so a retry resumes rather than
# restarts -- which makes retrying the right response to a dropped connection.
for st in "${STAGES[@]}"; do
  for attempt in 1 2 3 4 5; do
    say "--- $st attempt $attempt"
    if stage "$st" 21600; then break; fi
    grab "$st"                      # salvage whatever folds completed before the drop
    if grep -q "STAGE_FAILED" "$LOG" && ! tail -400 "$LOG" | grep -q "Connection was lost"; then
      say "$st failed for a reason other than a dropped connection; not retrying"
      break
    fi
    say "$st lost its connection; resuming from the folds already on disk"
  done
  grab "$st"
done
stage compare 600
grab final
say "stopping session"
timeout 300 colab stop -s "$SESSION" >>"$LOG" 2>&1 && say "stopped" || say "STOP FAILED - CHECK"
say "done -- log at $LOG"
