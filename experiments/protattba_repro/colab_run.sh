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
  # makedirs before chdir, or a fresh VM deadlocks: /content is wiped when a session is
  # replaced, so the wrapper chdir'd into a directory that only `bootstrap` would have
  # created, and bootstrap could therefore never run. Every stage then failed in under a
  # second with FileNotFoundError and the retry budget drained without computing anything.
  { printf 'import sys, os\nsys.argv = ["colab_job.py", "%s"]\nos.makedirs("%s", exist_ok=True)\nos.chdir("%s")\n' \
      "$1" "$REMOTE" "$REMOTE"
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
stage bootstrap 2400 || say "bootstrap had trouble; the fold loop will retry it"
stage verify 900
# Colab's python 3.13 forces a much newer transformers than their pin. Prove the encoder is
# unchanged before spending GPU hours on a run that would otherwise be unattributable.
if ! stage crosscheck 900; then
  if grep -q "EXCEEDS TOLERANCE" "$LOG"; then
    say "ENCODER DIFFERS across stacks -- refusing to run, the numbers would not be theirs"
    exit 1
  fi
  say "crosscheck could not run (not an encoder mismatch); continuing, see $LOG"
fi
stage extract 3600 || say "extract had trouble; the fold loop will retry it"
# One fold per exec, with the fold's predictions pulled down as soon as it finishes and any
# folds we already hold pushed back up first. Two things forced this:
#
#   * `colab exec` dropped its connection 35 minutes into a 10-fold call
#     ("RuntimeError: Connection was lost."), and
#   * the session itself was later reclaimed ("Session 'pab' appears to be lost (404/401)").
#
# At ~25 s an epoch a protocol is 3-5 hours, which is longer than a session can be relied on,
# so the unit of work has to be smaller than the run. run_cv.py writes one file per fold and
# reuses any already present, so this loop is resumable across a VM loss: whatever folds are
# on local disk are restored to the fresh VM and only the missing ones are computed.
push_folds () {   # push_folds <protocol>
  local n=0
  shopt -s nullglob
  for f in "$HERE"/results/"$1"_fold*.csv; do
    timeout 180 colab upload -s "$SESSION" "$f" \
      "$REMOTE/results/$(basename "$f")" >>"$LOG" 2>&1 && n=$((n+1))
  done
  shopt -u nullglob
  [[ $n -gt 0 ]] && say "  restored $n completed fold(s) to the VM"
  return 0
}

pull_folds () {   # pull_folds <protocol>
  for k in 0 1 2 3 4 5 6 7 8 9; do
    [[ -f "$HERE/results/$1_fold$k.csv" ]] && continue
    timeout 180 colab download -s "$SESSION" "$REMOTE/results/$1_fold$k.csv" \
      "$HERE/results/$1_fold$k.csv" >>"$LOG" 2>&1 \
      && say "  pulled $1_fold$k.csv"
  done
  return 0
}

first_missing () {   # first_missing <protocol> -> fold index, or 10 when all are held
  for k in 0 1 2 3 4 5 6 7 8 9; do
    [[ -s "$HERE/results/$1_fold$k.csv" ]] || { echo "$k"; return; }
  done
  echo 10
}

for st in "${STAGES[@]}"; do
  # Always work on the FIRST fold we do not hold, and only advance when it lands. An earlier
  # version looped k=1..10 and moved on after three failed attempts, which meant a session
  # that was broken for session reasons -- reclaimed VM, lost connection -- burned through all
  # ten folds in a couple of minutes without computing anything. Progress is now defined by
  # files on local disk, so a bad patch costs attempts rather than folds.
  budget=$(( 10 * 4 ))       # attempts across the whole protocol; ~4 per fold
  while :; do
    k=$(first_missing "$st")
    [[ $k -ge 10 ]] && { say "### $st: all 10 folds held locally"; break; }
    [[ $budget -le 0 ]] && { say "### $st: attempt budget exhausted at fold $k"; break; }
    budget=$((budget - 1))
    say "--- $st fold $k  (attempts left $budget)"

    if ! ensure; then
      say "  no session available; waiting 120s"
      sleep 120
      continue
    fi
    # Re-push the harness every attempt, not once at startup. /content is wiped when a session
    # is replaced, so after a VM swap the eight .py files and the vendored csv are gone and
    # `python run_cv.py` cannot run. Fourteen small uploads cost ~30 s against a ~25 min fold,
    # and being idempotent is worth more here than being minimal.
    push
    push_folds "$st"
    before=$(ls "$HERE"/results/"$st"_fold*.csv 2>/dev/null | wc -l)
    # A fresh VM needs the environment and the cache back before it can train. Both stages are
    # idempotent and skip their own work when the outputs are present -- seconds on a session
    # that has already run them, ~4 min on a brand new VM -- so they are called every time
    # rather than tracked.
    stage bootstrap 2400
    stage extract 3600
    stage "$st:$((k + 1))" 7200
    pull_folds "$st"

    # Back off when an attempt made no progress. Without this a fast-failing session (a swap,
    # a 500 from the contents API) burns the whole budget in a couple of minutes; a fold that
    # is genuinely training takes ~25 min, so a sub-minute attempt means something is wrong
    # rather than slow.
    after=$(ls "$HERE"/results/"$st"_fold*.csv 2>/dev/null | wc -l)
    if [[ "$after" -le "$before" ]]; then
      say "  no progress on fold $k; backing off 90s"
      sleep 90
    fi
  done
  say "### $st: $(ls "$HERE"/results/"$st"_fold*.csv 2>/dev/null | wc -l) of 10 folds on disk"
  grab "$st"
done
stage compare 600
grab final
say "stopping session"
timeout 300 colab stop -s "$SESSION" >>"$LOG" 2>&1 && say "stopped" || say "STOP FAILED - CHECK"
say "done -- log at $LOG"
