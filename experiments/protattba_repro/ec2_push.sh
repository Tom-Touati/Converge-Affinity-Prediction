#!/usr/bin/env bash
# Put everything a training run needs onto one EC2 box.
#
# The caches are gitignored and live only in the worktree that built them, so this copies
# from there rather than from the checkout this script sits in. The ESM token store is 823 MB
# and is the whole cost of this transfer; everything else together is under 30 MB.
#
# The trainer already reads PERTURB_ROOT from the environment, so nothing needs patching --
# it looks in /home/ubuntu/perturb here instead of /content/perturb on Colab.
#
#   bash ec2_push.sh <ip> [src-worktree]
set -uo pipefail
IP=${1:?usage: ec2_push.sh <ip> [src]}
SRC=${2:-/c/Users/tomto/workspaces/converge_bind/.claude/worktrees/baseline-protattba-repro-82952e}
KEY=${KEY:-C:/Users/tomto/.ssh/tom_aws.pem}
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20"
say() { echo "[$(date -u +%H:%M:%S)] $IP  $*"; }

$SSH ubuntu@"$IP" 'mkdir -p /home/ubuntu/perturb/out' 2>/dev/null

# 1. the small things first, so a slow ESM transfer cannot hold up a smoke test
say "harness + small caches"
for f in "$SRC/experiments/protattba_repro/_perturb_v2_colab.py:_perturb_v2_colab.py" \
         "$SRC/experiments/protattba_repro/_run_ladder.py:_run_ladder.py" \
         "$SRC/src/perturb/model_simple.py:model_simple.py" \
         "$SRC/src/perturb/model_v2.py:model_v2.py" \
         "$SRC/.perturb_local/perturb_rows.parquet:perturb_rows.parquet" \
         "$SRC/.perturb_local/perturb_crops.npz:perturb_crops.npz" \
         "$SRC/.perturb_local/esm2_650m_index.pkl:esm2_650m_index.pkl" \
         "$SRC/data/features/chem_perturb.parquet:chem_perturb.parquet"; do
  src="${f%%:*}"; dst="${f##*:}"
  [ -f "$src" ] || { say "  MISSING $src"; continue; }
  scp -q -i "$KEY" -o StrictHostKeyChecking=no "$src" ubuntu@"$IP":/home/ubuntu/perturb/"$dst" \
    && say "  ok $dst" || say "  FAILED $dst"
done

# 2. the per-residue ProteinMPNN cache: many small files, so tar it through the pipe rather
#    than paying an scp round trip per complex
say "mpnn cache (streamed as tar)"
tar -C "$SRC/data/features" -cf - mpnn_per_residue \
  | $SSH ubuntu@"$IP" 'tar -C /home/ubuntu/perturb -xf -' \
  && say "  ok mpnn_per_residue" || say "  FAILED mpnn"

# 3. the big one, in chunks.
#    scp of a single 823 MB file restarts from zero on any dropped connection, and rsync is
#    not available in Git Bash on Windows. Splitting means a failure costs one chunk, and a
#    re-run skips the chunks that already landed with the right size.
BIG="$SRC/.perturb_local/esm2_650m_tokens.npy"
say "esm token store, $(du -h "$BIG" | cut -f1) -- the slow part, sent in 100 MB chunks"
WANT=$(stat -c %s "$BIG")
HAVE=$($SSH ubuntu@"$IP" 'stat -c %s /home/ubuntu/perturb/esm2_650m_tokens.npy 2>/dev/null' 2>/dev/null | tr -d '
')
if [ "${HAVE:-0}" = "$WANT" ]; then
  say "  already present and the right size, skipping"
else
  TMP=$(mktemp -d)
  split -b 100m "$BIG" "$TMP/esm."
  n=$(ls "$TMP" | wc -l | tr -d ' ')
  say "  $n chunks"
  $SSH ubuntu@"$IP" 'mkdir -p /home/ubuntu/perturb/.esm_parts' 2>/dev/null
  i=0
  for c in "$TMP"/esm.*; do
    i=$((i + 1)); b=$(basename "$c"); want=$(stat -c %s "$c")
    got=$($SSH ubuntu@"$IP" "stat -c %s /home/ubuntu/perturb/.esm_parts/$b 2>/dev/null" 2>/dev/null | tr -d '
')
    if [ "${got:-0}" = "$want" ]; then say "  $i/$n $b already there"; continue; fi
    for attempt in 1 2 3; do
      scp -q -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20           "$c" ubuntu@"$IP":/home/ubuntu/perturb/.esm_parts/ && break
      say "  $i/$n $b attempt $attempt failed"; sleep 5
    done
    say "  $i/$n $b sent"
  done
  rm -rf "$TMP"
  say "  reassembling on the instance"
  $SSH ubuntu@"$IP" 'cd /home/ubuntu/perturb && cat .esm_parts/esm.* > esm2_650m_tokens.npy      && rm -rf .esm_parts' 2>/dev/null
  got=$($SSH ubuntu@"$IP" 'stat -c %s /home/ubuntu/perturb/esm2_650m_tokens.npy 2>/dev/null' 2>/dev/null | tr -d '
')
  # a truncated token store fails later and confusingly, so it is checked here
  [ "$got" = "$WANT" ] && say "  size verified: $got bytes"                        || say "  SIZE MISMATCH want=$WANT got=${got:-none}"
fi

say "verifying"
$SSH ubuntu@"$IP" 'cd /home/ubuntu/perturb && du -sh . && ls | tr "\n" " "' 2>/dev/null
say "PUSH DONE"
