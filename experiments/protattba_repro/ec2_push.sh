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

# 3. the big one
say "esm token store, 823 MB -- this is the slow part"
scp -q -i "$KEY" -o StrictHostKeyChecking=no \
  "$SRC/.perturb_local/esm2_650m_tokens.npy" ubuntu@"$IP":/home/ubuntu/perturb/ \
  && say "  ok esm tokens" || say "  FAILED esm tokens"

say "verifying"
$SSH ubuntu@"$IP" 'cd /home/ubuntu/perturb && du -sh . && ls | tr "\n" " "' 2>/dev/null
say "PUSH DONE"
