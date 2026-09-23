"""Run configurations two at a time in one session.

These models are 50-80k parameters on ~85-token crops and the GPU sat at 0-4% utilisation
running one of them, so they are data-loader bound rather than compute bound: a second
process on the same GPU costs little. Sessions are reclaimed after roughly an hour, so
two at a time is the difference between finishing a pair and finishing half of one.
"""
import os, subprocess, sys, time
from pathlib import Path

R = Path("/content/perturb")
PAIRS = [['gf_reg2', 'nopca_reg2'], ['gf_reg3', 'nopca_reg3']]

# whatever the bring-up launched, stop it -- this session runs only what is pinned here
for pat in ("_split_driver", "_run_ladder", "_perturb_v2_colab"):
    for line in subprocess.run(["bash", "-lc", "pgrep -af " + pat],
                               capture_output=True, text=True).stdout.splitlines():
        pid = int(line.split()[0])
        if pid != os.getpid():
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
time.sleep(3)

while True:                                   # the ESM cache must exist first
    b = R / "bootstrap.log"
    if b.exists() and "READY" in b.read_text():
        break
    time.sleep(15)

for pair in PAIRS:
    procs = []
    for name in pair:
        f = open(R / (name + ".log"), "w")
        procs.append((name, subprocess.Popen(
            [sys.executable, "_run_ladder.py", name], cwd=R,
            stdout=f, stderr=subprocess.STDOUT)))
    print("started", [n for n, _ in procs], flush=True)
    for n, p in procs:
        print(n, "rc", p.wait(), flush=True)
print("PAIRS DONE", flush=True)
