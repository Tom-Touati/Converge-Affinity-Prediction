"""np9: the one-layer head ALONE first, then the two-layer concat pair behind it.

Alone rather than paired on purpose. The point of this run is a clean read on whether
removing the second 128x128 block costs anything, and a co-tenant process changes the
epoch timing and the early-stopping point enough to muddy a 0.03-sized comparison.
"""
import os, subprocess, sys, time
from pathlib import Path

R = Path("/content/perturb")
PAIRS = [["cat128_reg2_l1"], ["cat128_reg2", "cat128_reg3"]]

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
print("cleared:", subprocess.run(["bash", "-lc", "pgrep -af _perturb_v2_colab"],
                                 capture_output=True, text=True).stdout.strip() or "none",
      flush=True)

while True:
    b = R / "bootstrap.log"
    if b.exists() and "READY" in b.read_text():
        break
    time.sleep(15)

for group in PAIRS:
    procs = []
    for name in group:
        f = open(R / (name + ".log"), "w")
        procs.append((name, subprocess.Popen(
            [sys.executable, "_run_ladder.py", name], cwd=R,
            stdout=f, stderr=subprocess.STDOUT)))
    print("started", [n for n, _ in procs], flush=True)
    for n, p in procs:
        print(n, "rc", p.wait(), flush=True)
print("PAIRS DONE", flush=True)
