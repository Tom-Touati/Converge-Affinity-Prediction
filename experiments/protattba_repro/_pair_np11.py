"""np11: the classification table, three configurations at a time."""
import os, subprocess, sys, time
from pathlib import Path

R = Path("/content/perturb")
PAIRS = [["cls_nostruct", "cls_concat", "cls_gated"],
         ["cls_film", "cls_xattn", "cls_xattn_rev"],
         ["cls_abag_xattn", "cls_areapool", "cls_sitepool"]]

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
            [sys.executable, "_run_ladder.py", name], cwd=R, stdout=f,
            stderr=subprocess.STDOUT)))
    print("started", [n for n, _ in procs], flush=True)
    for n, p in procs:
        print(n, "rc", p.wait(), flush=True)
print("PAIRS DONE", flush=True)
