"""Run a group of configurations on one EC2 box, concurrently, until they are done.

Unlike the Colab driver this does not have to survive anything: the instance is ours and is
not reclaimed, so there is no resume dance, no supervisor and no collector. It just runs the
names it is given and writes results where PERTURB_ROOT points.

Concurrency is set from the vCPU count, not the GPU. These models leave a T4 at 0-4%
utilisation -- they are data-loader bound -- so the limit is cores, and g4dn.xlarge has four.
Three at once was what Colab sustained on the same shape of machine.

    PERTURB_ROOT=/home/ubuntu/perturb /home/ubuntu/venv/bin/python ec2_run.py a b c
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PERTURB_ROOT", "/home/ubuntu/perturb"))


def main() -> None:
    names = sys.argv[1:]
    if not names:
        raise SystemExit("usage: ec2_run.py <exp> [exp ...]")
    ROOT.joinpath("out").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PERTURB_ROOT=str(ROOT))

    procs = []
    for n in names:
        log = open(ROOT / f"{n}.log", "w")
        procs.append((n, subprocess.Popen(
            [sys.executable, str(ROOT / "_run_ladder.py"), n],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)))
        print(f"started {n}", flush=True)

    t0 = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(60)
        done = []
        for n, _ in procs:
            f = ROOT / "out" / f"{n}_results.csv"
            k = len(f.read_text().strip().splitlines()) - 1 if f.exists() else 0
            done.append(f"{n}:{k}")
        print(f"[{(time.time()-t0)/60:5.1f} min] " + " ".join(done), flush=True)

    for n, p in procs:
        print(f"{n} rc {p.returncode}", flush=True)
    print("GROUP DONE", flush=True)


if __name__ == "__main__":
    main()
