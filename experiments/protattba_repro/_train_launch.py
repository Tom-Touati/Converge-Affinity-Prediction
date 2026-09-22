import subprocess, os, sys
os.chdir("/content/perturb")
exp = os.environ.get("EXP", "full")
folds = os.environ.get("FOLDS", "0 1 2 3 4")
seeds = os.environ.get("SEEDS", "0")
ov = os.environ.get("OVERRIDES", "{}")
cmd = f"python _perturb_colab.py --exp {exp} --folds {folds} --seeds {seeds} --overrides '{ov}'"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1, errors="replace")
for line in p.stdout:
    line = line.rstrip()
    if line and not any(s in line for s in ("newly initialized","should probably","LOAD REPORT",
                                            "UNEXPECTED","MISSING","Loading weights","| Status",
                                            "---","Notes:","- ")):
        print(line, flush=True)
print("rc", p.wait(), flush=True)
