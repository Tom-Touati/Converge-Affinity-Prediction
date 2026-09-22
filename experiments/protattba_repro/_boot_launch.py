import subprocess, os
os.chdir("/content/perturb")
p = subprocess.Popen("python _perturb_bootstrap.py", shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
for line in p.stdout:
    l = line.rstrip()
    if l and not any(s in l for s in ("newly initialized","should probably","LOAD REPORT",
                                      "UNEXPECTED","MISSING","Loading weights","| Status",
                                      "---","Notes:","- ","Requirement already")):
        print(l, flush=True)
print("rc", p.wait(), flush=True)
