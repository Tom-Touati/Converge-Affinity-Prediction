import subprocess, os
os.makedirs("/content/protattba_repro", exist_ok=True)
os.chdir("/content/protattba_repro")
p = subprocess.Popen("python _esm_colab.py", shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
for line in p.stdout:
    line = line.rstrip()
    if any(k in line for k in ("rows,", "tokens,", "seqs", "DONE", "Error", "Traceback")):
        print(line, flush=True)
print("rc", p.wait(), flush=True)
