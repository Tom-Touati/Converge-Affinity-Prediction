import subprocess, os
os.chdir("/content/perturb")
p = subprocess.Popen("python _perturb_bootstrap.py", shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
for line in p.stdout:
    print(line.rstrip(), flush=True)
print("rc", p.wait(), flush=True)
