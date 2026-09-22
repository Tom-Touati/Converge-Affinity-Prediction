"""Finish the by-complex run (resuming from 4 folds) then run the cluster split.

Launched detached with nohup on the VM so it is not tied to an exec client: the client times
out long before either run finishes, and a dropped client must not stop the training.
"""
import subprocess, os
os.chdir("/content/perturb")
cmd = ("nohup sh -c '"
       "python _perturb_colab.py --exp full --folds 0 1 2 3 4 --seeds 0 "
       "  >> out/full.log 2>&1; "
       "python _perturb_colab.py --exp full_cluster --rows perturb_rows_cluster.parquet "
       "  --folds 0 1 2 3 --seeds 0 >> out/full_cluster.log 2>&1"
       "' > out/driver.log 2>&1 &")
subprocess.Popen(cmd, shell=True)
print("launched detached; logs at /content/perturb/out/*.log", flush=True)
