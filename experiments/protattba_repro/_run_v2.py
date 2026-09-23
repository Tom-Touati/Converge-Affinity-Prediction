import subprocess, os
os.chdir("/content/perturb")
cmd = ("nohup sh -c '"
       "python _perturb_v2_colab.py --exp v2_full --folds 0 1 2 3 4 --seeds 0 "
       "  >> out/v2_full.log 2>&1; "
       "python _perturb_v2_colab.py --exp v2_full_cluster --rows perturb_rows_cluster.parquet "
       "  --folds 0 1 2 3 --seeds 0 >> out/v2_full_cluster.log 2>&1"
       "' > out/v2_driver.log 2>&1 &")
subprocess.Popen(cmd, shell=True)
print("launched detached", flush=True)
