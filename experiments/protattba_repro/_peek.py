import os
for f in ("out/full.log", "out/full_cluster.log"):
    p = f"/content/perturb/{f}"
    if os.path.exists(p):
        lines = [l for l in open(p).read().split("\n") if l.strip() and not any(
            s in l for s in ("newly initialized","should probably","LOAD REPORT","UNEXPECTED",
                             "MISSING","Loading weights","| Status","---","Notes:","- "))]
        print(f"== {f} ==")
        for l in lines[-6:]: print("  ", l)
print("results:", sorted(os.listdir("/content/perturb/out")))
