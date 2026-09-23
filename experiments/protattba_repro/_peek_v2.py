import os
for f in ("out/v2_full.log","out/v2_full_cluster.log"):
    p=f"/content/perturb/{f}"
    if os.path.exists(p):
        ls=[l for l in open(p).read().split("\n") if l.strip() and not any(
            s in l for s in ("newly initialized","should probably","LOAD REPORT","UNEXPECTED",
                             "MISSING","Loading weights","| Status","---","Notes:","- "))]
        print(f"== {f} =="); [print("  ",l) for l in ls[-8:]]
print("out:", sorted(os.listdir("/content/perturb/out")))
