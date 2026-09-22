import os, shutil
os.makedirs("/content/perturb/out", exist_ok=True)
# the ESM cache was extracted earlier into /content/protattba_repro; reuse it in place
for f in ("esm2_650m_tokens.npy", "esm2_650m_index.pkl"):
    src, dst = f"/content/protattba_repro/{f}", f"/content/perturb/{f}"
    if os.path.exists(src) and not os.path.exists(dst):
        os.link(src, dst) if os.stat(src).st_dev == os.stat("/content/perturb").st_dev else shutil.copy(src, dst)
if not os.path.exists("/content/perturb/model") and os.path.exists("/content/protattba_repro/model"):
    os.symlink("/content/protattba_repro/model", "/content/perturb/model")
print("ready:", sorted(os.listdir("/content/perturb")))
