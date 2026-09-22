"""Bring a fresh Colab VM to the point where perturbation training can run.

Sessions get reclaimed and /content is wiped with them, so this is written to be run on a
brand-new VM and to skip any step whose output is already present. Nothing large is uploaded:
the 0.86 GB ESM cache is re-extracted on the GPU in about three minutes, which beats pushing
it over the wire.
"""
import os, subprocess, sys, tarfile, time

ROOT = "/content/perturb"

def sh(cmd, cwd=ROOT):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    for line in p.stdout:
        line = line.rstrip()
        if line and not any(s in line for s in ("newly initialized", "should probably",
                                                "Requirement already", "WARNING:")):
            print(line, flush=True)
    return p.wait()

os.makedirs(ROOT, exist_ok=True)
os.chdir(ROOT)

print("== deps ==", flush=True)
sh("pip install -q biopython joblib 2>&1 | tail -2", cwd="/content")

print("== ESM2-650M weights ==", flush=True)
if not os.path.exists(f"{ROOT}/model/esm2_650m/model.safetensors"):
    from huggingface_hub import snapshot_download
    snapshot_download("facebook/esm2_t33_650M_UR50D", local_dir=f"{ROOT}/model/esm2_650m",
                      allow_patterns=["config.json", "vocab.txt", "tokenizer_config.json",
                                      "special_tokens_map.json", "model.safetensors"])
    print("  downloaded", flush=True)
else:
    print("  already present", flush=True)

print("== unpack caches ==", flush=True)
for tgz in ("_mpnn_cache.tgz", "_dist_cache.tgz"):
    if os.path.exists(f"{ROOT}/{tgz}"):
        with tarfile.open(f"{ROOT}/{tgz}") as t:
            t.extractall(ROOT)
        print(f"  {tgz} ->", sorted(os.listdir(ROOT))[:8], flush=True)

print("== per-residue ESM tokens ==", flush=True)
if os.path.exists(f"{ROOT}/esm2_650m_index.pkl"):
    print("  already present", flush=True)
else:
    rc = sh("python _esm_colab.py")
    print("  extraction rc", rc, flush=True)

print("READY", sorted(os.listdir(ROOT)), flush=True)
