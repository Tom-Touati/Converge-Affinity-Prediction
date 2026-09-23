"""Run the ablation ladder on the VM, one configuration after another.

Sequential inside a single process, so nothing polls for anything. An earlier attempt
chained jobs with ``while pgrep -f ...``; each waiter's pattern matched the OTHER waiter's
own command line, so neither ever saw the queue empty and the GPU sat idle. Here the order
is just the order of the list.

Each job is launched with subprocess and an argument LIST, never a shell string, so the
JSON in --overrides needs no quoting and cannot be mangled by a shell.

Ordered by what is learned per minute, because sessions ARE reclaimed -- three times so
far, each costing a rebuild. Anything with five folds already on disk is skipped, so a
rebuilt session resumes rather than restarts; push the finished results back before
relaunching and nothing is retrained.

    python _run_ladder.py                 # everything still missing a results file
    python _run_ladder.py abl_no_delta    # just these
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/perturb")
OUT = ROOT / "out"

#: The two changes that actually moved per-complex Pearson, combined. site_mean was the
#: best single config (+0.172) and dropping BLOSUM the second (+0.165); they had never been
#: run together. Every abl_* below is this minus one component, so each line answers "what
#: does this part contribute" against one fixed reference rather than against whatever
#: happened to be best that hour.
BASE = {"pool": "site_mean", "use_blosum": False, "film_init": 0.02}

#: The simple model: LayerNorm + Linear, delta-FiLMed structure, mean over the mutated
#: residues, concat, MLP. No attention, RoPE, distance bias, chain embedding or BLOSUM.
#: It runs first because it is the one remaining hypothesis with a mechanism behind it,
#: and because sessions get reclaimed.
SIMPLE = {"arch": "simple"}

ST = {"arch": "sitetok", "pca_dim": 128, "proj": 64, "subtract": True,
      "use_site_pool": True, "hidden": 128, "layers": 2,
      "input_noise": 0.1, "feature_dropout": 0.2}

REG = {"input_noise": 0.1, "feature_dropout": 0.2}
TT = {"arch": "twotower", "pca_dim": 128, "proc": 128, "hidden": 128, "layers": 2, **REG}

LADDER = [
    # No residual at all: the token does not pass through, so the head's delta becomes
    #     o((attn_mt - attn_wt) @ V)
    # K and V come from the WILD-TYPE structure in both branches, so V is identical and the
    # entire edit has to be expressed as a change in WHERE the sequence looks. The raw
    # sequence difference no longer reaches the head by any path. That is a much stronger
    # claim than weighting the residual down, and it either works or collapses.
    ("st64_xattn_r01", dict(ST, chem_dim=26, cross_attn=True, n_heads=4, mpnn_proj=64,
                            res_pre=0.0, res_post=1.0)),
    # 0.2/0.8 scored +0.208 and 1.0/1.0 scored +0.192; these two are already done and are
    # listed so a resumed ladder does not re-run them.
    ("st64_xattn_r28", dict(ST, chem_dim=26, cross_attn=True, n_heads=4, mpnn_proj=64,
                            res_pre=0.2, res_post=0.8)),
    ("st64_nochem", dict(ST)),
]


def run(name: str, ov: dict) -> int:
    # "arch" selects the model and is a CLI flag, not a config field, so it is lifted out
    # of the overrides before they are handed to the config constructor.
    ov = dict(ov)
    arch = ov.pop("arch", "v2")
    cmd = [sys.executable, "_perturb_v2_colab.py", "--exp", name, "--arch", arch,
           "--folds", "0", "1", "2", "3", "4", "--seeds", "0",
           "--clip", "4", "--select-on", "per_complex",
           "--overrides", json.dumps(ov)]
    log = ROOT / f"{name}.log"
    print(f"\n=== {name} {ov} ===", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"    rc {rc} in {(time.time()-t0)/60:.1f} min -> {log.name}", flush=True)
    if rc != 0:
        print("    tail:", log.read_text()[-400:].replace("\n", " | "), flush=True)
    return rc


def main() -> None:
    want = set(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ov in LADDER:
        if want and name not in want:
            continue
        done = OUT / f"{name}_results.csv"
        if done.exists() and len(done.read_text().strip().splitlines()) >= 6:
            print(f"=== {name}: already has 5 folds, skipping ===", flush=True)
            continue
        run(name, ov)
    print("\nLADDER DONE", flush=True)


if __name__ == "__main__":
    main()
