"""Run the v3 ablation ladder on the VM, one configuration after another.

Sequential inside a single process, so nothing polls for anything. The previous attempt
chained jobs with ``while pgrep -f ...``; each waiter's pattern matched the OTHER waiter's
own command line, so neither ever saw the queue empty and the GPU sat idle. Here the order
is just the order of the list.

Each job is launched with subprocess and an argument LIST, never a shell string, so the
JSON in --overrides needs no quoting and cannot be mangled by a shell.

Ordered by what is learned per minute, because a session can be reclaimed at any point:
the baseline and the two questions that motivated the work come first, the controls after.

    python _run_ladder.py            # everything still missing a results file
    python _run_ladder.py v3_site    # just these
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/perturb")
OUT = ROOT / "out"

#: (name, config overrides). Every job also gets --clip 4 and --select-on per_complex.
LADDER = [
    # the new baseline: unchanged pooling, but with the shared-dropout fix, the wider
    # validation split and per-complex selection. Everything else is measured against it.
    ("v3_base", {}),
    # the question that started this: does pooling at the site beat averaging the crop
    ("v3_site", {"pool": "site"}),
    # the control for the dropout fix. Same as v3_base with the branches drawing their own
    # masks again, which is what every earlier run did.
    ("v3_indep", {"shared_branch_dropout": False}),
    # regularisation, against 85 parameters per training row
    ("v3_reg", {"input_noise": 0.1, "feature_dropout": 0.2}),
    # keep the global view as well as the site, at 59,281 parameters instead of 51,089
    ("v3_sitemean", {"pool": "site_mean"}),
    # the reverse-mutation augmentation sets the training target's mean to exactly zero
    ("v3_noaug", {"_augment": False}),
    # the two that looked best, together
    ("v3_site_reg", {"pool": "site", "input_noise": 0.1, "feature_dropout": 0.2}),

    # --- v4: BLOSUM out, and the delta path live from step 0.
    # gamma and beta are zero-initialised, so (1 + gamma(d)) * t + beta(d) == t exactly and
    # the ESM delta contributes NOTHING at step 0 (measured: 0.000000) while branch.mut
    # ships with ordinary init and injects BLOSUM immediately (0.048477). The model is
    # therefore a BLOSUM predictor with learned complex context, and BLOSUM is the one
    # module carrying more gradient per parameter than attention while being 5% of the
    # weights. Dropping it without also lifting the FiLM off zero would leave the model
    # with no mutation signal at all for its first epochs, so the two go together.
    ("v4_noblosum", {"use_blosum": False, "film_init": 0.02}),
    # with the regularisation that v3_reg showed is doing real work
    ("v4_noblosum_reg", {"use_blosum": False, "film_init": 0.02,
                         "input_noise": 0.1, "feature_dropout": 0.2}),
]


def run(name: str, ov: dict) -> int:
    cmd = [sys.executable, "_perturb_v2_colab.py", "--exp", name,
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
        # A finished job leaves its results file; the trainer resumes per fold anyway, so
        # re-running one is cheap, but skipping keeps a restarted ladder moving.
        done = OUT / f"{name}_results.csv"
        if done.exists() and len(done.read_text().strip().splitlines()) >= 6:
            print(f"=== {name}: already has 5 folds, skipping ===", flush=True)
            continue
        run(name, ov)
    print("\nLADDER DONE", flush=True)


if __name__ == "__main__":
    main()
