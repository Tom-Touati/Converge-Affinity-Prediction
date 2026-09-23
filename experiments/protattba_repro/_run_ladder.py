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

LADDER = [
    # --- the floor: pool the ESM delta at the mutated residues, regress. No structure,
    # no attention, no FiLM, no crop geometry -- one 512-vector per row. If the rest of
    # the architecture does not clearly beat this, it is not earning its place.
    # Regularised first. Every config in this session that had input noise and feature
    # dropout beat its unregularised twin, and this model sees 512 inputs for 752 training
    # rows, so it is the one most exposed. The plain version follows as the control.
    ("mlp_delta_pca256_reg", {"arch": "mlp", "input_noise": 0.1, "feature_dropout": 0.2}),
    ("mlp_delta_pca256", {"arch": "mlp"}),
    ("mlp_delta_pca256_h128_reg", {"arch": "mlp", "hidden": 128, "layers": 2,
                                   "input_noise": 0.1, "feature_dropout": 0.2}),
    ("mlp_delta_pca128_reg", {"arch": "mlp", "pca_dim": 128,
                              "input_noise": 0.1, "feature_dropout": 0.2}),

    # --- the ESM delta plus the forest's own columns (chem + ProteinMPNN log-likelihood
    # ratios, 26 of them). The forest beats every net here; this separates "the features
    # are better" from "the model class is better". chem_only is the control: the same
    # head on the chemistry ALONE, with no embedding at all.
    ("mlp_delta_chem_reg", {"arch": "mlp", "hidden": 128, "layers": 2, "chem_dim": 26,
                            "input_noise": 0.1, "feature_dropout": 0.2}),
    ("mlp_chem_only", {"arch": "mlp", "hidden": 128, "layers": 2, "chem_dim": 26,
                       "pca_dim": 1, "input_noise": 0.0, "feature_dropout": 0.0}),

    # width 64 (41,792 params) and width 32 (15,008: 25 per training row, against 85 for
    # the attention model, which is what the overfitting measurement asks for)
    ("v5_simple", dict(SIMPLE)),
    ("v5_simple_w32", dict(SIMPLE, width=32, hidden=32)),
    ("v5_simple_w32_reg", dict(SIMPLE, width=32, hidden=32,
                               input_noise=0.1, feature_dropout=0.2)),
    # does ProteinMPNN earn its place once everything else is gone? 16,768 params.
    ("v5_simple_noseqstruct", dict(SIMPLE, width=32, hidden=32, use_structure=False)),

    # --- the combination, and the combination plus regularisation
    ("v4_sm_nb", dict(BASE)),
    ("v4_sm_nb_reg", dict(BASE, input_noise=0.1, feature_dropout=0.2)),

    # --- the two configurations a reclaimed session never got to
    ("v3_noaug", {"_augment": False}),
    ("v3_site_reg", {"pool": "site", "input_noise": 0.1, "feature_dropout": 0.2}),

    # --- component ablations, each removing one thing from BASE.
    # Does ProteinMPNN contribute anything, and is it the structure or just the token
    # count? shuffle is the control: same tensors, positions permuted.
    ("abl_no_structure", dict(BASE, use_structure=False)),
    ("abl_shuffle_structure", dict(BASE, shuffle_structure=True)),
    # With BLOSUM gone the ESM delta is the ONLY carrier of which substitution was made.
    # Removing it should collapse the model to chance; if it does not, the model is
    # scoring on something other than the mutation.
    ("abl_no_delta", dict(BASE, use_delta=False)),
    ("abl_no_film", dict(BASE, delta_film=False)),
    # geometry
    ("abl_no_dist_bias", dict(BASE, dist_bias=False)),
    ("abl_no_cross_chain", dict(BASE, cross_chain=False)),
    ("abl_no_chain_emb", dict(BASE, chain_embedding=False)),
    # the two-branch difference itself, which is the whole premise
    ("abl_one_branch", dict(BASE, two_branch=False)),
    ("abl_concat_head", dict(BASE, concat_head=True)),
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
