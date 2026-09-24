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

NOPCA = {"group_reduce": 128, "group_out": 16, "flags": ["--no-pca"]}

#: Harder than anything run so far. The ladder has used dropout 0.2 / noise 0.1 /
#: feature-dropout 0.2 / weight-decay 1e-2 throughout, and weight decay was a module
#: constant with no way to vary it. The case for turning all of it up is the variance:
#: the same model on the same fold scored 0.049 and 0.425 on two seeds, which is what
#: a model with too much freedom for 752 training rows looks like.
REG2 = {"dropout": 0.35, "input_noise": 0.25, "feature_dropout": 0.35, "wd": 0.1}
REG3 = {"dropout": 0.50, "input_noise": 0.40, "feature_dropout": 0.50, "wd": 0.3}

#: The two strongest models, by mean over the seeds we have.
GF = dict(ST, use_site_pool=False, chem_dim=26, gated_fusion=True, mpnn_proj=64)
NP = dict(ST, use_site_pool=False, chem_dim=26, **NOPCA)

#: Clip at 10 rather than 5. The measured global gradient norm is 5.0-5.3, so a threshold of
#: 5 rescaled about half of all steps and left the other half alone -- the most intermittent
#: setting available, and one that differed systematically between arms because the heavier
#: regularisation levels clip more often. At 10 essentially nothing reaches it.
CLIP10 = {"grad_clip": 10.0}

#: The night queue. Sequence read AT the mutation, structure read as the mean of the binding
#: AREA -- the two modalities answering different questions rather than both being asked
#: what is at the mutated residue.
AREA = dict(ST, use_site_pool=False, chem_dim=26, mpnn_proj=64, layers=1,
            struct_area_pool=True, seeds=[0, 1, 2], **REG2, **CLIP10)

#: The best three-seed configuration we have: one hidden layer in the head, structure pooled
#: at the MUTATED RESIDUES, chem columns, reg2, clip 10. +0.239 over three seeds.
#: area_concat pooled the structure over the whole binding area instead and scored +0.189 --
#: worse on per-complex AND on pooled r -- so the fusion mechanisms below are built on the
#: site-pooled representation rather than the area one. ProteinMPNN's signal here is local;
#: averaged over ~85 interface residues it washes out and adds a near-constant per complex.
L1 = dict(ST, use_site_pool=False, chem_dim=26, mpnn_proj=64, layers=1,
          seeds=[0, 1, 2], **REG2, **CLIP10)

#: The classification table: one row per architecture family, each at its best known
#: configuration, all with the ordinal head, two seeds each.
#:
#: Early stopping is loosened from patience 10 to 25. The ordinal loss moves in much smaller
#: steps than MSE -- measured gradient norms dropped from ~5.1 to 0.4-0.6 -- so a patience
#: tuned for the regression runs stops these far too early.
CLS = dict(ST, use_site_pool=False, chem_dim=26, mpnn_proj=64, layers=1,
           ordinal=2, seeds=[0, 1], patience=25, **REG2, **CLIP10)

LADDER = [
    # no fusion at all: the control every other row has to beat
    ("cls_nostruct", dict(CLS)),
    # plain concatenation of the two modalities
    ("cls_concat", dict(CLS, concat_struct=True)),
    # gated fusion -- best regression net at +0.249
    ("cls_gated", dict(CLS, gated_fusion=True)),
    # FiLM: structure modulates the sequence edit
    ("cls_film", dict(CLS, film_struct=True)),
    # sequence queries structure
    ("cls_xattn", dict(CLS, cross_attn=True, n_heads=4,
                       attn_direction="seq_to_struct")),
    # structure queries sequence -- the direction that led under PCA
    ("cls_xattn_rev", dict(CLS, cross_attn=True, n_heads=4,
                           attn_direction="struct_to_seq")),
    # ProtAttBA's mechanism: antibody tokens attend across the interface to antigen tokens
    ("cls_abag_xattn", dict(CLS, ab_ag_attn=True, n_heads=4)),
    # binding-site mean alongside the mutation mean: structure pooled over the whole area
    ("cls_areapool", dict(CLS, concat_struct=True, struct_area_pool=True)),
    # the wild-type binding-site pools, which are a complex-identity channel by construction
    ("cls_sitepool", dict(CLS, concat_struct=True, use_site_pool=True)),
]


def run(name: str, ov: dict) -> int:
    # "arch" selects the model and is a CLI flag, not a config field, so it is lifted out
    # of the overrides before they are handed to the config constructor.
    ov = dict(ov)
    arch = ov.pop("arch", "v2")
    flags = list(ov.pop("flags", []))
    # optimiser settings are CLI flags, not config fields
    for k in ("wd", "lr", "patience", "grad_clip"):
        if k in ov:
            flags += ["--" + k.replace("_", "-"), str(ov.pop(k))]
    seeds = ov.pop("seeds", [ov.pop("seed", 0)])
    seeds = [str(x) for x in (seeds if isinstance(seeds, list) else [seeds])]
    cmd = [sys.executable, "_perturb_v2_colab.py", "--exp", name, "--arch", arch,
           "--folds", "0", "1", "2", "3", "4", "--seeds", *seeds, *flags,
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
        want_rows = 5 * len(ov.get("seeds", [0]) if isinstance(ov.get("seeds"), list) else [0])
        if done.exists() and len(done.read_text().strip().splitlines()) >= want_rows + 1:
            print(f"=== {name}: already has 5 folds, skipping ===", flush=True)
            continue
        run(name, ov)
    print("\nLADDER DONE", flush=True)


if __name__ == "__main__":
    main()
