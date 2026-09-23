"""Is the mutation where we think it is, and does ESM see it there?

Three independent failures all look identical from the loss curve -- the model simply does
not learn -- so each is checked separately here and a failure names itself:

  1. the substitution was applied to a DIFFERENT sequence than the one we embed
     (SEQRES vs the ATOM-derived chain), which shows up as the WT and MT strings
     differing somewhere other than the recorded sites, or by a different count;
  2. an off-by-one, typically from a PDB insertion code, which shows up as the residue
     at the index not matching the mutation string's WT letter;
  3. a chain mix-up, which shows up as the sites for one side landing in the other's
     sequence -- caught by the same letter check, since the letters would not agree.

Only the fourth check needs embeddings: ||t_mt - t_wt|| per residue should be sharply
peaked at the mutated position. If it is not, nothing downstream can be learning from the
delta, whatever the sequence bookkeeping says.

    python -m src.perturb.check_alignment [-n 20] [--seed 0]

Runs against the VM's copies when they are present, otherwise the repo's own, and skips
the embedding check when no token cache is reachable.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

VM = pathlib.Path("/content/perturb")
# `colab exec` runs this file as a notebook cell, where __file__ does not exist and argv
# carries the kernel's own -f flag. Both are handled rather than requiring a second copy
# of the script for the VM.
try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except NameError:
    ROOT = pathlib.Path.cwd()


def locate():
    if (VM / "perturb_rows.parquet").exists():
        return VM / "perturb_rows.parquet", VM / "perturb_crops.npz", VM
    return (ROOT / "experiments" / "protattba_repro" / "cache" / "perturb_rows.parquet",
            ROOT / "data" / "features" / "perturb_crops.npz", VM)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--rows", dest="n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a, _ = ap.parse_known_args()

    rows_p, crops_p, vm = locate()
    rows = pd.read_parquet(rows_p)
    crops = np.load(crops_p, allow_pickle=False)
    print(f"rows  {rows_p}\ncrops {crops_p}  ({len(crops.files)} keys)")

    cache = None
    if (vm / "esm2_650m_tokens.npy").exists():
        import importlib.util
        # the trainer imports model_v2 / model_simple as top-level modules, so its own
        # directory has to be importable -- the notebook cwd is not it
        sys.path.insert(0, str(vm))
        spec = importlib.util.spec_from_file_location("trainer", vm / "_perturb_v2_colab.py")
        T = importlib.util.module_from_spec(spec)
        sys.modules["trainer"] = T
        spec.loader.exec_module(T)
        cache = T.Cache()
        print("token cache: present, embedding check ON")
    else:
        print("token cache: absent, embedding check SKIPPED (sequence checks still run)")

    rng = np.random.default_rng(a.seed)
    pick = rows.iloc[rng.choice(len(rows), min(a.n, len(rows)), replace=False)]

    bad_letter = bad_span = 0
    ranks, ratios, n_sites = [], [], 0
    topk_hit = topk_tot = 0
    hdr = f"{'row':<40}{'side':>5}{'pos':>6}{'wt':>4}{'mt':>4}{'seqWT':>7}{'seqMT':>7}"
    if cache:
        hdr += f"{'rank':>6}{'d/med':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for r in pick.itertuples():
        for side in ("ab", "ag"):
            k = r.row_id
            if f"{k}|{side}_site" not in crops:
                continue
            flags = crops[f"{k}|{side}_site"]
            idx = crops[f"{k}|{side}_idx"]
            pos = idx[np.flatnonzero(flags)]
            if not len(pos):
                continue
            l_wt = crops[f"{k}|{side}_wt_aa"]
            l_mt = crops[f"{k}|{side}_mt_aa"]
            wt_s, mt_s = getattr(r, f"{side}_wt"), getattr(r, f"{side}_mt")

            # 1. the substitution landed on the sequence we embed, and ONLY there
            diff = sorted(i for i, (x, y) in enumerate(zip(wt_s, mt_s)) if x != y)
            if diff != sorted(int(p) for p in pos) or len(wt_s) != len(mt_s):
                bad_span += 1
                print(f"  !! {k[:38]} {side}: strings differ at {diff[:6]}, "
                      f"sites say {sorted(int(p) for p in pos)[:6]}")

            d = order = med = None
            if cache:
                t_wt, t_mt = cache.seq(side, wt_s), cache.seq(side, mt_s)
                d = np.linalg.norm(t_mt - t_wt, axis=1)
                order = np.argsort(-d)
                med = float(np.median(d)) or 1e-9

            # With k mutations on a side only ONE can be rank 0, so "rank 0" understates
            # a clean alignment on multi-point rows. The honest test is whether the k
            # mutated sites occupy the top k of ||delta|| -- i.e. every mutated residue
            # moves more than every unmutated one.
            if cache and d is not None:
                nmut = len(pos)
                top = set(int(x) for x in order[:nmut])
                topk_tot += 1
                topk_hit += top == set(int(x) for x in pos if int(x) < len(d))

            for j, p in enumerate(pos):
                p = int(p)
                ok_wt = p < len(wt_s) and wt_s[p] == str(l_wt[j])
                ok_mt = p < len(mt_s) and mt_s[p] == str(l_mt[j])
                bad_letter += (not ok_wt) or (not ok_mt)
                line = (f"{k[:38]:<40}{side:>5}{p:>6}{str(l_wt[j]):>4}{str(l_mt[j]):>4}"
                        f"{('OK' if ok_wt else 'BAD'):>7}{('OK' if ok_mt else 'BAD'):>7}")
                if cache and p < len(d):
                    rank = int(np.flatnonzero(order == p)[0])
                    ratio = float(d[p] / med)
                    ranks.append(rank); ratios.append(ratio)
                    line += f"{rank:>6}{ratio:>8.1f}"
                n_sites += 1
                print(line)

    print(f"\n{n_sites} mutated sites over {len(pick)} rows")
    print(f"residue at the index disagrees with the mutation string: {bad_letter}")
    print(f"WT/MT strings differ anywhere other than the recorded sites: {bad_span}")
    if ranks:
        rk, rt = np.array(ranks), np.array(ratios)
        print(f"||delta|| largest in the WHOLE sequence at the site: "
              f"{(rk == 0).sum()}/{len(rk)} ({(rk == 0).mean():.0%})")
        print(f"rank <= 2: {(rk <= 2).sum()}/{len(rk)}, median rank {np.median(rk):.0f}")
        print(f"||delta|| at site / median: median {np.median(rt):.1f}x, "
              f"min {rt.min():.1f}x, max {rt.max():.1f}x")
        if topk_tot:
            print(f"mutated sites occupy the TOP-k of ||delta||: "
                  f"{topk_hit}/{topk_tot} sides ({topk_hit / topk_tot:.0%})")
        if (rk == 0).mean() < 0.8 or np.median(rt) < 3:
            print("\nNOT sharply peaked -- the alignment is suspect")
        else:
            print("\nsharply peaked at the mutated site")


if __name__ == "__main__":
    main()
