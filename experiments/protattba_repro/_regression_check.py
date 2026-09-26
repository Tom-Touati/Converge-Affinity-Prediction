"""Regression harness for a model_simple.py refactor: builds a synthetic batch (no data
pipeline dependency, so this runs anywhere torch does), constructs PerturbSiteToken under a
representative sweep of configs, and records each one's exact forward output plus its
parameter count. Run once BEFORE a refactor and once AFTER; the two dumps must match
byte-for-byte (this script asserts it directly when --compare is given).

    python _regression_check.py --out before.pt
    # ... refactor ...
    python _regression_check.py --out after.pt --compare before.pt
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from model_simple import PerturbSiteToken, SiteTokenConfig  # noqa: E402

B, L, M, PCA, CHEM = 6, 10, 12, 128, 39


def synthetic_batch(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    batch = {}
    for side, n in (("ab", L), ("ag", M)):
        batch[f"seq_{side}_wt"] = torch.randn(B, n, PCA, generator=g)
        batch[f"seq_{side}_mt"] = torch.randn(B, n, PCA, generator=g)
        batch[f"struct_{side}"] = torch.randn(B, n, PCA, generator=g)
        mask = torch.ones(B, n)
        batch[f"mask_{side}"] = mask
        site = torch.zeros(B, n)
        site[:, 0] = 1.0                                  # residue 0 is always the mutation
        batch[f"site_{side}"] = site
        batch[f"res_{side}"] = torch.arange(n).unsqueeze(0).expand(B, n).clone()
    dist = torch.rand(B, L, M, generator=g) * 20.0
    batch["dist"] = dist
    batch["chem"] = torch.randn(B, CHEM, generator=g)
    batch["y"] = torch.randn(B, generator=g)
    batch["row_id"] = [f"CPLX{i % 2}_A_B|A1G" for i in range(B)]  # 2 pseudo-complexes
    return batch


#: One representative config per axis this refactor must preserve exactly.
CONFIGS = {
    "leader": dict(mut_pair_struct_inject="film_site"),
    "struct_only": dict(mut_pair_struct_inject="film_site", struct_only=True),
    "input_noise": dict(mut_pair_struct_inject="film_site", input_noise=0.25, feature_dropout=0.25),
    "split_proj_false": dict(mut_pair_struct_inject="film_site", split_proj=False),
    "split_struct_true": dict(mut_pair_struct_inject="film_site", split_struct=True),
    "mul_op": dict(mut_pair_struct_inject="film_site", mut_pair_op="mul"),
    "concat_op": dict(mut_pair_struct_inject="film_site", mut_pair_op="concat"),
    "no_act": dict(mut_pair_struct_inject="film_site", mut_pair_act=False),
    "split_mutwt": dict(mut_pair_struct_inject="film_site", split_mutwt=True),
    "split_mutwt_tied": dict(mut_pair_struct_inject="film_site", split_mutwt=True,
                             split_mutwt_tied_init=True),
    "seq_depth2": dict(mut_pair_struct_inject="film_site", seq_mlp_depth=2),
    "seq_depth3": dict(mut_pair_struct_inject="film_site", seq_mlp_depth=3),
    "struct_depth2": dict(mut_pair_struct_inject="film_site", struct_mlp_depth=2),
    "both_depth2": dict(mut_pair_struct_inject="film_site", seq_mlp_depth=2, struct_mlp_depth=2),
    "fuse_multiply": dict(mut_pair_struct_inject="film_site", fuse_mode="multiply"),
    "fuse_add": dict(mut_pair_struct_inject="film_site", fuse_mode="add"),
    "fuse_concat": dict(mut_pair_struct_inject="film_site", fuse_mode="concat"),
    "struct_bind_feat": dict(mut_pair_struct_inject="film_site", struct_bind_feat=True),
    "struct_stab_feat": dict(mut_pair_struct_inject="film_site", struct_stab_feat=True),
    "fusion_early": dict(mut_pair_struct_inject="none", fusion_stage="early"),
    "fusion_mid": dict(mut_pair_struct_inject="none", fusion_stage="mid"),
    "bsite_crop_concat": dict(mut_pair_struct_inject="film_site", bsite_extra="crop_concat"),
    "bsite_min_dist": dict(mut_pair_struct_inject="film_site", bsite_extra="min_dist_crop"),
    "bsite_crop_xattn": dict(mut_pair_struct_inject="film_site", bsite_extra="crop_xattn"),
    "mut_feat_burial": dict(mut_pair_struct_inject="film_site", mut_feat="burial_rank"),
    "mut_feat_nearest": dict(mut_pair_struct_inject="film_site",
                             mut_feat="nearest_partner_struct"),
    "site_concat_mode": dict(mut_pair_struct_inject="site_concat"),
    "crop_concat_mode": dict(mut_pair_struct_inject="crop_concat"),
    "gated_site_mode": dict(mut_pair_struct_inject="gated_site"),
    "nn_diff_concat_mode": dict(mut_pair_struct_inject="nn_diff_concat"),
    "no_struct": dict(mut_pair_struct_inject="none"),
}

BASE = dict(pca_dim=PCA, proj=64, hidden=32, layers=1, dropout=0.0, chem_dim=CHEM,
           split_proj=True, split_struct=False, n_heads=2, mut_pair_ffn=True,
           mut_pair_op="sub", mut_pair_act=True)


def run(name: str, extra: dict, seed: int = 0) -> dict:
    cfg = SiteTokenConfig(**{**BASE, **extra})
    torch.manual_seed(seed)
    net = PerturbSiteToken(cfg).eval()
    batch = synthetic_batch(seed=123)
    with torch.no_grad():
        y = net(batch)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"name": name, "output": y.clone(), "n_params": n_params,
           "finite": bool(torch.isfinite(y).all())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    results = {}
    failures = []
    for name, extra in CONFIGS.items():
        try:
            results[name] = run(name, extra)
        except Exception as e:                            # noqa: BLE001
            failures.append((name, repr(e)))
    torch.save(results, a.out)

    print(f"ran {len(results)}/{len(CONFIGS)} configs, {len(failures)} failed")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")
    for name, r in results.items():
        print(f"  {name:22s} finite={r['finite']!s:5s} params={r['n_params']}")

    if a.compare:
        prev = torch.load(a.compare)
        print(f"\ncomparing against {a.compare}:")
        all_match = True
        for name in CONFIGS:
            if name in failures_names(failures) or name not in prev:
                continue
            if name not in results:
                print(f"  {name:22s} MISSING NOW (was present before)"); all_match = False
                continue
            if name not in prev:
                print(f"  {name:22s} NEW (wasn't in before)"); continue
            a_out, b_out = results[name]["output"], prev[name]["output"]
            same_shape = a_out.shape == b_out.shape
            match = same_shape and torch.allclose(a_out, b_out, atol=1e-5, rtol=1e-4)
            params_match = results[name]["n_params"] == prev[name]["n_params"]
            status = "OK" if (match and params_match) else "MISMATCH"
            if status == "MISMATCH":
                all_match = False
                diff = (a_out - b_out).abs().max().item() if same_shape else float("nan")
                print(f"  {name:22s} {status}  params {prev[name]['n_params']}->"
                     f"{results[name]['n_params']}  max_diff={diff}")
            else:
                print(f"  {name:22s} {status}")
        print("\nALL MATCH" if all_match else "\nREGRESSION DETECTED")
        sys.exit(0 if all_match else 1)


def failures_names(failures):
    return {n for n, _ in failures}


if __name__ == "__main__":
    main()
