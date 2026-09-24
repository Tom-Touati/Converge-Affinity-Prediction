"""Can the rotary encoding carry information at THIS sequence length?

RoPE's frequencies are a geometric series set by ``base``, and 10000 is the value chosen
for language models reading thousands of tokens. A crop here is about 50-90 residues. If
most of the series turns too slowly to complete an appreciable fraction of a turn across
that span, those channels are constant, and a constant is not information -- the encoding
would be nominally present and practically absent.

Three things are measured, on the real crops rather than on an assumption:

1. the distribution of residue separations the attention actually sees;
2. per frequency channel, the phase swing across that distribution -- a channel needs
   enough to distinguish near from far, but not so much that it wraps and aliases;
3. what the rotation does to the attention logits: how much they move, and whether the
   movement is a FUNCTION of separation rather than noise. A relative encoding must give
   two pairs at the same separation the same treatment; that is the property being
   checked, and it is what distinguishes an encoding from a perturbation.

    PERTURB_ROOT=/home/ubuntu/perturb python probe_rope.py [--base 10000]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("PERTURB_ROOT", "/home/ubuntu/perturb"))
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--rows", type=int, default=300)
    a = ap.parse_args()

    T = load("trainer", ROOT / "_perturb_v2_colab.py")
    import pandas as pd
    from model_simple import rotary

    cache = T.Cache()
    rows = pd.read_parquet(ROOT / "perturb_rows.parquet")

    # 1. what separations does the attention actually see?
    seps, lens, cross = [], [], 0
    for rid in rows.row_id[: a.rows]:
        c = cache.crop(rid)
        for side in ("ab", "ag"):
            r, ch = c[f"{side}_res"], c[f"{side}_chain"]
            if len(r) < 2:
                continue
            lens.append(len(r))
            same = ch[:, None] == ch[None, :]
            d = np.abs(r[:, None] - r[None, :])[same]
            seps.append(d[d > 0])
            cross += int((~same).sum())
    sep = np.concatenate(seps)
    q = np.percentile(sep, [50, 90, 99])
    print(f"crops: median length {np.median(lens):.0f}")
    print(f"same-chain residue separations: median {q[0]:.0f}, p90 {q[1]:.0f}, "
          f"p99 {q[2]:.0f}, max {sep.max():.0f}")
    print(f"cross-chain key/query pairs: {cross/ (cross + len(sep)):.0%} of all pairs\n")

    # 2. per-channel phase swing over the separation range that matters
    half = a.head_dim // 2
    inv = a.base ** (-np.arange(half) / half)
    swing = inv * q[1]                       # radians across the p90 separation
    live = int(((swing > 0.5) & (swing < 2 * np.pi)).sum())
    print(f"base {a.base:g}, head_dim {a.head_dim} -> {half} frequency channels")
    print(f"  phase swing across p90 separation ({q[1]:.0f} residues), per channel:")
    print("   " + "  ".join(f"{s:.2f}" for s in swing))
    print(f"  channels that wrap (>2pi, aliased):      "
          f"{int((swing > 2 * np.pi).sum()):2d}/{half}")
    print(f"  channels that barely move (<0.5 rad):    "
          f"{int((swing < 0.5).sum()):2d}/{half}")
    print(f"  channels usefully tuned to this range:   {live:2d}/{half}\n")

    # 3. does the rotation move attention logits AS A FUNCTION of separation?
    torch.manual_seed(0)
    L = int(np.median(lens))
    pos = torch.arange(L).float()[None]
    qv = torch.randn(1, 2, L, a.head_dim)
    kv = torch.randn(1, 2, L, a.head_dim)
    plain = (qv @ kv.transpose(-2, -1))[0, 0] / a.head_dim ** 0.5
    rot = (rotary(qv, pos, a.base) @ rotary(kv, pos, a.base).transpose(-2, -1))[0, 0] \
        / a.head_dim ** 0.5
    d = (pos[0][:, None] - pos[0][None, :]).abs().numpy()
    delta = (rot - plain).numpy()
    print(f"logit change from rotation: sd {delta.std():.3f} against the plain logits' "
          f"own sd {plain.numpy().std():.3f}")
    # the defining property: pairs at equal separation must be treated identically, so the
    # rotation's effect must be a function of |i-j| alone
    band = [delta[d == s].std() for s in range(1, min(40, L))]
    print(f"spread WITHIN a separation band: mean {np.mean(band):.3f} "
          f"(0 would mean the effect depends on separation alone)")
    r = np.corrcoef(d.ravel(), np.abs(delta).ravel())[0, 1]
    print(f"|logit change| vs separation: r = {r:+.3f}")
    far = np.abs(delta)[d > q[1]].mean() if (d > q[1]).any() else float("nan")
    print(f"mean |change| near (<={q[0]:.0f} apart) {np.abs(delta)[d <= q[0]].mean():.3f} "
          f"vs far (>{q[1]:.0f}) {far:.3f}")


if __name__ == "__main__":
    main()
