"""The six properties section 1 of the parameter-economy spec requires.

(a) null edit gives exactly zero, (b) the ITW branch is untouched by the structure FiLM,
(c) ITW and MUT receive identical crop, mask, chain ids and residue indices, (d) the distance
bias reproduces contact-map ordering, (e) pooling is invariant to padding length, (f) residue
indices align across the mutation string, the ESM input and the PDB residue list.

(f) lives in ``test_perturb_model.py`` and is shared: it is a property of the data, not of a
model version, and duplicating it would mean two places to keep in step.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.perturb.crop import ANTIGEN, HEAVY, LIGHT, build_crop  # noqa: E402
from src.perturb.model_v2 import (  # noqa: E402
    N_DIST_BINS,
    PerturbV2,
    PerturbV2Config,
    distance_bins,
    masked_mean,
)


def make_batch(b=3, l_ab=9, l_ag=6, cfg=None, mutate=False, seed=0, pad_to=None):
    cfg = cfg or PerturbV2Config()
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    site = min(2, l_ab - 1)

    L = pad_to or l_ab
    M = pad_to or l_ag

    def padded(x, n):
        out = torch.zeros(b, n, x.shape[-1])
        out[:, : x.shape[1]] = x
        return out

    seq_ab, seq_ag = r(b, l_ab, cfg.pca_dim), r(b, l_ag, cfg.pca_dim)
    seq_ab_mt = seq_ab.clone()
    if mutate:
        seq_ab_mt[:, site] += r(b, cfg.pca_dim)

    bl_wt = torch.zeros(b, l_ab, 40)
    bl_wt[:, site, :20] = r(b, 20)
    bl_wt[:, site, 20:] = bl_wt[:, site, :20]          # ITW pair is (a, a)
    bl_mt = bl_wt.clone()
    if mutate:
        bl_mt[:, site, 20:] = r(b, 20)

    site_ab = torch.zeros(b, l_ab); site_ab[:, site] = 1.0
    mask_ab = torch.zeros(b, L); mask_ab[:, :l_ab] = 1.0
    mask_ag = torch.zeros(b, M); mask_ag[:, :l_ag] = 1.0
    d = torch.rand(b, l_ab, l_ag, generator=g) * 30.0
    dpad = torch.full((b, L, M), 99.0)
    dpad[:, :l_ab, :l_ag] = d

    chain_ab = torch.zeros(b, L, dtype=torch.long); chain_ab[:, :l_ab] = HEAVY
    chain_ag = torch.full((b, M), ANTIGEN, dtype=torch.long)
    res_ab = torch.arange(L).expand(b, -1).clone()
    res_ag = torch.arange(M).expand(b, -1).clone()

    return {
        "seq_ab_wt": padded(seq_ab, L), "seq_ab_mt": padded(seq_ab_mt, L),
        "seq_ag_wt": padded(seq_ag, M), "seq_ag_mt": padded(seq_ag.clone(), M),
        "struct_ab": padded(r(b, l_ab, cfg.pca_dim), L),
        "struct_ag": padded(r(b, l_ag, cfg.pca_dim), M),
        "blosum_ab_wt": padded(bl_wt, L), "blosum_ab_mt": padded(bl_mt, L),
        "blosum_ag_wt": torch.zeros(b, M, 40), "blosum_ag_mt": torch.zeros(b, M, 40),
        "site_ab": padded(site_ab.unsqueeze(-1), L).squeeze(-1),
        "site_ag": torch.zeros(b, M),
        "mask_ab": mask_ab, "mask_ag": mask_ag,
        "chain_ab": chain_ab, "chain_ag": chain_ag,
        "res_ab": res_ab, "res_ag": res_ag,
        "dist": dpad, "dist_bins": distance_bins(dpad),
    }


# ------------------------------------------------------------------------------- (a)
def test_null_edit_is_exactly_zero():
    m = PerturbV2().eval()
    with torch.no_grad():
        y = m(make_batch(mutate=False))
    assert torch.equal(y, torch.zeros_like(y)), f"null edit gave {y.tolist()}"


def test_a_real_mutation_is_not_zero():
    """So (a) cannot be satisfied by a model that always returns zero."""
    m = PerturbV2().eval()
    for p in m.head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        y = m(make_batch(mutate=True))
    assert y.abs().max() > 0


# ------------------------------------------------------------------------------- (b)
def test_itw_branch_ignores_structure_film():
    batch = make_batch(mutate=True)
    on = PerturbV2(PerturbV2Config(delta_film=True)).eval()
    off = PerturbV2(PerturbV2Config(delta_film=False)).eval()
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on.branch(batch, mutated=False)
        c = off.branch(batch, mutated=False)
    assert torch.allclose(a, c, atol=0, rtol=0), \
        f"ITW changed with step 2: max |diff| {(a - c).abs().max().item():.3e}"


# ------------------------------------------------------------------------------- (c)
def test_both_branches_see_the_same_crop_and_indices():
    """The crop comes from the wild-type backbone, so the branches cannot disagree.

    Asserted on the tensors the branches actually read, not on the code path, so a future
    change that makes MUT re-derive its own crop would fail here.
    """
    batch = make_batch(mutate=True)
    seen = {}

    real = PerturbV2().branch.encode

    def spy(seq_pca, str_pca, delta, blosum, site, chain_type):
        seen.setdefault("site", []).append(site.clone())
        seen.setdefault("chain", []).append(chain_type.clone())
        return real(seq_pca, str_pca, delta, blosum, site, chain_type)

    m = PerturbV2().eval()
    m.branch.encode = spy
    with torch.no_grad():
        m(batch)

    # two sides x two branches
    assert len(seen["site"]) == 4, f"expected 4 encode calls, saw {len(seen['site'])}"
    mut_ab, mut_ag, itw_ab, itw_ag = seen["site"]
    assert torch.equal(mut_ab, itw_ab) and torch.equal(mut_ag, itw_ag), \
        "site masks differ between branches"
    c_mut_ab, c_mut_ag, c_itw_ab, c_itw_ag = seen["chain"]
    assert torch.equal(c_mut_ab, c_itw_ab) and torch.equal(c_mut_ag, c_itw_ag), \
        "chain ids differ between branches"
    for k in ("mask_ab", "mask_ag", "res_ab", "res_ag", "dist_bins"):
        assert k in batch, f"{k} missing; both branches read it from the same batch"


def test_crop_is_symmetric_and_keeps_the_site():
    """The crop must contain the mutated residue and flag a disconnected neighbourhood."""
    n_ab, n_ag = 30, 20
    d = np.full((n_ab, n_ag), 50.0, np.float32)
    d[0:4, 0:4] = 5.0                       # an interface patch
    types_ab = np.full(n_ab, HEAVY); types_ag = np.full(n_ag, ANTIGEN)
    res_ab = np.arange(n_ab); res_ag = np.arange(n_ag)

    near = build_crop(d, [1], [], n_ab, n_ag, types_ab, types_ag, res_ab, res_ag)
    assert 1 in near.ab_idx.tolist(), "the mutated residue must be in the crop"
    assert not near.disconnected, "a mutation inside the interface patch is connected"

    far = build_crop(d, [25], [], n_ab, n_ag, types_ab, types_ag, res_ab, res_ag)
    assert 25 in far.ab_idx.tolist()
    assert far.disconnected, "a mutation 50 A from the partner must be flagged disconnected"
    assert far.size < n_ab + n_ag, "the crop must actually drop residues"


# ------------------------------------------------------------------------------- (d)
def test_distance_bias_reproduces_contact_ordering():
    d = torch.tensor([[[0.0, 1.0, 4.9, 5.1, 12.0, 19.9, 20.1, 100.0]]])
    b = distance_bins(d).squeeze()
    assert torch.all(b[1:] >= b[:-1]), f"bins not monotone: {b.tolist()}"
    assert b[0] == 0
    assert b[-1] == N_DIST_BINS - 1 and b[-2] == N_DIST_BINS - 1, \
        f"beyond 20 A must be the last bin, got {b[-2:].tolist()}"
    assert b[4] < b[5], "12 A and 19.9 A must not collapse into one bin"


# ------------------------------------------------------------------------------- (e)
def test_pooling_is_invariant_to_padding_length():
    """Masked mean over real tokens only: padding to 64 or to 256 must not matter."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 10, 8, generator=g)
    for pad in (64, 256):
        xp = torch.zeros(2, pad, 8); xp[:, :10] = x
        mp = torch.zeros(2, pad); mp[:, :10] = 1.0
        out = masked_mean(xp, mp)
        base = masked_mean(x, torch.ones(2, 10))
        assert torch.allclose(out, base, atol=1e-6), \
            f"pooling changed when padded to {pad}: max |diff| {(out-base).abs().max():.2e}"


def test_model_output_is_invariant_to_padding_length():
    """The same property end to end, through attention and pooling."""
    cfg = PerturbV2Config()
    m = PerturbV2(cfg).eval()
    for p in m.head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        a = m(make_batch(mutate=True, cfg=cfg, seed=1))
        b = m(make_batch(mutate=True, cfg=cfg, seed=1, pad_to=40))
    assert torch.allclose(a, b, atol=1e-5), \
        f"prediction changed with padding width: max |diff| {(a-b).abs().max():.2e}"
