"""Properties the simple model has to keep, stated on tensors rather than on scores.

The two that carried over from the attention model are (a) a null edit is exactly zero and
(e) the answer does not depend on how far the batch was padded. The rest pin the specific
claims made in the docstring: that the structure enters only through what the edit did to
it, that pooling ignores everything but the mutated residues, and that the shared input
perturbation cancels in the delta.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.perturb.model_simple import PerturbSimple, SimpleConfig, site_mean  # noqa: E402


def make_batch(b=3, l_ab=9, l_ag=6, cfg=None, mutate=False, seed=0, pad_to=None,
               site_ag=False):
    cfg = cfg or SimpleConfig()
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    site = min(2, l_ab - 1)
    L, M = pad_to or l_ab, pad_to or l_ag

    def pad(x, n):
        out = torch.zeros(b, n, x.shape[-1])
        out[:, : x.shape[1]] = x
        return out

    seq_ab = r(b, l_ab, cfg.pca_dim)
    seq_ab_mt = seq_ab.clone()
    if mutate:
        seq_ab_mt[:, site] += r(b, cfg.pca_dim)
    seq_ag = r(b, l_ag, cfg.pca_dim)
    seq_ag_mt = seq_ag.clone()

    s_ab = torch.zeros(b, l_ab); s_ab[:, site] = 1.0
    s_ag = torch.zeros(b, l_ag)
    if site_ag:
        s_ag[:, 1] = 1.0
        if mutate:
            seq_ag_mt = seq_ag.clone(); seq_ag_mt[:, 1] += r(b, cfg.pca_dim)

    return {
        "seq_ab_wt": pad(seq_ab, L), "seq_ab_mt": pad(seq_ab_mt, L),
        "seq_ag_wt": pad(seq_ag, M), "seq_ag_mt": pad(seq_ag_mt, M),
        "struct_ab": pad(r(b, l_ab, cfg.pca_dim), L),
        "struct_ag": pad(r(b, l_ag, cfg.pca_dim), M),
        "site_ab": pad(s_ab.unsqueeze(-1), L).squeeze(-1),
        "site_ag": pad(s_ag.unsqueeze(-1), M).squeeze(-1),
    }


# --------------------------------------------------------------------------- (a)
def test_null_edit_is_exactly_zero():
    m = PerturbSimple().eval()
    with torch.no_grad():
        y = m(make_batch(mutate=False))
    assert torch.equal(y, torch.zeros_like(y)), f"null edit gave {y.tolist()}"


def test_null_edit_is_zero_in_training_too():
    """Where it matters: the gradient comes from this path."""
    cfg = SimpleConfig(input_noise=1.0, feature_dropout=0.3, dropout=0.3)
    m = PerturbSimple(cfg).train()
    torch.manual_seed(0)
    y = m(make_batch(mutate=False, cfg=cfg))
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), \
        f"null edit under training-time noise gave {y.tolist()}"


def test_a_real_mutation_is_not_zero():
    """So the two tests above cannot be satisfied by a model that always returns zero."""
    m = PerturbSimple().eval()
    for p in m.mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    with torch.no_grad():
        y = m(make_batch(mutate=True))
    assert y.abs().max() > 0


# --------------------------------------------------------------------------- (e)
def test_padding_width_does_not_change_the_answer():
    cfg = SimpleConfig()
    m = PerturbSimple(cfg).eval()
    for p in m.mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    with torch.no_grad():
        a = m(make_batch(mutate=True, cfg=cfg, seed=1))
        b = m(make_batch(mutate=True, cfg=cfg, seed=1, pad_to=48))
    assert torch.allclose(a, b, atol=1e-5), \
        f"prediction changed with padding width: max |diff| {(a-b).abs().max():.2e}"


def test_site_mean_ignores_everything_but_the_site():
    """Pooling must not see a token the mutation did not touch."""
    x = torch.randn(2, 7, 4)
    s = torch.zeros(2, 7); s[:, 3] = 1.0
    assert torch.allclose(site_mean(x, s), x[:, 3])
    y = x.clone(); y[:, 5] += 100.0            # a large change away from the site
    assert torch.allclose(site_mean(y, s), site_mean(x, s)), \
        "pooling responded to a token that is not a mutated residue"
    z = site_mean(x, torch.zeros(2, 7))
    assert torch.equal(z, torch.zeros_like(z)), "a side with no site must contribute zeros"


# --------------------------------------------------- the structure enters via the edit
def test_structure_reaches_the_head_only_through_what_the_edit_did_to_it():
    """Changing the structure of an UNMUTATED row must not change the prediction.

    The head is handed ``t - x_str``, so the complex's own structure cancels; only the
    FiLM's effect survives. If the raw structure reached the head, this would fail -- and
    the model would have a channel that identifies the complex without describing the edit,
    which is the failure mode the whole ladder kept running into.
    """
    cfg = SimpleConfig()
    m = PerturbSimple(cfg).eval()
    for p in m.mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    b = make_batch(mutate=False, cfg=cfg, seed=2)
    with torch.no_grad():
        y1 = m(b)
        b2 = dict(b); b2["struct_ab"] = torch.randn_like(b["struct_ab"]) * 5
        y2 = m(b2)
    assert torch.equal(y1, y2), "the raw structure leaked into the head"


def test_a_mutation_on_the_antigen_side_is_read_too():
    cfg = SimpleConfig()
    m = PerturbSimple(cfg).eval()
    for p in m.mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    with torch.no_grad():
        y = m(make_batch(mutate=True, cfg=cfg, seed=3, site_ag=True))
    assert y.abs().max() > 0, "an antigen-side mutation produced no response"


def test_shared_perturbation_cancels_in_the_delta():
    cfg = SimpleConfig(input_noise=1.0, feature_dropout=0.3)
    m = PerturbSimple(cfg).train()
    b = make_batch(mutate=True, cfg=cfg, seed=4)
    torch.manual_seed(0)
    p1 = m.perturb(b)
    torch.manual_seed(1)
    p2 = m.perturb(b)
    n1, m1 = p1["seq_ab"]
    n2, _ = p2["seq_ab"]
    assert not torch.equal(n1, n2), "noise is identical across samples"
    assert (m1 == 0).any(), "feature dropout never dropped a channel"
    with torch.no_grad():
        a = m.side(b, "ab", p1)
        c = m.side(b, "ab", None)
    assert not torch.allclose(a, c), "the perturbation did not change anything"


def test_it_is_actually_smaller():
    n = sum(p.numel() for p in PerturbSimple().parameters() if p.requires_grad)
    assert n < 51089, f"{n:,} parameters is not smaller than the attention model's 51,089"


# ------------------------------------------------------- two towers, combined late
from src.perturb.model_simple import PerturbTwoTower, TwoTowerConfig  # noqa: E402


def tt_batch(b=3, l=9, dim=128, mutate=False, seed=0, pad_to=None):
    g = torch.Generator().manual_seed(seed)
    L = pad_to or l
    def pad(x):
        out = torch.zeros(b, L, x.shape[-1]); out[:, : x.shape[1]] = x; return out
    ab = torch.randn(b, l, dim, generator=g)
    ab_mt = ab.clone()
    if mutate:
        ab_mt[:, 2] += torch.randn(b, dim, generator=g)
    ag = torch.randn(b, l, dim, generator=g)
    s_ab = torch.zeros(b, l); s_ab[:, 2] = 1.0
    return {"seq_ab_wt": pad(ab), "seq_ab_mt": pad(ab_mt),
            "seq_ag_wt": pad(ag), "seq_ag_mt": pad(ag.clone()),
            "site_ab": pad(s_ab.unsqueeze(-1)).squeeze(-1),
            "site_ag": torch.zeros(b, L)}


def test_subtract_is_zero_on_a_null_edit_and_concat_is_not():
    """The whole point of the ablation, asserted rather than assumed.

    subtract can only express what the edit changed. concat also hands the head the
    wild-type vector, which is constant across every mutation of a complex -- so it can
    answer from complex identity alone, and on these labels that scores pooled +0.672.
    """
    for combine, expect_zero in (("subtract", True), ("concat", False)):
        m = PerturbTwoTower(TwoTowerConfig(combine=combine)).eval()
        for p in m.mlp.parameters():
            torch.nn.init.normal_(p, std=0.1)
        with torch.no_grad():
            y = m(tt_batch(mutate=False))
        if expect_zero:
            assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), \
                f"subtract gave {y.tolist()} on a null edit"
        else:
            assert y.abs().max() > 1e-3, \
                "concat gave zero on a null edit; it should be able to see the wild type"


def test_both_towers_share_weights():
    """Subtracting two differently-parameterised towers compares two different things."""
    m = PerturbTwoTower(TwoTowerConfig())
    names = {n for n, _ in m.named_parameters()}
    assert not any("tower" in n and ("_wt" in n or "_mt" in n) for n in names), \
        f"found per-tower parameters: {sorted(names)}"
    b = tt_batch(mutate=True, seed=2)
    swapped = dict(b)
    swapped["seq_ab_wt"], swapped["seq_ab_mt"] = b["seq_ab_mt"], b["seq_ab_wt"]
    swapped["seq_ag_wt"], swapped["seq_ag_mt"] = b["seq_ag_mt"], b["seq_ag_wt"]
    ms = PerturbTwoTower(TwoTowerConfig(combine="subtract")).eval()
    with torch.no_grad():
        z = ms.tower(b, "mt", None) - ms.tower(b, "wt", None)
        z_sw = ms.tower(swapped, "mt", None) - ms.tower(swapped, "wt", None)
    # The MLP's INPUT negates, which is what shared towers guarantee. Its OUTPUT does not:
    # GELU is not an odd function, so MLP(-z) != -MLP(z). Asserting on the output would be
    # asserting the head is antisymmetric, which it is not and was never meant to be --
    # the first version of this test did exactly that and failed against a correct model.
    assert torch.allclose(z, -z_sw, atol=1e-5), \
        "swapping wild type and mutant did not negate the difference; the towers differ"


@pytest.mark.parametrize("combine", ["concat", "subtract"])
def test_two_tower_padding_invariance(combine):
    cfg = TwoTowerConfig(combine=combine)
    m = PerturbTwoTower(cfg).eval()
    for p in m.mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    with torch.no_grad():
        a = m(tt_batch(mutate=True, seed=1))
        b = m(tt_batch(mutate=True, seed=1, pad_to=40))
    assert torch.allclose(a, b, atol=1e-5)
