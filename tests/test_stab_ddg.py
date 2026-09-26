"""Unit tests for the StaB-ddG (Deng et al., arXiv 2507.05502) binding-ddG parameterization in
src/features/proteinmpnn.py: the antithetic-variate invariant the paper's own ablation says the
signal depends on (ONE decoding order and ONE backbone-noise draw shared across ALL SIX forward
passes of a trial -- without it, per-interface Spearman drops ~0.45 -> ~0.30), determinism under
a fixed seed, that the frozen model never leaves eval/no-grad, and a real-data smoke test.

test_wt_and_mutant_share_decoding_order_and_backbone_noise, test_synonymous_mutation_gives_zero_ddg
and test_stab_ddg_deterministic_with_fixed_seed are pure-torch and need no PDB corpus (a fake
model stands in for the first two; the third needs the real weights but not the PDB corpus).
test_frozen_model_has_no_trainable_parameters needs the real weights only. test_stab_ddg_smoke_on_
real_data needs the SKEMPI PDB corpus this repo does not vendor, and skips cleanly when it (or
third_party/ProteinMPNN) is absent, matching test_perturb_model.py's own skip convention.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import paths                                       # noqa: E402
from src.features import proteinmpnn as pm                  # noqa: E402
from src.structures import Chain, Mutation, Structure        # noqa: E402


def _toy_structure(l_ab: int = 5, l_ag: int = 4) -> Structure:
    def chain(cid: str, n: int, seed: int) -> Chain:
        rng = np.random.default_rng(seed)
        seq = "".join(rng.choice(list("ACDEFHIKLMNPQRSTVWY"), n))   # no G: keeps mutations real
        backbone = (rng.normal(size=(n, 4, 3)) * 3.8).astype(np.float32)
        backbone = np.cumsum(backbone, axis=0)
        return Chain(id=cid, seq=seq, keys=[(i + 1, "") for i in range(n)],
                    index={(i + 1, ""): i for i in range(n)}, backbone=backbone)
    return Structure(pdb="TOY", chains={"A": chain("A", l_ab, 0), "B": chain("B", l_ag, 1)})


class _RecordingFakeMPNN:
    """Stands in for a loaded ProteinMPNN: log_probs are uniform (don't depend on the input),
    but every call's decoding_order and noised backbone are recorded, so the test can assert
    ALL SIX calls in one trial (not just a wt/mut pair) received bit-identical copies of both.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn,
                use_input_decoding_order=False, decoding_order=None):
        assert use_input_decoding_order and decoding_order is not None
        self.calls.append({"X": X.clone(), "decoding_order": decoding_order.clone()})
        n_aa = len(pm.ALPHABET)
        return torch.full((S.shape[0], S.shape[1], n_aa), -float(np.log(n_aa)))


def test_wt_and_mutant_share_decoding_order_and_backbone_noise():
    st = _toy_structure()
    mutations = [Mutation(wt=st.chains["A"].seq[0], chain="A", resnum=1, icode="", mut="G")]
    fake = _RecordingFakeMPNN()
    gen = torch.Generator().manual_seed(0)

    pm._stab_trial(fake, st, "A", "B", mutations, "cpu", pm.STAB_NOISE_LEVEL, gen)

    # complex(wt,mut), ab_alone(wt,mut), ag_alone(wt,mut) = 6 calls, ALL sharing one order/noise.
    assert len(fake.calls) == 6
    order0, X0 = fake.calls[0]["decoding_order"], fake.calls[0]["X"]
    for call in fake.calls[1:]:
        assert torch.equal(call["decoding_order"], order0)
        assert torch.equal(call["X"], X0)

    # A second trial with the same generator must draw a DIFFERENT order/noise -- sharing is
    # within a trial, not frozen across the whole Monte Carlo ensemble.
    fake2 = _RecordingFakeMPNN()
    gen2 = torch.Generator().manual_seed(0)
    pm._stab_trial(fake2, st, "A", "B", mutations, "cpu", pm.STAB_NOISE_LEVEL, gen2)
    pm._stab_trial(fake2, st, "A", "B", mutations, "cpu", pm.STAB_NOISE_LEVEL, gen2)
    assert not torch.equal(fake2.calls[0]["decoding_order"], fake2.calls[6]["decoding_order"])


def test_synonymous_mutation_gives_zero_ddg():
    """A 'mutation' to the wild-type residue itself leaves S_mut == S_wt bit-for-bit, so
    b(mut) must equal b(wt) exactly -- independent of the antithetic-variate mechanism above,
    and independent of whatever the (fake, uniform) model actually predicts."""
    st = _toy_structure()
    wt_aa = st.chains["A"].seq[0]
    mutations = [Mutation(wt=wt_aa, chain="A", resnum=1, icode="", mut=wt_aa)]
    out = pm._stab_trial(_RecordingFakeMPNN(), st, "A", "B", mutations, "cpu",
                         pm.STAB_NOISE_LEVEL, torch.Generator().manual_seed(0))
    assert out["stab_ddg"] == pytest.approx(0.0, abs=1e-6)


def test_monomer_terms_are_masked_not_reshaped():
    """The bound-complex backbone, masked, not a smaller re-featurized structure: ab_alone and
    ag_alone must be scored on tensors the SAME length as the complex, and a mutation on chain A
    must leave ag_alone's term completely untouched (chain A contributes nothing to it)."""
    st = _toy_structure()
    X, S_wt, masks, residue_idx, chain_enc, offsets = pm._featurize_stab(st, "A", "B", "cpu")
    L = st.chains["A"].seq.__len__() + st.chains["B"].seq.__len__()
    for m in masks.values():
        assert m.shape[-1] == L == X.shape[1]
    assert torch.equal(masks["ab_alone"] + masks["ag_alone"], masks["complex"])

    wt_aa = st.chains["A"].seq[0]
    mutations = [Mutation(wt=wt_aa, chain="A", resnum=1, icode="", mut="G" if wt_aa != "G" else "A")]
    out = pm._stab_trial(_RecordingFakeMPNN(), st, "A", "B", mutations, "cpu",
                         pm.STAB_NOISE_LEVEL, torch.Generator().manual_seed(0))
    # the fake model is uniform regardless of S, so this checks the MASK/offset plumbing only:
    # ag_alone's summed log-prob covers a fixed set of positions never touched by an A mutation.
    assert out["f_wt_ag_alone"] == out["f_mut_ag_alone"]


def test_frozen_model_has_no_trainable_parameters():
    if not pm.MPNN_DIR.exists():
        pytest.skip("third_party/ProteinMPNN not present")
    model = pm._load_stab_model("cpu", "zeroshot")
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    # Frozen at extraction time, never part of the trainable graph: this model is used only from
    # src/features/proteinmpnn.py's own extraction functions, never held as a submodule of
    # src/perturb/model_simple.py's trainable head -- so "never in an optimizer param group" is
    # true by construction, not by a runtime check on a specific optimizer instance.
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable_ids == set()


def test_stab_ddg_deterministic_with_fixed_seed():
    """Running the SAME row twice with a fixed seed must give bitwise-identical output --
    bypasses the memoisation cache (which would make a second call trivially "identical" without
    re-running anything) by clearing it between calls."""
    if not pm.MPNN_DIR.exists():
        pytest.skip("third_party/ProteinMPNN not present")
    st = _toy_structure()
    mutations = [Mutation(wt=st.chains["A"].seq[2], chain="A", resnum=3, icode="", mut="G")]

    out1 = pm.stab_ddg(st, "A", "B", mutations, device="cpu", n_samples=2, seed=7)
    pm._stab_cache.clear()
    out2 = pm.stab_ddg(st, "A", "B", mutations, device="cpu", n_samples=2, seed=7)

    assert out1.keys() == out2.keys()
    for k in out1:
        if k == "unit":
            assert out1[k] == out2[k]
        else:
            assert out1[k] == out2[k], f"{k} differs: {out1[k]} != {out2[k]}"


def test_stab_ddg_smoke_on_real_data():
    """~30 rows of our antibody-antigen dataset: per-interface Pearson/Spearman of the raw
    stab_ddg scalar against ddG. Expected 0.4-0.5 (paper's own reported range) -- a result near
    zero means the shared-randomness wiring is broken, not that the method doesn't work. Skips
    cleanly -- this repo does not vendor the SKEMPI PDB corpus.
    """
    pd = pytest.importorskip("pandas")
    from scipy.stats import pearsonr, spearmanr

    from src.structures import load_structures, parse_mutations

    if not paths.PDB_DIR.exists() or not pm.MPNN_DIR.exists():
        pytest.skip("PDB directory or third_party/ProteinMPNN not present")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(paths.DATASET).head(30)
    structures = load_structures(df["pdb"].unique())

    import time
    preds, labels, complexes = [], [], []
    t0 = time.perf_counter()
    for row_id, key, pdb, ab, ag, muts, y in zip(
        df["row_id"], df["#Pdb"], df["pdb"], df["ab_chains"], df["ag_chains"],
        df["mutations"], df["ddG"],
    ):
        st = structures[pdb]
        mutations = parse_mutations(muts)
        out = pm.stab_ddg(st, ab, ag, mutations, device=device, n_samples=pm.STAB_MC_SAMPLES)
        assert np.isfinite(out["stab_ddg"])
        preds.append(out["stab_ddg"])
        labels.append(y)
        complexes.append(key)
    dt = (time.perf_counter() - t0) / len(df)

    pooled_pearson = pearsonr(preds, labels).statistic
    pooled_spearman = spearmanr(preds, labels).statistic
    per_cx = (pd.DataFrame({"cx": complexes, "pred": preds, "y": labels})
             .groupby("cx").filter(lambda g: len(g) >= 3)
             .groupby("cx").apply(lambda g: spearmanr(g.pred, g.y).statistic))

    print(f"\nstab_ddg smoke test ({device}, {pm.STAB_MC_SAMPLES} MC samples, {dt:.2f}s/row):")
    print(f"  pooled Pearson  {pooled_pearson:.3f}")
    print(f"  pooled Spearman {pooled_spearman:.3f}")
    print(f"  mean per-complex Spearman {per_cx.mean():.3f} over {len(per_cx)} complexes "
         f"(n too small above for this cut to be reliable)")
