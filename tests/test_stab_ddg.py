"""Unit tests for the StaB-ddG (Deng et al., arXiv 2507.05502) binding-ddG parameterization
added to src/features/proteinmpnn.py: the antithetic-variate invariant the paper's own ablation
says the signal depends on (shared decoding order + backbone noise between the wild-type and
mutant forward pass -- without it, per-interface Spearman drops 0.45 -> 0.30), plus a real-data
smoke test.

The first two tests are pure-torch, need no PDB corpus and no ProteinMPNN weights (a fake model
stands in), and run anywhere -- same split as test_perturb_model.py's (a)-(c) vs (d). The smoke
test needs the real SKEMPI PDB structures this repo does not vendor, and skips cleanly when
they (or third_party/ProteinMPNN) are absent, matching that file's own skip convention.
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
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), n))
        backbone = rng.normal(size=(n, 4, 3)).astype(np.float32)
        return Chain(id=cid, seq=seq, keys=[(i + 1, "") for i in range(n)],
                    index={(i + 1, ""): i for i in range(n)}, backbone=backbone)
    return Structure(pdb="TOY", chains={"A": chain("A", l_ab, 0), "B": chain("B", l_ag, 1)})


class _RecordingFakeMPNN:
    """Stands in for a loaded ProteinMPNN: log_probs are uniform (don't depend on the input),
    but every call's decoding_order and noised backbone are recorded, so the test can assert
    the wild-type and mutant calls for one system received bit-identical copies of both.
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

    # Two calls per system (wt, then mut) x 3 systems (complex, ab_alone, ag_alone) = 6.
    assert len(fake.calls) == 6
    for wt_call, mut_call in zip(fake.calls[0::2], fake.calls[1::2]):
        assert torch.equal(wt_call["decoding_order"], mut_call["decoding_order"])
        assert torch.equal(wt_call["X"], mut_call["X"])
    # And trials are NOT frozen across MC samples -- a second trial must redraw both.
    fake2 = _RecordingFakeMPNN()
    gen2 = torch.Generator().manual_seed(0)
    pm._stab_trial(fake2, st, "A", "B", mutations, "cpu", pm.STAB_NOISE_LEVEL, gen2)
    pm._stab_trial(fake2, st, "A", "B", mutations, "cpu", pm.STAB_NOISE_LEVEL, gen2)
    first_order = fake2.calls[0]["decoding_order"]
    second_trial_order = fake2.calls[6]["decoding_order"]
    assert not torch.equal(first_order, second_trial_order)


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


def test_stab_ddg_smoke_on_real_data():
    """~30 rows of our antibody-antigen dataset: per-complex Spearman of stab_ddg against
    ddG, no NaNs, and runtime per mutation against the existing unconditional_probs pass
    (_log_probs/extract). Skips cleanly -- this repo does not vendor the SKEMPI PDB corpus.
    """
    pd = pytest.importorskip("pandas")
    from scipy.stats import spearmanr

    from src.structures import load_structures, parse_mutations

    if not paths.PDB_DIR.exists() or not pm.MPNN_DIR.exists():
        pytest.skip("PDB directory or third_party/ProteinMPNN not present")

    df = pd.read_parquet(paths.DATASET).head(30)
    structures = load_structures(df["pdb"].unique())
    model = pm._load_stab_model("cpu", "zeroshot")

    import time
    preds, labels, complexes = [], [], []
    t_stab0 = time.perf_counter()
    for row_id, key, pdb, ab, ag, muts, y in zip(
        df["row_id"], df["#Pdb"], df["pdb"], df["ab_chains"], df["ag_chains"],
        df["mutations"], df["ddG"],
    ):
        st = structures[pdb]
        mutations = parse_mutations(muts)
        out = pm.stab_ddg(st, ab, ag, mutations, device="cpu", n_samples=4)
        assert np.isfinite(out["stab_ddg"])
        preds.append(out["stab_ddg"])
        labels.append(y)
        complexes.append(key)
    t_stab = (time.perf_counter() - t_stab0) / len(df)

    t_unc0 = time.perf_counter()
    cx = df.drop_duplicates("#Pdb")
    for key, pdb, ab, ag in zip(cx["#Pdb"], cx["pdb"], cx["ab_chains"], cx["ag_chains"]):
        st = structures[pdb]
        pm._log_probs(model, st, ab + ag, "cpu")
        pm._log_probs(model, st, ab, "cpu")
        pm._log_probs(model, st, ag, "cpu")
    t_unc = (time.perf_counter() - t_unc0) / len(df)

    per_cx = (pd.DataFrame({"cx": complexes, "pred": preds, "y": labels})
             .groupby("cx").filter(lambda g: len(g) >= 3)
             .groupby("cx").apply(lambda g: spearmanr(g.pred, g.y).statistic))
    print(f"\nstab_ddg: mean per-complex Spearman {per_cx.mean():.3f} over {len(per_cx)} "
         f"complexes; {t_stab:.2f}s/mutation vs unconditional_probs' {t_unc:.2f}s/mutation "
         f"({t_stab / t_unc:.0f}x)")
