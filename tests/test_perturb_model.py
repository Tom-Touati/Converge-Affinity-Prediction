"""The four properties section 1 requires, pinned as tests.

(a) a null edit gives exactly zero, (b) the ITW branch is untouched by the structure
refinement, (c) the distance bias respects contact-map ordering, (d) residue indices align
across the mutation string, the ESM input sequence and the PDB-derived residue list.

(a)-(c) are pure-torch and run anywhere. (d) is a data audit over the real dataset and skips
cleanly when the structures or the sequence table are absent.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.perturb.model import (  # noqa: E402
    DIST_MAX,
    N_DIST_BINS,
    PerturbConfig,
    PerturbModel,
    distance_bins,
)


def make_batch(b=3, l_ab=11, l_ag=7, cfg=None, mutate=False, seed=0):
    """A synthetic batch. With ``mutate=False`` the mutant tokens equal the wild type."""
    cfg = cfg or PerturbConfig()
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)

    seq_ab, seq_ag = r(b, l_ab, cfg.seq_dim), r(b, l_ag, cfg.seq_dim)
    site = min(3, l_ab - 1)                   # the toy complex has a single antibody residue
    site_ab = torch.zeros(b, l_ab)
    site_ab[:, site] = 1.0                    # one mutated position on the antibody side
    site_ag = torch.zeros(b, l_ag)

    blos_wt_ab = torch.zeros(b, l_ab, 40)
    blos_wt_ab[:, site, :20] = r(b, 20)          # BLOSUM row of the wild-type residue
    blos_wt_ab[:, site, 20:] = blos_wt_ab[:, site, :20]        # ITW pair is (a, a)
    blos_mt_ab = blos_wt_ab.clone()
    if mutate:
        blos_mt_ab[:, site, 20:] = r(b, 20)      # mutant residue's row differs

    seq_ab_mt = seq_ab.clone()
    if mutate:
        seq_ab_mt[:, site] = seq_ab_mt[:, site] + r(b, cfg.seq_dim)

    dist = torch.rand(b, l_ab, l_ag, generator=g) * 30.0
    return {
        "seq_ab_wt": seq_ab, "seq_ab_mt": seq_ab_mt,
        "seq_ag_wt": seq_ag, "seq_ag_mt": seq_ag.clone(),
        "struct_ab": r(b, l_ab, cfg.struct_dim), "struct_ag": r(b, l_ag, cfg.struct_dim),
        "blosum_ab_wt": blos_wt_ab, "blosum_ab_mt": blos_mt_ab,
        "blosum_ag_wt": torch.zeros(b, l_ag, 40), "blosum_ag_mt": torch.zeros(b, l_ag, 40),
        "site_ab": site_ab, "site_ag": site_ag,
        "near_ab": torch.ones(b, l_ab), "near_ag": torch.ones(b, l_ag),
        "mask_ab": torch.ones(b, l_ab), "mask_ag": torch.ones(b, l_ag),
        "dist_bins": distance_bins(dist),
    }


# --------------------------------------------------------------------------- (a)
def test_null_edit_is_exactly_zero():
    """No mutation must give ddG == 0 exactly, not approximately.

    The head is bias-free precisely so this holds: with a null edit both branches produce the
    same vector, the difference is zero, and a bias-free MLP maps zero to zero.
    """
    model = PerturbModel().eval()
    with torch.no_grad():
        y = model(make_batch(mutate=False))
    assert torch.equal(y, torch.zeros_like(y)), f"null edit gave {y.tolist()}"


def test_a_real_mutation_is_not_zero():
    """The converse, so (a) cannot be passed by a model that always returns zero."""
    model = PerturbModel().eval()
    for p in model.head.parameters():          # FiLMs start at identity, so wake the head
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        y = model(make_batch(mutate=True))
    assert y.abs().max() > 0, "a real mutation produced exactly zero"


# --------------------------------------------------------------------------- (b)
def test_itw_branch_ignores_structure_refinement():
    """Step 3 must be a no-op on ITW whether it is enabled or disabled."""
    batch = make_batch(mutate=True)
    on = PerturbModel(PerturbConfig(delta_film=True)).eval()
    off = PerturbModel(PerturbConfig(delta_film=False)).eval()
    off.load_state_dict(on.state_dict())       # identical weights, only the flag differs
    with torch.no_grad():
        f_on = on.branch(batch, mutated=False)
        f_off = off.branch(batch, mutated=False)
    assert torch.allclose(f_on, f_off, atol=0, rtol=0), \
        f"ITW changed with step 3: max |diff| {(f_on - f_off).abs().max().item():.3e}"


# --------------------------------------------------------------------------- (c)
def test_distance_bias_reproduces_contact_ordering():
    """Closer pairs must land in lower bins, and everything past 20 A in the final bin."""
    d = torch.tensor([[[0.0, 1.0, 4.9, 5.1, 12.0, 19.9, 20.1, 100.0]]])
    b = distance_bins(d).squeeze()
    assert torch.all(b[1:] >= b[:-1]), f"bins not monotone in distance: {b.tolist()}"
    assert b[0] == 0, "zero distance must be the first bin"
    assert b[-1] == N_DIST_BINS - 1 and b[-2] == N_DIST_BINS - 1, \
        f"beyond {DIST_MAX} A must be the last bin, got {b[-2:].tolist()}"
    assert b[4] < b[5], "12 A and 19.9 A must not collapse into one bin"


def test_distance_bias_orders_attention_on_a_toy_complex():
    """With the content term off, a decreasing bias must rank keys by proximity."""
    cfg = PerturbConfig(dist_bias=True, dist_bias_only=True)
    model = PerturbModel(cfg).eval()
    with torch.no_grad():                      # monotone decreasing: near bins score highest
        w = torch.linspace(1.0, -1.0, N_DIST_BINS).unsqueeze(-1)
        model.branch.attn_ab.bias.weight.copy_(w)
        model.branch.attn_ag.bias.weight.copy_(w)
        batch = make_batch(b=1, l_ab=1, l_ag=5, cfg=cfg)
        d = torch.tensor([[[1.0, 6.0, 11.0, 16.0, 21.0]]])
        batch["dist_bins"] = distance_bins(d)
        model.branch(batch, mutated=False)
        attn = model.branch.attn_ab.last_attn[0, 0, 0]
    assert torch.all(attn[:-1] >= attn[1:] - 1e-6), \
        f"attention not ordered by contact distance: {attn.tolist()}"


# --------------------------------------------------------------------------- (d)
def test_residue_indices_align_across_sequence_and_structure():
    """For every row, the mutation's wild-type residue must match both the ESM input
    sequence and the PDB-derived residue list at that index. Mismatches are logged, not
    silently dropped."""
    pd = pytest.importorskip("pandas")
    from src import paths
    from src.structures import load_structures, parse_mutations

    table = (paths.ROOT / "experiments" / "protattba_repro" / "cache"
             / "project_sequences.parquet")
    if not table.exists() or not paths.PDB_DIR.exists():
        pytest.skip("sequence table or PDB directory not present")

    seqs = pd.read_parquet(table)
    data = pd.read_parquet(paths.DATASET)[["row_id", "pdb", "mutations", "ab_chains",
                                           "ag_chains", "side1", "side2"]]
    # project_sequences.parquet already carries ab_chains/ag_chains; drop them so the merge
    # does not produce _x/_y suffixes and the dataset's copy is the one used.
    seqs = seqs.drop(columns=[c for c in ("ab_chains", "ag_chains") if c in seqs.columns])
    m = seqs.merge(data, on="row_id", validate="1:1")
    structures = load_structures(m.pdb.unique())

    bad_struct, bad_seq = [], []
    for r in m.itertuples():
        st = structures[r.pdb]
        ab = r.ab_chains or r.side1
        ag = r.ag_chains or r.side2
        for mut in parse_mutations(r.mutations):
            chain = st.chains.get(mut.chain)
            if chain is None or mut.key not in chain.index:
                bad_struct.append((r.row_id, str(mut.chain), "absent"))
                continue
            pos = chain.index[mut.key]
            if chain.seq[pos] != mut.wt:
                bad_struct.append((r.row_id, str(mut.chain), f"{chain.seq[pos]}!={mut.wt}"))
                continue
            # the same residue, located inside the concatenated side sequence
            group = ab if mut.chain in ab else (ag if mut.chain in ag else None)
            if group is None:
                bad_seq.append((r.row_id, str(mut.chain), "chain in neither side"))
                continue
            offset = sum(len(st.chains[c].seq) for c in group[: group.index(mut.chain)])
            whole = r.ab_wt if group is ab or mut.chain in ab else r.ag_wt
            if offset + pos >= len(whole) or whole[offset + pos] != mut.wt:
                got = whole[offset + pos] if offset + pos < len(whole) else "<past end>"
                bad_seq.append((r.row_id, str(mut.chain), f"{got}!={mut.wt}"))

    print(f"\nalignment audit over {len(m)} rows: "
          f"{len(bad_struct)} structure mismatches, {len(bad_seq)} sequence mismatches")
    for row in (bad_struct + bad_seq)[:10]:
        print("   ", row)
    assert not bad_struct, f"{len(bad_struct)} rows disagree with the PDB, e.g. {bad_struct[:3]}"
    assert not bad_seq, f"{len(bad_seq)} rows disagree with the ESM input, e.g. {bad_seq[:3]}"


def test_blosum_letters_follow_the_site_not_its_rank():
    """The substitution stored at a crop's flagged token must be the one that lands there.

    ``Row.wt_aa`` lists the parts of a multi-point mutation in the order the mutation string
    writes them, which is neither per-side nor sorted by position. Counting within one side --
    ``wt_aa[j]`` for the j-th flagged token -- therefore reads a different mutation's residue
    whenever the two orders disagree: on 8.7% of rows, including every row mutated on both
    sides. It produced no error, only a wrong BLOSUM row, so the check is against the mutation
    string itself rather than against the code that writes the crop.
    """
    pd = pytest.importorskip("pandas")
    from src import paths
    from src.perturb import data as D
    from src.perturb.build_crops import OUT
    from src.structures import load_structures, parse_mutations

    if not OUT.exists() or not paths.PDB_DIR.exists():
        pytest.skip("crop file or PDB directory not present")

    crops = np.load(OUT, allow_pickle=False)
    rows = D.load_rows()
    meta = pd.read_parquet(paths.DATASET).set_index("row_id")
    structures = load_structures({r.pdb for r in rows})

    bad, n_interleaved = [], 0
    for r in rows:
        st = structures[r.pdb]
        ab = meta.loc[r.row_id, "ab_chains"] or meta.loc[r.row_id, "side1"]
        ag = meta.loc[r.row_id, "ag_chains"] or meta.loc[r.row_id, "side2"]
        truth = {}
        for mut in parse_mutations(meta.loc[r.row_id, "mutations"]):
            group = ab if mut.chain in ab else ag
            offset = sum(len(st.chains[c].seq) for c in group[: group.index(mut.chain)])
            side = "ab" if mut.chain in ab else "ag"
            truth[(side, offset + st.chains[mut.chain].index[mut.key])] = (mut.wt, mut.mut)

        for side in ("ab", "ag"):
            idx = crops[f"{r.row_id}|{side}_idx"]
            flags = crops[f"{r.row_id}|{side}_site"]
            wt = crops[f"{r.row_id}|{side}_wt_aa"]
            mt = crops[f"{r.row_id}|{side}_mt_aa"]
            hits = np.flatnonzero(flags)
            assert len(wt) == len(mt) == len(hits), \
                f"{r.row_id} {side}: {len(wt)} letters for {len(hits)} flagged tokens"
            for j, t in enumerate(hits):
                want = truth.get((side, int(idx[t])))
                assert want is not None, f"{r.row_id} {side}: flagged token {idx[t]} is not mutated"
                if want != (str(wt[j]), str(mt[j])):
                    bad.append((r.row_id, side, int(idx[t]), want, (str(wt[j]), str(mt[j]))))
                # what indexing the interleaved list by rank-within-side would have read
                if j < len(r.wt_aa) and r.wt_aa[j] != want[0]:
                    n_interleaved += 1

    print(f"\nBLOSUM letter audit over {len(rows)} rows: {len(bad)} wrong; "
          f"rank-within-side would have been wrong at {n_interleaved} sites")
    assert n_interleaved > 0, "no row exercises the interleaving; the test proves nothing"
    assert not bad, f"{len(bad)} sites carry another mutation's residue, e.g. {bad[:3]}"
