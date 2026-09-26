"""Structure modality, part 2: ProteinMPNN inverse-folding log-odds.

Why ProteinMPNN rather than ESM-IF1 as the primary structure encoder: on the closest published
setup to ours -- SKEMPI antibody-antigen interface single-point mutations -- CIR-DDG measures
zero-shot ProteinMPNN at Spearman 0.172 against ESM-IF's 0.119, and ProteinMPNN is pure PyTorch
with no torch-geometric dependency tree. See BACKBONE_COMPARISON.md.

The score is ``unconditional_probs``: p(amino acid at position i | backbone geometry), with no
sequence input at all. Three consequences, all of them good for us:

* It is deterministic -- no decoding order, no sampling, and ``augment_eps`` is forced to 0.
* Wild type and mutant share one forward pass, because the backbone is identical and only the
  identity read off it changes. LLR = log p(mut) - log p(wt) at the mutated position.
* It is a pure structure signal, uncontaminated by the sequence model we already have.

**The control is the point.** de Kanter & Greiff showed ESM-IF1 predicts a *control* epitope's
affinity change as well as the real one (r 0.64 vs 0.59), i.e. inverse-folding likelihood tracks
protein *quality* rather than *interaction*. Their run was single-chain, so multichain
conditioning is the untested variable. This module therefore scores every mutation twice:

* ``llr_complex`` -- conditioned on the whole antibody-antigen complex.
* ``llr_alone``   -- conditioned on the mutated residue's own side only, partner deleted.
* ``llr_delta``   -- the difference, i.e. the part of the score that the partner's presence
  creates. This is the only column that can contain binding-specific information.

If ``llr_complex`` predicts ddG no better than ``llr_alone``, our structure modality is a
foldability predictor wearing a binding hat, and ``llr_delta`` is where we find out.

    python -m src.features.proteinmpnn --device auto
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..structures import load_structures, parse_mutations
from ..util import resolve_device

MPNN_DIR = paths.ROOT / "third_party" / "ProteinMPNN"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}
DEFAULT_WEIGHTS = "v_48_020.pt"


def _load_model(device: str, weights: str = DEFAULT_WEIGHTS, weights_path=None,
                k_neighbors: int | None = None):
    """``weights_path`` overrides the default ``vanilla_model_weights/`` lookup -- StaB-ddG's
    own checkpoints (see ``_load_stab_model``) live outside the vanilla dauparas repo and are
    not shaped like its checkpoints: no guaranteed ``num_edges``/``noise_level`` metadata, and
    sometimes no ``model_state_dict`` wrapper at all (``run_stabddg.py`` itself branches on
    this). ``k_neighbors`` lets a caller supply the value directly when the checkpoint can't.
    """
    if not MPNN_DIR.exists():
        raise FileNotFoundError(
            f"{MPNN_DIR} not found. Clone it with:\n"
            f"  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git "
            f"third_party/ProteinMPNN"
        )
    sys.path.insert(0, str(MPNN_DIR))
    from protein_mpnn_utils import ProteinMPNN

    path = weights_path or (MPNN_DIR / "vanilla_model_weights" / weights)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0,                    # training-time coordinate noise off: determinism
        k_neighbors=k_neighbors if k_neighbors is not None else ckpt["num_edges"],
    )
    model.load_state_dict(state_dict)
    return model.eval().to(device), ckpt


def _featurize(structure, chain_group: str, device: str):
    """Build ProteinMPNN's tensors straight from our verified Structure.

    Deliberately not using the repo's own ``parse_PDB``: a second parser could disagree with the
    one the EDA validated, and every residue index here has to line up with ``chain.index``.
    Conventions copied from ``tied_featurize``: residue_idx carries a 100-offset per chain so the
    relative positional encoding cannot bridge chains, and chain_encoding is 1-based.
    """
    xs, res_idx, chain_enc, offsets = [], [], [], {}
    cursor = 0
    for c_i, cid in enumerate(chain_group):
        ch = structure.chains.get(cid)
        if ch is None:
            continue
        n = len(ch.seq)
        xs.append(ch.backbone)                              # (n, 4, 3) N, CA, C, O
        res_idx.append(100 * c_i + np.arange(n))
        chain_enc.append(np.full(n, c_i + 1))
        offsets[cid] = cursor
        cursor += n

    X = np.concatenate(xs, 0)[None]                         # (1, L, 4, 3)
    finite = np.isfinite(X).all(axis=(2, 3))                # (1, L)
    X = np.nan_to_num(X, nan=0.0)
    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    return (t(X, torch.float32), t(finite.astype(np.float32), torch.float32),
            t(np.concatenate(res_idx)[None], torch.long),
            t(np.concatenate(chain_enc)[None], torch.long), offsets)


@torch.no_grad()
def _log_probs(model, structure, chain_group: str, device: str):
    """Per-position log p(aa | backbone) for one chain group, plus that group's index offsets."""
    X, mask, residue_idx, chain_enc, offsets = _featurize(structure, chain_group, device)
    lp = model.unconditional_probs(X, mask, residue_idx, chain_enc)
    return lp[0].float().cpu().numpy(), offsets                 # (L, 21)


# ---------------------------------------------------------------------------------------------
# StaB-ddG (Deng et al., arXiv 2507.05502, github.com/LDeng0205/StaB-ddG): the SAME frozen
# ProteinMPNN weights loaded above, scored with the paper's own folding-energy cycle instead of
# unconditional_probs' single-pass per-position marginal. unconditional_probs treats every
# position as independent of every other position's identity -- a pseudo-likelihood. StaB-ddG's
# f(s) = log p(s | backbone) is the actual autoregressive sequence log-likelihood (ProteinMPNN's
# OWN training objective), decomposed along one decoding order, one position at a time, each
# conditioned on the others already decoded. That is a materially different quantity, not just a
# reweighting of the same one -- hence a new code path rather than a new column on the old one.
#
# No vendored StaB-ddG code and no new checkpoint for the zero-shot variant: the paper's own
# ProteinMPNN.forward() modification (stabddg/mpnn_utils.py) turns out to be unnecessary for us
# -- the vanilla dauparas forward() we already load already accepts a fixed decoding order via
# ``use_input_decoding_order``/``decoding_order``; the only other piece of "fixed randomness" the
# paper needs, shared backbone coordinate noise, we add ourselves before calling forward (with
# augment_eps=0 the model contributes none of its own), which is behaviourally identical to their
# ``fix_backbone_noise`` argument without touching the model class at all.
#
# The two monomer terms (f(A), f(B)) are the bound-complex backbone with the partner MASKED out
# of ProteinMPNN's own ``mask`` argument (the padding mechanism ``ProteinFeatures._dist`` already
# uses to exclude positions from the k-nearest-neighbour graph) -- never a separately modelled
# apo structure, and never a separately re-featurized smaller PDB. All three systems (complex,
# ab-alone, ag-alone) and both sequences (wild-type, mutant) share ONE X tensor and ONE
# (decoding_order, backbone_noise) draw per Monte Carlo trial -- six forward passes, one shared
# source of randomness, not three independent ones. This is not a style choice: the paper reports
# per-interface Spearman moves from ~0.30 to ~0.45 on this alone.
# ---------------------------------------------------------------------------------------------

STAB_MC_SAMPLES = 20            # the paper's own default ensemble size
STAB_NOISE_LEVEL = 0.1          # the paper's own default backbone-noise magnitude (Angstrom)
#: variant -> (weights filename, whether it needs the vanilla dauparas weights directory).
#: "stability_finetuned" is ProteinMPNN fine-tuned on Megascale folding-stability data only --
#: never on SKEMPI, so it carries no leakage risk against our own homology-clustered eval, and
#: (per the paper) it is calibrated to real kcal/mol rather than raw log-likelihood units. The
#: SKEMPI-fine-tuned checkpoint (``stabddg.pt``) is deliberately absent from this dict: loading
#: it would mean scoring our own SKEMPI-derived eval rows with a model fine-tuned on SKEMPI.
STAB_VARIANTS = {
    "zeroshot": (DEFAULT_WEIGHTS, True),
    "stability_finetuned": ("stability_finetuned.pt", False),
}
STAB_CKPT_DIR = paths.ROOT / "third_party" / "StaB-ddG-ckpts"
_stab_model_cache: dict = {}     # (device, variant) -> frozen nn.Module, load once per process
_stab_cache: dict = {}           # (pdb, ab, ag, mutation_string, variant, n_samples, noise, seed) -> result dict


def _load_stab_model(device: str, variant: str = "zeroshot"):
    if variant not in STAB_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(STAB_VARIANTS)}, got {variant!r}")
    weights, use_vanilla_dir = STAB_VARIANTS[variant]
    assert "skempi" not in weights.lower(), (
        f"refusing to load {weights!r}: a SKEMPI-fine-tuned checkpoint would leak our own eval "
        f"labels into a feature we are treating as zero-shot/leak-free"
    )
    key = (device, variant)
    if key in _stab_model_cache:
        return _stab_model_cache[key]
    if use_vanilla_dir:
        model, _ = _load_model(device, weights)
    else:
        path = STAB_CKPT_DIR / weights
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download it (after checking with the user -- it is an "
                f"~6.7 MB file from a third party) with:\n"
                f"  curl -L -o {path} "
                f"https://raw.githubusercontent.com/LDeng0205/StaB-ddG/main/model_ckpts/{weights}"
            )
        model, _ = _load_model(device, weights, weights_path=path, k_neighbors=48)
    for p in model.parameters():
        p.requires_grad_(False)             # frozen: not eligible for any optimizer param group
    model = model.eval()
    _stab_model_cache[key] = model
    return model


def _seq_ids(seq: str) -> np.ndarray:
    return np.array([AA_IDX.get(a, AA_IDX["X"]) for a in seq], dtype=np.int64)


def _featurize_stab(structure, ab: str, ag: str, device: str):
    """ONE complex-sized tensor for ALL THREE systems (complex, ab-alone, ag-alone) -- the
    paper's rigid-backbone assumption taken literally. The two monomer terms are the SAME
    backbone with the partner's positions masked out of ProteinMPNN's own ``mask`` argument
    (the mechanism ``ProteinFeatures._dist`` already uses to exclude padding from the k-nearest-
    neighbour graph), never a separately modelled apo structure -- there is exactly one X here,
    reused for every one of the six forward passes in ``_stab_trial``.

    Chain-local ``residue_idx``/``chain_encoding`` numbering (``100*c_i + arange(n)``, one block
    per chain in ``ab+ag`` order) makes this safe: a chain's own relative offsets are identical
    whether its partner is masked in or out, so masking never perturbs the positional encoding
    the way concatenating a *different* set of chains would.
    """
    xs, seqs, res_idx, chain_enc, offsets = [], [], [], [], {}
    cursor = 0
    chain_group = ab + ag
    for c_i, cid in enumerate(chain_group):
        ch = structure.chains.get(cid)
        if ch is None:
            continue
        n = len(ch.seq)
        xs.append(ch.backbone)
        seqs.append(_seq_ids(ch.seq))
        res_idx.append(100 * c_i + np.arange(n))
        chain_enc.append(np.full(n, c_i + 1))
        offsets[cid] = cursor
        cursor += n

    X = np.concatenate(xs, 0)[None]
    S = np.concatenate(seqs, 0)[None]
    L = X.shape[1]
    finite = np.isfinite(X).all(axis=(2, 3)).astype(np.float32)   # (1, L): real residue vs gap
    X = np.nan_to_num(X, nan=0.0)

    is_ab = np.zeros((1, L), dtype=np.float32)
    for cid in ab:
        if cid in offsets:
            n = len(structure.chains[cid].seq)
            is_ab[0, offsets[cid]:offsets[cid] + n] = 1.0
    masks_np = {"complex": finite, "ab_alone": finite * is_ab, "ag_alone": finite * (1.0 - is_ab)}

    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    return (t(X, torch.float32), t(S, torch.long),
            {name: t(m, torch.float32) for name, m in masks_np.items()},
            t(np.concatenate(res_idx)[None], torch.long),
            t(np.concatenate(chain_enc)[None], torch.long), offsets)


@torch.no_grad()
def _stab_logp(model, X, S, mask, residue_idx, chain_enc, decoding_order, backbone_noise):
    """StaB-ddG's f(s): the teacher-forced sequence log-likelihood log p(S | noised backbone),
    restricted to ``mask``'s positions, decomposed along ONE FIXED decoding order, under ONE
    FIXED backbone-coordinate noise draw shared across every call in a trial (see
    ``_stab_trial``) -- the paper reports this is where the signal comes from: without shared
    randomness, per-interface Spearman drops from ~0.45 to ~0.30.
    """
    chain_M = torch.ones_like(mask)               # nothing is "given"; every position is scored
    randn_dummy = torch.zeros_like(mask)           # unused: use_input_decoding_order=True below
    log_probs = model(X + backbone_noise, S, mask, chain_M, residue_idx, chain_enc, randn_dummy,
                      use_input_decoding_order=True, decoding_order=decoding_order)
    ll = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)   # (B, L)
    return (ll * mask).sum(dim=1)                                  # (B,)


def _mutate(S_wt: torch.Tensor, structure, offsets: dict, mutations) -> torch.Tensor:
    S_mut = S_wt.clone()
    for m in mutations:
        if m.chain not in offsets:
            continue
        pos = offsets[m.chain] + structure.chains[m.chain].index[m.key]
        S_mut[0, pos] = AA_IDX[m.mut]
    return S_mut


def _stab_trial(model, structure, ab: str, ag: str, mutations, device: str,
                noise_level: float, gen: torch.Generator) -> dict:
    """One Monte Carlo trial of StaB-ddG's cycle: b(s) = f(AB) - f(A) - f(B), for both the
    wild-type and the mutant sequence. ONE decoding permutation and ONE backbone-noise draw are
    sampled here and reused across ALL SIX forward passes below (not just within a wt/mut pair)
    -- mandatory, per the paper: this is the variance-reduction mechanism the reported
    Spearman ~0.45 depends on, not an optional refinement.
    """
    X, S_wt, masks, residue_idx, chain_enc, offsets = _featurize_stab(structure, ab, ag, device)
    S_mut = _mutate(S_wt, structure, offsets, mutations)
    order = torch.argsort(torch.abs(torch.randn(masks["complex"].shape, generator=gen))).to(device)
    noise = (noise_level * torch.randn(X.shape, generator=gen)).to(device)

    terms = {}
    for name, mask in masks.items():                  # complex, ab_alone, ag_alone, in order
        f_wt = _stab_logp(model, X, S_wt, mask, residue_idx, chain_enc, order, noise)
        f_mut = _stab_logp(model, X, S_mut, mask, residue_idx, chain_enc, order, noise)
        terms[f"f_wt_{name}"] = float(f_wt.item())
        terms[f"f_mut_{name}"] = float(f_mut.item())

    b_wt = terms["f_wt_complex"] - terms["f_wt_ab_alone"] - terms["f_wt_ag_alone"]
    b_mut = terms["f_mut_complex"] - terms["f_mut_ab_alone"] - terms["f_mut_ag_alone"]
    # StaB-ddG's own convention (run_stabddg.py negates its raw cycle output): raw b_mut - b_wt
    # is positive when the MUTANT is the more favourable (higher-log-likelihood) sequence, i.e.
    # positive = stabilising. Our ddg_true, like SKEMPI's own convention, is positive =
    # DESTABILISING -- so flip the sign once here, at the source, not at every consumer.
    terms["stab_ddg"] = -(b_mut - b_wt)
    return terms


def stab_ddg(structure, ab: str, ag: str, mutations, *, device: str = "auto",
            variant: str = "zeroshot", n_samples: int = STAB_MC_SAMPLES,
            noise_level: float = STAB_NOISE_LEVEL, seed: int = 0) -> dict:
    """StaB-ddG's binding ddG for one mutation, averaged over ``n_samples`` antithetic Monte
    Carlo trials. Returns the six per-system f(...) terms (also averaged over trials -- our own
    head may recombine them better than the fixed b = f(AB) - f(A) - f(B) cycle does), the final
    ``stab_ddg`` scalar, and ``unit`` ("log-likelihood" for zero-shot, "kcal/mol" for
    stability_finetuned -- branch on this downstream rather than assuming one scale).

    Cached by every argument that changes the answer: a complex is scored by many mutations, and
    re-running six ProteinMPNN forward passes per row when only the mutation differs is pure
    waste. ``noise_level`` is included even though the task's own cache-key list omitted it --
    leaving it out would silently reuse a stale answer for anyone who sweeps it.
    """
    device = resolve_device(device)
    mut_str = ",".join(str(m) for m in mutations)
    key = (structure.pdb, ab, ag, mut_str, variant, n_samples, noise_level, seed)
    if key in _stab_cache:
        return _stab_cache[key]

    model = _load_stab_model(device, variant)
    gen = torch.Generator().manual_seed(seed)
    trials = [_stab_trial(model, structure, ab, ag, mutations, device, noise_level, gen)
             for _ in range(n_samples)]
    out = {k: float(np.mean([t[k] for t in trials])) for k in trials[0]}
    out["unit"] = "log-likelihood" if variant == "zeroshot" else "kcal/mol"
    _stab_cache[key] = out
    return out


def extract(device: str = "auto", weights: str = DEFAULT_WEIGHTS,
            verbose: bool = True) -> pd.DataFrame:
    paths.ensure_dirs()
    device = resolve_device(device)
    df = pd.read_parquet(paths.DATASET)
    structures = load_structures(df["pdb"].unique())
    model, ckpt = _load_model(device, weights)
    if verbose:
        print(f"ProteinMPNN {weights}: k_neighbors={ckpt['num_edges']} "
              f"noise_level={ckpt['noise_level']} device={device}", flush=True)

    cx = df.drop_duplicates("#Pdb")
    cache = {}
    t0 = time.perf_counter()
    for n, (key, pdb, abc, agc) in enumerate(
        zip(cx["#Pdb"], cx["pdb"], cx["ab_chains"], cx["ag_chains"])
    ):
        st = structures[pdb]
        ab = abc or key.split("_")[1]
        ag = agc or key.split("_")[2]
        entry = {"complex": _log_probs(model, st, ab + ag, device)}
        # each side scored with the partner deleted -- the control
        entry["ab"] = _log_probs(model, st, ab, device)
        entry["ag"] = _log_probs(model, st, ag, device)
        entry["sides"] = {c: "ab" for c in ab}
        entry["sides"].update({c: "ag" for c in ag})
        cache[key] = entry
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(cx)} complexes  {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for row_id, key, pdb, mutations in zip(df["row_id"], df["#Pdb"], df["pdb"], df["mutations"]):
        st = structures[pdb]
        e = cache[key]
        llr_c = llr_a = 0.0
        p_wt_c = p_mut_c = 0.0
        for m in parse_mutations(mutations):
            pos_in_chain = st.chains[m.chain].index[m.key]
            iw, im = AA_IDX[m.wt], AA_IDX[m.mut]

            lp_c, off_c = e["complex"]
            i_c = off_c[m.chain] + pos_in_chain
            llr_c += float(lp_c[i_c, im] - lp_c[i_c, iw])
            p_wt_c += float(lp_c[i_c, iw])
            p_mut_c += float(lp_c[i_c, im])

            side = e["sides"][m.chain]
            lp_a, off_a = e[side]
            i_a = off_a[m.chain] + pos_in_chain
            llr_a += float(lp_a[i_a, im] - lp_a[i_a, iw])

        rows.append({
            "row_id": row_id,
            "llr_complex": llr_c,
            "llr_alone": llr_a,
            "llr_delta": llr_c - llr_a,      # the only binding-specific column
            "logp_wt_complex": p_wt_c,
            "logp_mut_complex": p_mut_c,
        })

    block = pd.DataFrame(rows).set_index("row_id").astype(np.float32)
    out = paths.FEATURES / "mpnn.parquet"
    block.to_parquet(out)
    if verbose:
        print(f"wrote {out.relative_to(paths.ROOT)}  {block.shape}  "
              f"in {time.perf_counter()-t0:.1f}s")
    return block


def build(df: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    return extract(verbose=verbose)


def extract_stab(device: str = "auto", variant: str = "zeroshot",
                 n_samples: int = STAB_MC_SAMPLES, noise_level: float = STAB_NOISE_LEVEL,
                 seed: int = 0, n_rows: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Writes ``data/features/stab_ddg.parquet``: the seven StaB-ddG columns (six per-system
    f(...) terms, kept separate -- see ``_stab_trial`` -- plus the assembled ``stab_ddg``
    double-difference) for every row, alongside ``unit`` so a downstream consumer can tell
    log-likelihood (zero-shot) from kcal/mol (stability_finetuned) apart. ``n_rows`` truncates
    the dataset for a quick run (the smoke test uses it); omit it for the full extraction.
    """
    paths.ensure_dirs()
    device = resolve_device(device)
    df = pd.read_parquet(paths.DATASET)
    if n_rows is not None:
        df = df.head(n_rows)
    structures = load_structures(df["pdb"].unique())
    _load_stab_model(device, variant)          # loaded once here; stab_ddg() reuses it via cache
    if verbose:
        print(f"StaB-ddG {variant}: n_samples={n_samples} noise_level={noise_level} "
              f"device={device}", flush=True)

    rows = []
    t0 = time.perf_counter()
    for n, (row_id, pdb, ab, ag, muts) in enumerate(
        zip(df["row_id"], df["pdb"], df["ab_chains"], df["ag_chains"], df["mutations"])
    ):
        st = structures[pdb]
        mutations = parse_mutations(muts)
        out = stab_ddg(st, ab, ag, mutations, device=device, variant=variant,
                       n_samples=n_samples, noise_level=noise_level, seed=seed)
        rows.append({"row_id": row_id, **{f"stab__{k}": v for k, v in out.items() if k != "unit"},
                    "stab__unit": out["unit"]})
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(df)} rows  {time.perf_counter()-t0:.0f}s", flush=True)

    # row_id stays a plain COLUMN (not the index) -- matches chem_perturb_v2.parquet's own
    # on-disk convention, which experiments/protattba_repro/_perturb_v2_colab.py's Cache reads
    # via ``pd.read_parquet(f).set_index("row_id")``, not by assuming an index is already there.
    # stab__unit is a genuine per-row string column, not a DataFrame.attrs sidecar -- attrs is
    # not guaranteed to round-trip through every parquet engine/version, and the unit distinction
    # (log-likelihood vs kcal/mol) is exactly the kind of thing that must not go missing silently.
    block = pd.DataFrame(rows)
    num_cols = [c for c in block.columns if c not in ("row_id", "stab__unit")]
    block[num_cols] = block[num_cols].astype(np.float32)
    suffix = "" if variant == "zeroshot" else f"_{variant}"
    out_path = paths.FEATURES / f"stab_ddg{suffix}.parquet"
    block.to_parquet(out_path, index=False)
    if verbose:
        print(f"wrote {out_path.relative_to(paths.ROOT)}  {block.shape}  unit={unit}  "
              f"in {time.perf_counter()-t0:.1f}s")
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--stab", action="store_true",
                   help="extract StaB-ddG's 7 columns (data/features/stab_ddg.parquet) instead "
                        "of the unconditional_probs block")
    p.add_argument("--stab-variant", default="zeroshot", choices=sorted(STAB_VARIANTS))
    p.add_argument("--stab-mc-samples", type=int, default=STAB_MC_SAMPLES)
    p.add_argument("--stab-noise-level", type=float, default=STAB_NOISE_LEVEL)
    p.add_argument("--stab-seed", type=int, default=0)
    p.add_argument("--stab-n-rows", type=int, default=None)
    a = p.parse_args()
    if a.stab:
        extract_stab(a.device, a.stab_variant, a.stab_mc_samples, a.stab_noise_level,
                    a.stab_seed, a.stab_n_rows)
    else:
        extract(a.device, a.weights)


if __name__ == "__main__":
    main()
