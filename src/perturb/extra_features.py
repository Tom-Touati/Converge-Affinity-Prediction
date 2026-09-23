"""Build every cache the perturbation trainer needs for the extra AB645/AB1101 complexes.

``extra_rows.py`` produces the rows; this produces the features they refer to. The existing
builders all read ``paths.DATASET`` directly and so only ever cover our own 53 complexes --
rather than parameterise four separate modules, this does the same work for an arbitrary
rows table and writes into the same caches, so downstream code needs no change at all.

Four artefacts, in dependency order:

    1. Ca distance matrices      data/features/perturb_dist/<complex_key>.npy
    2. ProteinMPNN per residue   data/features/mpnn_per_residue/<complex_key>.npz
    3. crops                     merged into data/features/perturb_crops.npz
    4. sequences for the ESM pass appended to cache/project_sequences.parquet

The ESM extraction itself runs on the VM (``_esm_colab.py``), which reads that parquet, so
step 4 is what makes the extra sequences appear in the token cache on the next bootstrap.

Chain roles come from the rows table, not re-derived: ``extra_rows`` decided which chains
are the Fv pair, and deciding again here risks the two disagreeing silently.

Run: ``python -m src.perturb.extra_features [--skip-mpnn]``
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import paths
from src.perturb import data as D
from src.perturb.build_crops import OUT as CROPS
from src.perturb.crop import R_IFACE_DEFAULT, R_SITE_DEFAULT, build_crop, chain_layout

# Defined here rather than imported from extra_rows: that module pulls in src.fusion for
# the benchmark loaders, which this one does not need and which is not always present
# wherever the feature build has to run.
EXTRA_ROWS = (paths.ROOT / "experiments" / "protattba_repro" / "cache"
              / "perturb_rows_extra.parquet")
MPNN_CACHE = paths.FEATURES / "mpnn_per_residue"
SEQ_TABLE = paths.ROOT / "experiments" / "protattba_repro" / "cache" / "project_sequences.parquet"


def chains_of(complex_key: str) -> tuple[str, str]:
    """``1T83_AB_C`` -> ("AB", "C"). The key was built by extra_rows and carries the roles."""
    parts = complex_key.split("_")
    return parts[1], parts[2]


def build_dist(rows: pd.DataFrame, verbose: bool = True) -> int:
    from src.structures import load_structures
    from src.perturb.data import ca_distance_matrix   # lives with the cache it fills

    D.DIST_DIR.mkdir(parents=True, exist_ok=True)
    cx = rows.drop_duplicates("complex_key")
    structures = load_structures(cx.pdb.unique())
    n = 0
    for r in cx.itertuples():
        out = D.DIST_DIR / f"{r.complex_key}.npy"
        if out.exists():
            continue
        ab, ag = chains_of(r.complex_key)
        np.save(out, ca_distance_matrix(structures[r.pdb], ab, ag))
        n += 1
    if verbose:
        print(f"distance: {n} written, {len(list(D.DIST_DIR.glob('*.npy')))} total")
    return n


def build_mpnn(rows: pd.DataFrame, device: str = "auto", verbose: bool = True) -> int:
    from src.features.mpnn_repr import encoder_h_V
    from src.features.proteinmpnn import DEFAULT_WEIGHTS, _load_model
    from src.structures import load_structures
    from src.util import resolve_device

    dev = resolve_device(device)
    cx = rows.drop_duplicates("complex_key")
    todo = [r for r in cx.itertuples() if not (MPNN_CACHE / f"{r.complex_key}.npz").exists()]
    if verbose:
        print(f"mpnn: {len(todo)} complexes to compute, device {dev}")
    if not todo:
        return 0
    MPNN_CACHE.mkdir(parents=True, exist_ok=True)
    structures = load_structures([r.pdb for r in todo])
    model, _ = _load_model(dev, DEFAULT_WEIGHTS)

    for r in todo:
        st = structures[r.pdb]
        ab, ag = chains_of(r.complex_key)
        h_cx, off_cx = encoder_h_V(model, st, ab + ag, dev)
        h_ab, off_ab = encoder_h_V(model, st, ab, dev)
        h_ag, off_ag = encoder_h_V(model, st, ag, dev)
        # A residue's row must be findable later, so assert the offsets span the field.
        expected = sum(len(st.chains[c].seq) for c in ab + ag if c in st.chains)
        if h_cx.shape[0] != expected:
            raise RuntimeError(f"{r.complex_key}: complex field has {h_cx.shape[0]} rows "
                               f"but the chains total {expected}; offsets would be wrong")
        np.savez_compressed(
            MPNN_CACHE / f"{r.complex_key}.npz",
            h_complex=h_cx.astype(np.float16), h_ab=h_ab.astype(np.float16),
            h_ag=h_ag.astype(np.float16),
            meta=np.array(json.dumps({
                "complex_key": r.complex_key, "pdb": r.pdb, "chains_ab": ab,
                "chains_ag": ag, "off_complex": off_cx, "off_ab": off_ab,
                "off_ag": off_ag, "dim": int(h_cx.shape[1]), "weights": DEFAULT_WEIGHTS})))
        if verbose:
            print(f"  {r.complex_key}: {h_cx.shape}")
    return len(todo)


def build_crops(rows: pd.DataFrame, verbose: bool = True) -> int:
    """Append to the existing crop file rather than rebuilding it.

    The trainer opens one npz keyed by row id, so the extra rows have to live in the same
    file. Rewriting it from scratch would mean recomputing all 940 of ours for no reason
    and risking a difference between the two halves.
    """
    from src.structures import load_structures

    # Open the archive ONCE. Written as a comprehension over np.load(...)[k] it reopens and
    # re-decompresses the whole file for every one of the ~13,000 keys, which does not
    # error -- it just never finishes.
    store: dict[str, np.ndarray] = {}
    if CROPS.exists():
        with np.load(CROPS, allow_pickle=False) as z:
            for k in z.files:
                store[k] = z[k]
    before = len({k.rsplit("|", 1)[0] for k in store})
    structures = load_structures(rows.pdb.unique())

    for r in rows.itertuples():
        if f"{r.row_id}|ab_idx" in store:
            continue
        st = structures[r.pdb]
        ab, ag = chains_of(r.complex_key)
        lens = {c: len(st.chains[c].seq) for c in st.chains}
        t_ab, i_ab, _ = chain_layout(ab, lens, "ab")
        t_ag, i_ag, _ = chain_layout(ag, lens, "ag")
        s_ab = [int(x) for x in str(r.sites_ab).split(",") if x != ""]
        s_ag = [int(x) for x in str(r.sites_ag).split(",") if x != ""]

        dist = np.load(D.DIST_DIR / f"{r.complex_key}.npy")
        c = build_crop(dist, s_ab, s_ag, len(t_ab), len(t_ag), t_ab, t_ag, i_ab, i_ag,
                       r_iface=R_IFACE_DEFAULT, r_site=R_SITE_DEFAULT)
        d = np.full((len(t_ab), len(t_ag)), 99.0, np.float32)
        d[: dist.shape[0], : dist.shape[1]] = dist[: len(t_ab), : len(t_ag)]
        sub = d[np.ix_(c.ab_idx, c.ag_idx)] if c.ab_idx.size and c.ag_idx.size else \
            np.zeros((len(c.ab_idx), len(c.ag_idx)), np.float32)

        k = r.row_id
        store[f"{k}|ab_idx"] = c.ab_idx.astype(np.int32)
        store[f"{k}|ag_idx"] = c.ag_idx.astype(np.int32)
        store[f"{k}|ab_chain"] = c.ab_chain_type.astype(np.int8)
        store[f"{k}|ag_chain"] = c.ag_chain_type.astype(np.int8)
        store[f"{k}|ab_res"] = c.ab_res_index.astype(np.int32)
        store[f"{k}|ag_res"] = c.ag_res_index.astype(np.int32)
        store[f"{k}|ab_site"] = c.ab_is_site.astype(np.float32)
        store[f"{k}|ag_site"] = c.ag_is_site.astype(np.float32)
        store[f"{k}|dist"] = sub.astype(np.float16)
        # the substitution letters, per side, in crop order -- the same alignment the main
        # builder does, and for the same reason: indexing a row's global list by a count
        # within one side reads another mutation's residue
        wt_aa, mt_aa = list(str(r.wt_aa)), list(str(r.mt_aa))
        order_all = [("ab", p) for p in s_ab] + [("ag", p) for p in s_ag]
        for side, idx, flags, sites in (("ab", c.ab_idx, c.ab_is_site, s_ab),
                                        ("ag", c.ag_idx, c.ag_is_site, s_ag)):
            at = {p: order_all.index((side, p)) for p in sites}
            order = [at[int(idx[j])] for j in np.flatnonzero(flags)]
            store[f"{k}|{side}_wt_aa"] = np.array([wt_aa[o] for o in order], dtype="U1")
            store[f"{k}|{side}_mt_aa"] = np.array([mt_aa[o] for o in order], dtype="U1")

    np.savez_compressed(CROPS, **store)
    after = len({k.rsplit("|", 1)[0] for k in store})
    if verbose:
        print(f"crops: {before} -> {after} rows in {CROPS.relative_to(paths.ROOT)} "
              f"({CROPS.stat().st_size / 1e6:.1f} MB)")
    return after - before


def append_sequences(rows: pd.DataFrame, verbose: bool = True) -> int:
    """Put the extra sequences where ``_esm_colab.py`` will find them on the next pass."""
    if not SEQ_TABLE.exists():
        raise SystemExit(f"{SEQ_TABLE} not found")
    seq = pd.read_parquet(SEQ_TABLE)
    add = rows[["row_id", "ab_wt", "ag_wt", "ab_mt", "ag_mt"]].copy()
    add = add[~add.row_id.isin(set(seq.row_id))]
    for c in seq.columns:
        if c not in add.columns:
            add[c] = pd.NA
    out = pd.concat([seq, add[seq.columns]], ignore_index=True)
    out.to_parquet(SEQ_TABLE, index=False)
    if verbose:
        distinct = pd.unique(out[["ab_wt", "ag_wt", "ab_mt", "ag_mt"]].values.ravel())
        print(f"sequences: {len(seq)} -> {len(out)} rows, "
              f"{len([d for d in distinct if isinstance(d, str)])} distinct to embed")
    return len(add)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-mpnn", action="store_true",
                    help="ProteinMPNN needs its weights; skip to build the rest")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    rows = pd.read_parquet(EXTRA_ROWS)
    print(f"{len(rows)} extra rows over {rows.complex_key.nunique()} complexes\n")
    build_dist(rows)
    if not a.skip_mpnn:
        build_mpnn(rows, a.device)
    build_crops(rows)
    append_sequences(rows)
    print("\nremaining: re-run the ESM pass on the VM so the new sequences are embedded")


if __name__ == "__main__":
    main()
