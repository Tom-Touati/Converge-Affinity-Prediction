"""Cache ProteinMPNN's per-residue encoder field, unreduced. The structure half of E1-E5.

``src/features/mpnn_repr.py`` already computes exactly the tensor the fusion head needs --
``encoder_h_V`` returns ``(L, 128)`` per-residue geometric features plus the chain offsets that
locate a residue inside it -- and then reduces it to 384 pooled numbers per mutation for its
parquet. This module reuses that function verbatim and writes the field out instead of
reducing it. Nothing in ``src/features/`` is modified (plan rule 9), and there is no second
implementation of the encoder to drift out of step.

One cache entry per **complex**, not per mutation, for two reasons:

* ``h_V`` comes from backbone geometry alone -- no sequence enters the encoder -- so it
  describes the *site*, identically for whichever residue occupies it. It is therefore the
  same for the wild type and the mutant, which is exactly what the plan's E1 specifies
  ("MPNN features come from the WT complex and are shared by the WT and mutant branches").
* 53 complexes against 940 rows, so per-complex caching is ~18x less work and storage.

Three fields are stored per complex, matching what ``mpnn_repr.build`` computes: the encoder
run over the whole complex, over the antibody side alone, and over the antigen side alone. The
difference between the complex-conditioned and side-alone fields is the only place binding
context enters a structure feature, so both are kept rather than just the complex.

Layout, one ``.npz`` per complex keyed by ``#Pdb``:

    h_complex  (L_ab+L_ag, 128)   encoder over both sides together
    h_ab       (L_ab, 128)        antibody side alone
    h_ag       (L_ag, 128)        antigen side alone
    off_*      json in `meta`     {chain_id: start index} for each of the three
    chains_ab / chains_ag         the chain ids, after the 1DVF tie fallback

Run: ``python -m src.fusion.mpnn_per_residue [--device auto|cpu|cuda]``
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import paths
from src.features.mpnn_repr import encoder_h_V
from src.features.proteinmpnn import DEFAULT_WEIGHTS, _load_model
from src.util import resolve_device
from src.structures import load_structures

CACHE = paths.FEATURES / "mpnn_per_residue"
BENCH_CACHE = paths.FEATURES / "mpnn_per_residue_benchmarks"


def sides_for(key: str, ab_chains: str, ag_chains: str) -> tuple[str, str]:
    """(antibody chains, antigen chains) with the project's tie fallback.

    ``assign_roles`` returns empty strings for 1DVF, the anti-idiotope antibody-antibody pair,
    and every consumer in the repo recovers with ``r["ag_chains"] or r["side2"]``. ``#Pdb`` is
    ``<pdb>_<side1>_<side2>``, so the fallback is available from the key itself -- which is how
    ``mpnn_repr.build`` does it, and this matches so the two caches describe the same residues.
    """
    return (ab_chains or key.split("_")[1]), (ag_chains or key.split("_")[2])


def resolve_chains(structure, partners: str) -> tuple[str, str]:
    """(antibody chains, antigen chains) from a ``side1_side2`` string, with a fallback.

    Their csv's ``Partners`` column follows SKEMPI's chain naming, but five AB645/AB1101
    complexes ship a structure from AB-bind whose chains were renamed -- 1MLC is annotated
    ``AB_E`` while the file holds chains E, H and L. Taking the annotation literally drops
    those complexes silently.

    The fallback uses the immunoglobulin naming convention the files themselves follow: H
    and L are the antibody's heavy and light chains, so everything else is the antigen. It
    only fires when the annotation cannot be honoured, and only when the structure actually
    has an H or L chain, so it cannot quietly override a valid annotation.
    """
    side1, _, side2 = partners.partition("_")
    ab = "".join(c for c in side1 if c in structure.chains)
    ag = "".join(c for c in side2 if c in structure.chains)
    if ab and ag:
        return ab, ag

    heavy_light = "".join(c for c in "HL" if c in structure.chains)
    if heavy_light:
        rest = "".join(sorted(c for c in structure.chains if c not in heavy_light))
        if rest:
            return heavy_light, rest
    return "", ""

def build(device: str = "auto", weights: str = DEFAULT_WEIGHTS, force: bool = False,
          verbose: bool = True) -> int:
    device = resolve_device(device)
    df = pd.read_parquet(paths.DATASET)
    cx = df.drop_duplicates("#Pdb")
    CACHE.mkdir(parents=True, exist_ok=True)

    todo = [k for k in cx["#Pdb"] if force or not (CACHE / f"{k}.npz").exists()]
    if verbose:
        print(f"{len(cx)} complexes, {len(todo)} to compute, device {device}")
    if not todo:
        return 0

    structures = load_structures(cx["pdb"].unique())
    model, _ = _load_model(device, weights)
    t0 = time.perf_counter()
    written = 0

    for n, (key, pdb, abc, agc) in enumerate(
        zip(cx["#Pdb"], cx["pdb"], cx["ab_chains"], cx["ag_chains"])
    ):
        if key not in todo:
            continue
        st = structures[pdb]
        ab, ag = sides_for(key, abc, agc)

        h_cx, off_cx = encoder_h_V(model, st, ab + ag, device)
        h_ab, off_ab = encoder_h_V(model, st, ab, device)
        h_ag, off_ag = encoder_h_V(model, st, ag, device)

        # A residue's row must be findable later, so assert the offsets span the field.
        expected = sum(len(st.chains[c].seq) for c in ab + ag if c in st.chains)
        if h_cx.shape[0] != expected:
            raise RuntimeError(
                f"{key}: complex field has {h_cx.shape[0]} rows but the chains total "
                f"{expected} residues; offsets would be wrong"
            )

        np.savez_compressed(
            CACHE / f"{key}.npz",
            h_complex=h_cx.astype(np.float16), h_ab=h_ab.astype(np.float16),
            h_ag=h_ag.astype(np.float16),
            meta=np.array(json.dumps({
                "complex_key": key, "pdb": pdb, "chains_ab": ab, "chains_ag": ag,
                "off_complex": off_cx, "off_ab": off_ab, "off_ag": off_ag,
                "dim": int(h_cx.shape[1]), "weights": weights,
            })),
        )
        written += 1
        if verbose and written % 10 == 0:
            print(f"  {written}/{len(todo)}  {time.perf_counter() - t0:.0f}s", flush=True)

    if verbose:
        total = sum(p.stat().st_size for p in CACHE.glob("*.npz"))
        print(f"wrote {written} complexes to {CACHE.relative_to(paths.ROOT)} "
              f"({total / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f}s")
    return written


def load(complex_key: str) -> dict:
    """One complex's per-residue fields and the offsets that index them.

    fp16 on disk, returned as fp32: the field is a geometric descriptor with values of order 1,
    so half precision costs nothing that matters and halves a cache that E1-E5 will read every
    epoch.
    """
    path = CACHE / f"{complex_key}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run: python -m src.fusion.mpnn_per_residue"
        )
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {
        "h_complex": z["h_complex"].astype(np.float32),
        "h_ab": z["h_ab"].astype(np.float32),
        "h_ag": z["h_ag"].astype(np.float32),
        **meta,
    }


def residue_row(entry: dict, chain: str, position: int, field: str = "h_complex") -> np.ndarray:
    """The feature vector for one residue, located the same way ``mpnn_repr.build`` locates it.

    ``position`` is the index into ``structure.chains[chain].seq``, i.e. what
    ``chain.index[m.key]`` returns for a parsed mutation.
    """
    off_key = {"h_complex": "off_complex", "h_ab": "off_ab", "h_ag": "off_ag"}[field]
    offsets = entry[off_key]
    if chain not in offsets:
        raise KeyError(f"chain {chain!r} not in {field} (has {sorted(offsets)})")
    return entry[field][offsets[chain] + position]


def build_benchmarks(device: str = "auto", weights: str = DEFAULT_WEIGHTS,
                     force: bool = False, verbose: bool = True) -> int:
    """The same field for ProtAttBA's AB645, AB1101 and S1131 complexes.

    Their ``Partners`` column is ``<side1>_<side2>`` in exactly the convention this repo's
    ``#Pdb`` uses, so side1 is treated as the antibody side and side2 as the antigen side.
    For the AB sets that matches how their csv is laid out (antibody columns first); for
    S1131 the labels are conventions anyway, since it holds no antibodies at all.

    Structures come from their shipped PDB directories, with our own parser rather than
    theirs -- a second parser could disagree with the one the EDA validated, and every
    residue index has to line up with ``chain.index``.
    """
    from src.fusion import benchmark_data as BD
    from src.structures import parse_pdb

    rows = BD.load_all()
    keys = rows.drop_duplicates(["dataset", "pdb", "partners"])
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)

    def cache_name(ds, pdb, partners):
        return f"{ds}__{pdb}__{partners}".replace("/", "-")

    todo = [r for r in keys.itertuples()
            if force or not (BENCH_CACHE / f"{cache_name(r.dataset, r.pdb, r.partners)}.npz").exists()]
    if verbose:
        print(f"{len(keys)} (dataset, complex) entries, {len(todo)} to compute, device {device}")
    if not todo:
        return 0

    device = resolve_device(device)
    model, _ = _load_model(device, weights)
    t0 = time.perf_counter()
    written, skipped = 0, []

    for r in todo:
        path = BD.find_structure(r.pdb)
        if path is None:
            skipped.append((r.dataset, r.pdb, "no structure"))
            continue
        try:
            st = parse_pdb(path)
            ab, ag = resolve_chains(st, str(r.partners))
            if not ab or not ag:
                skipped.append((r.dataset, r.pdb,
                                f"partners {r.partners} not in {sorted(st.chains)}"))
                continue
            h_cx, off_cx = encoder_h_V(model, st, ab + ag, device)
            h_ab, off_ab = encoder_h_V(model, st, ab, device)
            h_ag, off_ag = encoder_h_V(model, st, ag, device)
            np.savez_compressed(
                BENCH_CACHE / f"{cache_name(r.dataset, r.pdb, r.partners)}.npz",
                h_complex=h_cx.astype(np.float16), h_ab=h_ab.astype(np.float16),
                h_ag=h_ag.astype(np.float16),
                meta=np.array(json.dumps({
                    "complex_key": cache_name(r.dataset, r.pdb, r.partners),
                    "dataset": r.dataset, "pdb": r.pdb, "partners": r.partners,
                    "chains_ab": ab, "chains_ag": ag,
                    "off_complex": off_cx, "off_ab": off_ab, "off_ag": off_ag,
                    "dim": int(h_cx.shape[1]), "weights": weights,
                })))
            written += 1
        except Exception as e:                       # one bad structure must not stop the set
            skipped.append((r.dataset, r.pdb, f"{type(e).__name__}: {e}"))
        if verbose and written and written % 20 == 0:
            print(f"  {written}/{len(todo)}  {time.perf_counter() - t0:.0f}s", flush=True)

    if verbose:
        total = sum(p.stat().st_size for p in BENCH_CACHE.glob("*.npz"))
        print(f"wrote {written} entries to {BENCH_CACHE.relative_to(paths.ROOT)} "
              f"({total / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f}s")
        if skipped:
            print(f"  skipped {len(skipped)}:")
            for row in skipped[:10]:
                print(f"    {row}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="recompute complexes already cached")
    ap.add_argument("--benchmarks", action="store_true",
                    help="extract ProtAttBA's AB645/AB1101/S1131 complexes instead of ours")
    cli = ap.parse_args()
    if cli.benchmarks:
        build_benchmarks(device=cli.device, force=cli.force)
    else:
        build(device=cli.device, force=cli.force)


if __name__ == "__main__":
    main()
