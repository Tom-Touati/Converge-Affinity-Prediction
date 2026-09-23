"""Build the extra AB645/AB1101 training rows, in the schema the perturbation trainer reads.

These rows exist to enlarge the TRAINING set only. They are written with ``fold = -1``,
which is what makes that true in the trainer without a special case: it takes
``test = rows[rows.fold == fold]`` over folds 0..4, so a -1 row is never tested on, and
``train = rows[rows.fold != fold]``, so it is always trained on.

What is excluded, and why it matters more than it looks:

* **Any complex that is ours.** AB645 and AB1101 carry ``HM_`` prefixed entries that are
  homology models of complexes in our own set -- ``HM_2NYY``'s chains are 100% identical to
  ``2NYY``'s. Matching on the PDB id as written treats them as new, and 1,285 rows sail
  through into training while their complexes sit in our test folds. ``base()`` strips the
  prefix so the comparison is on the underlying structure.
* **Any mutation that is ours**, after canonicalisation, for the same reason.
* **Complexes that are not antibody-antigen.** "AB" in these benchmark names does not mean
  every row is an antibody complex: of the seven that survive deduplication, only 1T83 and
  3WJJ carry an Fv pair. The rest are protein-protein (beta-lactamase/BLIP, TGF-beta/TbetaR2,
  cyclophilin/capsid). ``--allow-non-antibody`` keeps them.

Run: ``python -m src.perturb.extra_rows [--allow-non-antibody]``
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from src import paths
from src.fusion import benchmark_data as B
from src.fusion.splits_frozen import canonical_mutations

#: The benchmark tables write a mutation as CHAIN:WT POS MUT and run several together with
#: no separator -- "A:A327DB:A327D" is two, not one. Splitting on the field boundary first
#: is the same fix the leakage check needed; a findall would read that as position "327D"
#: and mutant "B", because a PDB insertion code makes the two readings ambiguous.
_FIELD = re.compile(r"(?=[A-Za-z0-9]:)")
_BENCH = re.compile(r"^([A-Za-z0-9]):([A-Z])(-?\d+[A-Za-z]?)([A-Z])$")


def to_skempi(mutations: str) -> str:
    """CHAIN:WT POS MUT  ->  WT CHAIN POS MUT, comma separated, which parse_mutations reads."""
    out = []
    for tok in _FIELD.split(str(mutations).strip()):
        tok = tok.strip().strip(",")
        if not tok:
            continue
        m = _BENCH.match(tok)
        if not m:
            raise ValueError(f"unparsable benchmark mutation {tok!r} in {mutations!r}")
        chain, wt, pos, mut = m.groups()
        out.append(f"{wt}{chain}{pos}{mut}")
    return ",".join(out)

OUT = paths.ROOT / "experiments" / "protattba_repro" / "cache" / "perturb_rows_extra.parquet"
#: two chains of roughly this length, side by side, is what an Fv looks like
FV_LO, FV_HI = 200, 240


def base(pdb) -> str:
    """HM_2NYY is a homology model OF 2NYY. For leakage they are the same complex."""
    p = str(pdb).upper()
    return p[3:] if p.startswith("HM_") else p


def has_fv(structure) -> bool:
    return sum(1 for c in structure.chains.values()
               if FV_LO <= len(c.seq) <= FV_HI) >= 2


def build(allow_non_antibody: bool = False, verbose: bool = True) -> pd.DataFrame:
    from src.structures import load_structures, parse_mutations

    ours = pd.read_parquet(paths.DATASET)
    ours_pdb = {base(p) for p in ours.pdb}
    ours_mut = {(base(r.pdb), m) for r in ours.itertuples()
                for m in canonical_mutations(r.mutations)}

    keep, drop_ours, drop_mut, drop_unparsable = [], 0, 0, 0
    for name in ("AB645", "AB1101"):
        d = B.load(name)
        col = "mutation" if "mutation" in d.columns else "mutations"
        for r in d.itertuples():
            b = base(r.pdb)
            if b in ours_pdb:
                drop_ours += 1
                continue
            muts = canonical_mutations(getattr(r, col))
            if any((b, m) in ours_mut for m in muts):
                drop_mut += 1
                continue
            try:
                skempi = to_skempi(getattr(r, col))
            except ValueError:
                drop_unparsable += 1
                continue
            keep.append({"src": name, "pdb": b, "raw_pdb": str(r.pdb),
                         "mutations": skempi, "ddG": float(r.ddG),
                         "key": (b, tuple(sorted(muts)))})
    f = pd.DataFrame(keep).drop_duplicates(subset="key").drop(columns="key")
    if verbose:
        print(f"dropped {drop_ours} rows whose complex is ours (HM_ resolved), "
              f"{drop_mut} more whose mutation is ours, "
              f"{drop_unparsable} unparsable")
        print(f"{len(f)} candidate rows over {f.pdb.nunique()} complexes")

    structures = load_structures(sorted(f.pdb.unique()))
    if not allow_non_antibody:
        fv = {p for p in f.pdb.unique() if has_fv(structures[p])}
        dropped = sorted(set(f.pdb.unique()) - fv)
        f = f[f.pdb.isin(fv)]
        if verbose:
            print(f"kept {len(fv)} antibody-antigen complexes; dropped {dropped}")

    rows = []
    skipped: list[tuple[str, str]] = []
    for r in f.itertuples():
        st = structures[r.pdb]
        # Chain roles are not annotated in the benchmark tables. The Fv pair is the
        # antibody; everything else is the antigen. Recorded per row so it is auditable
        # rather than implicit.
        ab = [c for c, ch in st.chains.items() if FV_LO <= len(ch.seq) <= FV_HI]
        ag = [c for c in st.chains if c not in ab]
        if len(ab) < 2 or not ag:
            skipped.append((r.pdb, "no Fv pair or no antigen chain"))
            continue
        ab, ag = sorted(ab)[:2], sorted(ag)
        ab_wt = "".join(st.chains[c].seq for c in ab)
        ag_wt = "".join(st.chains[c].seq for c in ag)
        ab_mt, ag_mt = list(ab_wt), list(ag_wt)
        s_ab, s_ag, wt_aa, mt_aa, bad = [], [], [], [], False
        for mut in parse_mutations(r.mutations):
            grp = ab if mut.chain in ab else (ag if mut.chain in ag else None)
            if grp is None or mut.key not in st.chains[mut.chain].index:
                bad = True
                break
            off = sum(len(st.chains[c].seq) for c in grp[: grp.index(mut.chain)])
            pos = off + st.chains[mut.chain].index[mut.key]
            whole = ab_wt if grp is ab else ag_wt
            if pos >= len(whole) or whole[pos] != mut.wt:
                bad = True
                break
            (s_ab if grp is ab else s_ag).append(pos)
            (ab_mt if grp is ab else ag_mt)[pos] = mut.mut
            wt_aa.append(mut.wt)
            mt_aa.append(mut.mut)
        if bad or not (s_ab or s_ag):
            skipped.append((r.pdb, r.mutations))
            continue
        rows.append(dict(
            row_id=f"{r.pdb}_{''.join(ab)}_{''.join(ag)}|{r.mutations}",
            complex_key=f"{r.pdb}_{''.join(ab)}_{''.join(ag)}", pdb=r.pdb,
            ab_wt=ab_wt, ag_wt=ag_wt, ab_mt="".join(ab_mt), ag_mt="".join(ag_mt),
            sites_ab=",".join(map(str, s_ab)), sites_ag=",".join(map(str, s_ag)),
            wt_aa="".join(wt_aa), mt_aa="".join(mt_aa), ddg=float(r.ddG), fold=-1))

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\n{len(out)} rows built over {out.complex_key.nunique()} complexes "
              f"(fold = -1, train only)")
        if skipped:
            print(f"{len(skipped)} rows skipped (chain or residue did not resolve), "
                  f"e.g. {skipped[:3]}")
        if len(out):
            print(out.groupby("complex_key").size().rename("rows").to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(paths.ROOT)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-non-antibody", action="store_true")
    a = ap.parse_args()
    build(a.allow_non_antibody)


if __name__ == "__main__":
    main()
