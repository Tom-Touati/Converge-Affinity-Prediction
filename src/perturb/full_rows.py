"""Perturbation rows for ALL of SKEMPI 2.0, not only the antibody-antigen subset.

The project trains on 940 rows over 53 complexes. SKEMPI holds 7,085 rows over 348, of which
1,211 are AB/AG. The rest are protein-protein, TCR/pMHC and unclassified complexes, and the
question this exists to ask is whether a model learns transferable binding physics from them
that it cannot learn from 940 antibody rows alone.

Chain roles need no Ig-motif detection here. ``#Pdb`` already names both groups -- ``1ACB_E_I``
is chains E against chain I -- so side1 and side2 substitute directly for the antibody and
antigen slots the downstream code expects. For AB/AG complexes that assignment agrees with the
project's own, because the project derived it from the same field.

The label pipeline is deliberately identical to ``src/data.py``: the same temperature parsing,
the same ddG formula, and the same censored-affinity drop. A corpus built with different
cleaning would answer a different question.

    python -m src.perturb.full_rows [--out cache/perturb_rows_full.parquet]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from src import paths

R_GAS = 0.0019872041          # kcal / (mol K)


def build(verbose: bool = True) -> pd.DataFrame:
    from src.structures import load_structures, parse_mutations

    raw = pd.read_csv(paths.RAW_CSV, sep=";", low_memory=False)
    if verbose:
        print(f"SKEMPI 2.0: {len(raw)} rows, {raw['#Pdb'].nunique()} complexes")

    d = raw.copy()
    d["is_abag"] = d["Hold_out_type"].fillna("").str.contains("AB/AG")

    # label, exactly as src/data.py computes it
    tnum = d["Temperature"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    d["T"] = tnum.fillna(298.0)
    for side in ("mut", "wt"):
        d[f"{side}_is_bound"] = (
            d[f"Affinity_{side} (M)"].astype(str).str.contains(r"[<>]", regex=True))
    d["ddG"] = R_GAS * d["T"] * np.log(d["Affinity_mut_parsed"] / d["Affinity_wt_parsed"])
    d = d[np.isfinite(d["ddG"])]
    # censored affinities are a lower bound presented as a measurement; dropped here for the
    # same reason and with the same consequence as in the antibody pipeline
    d = d[~(d["mut_is_bound"] | d["wt_is_bound"])].copy()
    if verbose:
        print(f"after label cleaning: {len(d)} rows, {d['#Pdb'].nunique()} complexes "
              f"({int(d.is_abag.sum())} AB/AG)")

    d["pdb"] = d["#Pdb"].str.split("_").str[0]
    d["side1"] = d["#Pdb"].str.split("_").str[1]
    d["side2"] = d["#Pdb"].str.split("_").str[2]
    d["mutations"] = d["Mutation(s)_cleaned"].str.strip()
    d["row_id"] = d["#Pdb"] + "|" + d["mutations"]
    d = d.drop_duplicates("row_id")

    structures = load_structures(sorted(d.pdb.unique()))
    have = set(structures)
    missing = sorted(set(d.pdb.unique()) - have)
    if missing and verbose:
        print(f"no structure for {len(missing)} pdb ids, dropping their rows: {missing[:6]}")
    d = d[d.pdb.isin(have)]

    rows, skipped = [], {}
    for r in d.itertuples():
        st = structures[r.pdb]
        g1 = [c for c in r.side1 if c in st.chains]
        g2 = [c for c in r.side2 if c in st.chains]
        if not g1 or not g2:
            skipped["chain missing"] = skipped.get("chain missing", 0) + 1
            continue
        s1 = "".join(st.chains[c].seq for c in g1)
        s2 = "".join(st.chains[c].seq for c in g2)
        m1, m2 = list(s1), list(s2)
        p1, p2, wt_aa, mt_aa, bad = [], [], [], [], None
        for mut in parse_mutations(r.mutations):
            grp = g1 if mut.chain in g1 else (g2 if mut.chain in g2 else None)
            if grp is None or mut.key not in st.chains[mut.chain].index:
                bad = "chain or residue not resolved"; break
            off = sum(len(st.chains[c].seq) for c in grp[: grp.index(mut.chain)])
            pos = off + st.chains[mut.chain].index[mut.key]
            whole = s1 if grp is g1 else s2
            if pos >= len(whole) or whole[pos] != mut.wt:
                bad = "wild-type residue disagrees with the structure"; break
            (p1 if grp is g1 else p2).append(pos)
            (m1 if grp is g1 else m2)[pos] = mut.mut
            wt_aa.append(mut.wt); mt_aa.append(mut.mut)
        if bad or not (p1 or p2):
            skipped[bad or "no site"] = skipped.get(bad or "no site", 0) + 1
            continue
        key = f"{r.pdb}_{''.join(g1)}_{''.join(g2)}"
        rows.append(dict(
            row_id=f"{key}|{r.mutations}", complex_key=key, pdb=r.pdb,
            ab_wt=s1, ag_wt=s2, ab_mt="".join(m1), ag_mt="".join(m2),
            sites_ab=",".join(map(str, p1)), sites_ag=",".join(map(str, p2)),
            wt_aa="".join(wt_aa), mt_aa="".join(mt_aa),
            ddg=float(r.ddG), is_abag=bool(r.is_abag), fold=-1))

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\n{len(out)} rows over {out.complex_key.nunique()} complexes")
        print(f"  AB/AG      {int(out.is_abag.sum()):>5} rows over "
              f"{out[out.is_abag].complex_key.nunique()} complexes")
        print(f"  other      {int((~out.is_abag).sum()):>5} rows over "
              f"{out[~out.is_abag].complex_key.nunique()} complexes")
        if skipped:
            print("  skipped: " + ", ".join(f"{v} {k}" for k, v in
                                            sorted(skipped.items(), key=lambda x: -x[1])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/protattba_repro/cache/perturb_rows_full.parquet")
    a, _ = ap.parse_known_args()
    out = build()
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
