"""Put this project's 940 rows into the four-sequence shape ProtAttBA's head consumes.

ProtAttBA is sequence-only: per row it wants an antibody-side sequence and an antigen-side
sequence, wild-type and mutant, and nothing else. S1131 ships those as columns ``a``, ``b``,
``a_mut``, ``b_mut``. This project's ``dataset.parquet`` has no sequence columns at all -- it
carries ``ab_chains`` / ``ag_chains`` and reads residues out of the PDB files -- so the four
sequences have to be built before the head can run on our split.

Built to match their ``utils.common.get_s1131_data`` convention exactly:

* one sequence per **side**, formed by concatenating that side's chains in the order
  ``ab_chains`` / ``ag_chains`` record them;
* the mutant sequences come from ``structures.apply_mutations``, so only the mutated chain
  differs and the partner side is usually byte-identical to its wild type.

That concatenation is worth flagging rather than hiding. ``src/features/esm2.py`` deliberately
refuses to do it -- "ESM-2 is a single-chain model: feeding it a concatenated complex would
invent a covalent link that does not exist" -- and embeds only the mutated chain. ProtAttBA
concatenates light+heavy and both antigen chains and feeds the result to ESM2 anyway. So this
is not us relaxing the project's standard; it is a real difference between the two models, and
the reason their antibody stream sees framework context that our ESM-2 features never do.

Run: ``python build_project_sequences.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import paths, structures  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache" / "project_sequences.parquet"


def sides_for(row) -> tuple[str, str]:
    """(antibody chains, antigen chains), with the project's tie fallback.

    ``structures.assign_roles`` returns ``("", "", "tie")` when both chain groups score equally
    on Ig framework motifs, which is the right answer for 1DVF_AB_CD -- an anti-idiotype pair,
    antibody bound to antibody, with no antigen at all. 38 of our 940 rows are that complex.

    Every existing consumer handles it with the same idiom rather than dropping the rows:
    ``splits.py`` and ``tmscore.py`` use ``r["ag_chains"] or r["side2"]``,
    ``features/geometry.py`` has an explicit ``# 1DVF: antibody vs antibody`` branch, and
    ``features/antibody_plm.py`` falls back to ``side1 + side2``. We follow it, so the
    reproduction covers the same 940 rows as the forest it is being compared against. For this
    one complex "antibody side" and "antigen side" are labels of convenience.
    """
    return (row.ab_chains or row.side1), (row.ag_chains or row.side2)


def side_sequence(structure, chain_ids: str, mutated: dict | None = None) -> str:
    """Concatenate one side's chains, ProtAttBA-style.

    ``mutated`` is ``apply_mutations``' chain -> sequence map; chains it does not mention are
    unchanged, which is why the partner side of a single-point mutation comes back identical.
    """
    parts = []
    for ch in chain_ids:
        if mutated and ch in mutated:
            parts.append(mutated[ch])
        else:
            parts.append(structure.chains[ch].seq)
    return "".join(parts)


def main() -> None:
    data = pd.read_parquet(paths.DATASET)
    paths.ensure_pdbs()
    structs = structures.load_structures(sorted(data.pdb.unique()))
    print(f"{len(data)} rows, {data.pdb.nunique()} pdb ids, {len(structs)} structures parsed")

    rows, skipped = [], []
    for r in data.itertuples():
        st = structs.get(r.pdb)
        if st is None:
            skipped.append((r.row_id, "no structure"))
            continue
        try:
            muts = structures.parse_mutations(r.mutations)
            mutated = structures.apply_mutations(st, muts)
            ab_ch, ag_ch = sides_for(r)
            rows.append({
                "row_id": r.row_id,
                "ab_chains": ab_ch,
                "ag_chains": ag_ch,
                "ab_wt": side_sequence(st, ab_ch),
                "ag_wt": side_sequence(st, ag_ch),
                "ab_mt": side_sequence(st, ab_ch, mutated),
                "ag_mt": side_sequence(st, ag_ch, mutated),
                "ddG": r.ddG,
            })
        except Exception as e:  # a row we cannot build must be visible, not silently dropped
            skipped.append((r.row_id, f"{type(e).__name__}: {e}"))

    out = pd.DataFrame(rows).merge(
        data[["row_id", "#Pdb", "n_mut", "mut_side"]], on="row_id", validate="1:1"
    ).rename(columns={"#Pdb": "complex"})

    if skipped:
        print(f"\n{len(skipped)} rows could not be built:")
        for rid, why in skipped[:10]:
            print(f"    {rid}  {why}")

    # A mutation must change the side it is attributed to, and only that side.
    changed_ab = (out.ab_wt != out.ab_mt)
    changed_ag = (out.ag_wt != out.ag_mt)
    print(f"\nrows where the antibody side changed  {int(changed_ab.sum()):4d}")
    print(f"rows where the antigen side changed   {int(changed_ag.sum()):4d}")
    print(f"rows where neither side changed       {int((~changed_ab & ~changed_ag).sum()):4d}"
          "   <- must be 0")
    print(f"rows where both sides changed         {int((changed_ab & changed_ag).sum()):4d}")
    assert int((~changed_ab & ~changed_ag).sum()) == 0, "a mutation changed no sequence"

    lengths = pd.concat([out.ab_wt.str.len(), out.ag_wt.str.len()])
    distinct = set(out.ab_wt) | set(out.ag_wt) | set(out.ab_mt) | set(out.ag_mt)
    residues = sum(len(s) for s in distinct)
    over = sum(1 for s in distinct if len(s) > 1022)
    print(f"\nside lengths: median {int(lengths.median())}, max {int(lengths.max())}")
    print(f"distinct sequences across the four columns  {len(distinct)}  "
          f"({len(out) * 4} cells)")
    print(f"residues to embed once                      {residues}")
    print(f"distinct sequences longer than 1022          {over}   "
          "<- ESM2's position limit, these need a window")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
