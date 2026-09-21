"""Homology clustering and the frozen cross-validation split.

The split is built once and committed as ``data/folds.csv``. Nothing downstream may regenerate
it, so every later number is comparable and no score gain can come from a lucky re-split.

Why clusters and not complexes. The field's de facto protocol (RDE-Network, DiffAffinity) splits
by *structure*. That is not enough here: thirteen of our 54 complexes are different antibodies
against hen egg-white lysozyme, 25% of all rows. A structure-level split spreads them across
folds and lets the model memorise the epitope. We therefore group complexes into homology
clusters and split on those, which is strictly harder -- and the write-up has to say so, or our
lower numbers read as underperformance rather than as a stricter evaluation.

Links between complexes come from three sources, unioned into connected components:

1. SKEMPI's own ``Hold_out_proteins`` annotation. Using the dataset authors' definition of
   homology is the most defensible default. The annotation is not repeated on every row, so it
   is unioned across all rows of a complex.
2. Pairwise sequence identity on either side. Local Smith-Waterman with BLOSUM62, identity taken
   over the shorter sequence. Antigens link at 30%, the usual homology threshold. Antibodies
   need a much higher bar -- shared framework means two unrelated antibodies sit near 70-80%
   identity -- so they link at 90%.
3. Curator notes. SKEMPI flags "HyHEL-10 and HyHEL-63 are very similar" on 151 rows; that is a
   manual link regardless of what the automatic thresholds return.
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import paths
from .structures import load_structures

AG_IDENTITY = 0.30
AB_IDENTITY = 0.90
DEFAULT_K = 4

_PDB_TOKEN = re.compile(r"^[0-9][A-Za-z0-9]{3}_")


def _aligner():
    from Bio import Align
    from Bio.Align import substitution_matrices

    a = Align.PairwiseAligner()
    a.mode = "local"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -11
    a.extend_gap_score = -1
    return a


def identity(aligner, a: str, b: str) -> float:
    """Fraction of the shorter sequence that aligns identically. 0 if either side is empty."""
    if not a or not b:
        return 0.0
    try:
        aln = next(iter(aligner.align(a, b)))
    except (StopIteration, OverflowError, MemoryError):
        return 0.0
    n_id = sum(
        sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
        for (s1, e1), (s2, e2) in zip(*aln.aligned)
    )
    return n_id / min(len(a), len(b))


def build_clusters(ds: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """Return a Series mapping complex (#Pdb) -> cluster id."""
    cx = ds.drop_duplicates("#Pdb").set_index("#Pdb")
    keys = list(cx.index)
    structures = load_structures(cx["pdb"].unique())

    def side_seq(k: str, which: str) -> str:
        r = cx.loc[k]
        group = r["ab_chains"] if which == "ab" else r["ag_chains"]
        if not group:  # 1DVF: antibody vs antibody, both sides are immunoglobulin
            group = r["side1"] if which == "ab" else r["side2"]
        return structures[r["pdb"]].group_seq(group)

    ab_seq = {k: side_seq(k, "ab") for k in keys}
    ag_seq = {k: side_seq(k, "ag") for k in keys}

    edges, why = set(), Counter()

    # (1) SKEMPI's explicit hold-out grouping. Read from the raw file because the annotation is
    #     split-metadata, not a per-measurement property, and is unioned across a complex's rows.
    raw = pd.read_csv(paths.RAW_CSV, sep=";", low_memory=False, usecols=["#Pdb", "Hold_out_proteins"])
    hop = raw[raw["#Pdb"].isin(keys)].groupby("#Pdb")["Hold_out_proteins"].apply(
        lambda s: {t.strip() for v in s.fillna("") for t in str(v).split(",") if t.strip()}
    )
    outside = set()
    for k, tokens in hop.items():
        for tok in tokens:
            if not _PDB_TOKEN.match(tok):
                continue
            hits = [k2 for k2 in keys if k2.split("_")[0] == tok.split("_")[0] and k2 != k]
            if hits:
                for k2 in hits:
                    edges.add(tuple(sorted((k, k2))))
                    why["skempi hold-out group"] += 1
            else:
                outside.add(tok)

    # (2) sequence identity on either side
    aligner = _aligner()
    for a, b in itertools.combinations(keys, 2):
        if identity(aligner, ag_seq[a], ag_seq[b]) >= AG_IDENTITY:
            edges.add(tuple(sorted((a, b))))
            why["shared antigen"] += 1
        elif identity(aligner, ab_seq[a], ab_seq[b]) >= AB_IDENTITY:
            edges.add(tuple(sorted((a, b))))
            why["shared antibody"] += 1

    # (3) curator note: HyHEL-10 / HyHEL-63 are called out explicitly as very similar
    names = (cx["Protein 1"].fillna("") + " " + cx["Protein 2"].fillna("")).str.lower()
    hyhel = [k for k in keys if "hyhel" in names.loc[k]]
    for a, b in itertools.combinations(hyhel, 2):
        if tuple(sorted((a, b))) not in edges:
            why["curator note (HyHEL)"] += 1
        edges.add(tuple(sorted((a, b))))

    # connected components by union-find
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cluster = pd.Series({k: find(k) for k in keys}, name="cluster")
    if verbose:
        print(f"links by source: {dict(why)}")
        print(f"unique links: {len(edges)}")
        if outside:
            print(f"hold-out partners outside the AB/AG subset (no link possible): {sorted(outside)}")
        print(f"{len(keys)} complexes collapse into {cluster.nunique()} clusters")
    return cluster


def assign_folds(ds: pd.DataFrame, cluster: pd.Series, k: int, verbose: bool = True) -> pd.Series:
    """Greedily pack clusters into k folds, balancing row count then ddG mean.

    Balance is a secondary concern here: the headline metric averages Spearman *over complexes*,
    so a fold holding more rows does not dominate it. What matters is that every complex appears
    in exactly one test fold, which grouped assignment guarantees for any k.
    """
    d = ds.assign(cluster=ds["#Pdb"].map(cluster))
    stats = (
        d.groupby("cluster")
        .agg(rows=("ddG", "size"), mean=("ddG", "mean"), complexes=("#Pdb", "nunique"))
        .sort_values("rows", ascending=False)
    )

    fold_rows = np.zeros(k)
    fold_sum = np.zeros(k)
    assignment = {}
    for cl, r in stats.iterrows():
        # prefer the emptiest fold; break ties towards the fold whose mean it pulls least
        order = np.lexsort((np.abs(fold_sum / np.maximum(fold_rows, 1) - r["mean"]), fold_rows))
        f = int(order[0])
        assignment[cl] = f
        fold_rows[f] += r["rows"]
        fold_sum[f] += r["rows"] * r["mean"]

    fold = d["cluster"].map(assignment)
    if verbose:
        tbl = (
            d.assign(fold=fold.values)
            .groupby("fold")
            .agg(rows=("ddG", "size"), complexes=("#Pdb", "nunique"),
                 clusters=("cluster", "nunique"), ddG_mean=("ddG", "mean"))
        )
        tbl["% rows"] = (tbl["rows"] / len(d) * 100).round(1)
        print()
        print(tbl.round(2).to_string())
    return fold


def leakage_report(ds: pd.DataFrame, folds: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Max antigen-side identity between any test complex and any training complex, per fold."""
    cx = ds.drop_duplicates("#Pdb").set_index("#Pdb")
    structures = load_structures(cx["pdb"].unique())
    fold_of = folds.drop_duplicates("#Pdb").set_index("#Pdb")["fold"]

    def ag(k):
        r = cx.loc[k]
        return structures[r["pdb"]].group_seq(r["ag_chains"] or r["side2"])

    seqs = {k: ag(k) for k in cx.index}
    aligner = _aligner()
    out = []
    for f in sorted(fold_of.unique()):
        test = [k for k in cx.index if fold_of[k] == f]
        train = [k for k in cx.index if fold_of[k] != f]
        worst, pair = 0.0, ("", "")
        for a in test:
            for b in train:
                v = identity(aligner, seqs[a], seqs[b])
                if v > worst:
                    worst, pair = v, (a, b)
        out.append({"fold": f, "max_test_train_ag_identity": round(worst, 3),
                    "test_complex": pair[0], "train_complex": pair[1]})
    rep = pd.DataFrame(out)
    if verbose:
        print()
        print("leakage check -- worst antigen identity across the split boundary:")
        print(rep.to_string(index=False))
    return rep


GROUPINGS = ("cluster", "complex", "random")
"""Available split strictness. `cluster` is the project default and the only one reported.

The three sit on a ladder this project has already measured -- random 0.722, complex 0.485,
cluster 0.353 on the same model -- so the choice deserves a sentence rather than a default.

* **cluster** groups by homology (antigen 30% identity, antibody 90%), so structural twins
  cannot straddle a fold boundary. It is the honest question and it is brutal: every complex
  comes out `hard` under the published TM-score tiering, because all 118 antigen pairs above
  TM 0.8 sit inside a fold.
* **complex** keeps each #Pdb whole but lets homologues fall on opposite sides, which is what
  much of the SKEMPI literature does. The gap between it and `cluster` measures directly what
  homology leakage is worth.
* **random** ignores structure entirely. It quantifies a ceiling; it is not a result.

`complex` and `random` write their own files and never touch data/folds.csv, so the frozen
split stays frozen.
"""


def _grouped_folds(ds: pd.DataFrame, by: pd.Series, k: int) -> pd.Series:
    """Greedy balanced assignment of whole groups to k folds, largest group first."""
    key = by.reindex(ds["#Pdb"]).to_numpy()
    sizes = pd.Series(key).groupby(key).size().sort_values(ascending=False)
    load = {f: 0 for f in range(k)}
    where = {}
    for g, n in sizes.items():
        f = min(load, key=load.get)
        where[g] = f
        load[f] += n
    return pd.Series([where[g] for g in key])


def build(k: int = DEFAULT_K, verbose: bool = True, grouping: str = "cluster") -> pd.DataFrame:
    if grouping not in GROUPINGS:
        raise SystemExit(f"grouping must be one of {GROUPINGS}")
    ds = pd.read_parquet(paths.DATASET)

    if grouping == "cluster":
        cluster = build_clusters(ds, verbose)
        fold = assign_folds(ds, cluster, k, verbose)
    elif grouping == "complex":
        cluster = pd.Series(ds["#Pdb"].unique(), index=ds["#Pdb"].unique())
        fold = _grouped_folds(ds, cluster, k)
    else:
        cluster = pd.Series(ds["#Pdb"].unique(), index=ds["#Pdb"].unique())
        fold = pd.Series(np.random.default_rng(0).integers(0, k, len(ds)))

    folds = pd.DataFrame({
        "row_id": ds["row_id"],
        "#Pdb": ds["#Pdb"],
        "cluster": ds["#Pdb"].map(cluster),
        "fold": fold.values,
    }).sort_values("row_id").reset_index(drop=True)

    # invariants -- these are the tests that matter, so they run on every build
    assert folds["row_id"].is_unique and len(folds) == len(ds)
    if grouping != "random":
        per_cx_folds = folds.groupby("#Pdb")["fold"].nunique()
        assert (per_cx_folds == 1).all(), "a complex leaked across folds"
    if grouping == "cluster":
        per_cluster_folds = folds.groupby("cluster")["fold"].nunique()
        assert (per_cluster_folds == 1).all(), "a cluster leaked across folds"

    out = (paths.FOLDS if grouping == "cluster"
           else paths.FOLDS.with_name(f"folds_{grouping}.csv"))
    folds.to_csv(out, index=False)
    if verbose:
        if grouping == "cluster":
            leakage_report(ds, folds, verbose)
        print(f"\nwrote {out.relative_to(paths.ROOT)}  "
              f"({len(folds):,} rows, k={k}, grouping={grouping})")
    return folds


def load(grouping: str = "cluster") -> pd.DataFrame:
    """Dataset joined to the frozen folds. The single entry point for every model.

    `grouping` selects an alternative split file and defaults to the frozen cluster-grouped
    one, so nothing changes unless a caller asks. The others are secondary and must be built
    first: `python -m src.splits --grouping complex`.
    """
    ds = pd.read_parquet(paths.DATASET)
    fp = (paths.FOLDS if grouping == "cluster"
          else paths.FOLDS.with_name(f"folds_{grouping}.csv"))
    if not fp.exists():
        raise SystemExit(f"{fp.name} not built -- run: "
                         f"python -m src.splits --grouping {grouping}")
    folds = pd.read_csv(fp)
    out = ds.merge(folds[["row_id", "cluster", "fold"]], on="row_id", how="left", validate="1:1")
    if out["fold"].isna().any():
        raise RuntimeError("dataset rows missing from folds.csv -- rebuild the split deliberately")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-k", type=int, default=DEFAULT_K, help="number of folds (default: 4)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--grouping", default="cluster", choices=GROUPINGS,
                   help="cluster (default, frozen) | complex (literature-comparable) | random")
    a = p.parse_args()
    build(k=a.k, verbose=not a.quiet, grouping=a.grouping)


if __name__ == "__main__":
    main()
