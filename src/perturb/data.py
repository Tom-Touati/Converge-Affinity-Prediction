"""Batches for the perturbation model: PCA-reduced ESM tokens, MPNN tokens, geometry, BLOSUM.

Everything a branch needs, assembled per row and collated per batch. Three pieces are cached
because they are expensive and fold-dependent in different ways:

``PCA``
    Refit **per fold, on the training fold's sequences only**, and cached under
    ``data/features/perturb_pca/<split>_fold<k>.joblib``. Fitting once on every sequence would
    choose components using the test complexes, which under a complex-grouped split is exactly
    the homology the split exists to withhold. The same projection is applied to wild-type and
    mutant tokens, so the sequence delta stays a difference in one basis rather than a
    difference between two bases.

``distance``
    Inter-chain Ca distance matrix per complex from the wild-type PDB, cached to
    ``data/features/perturb_dist/<complex>.npy``. Shared by both branches: the mutation is a
    sequence edit and the backbone is fixed, which is the model's central assumption and also
    its main limitation.

``BLOSUM``
    The BLOSUM62 row of the wild-type and mutant residue at each mutated site, 20 + 20. The
    ITW branch uses the pair ``(a, a)``, which is what makes a null edit exactly null.

Sequence lengths here are the *concatenated side*, up to ~1500 residues, so batches are padded
to the longest member and the mask is carried explicitly. Padded positions are masked with
-inf in attention and pooling, never with a small finite number.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths

PCA_DIR = paths.FEATURES / "perturb_pca"
DIST_DIR = paths.FEATURES / "perturb_dist"
SEQ_TABLE = (paths.ROOT / "experiments" / "protattba_repro" / "cache"
             / "project_sequences.parquet")

#: BLOSUM62 alphabet order used for the 20-dim rows.
AA = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AA)}


def blosum62() -> np.ndarray:
    """20x20 BLOSUM62 in ``AA`` order, from Biopython, normalised to unit scale."""
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("BLOSUM62")
    out = np.zeros((20, 20), dtype=np.float32)
    for i, a in enumerate(AA):
        for j, b in enumerate(AA):
            out[i, j] = m[a, b]
    return out / 4.0            # BLOSUM62 spans about -4..11; keep inputs order 1


def ca_distance_matrix(structure, ab_chains: str, ag_chains: str) -> np.ndarray:
    """(L_ab, L_ag) Ca-Ca distances, in the concatenated order the sequences use."""
    def coords(group):
        out = []
        for c in group:
            ch = structure.chains.get(c)
            if ch is None:
                continue
            out.append(np.asarray(ch.backbone)[:, 1, :])      # N, CA, C, O -> CA
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3), np.float32)

    a, g = coords(ab_chains), coords(ag_chains)
    if len(a) == 0 or len(g) == 0:
        return np.zeros((len(a), len(g)), np.float32)
    d = np.linalg.norm(a[:, None, :] - g[None, :, :], axis=-1)
    return d.astype(np.float32)


def build_distance_cache(verbose: bool = True) -> int:
    """One matrix per complex. Cheap, but there is no reason to recompute it every epoch."""
    from src.structures import load_structures

    data = pd.read_parquet(paths.DATASET)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    cx = data.drop_duplicates("#Pdb")
    structures = load_structures(cx.pdb.unique())
    n = 0
    for r in cx.itertuples():
        key = r.row_id.split("|")[0]          # row_id is "<#Pdb>|<mutations>"
        out = DIST_DIR / f"{key}.npy"
        if out.exists():
            continue
        st = structures[r.pdb]
        ab = r.ab_chains or r.side1
        ag = r.ag_chains or r.side2
        np.save(out, ca_distance_matrix(st, ab, ag))
        n += 1
    if verbose:
        print(f"distance cache: {n} complexes written, "
              f"{len(list(DIST_DIR.glob('*.npy')))} total")
    return n


#: Effects beyond this are clipped, both in training and when scoring. 68 of 940 rows (7.2%)
#: exceed it. They are not spread evenly: fold 0 holds the 3HFM alanine hot-spot series and
#: 39 of its 188 rows are past 4, which is why its label sd is 2.42 against 1.30-1.71 for the
#: other folds and why its RMSE reads ~3 while its Pearson is ordinary. SKEMPI's large values
#: are also the least trustworthy -- many sit at the assay's detection limit -- so a squared
#: loss spends most of its budget on the numbers we are least sure of.
DDG_CLIP = 4.0


@dataclass
class Row:
    """One dataset row, resolved to arrays. Sequences stay as strings until PCA is applied."""

    row_id: str
    complex_key: str
    pdb: str
    ab_wt: str
    ag_wt: str
    ab_mt: str
    ag_mt: str
    sites_ab: list[int]
    sites_ag: list[int]
    wt_aa: list[str]              # in mutation-string order, i.e. INTERLEAVED across sides
    mt_aa: list[str]
    # The same substitutions split per side and kept in step with sites_ab / sites_ag.
    # A multi-point mutation lists its parts in string order, which need not be the order
    # they fall in on either side, so indexing wt_aa by a position within one side reads
    # the wrong residue. Anything that pairs a letter with a site must use these.
    wt_ab: list[str]
    mt_ab: list[str]
    wt_ag: list[str]
    mt_ag: list[str]
    ddg: float
    fold: int


def load_rows(split_json: Path | None = None) -> list[Row]:
    """Every row with its mutated positions located inside the concatenated side sequence."""
    from src.fusion.splits_frozen import fold_of
    from src.structures import load_structures, parse_mutations

    seqs = pd.read_parquet(SEQ_TABLE)
    # The sequence table carries its own copies of several dataset columns. Dropping them
    # here keeps the merge free of _x/_y suffixes, so every attribute below is unambiguous.
    seqs = seqs.drop(columns=[c for c in ("ab_chains", "ag_chains", "ddG", "complex",
                                          "n_mut", "mut_side") if c in seqs.columns])
    data = pd.read_parquet(paths.DATASET)[["row_id", "#Pdb", "pdb", "mutations",
                                           "ab_chains", "ag_chains", "side1", "side2", "ddG"]]
    # "#Pdb" cannot be reached as an attribute on an itertuples row, and its positional
    # name (_2) shifts whenever a column is added. Rename once, explicitly.
    m = seqs.merge(data, on="row_id", validate="1:1").rename(columns={"#Pdb": "complex_key"})
    m["fold"] = fold_of(m.row_id)
    structures = load_structures(m.pdb.unique())

    rows: list[Row] = []
    for r in m.itertuples():
        st = structures[r.pdb]
        ab = r.ab_chains or r.side1
        ag = r.ag_chains or r.side2
        sites_ab, sites_ag, wt_aa, mt_aa = [], [], [], []
        wt_ab, mt_ab, wt_ag, mt_ag = [], [], [], []
        for mut in parse_mutations(r.mutations):
            pos = st.chains[mut.chain].index[mut.key]
            group = ab if mut.chain in ab else ag
            offset = sum(len(st.chains[c].seq) for c in group[: group.index(mut.chain)])
            on_ab = mut.chain in ab
            (sites_ab if on_ab else sites_ag).append(offset + pos)
            (wt_ab if on_ab else wt_ag).append(mut.wt)
            (mt_ab if on_ab else mt_ag).append(mut.mut)
            wt_aa.append(mut.wt)
            mt_aa.append(mut.mut)
        rows.append(Row(r.row_id, r.complex_key, r.pdb, r.ab_wt, r.ag_wt, r.ab_mt, r.ag_mt,
                        sites_ab, sites_ag, wt_aa, mt_aa, wt_ab, mt_ab, wt_ag, mt_ag,
                        float(np.clip(r.ddG, -DDG_CLIP, DDG_CLIP)), int(r.fold)))
    return rows


def fit_fold_pca(rows: list[Row], fold: int, n_components: int = 256,
                 split_name: str = "frozen5", seed: int = 0):
    """PCA fit on the training fold's sequences only, then cached.

    Fitting on every sequence would leak the test complexes' variance structure into the
    basis, which under a complex-grouped split is the homology the split withholds.
    """
    import joblib
    from sklearn.decomposition import PCA

    PCA_DIR.mkdir(parents=True, exist_ok=True)
    out = PCA_DIR / f"{split_name}_fold{fold}_pca{n_components}.joblib"
    if out.exists():
        return joblib.load(out)

    from transformers import AutoTokenizer

    from src.perturb.esm_tokens import ESM_DIR, load, tokens_for

    store, index, _ = load()
    tok = AutoTokenizer.from_pretrained(str(ESM_DIR))

    train_seqs: set[str] = set()
    for r in rows:
        if r.fold == fold:
            continue
        train_seqs.update((r.ab_wt, r.ag_wt, r.ab_mt, r.ag_mt))

    # Subsample residues rather than fitting on all ~270k training tokens: PCA on 1280
    # dimensions needs far fewer rows than that, and the full matrix is 1.4 GB in fp32.
    rng = np.random.default_rng(seed)
    chunks = []
    for s in sorted(train_seqs):
        t = tokens_for(s, tok, store, index)
        take = min(len(t), 64)
        chunks.append(t[rng.choice(len(t), take, replace=False)])
    X = np.concatenate(chunks, axis=0)
    pca = PCA(n_components=n_components, random_state=seed).fit(X)
    joblib.dump(pca, out)
    print(f"  fold {fold}: PCA fit on {X.shape[0]} residues from {len(train_seqs)} training "
          f"sequences, {pca.explained_variance_ratio_.sum():.1%} of variance in "
          f"{n_components} components")
    return pca
