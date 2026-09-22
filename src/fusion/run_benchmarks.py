"""E6b — our random forest on ProtAttBA's three benchmarks, against their published numbers.

Answers one question: how does this project's structure-aware tree baseline do on the
benchmarks ProtAttBA reports, scored the way ProtAttBA scores them.

**The feature set is not identical to the documented forest, and that matters.** The
documented run is ``chem + geom + geomrev + mpnn``. Here it is:

* ``chem`` -- substitution chemistry, computed from the mutation string alone, so it needs no
  structure and is exact for every row.
* ``mpnn_site`` -- the ProteinMPNN encoder field at the mutated position, read out of the
  per-residue cache already built for these complexes. This is the same quantity
  ``mpnnrep.parquet`` stores for our dataset (``hc*``/``ha*``), recovered by indexing rather
  than by re-running the model.

``geom`` and ``geomrev`` are **omitted**, because building them means running the interface
geometry extractor over 173 more complexes. So this is a *floor* on what the forest would do
with its full feature set, not a like-for-like transfer of it. That is stated in the output and
in the results table rather than left for the reader to infer.

**Two protocols, both reported.**

``paper``
    Their protocol: ``KFold(shuffle=True, random_state=3407)`` over **rows**, 10 folds on
    S1131 and AB645, 5 on AB1101. Comparable to their published figure, and leaky by
    construction -- a test row's complex is almost always in training.
``grouped``
    The same folds counted by complex instead of by row. Not comparable to their number, and
    the one that says whether the model generalises to an unseen complex.

Rows whose mutation does not verify against the shipped structure are dropped and counted:
411 of 2876, all in AB645/AB1101, where the csv's residue numbering disagrees with the PDB.
S1131 verifies completely.

Run: ``python -m src.fusion.run_benchmarks``
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src import paths
from src.features import chem
from src.fusion import benchmark_data as BD
from src.fusion import metrics, results
from src.fusion.mpnn_per_residue import BENCH_CACHE, resolve_chains
from src.model import MODELS
from src.structures import parse_mutations, parse_pdb, verify

#: Their fold counts, from the paper's protocol.
N_FOLDS = {"S1131": 10, "AB645": 10, "AB1101": 5}
SEED = 3407          # the seed their bash script sets, verified against their shipped folds

_PA = re.compile(r"^([A-Za-z0-9]):([A-Z])(-?\d+[A-Za-z]?)([A-Z])$")
_BOUND = re.compile(r"(?=[A-Za-z0-9]:)")


def to_skempi(text: str) -> str | None:
    """``C:D101A`` -> ``DC101A``, the form this repo's parsers expect."""
    out = []
    for part in str(text).split(","):
        for chunk in (c for c in _BOUND.split(part.strip()) if c):
            m = _PA.match(chunk)
            if not m:
                return None
            ch, wt, pos, mut = m.groups()
            out.append(f"{wt}{ch}{pos}{mut}")
    return ",".join(out) if out else None


def prepare() -> pd.DataFrame:
    """Benchmark rows that verify against their structure, with features attached."""
    rows = BD.load_all()
    rows["mutations"] = rows.mutation.map(to_skempi)

    structures, keep, chains = {}, [], []
    for r in rows.itertuples():
        if r.mutations is None:
            keep.append(False); chains.append(("", "")); continue
        if r.pdb not in structures:
            p = BD.find_structure(r.pdb)
            structures[r.pdb] = parse_pdb(p) if p else None
        st = structures[r.pdb]
        if st is None:
            keep.append(False); chains.append(("", "")); continue
        try:
            good = not verify(st, parse_mutations(r.mutations))
        except Exception:
            good = False
        keep.append(good)
        chains.append(resolve_chains(st, str(r.partners)) if good else ("", ""))

    rows["verified"] = keep
    rows["ab_chains"] = [c[0] for c in chains]
    rows["ag_chains"] = [c[1] for c in chains]
    dropped = rows.groupby("dataset").verified.agg(lambda s: int((~s).sum())).to_dict()
    print(f"rows failing structure verification (dropped): {dropped}")

    ok = rows[rows.verified].copy().reset_index(drop=True)
    ok["n_mut"] = ok.mutations.str.count(",") + 1

    # --- chemistry: no structure needed, exact for every surviving row -------------------
    X_chem = chem.build(ok[["row_id", "mutations"]]).reset_index(drop=True)

    # --- ProteinMPNN at the mutated site, read from the cached per-residue field ---------
    dim = None
    site = []
    cache: dict[str, dict] = {}
    for r in ok.itertuples():
        name = f"{r.dataset}__{r.pdb}__{r.partners}".replace("/", "-")
        if name not in cache:
            p = BENCH_CACHE / f"{name}.npz"
            if p.exists():
                z = np.load(p, allow_pickle=False)
                meta = json.loads(str(z["meta"]))
                cache[name] = {"h_complex": z["h_complex"].astype(np.float32),
                               "off": meta["off_complex"], "dim": meta["dim"]}
            else:
                cache[name] = None
        e = cache[name]
        if e is None:
            site.append(None); continue
        dim = e["dim"]
        st = structures[r.pdb]
        v = np.zeros(e["dim"], np.float32)
        for m in parse_mutations(r.mutations):
            if m.chain in e["off"]:
                pos = st.chains[m.chain].index[m.key]
                idx = e["off"][m.chain] + pos
                if 0 <= idx < e["h_complex"].shape[0]:
                    v += e["h_complex"][idx]
        site.append(v)

    dim = dim or 128
    M = np.vstack([v if v is not None else np.zeros(dim, np.float32) for v in site])
    X_mpnn = pd.DataFrame(M, columns=[f"hc{i}" for i in range(dim)])
    have = np.array([v is not None for v in site])
    print(f"ProteinMPNN site features available for {have.sum()} of {len(ok)} verified rows")

    X = pd.concat([X_chem.drop(columns=[c for c in ("wt_aa", "mut_aa") if c in X_chem]),
                   X_mpnn], axis=1)
    X = X.loc[:, X.nunique(dropna=False) > 1].astype(np.float32).fillna(0.0)
    print(f"feature matrix: {X.shape[1]} columns over {len(X)} rows "
          f"(chem + ProteinMPNN site; geom/geomrev omitted)")
    return ok.join(X.add_prefix("f__"))


def run(frame: pd.DataFrame, seeds=(0, 1, 2)) -> pd.DataFrame:
    fcols = [c for c in frame.columns if c.startswith("f__")]
    out = []
    sha = results.git_sha()

    for dataset, g in frame.groupby("dataset"):
        g = g.reset_index(drop=True)
        X, y = g[fcols], g.ddG.to_numpy(float)
        k = N_FOLDS[dataset]

        for protocol in ("paper", "grouped"):
            if protocol == "paper":
                splitter = KFold(n_splits=k, shuffle=True, random_state=SEED).split(X)
            else:
                splitter = GroupKFold(n_splits=k).split(X, groups=g.pdb)
            folds = np.empty(len(g), int)
            for i, (_, te) in enumerate(splitter):
                folds[te] = i

            for seed in seeds:
                oof = np.full(len(y), np.nan)
                t0 = time.time()
                for i in range(k):
                    te = folds == i
                    est = MODELS["rf"](seed)
                    est.fit(X[~te], y[~te])
                    oof[te] = est.predict(X[te])
                mins = (time.time() - t0) / 60

                per_fold = []
                for i in range(k):
                    te = folds == i
                    s = metrics.score(y[te], oof[te])
                    s.update(exp=f"rf_chem_mpnnsite__{protocol}", dataset=dataset,
                             fold=i, seed=seed, n_train=int((~te).sum()),
                             n_test=int(te.sum()), params="", git_sha=sha,
                             train_minutes=round(mins / k, 3),
                             note=f"chem+mpnn_site/rf/{protocol}")
                    results.append(s, path=results.BENCHMARKS_CSV)
                    per_fold.append(s)
                pf = pd.DataFrame(per_fold)
                out.append({"dataset": dataset, "protocol": protocol, "seed": seed,
                            "pearson": pf.pearson.mean(), "pearson_sd": pf.pearson.std(ddof=0),
                            "spearman": pf.spearman.mean(), "rmse": pf.rmse.mean(),
                            "rmse_sd": pf.rmse.std(ddof=0)})
                print(f"  {dataset:7s} {protocol:8s} seed {seed}: "
                      f"PCC {pf.pearson.mean():.3f} +/- {pf.pearson.std(ddof=0):.3f}  "
                      f"rho {pf.spearman.mean():.3f}  RMSE {pf.rmse.mean():.3f}")
    return pd.DataFrame(out)


#: ProtAttBA Table 1, ESM2 column. S1131 verified against their shipped predictions.
PUBLISHED = {"S1131": (0.84, 0.75, 1.31), "AB645": (0.66, 0.62, 1.40),
             "AB1101": (0.63, 0.59, 1.62)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cli = ap.parse_args()

    frame = prepare()
    print()
    summary = run(frame, seeds=tuple(cli.seeds))

    print("\n" + "=" * 78)
    print("our forest (chem + ProteinMPNN site) vs ProtAttBA (ESM2, sequence only)")
    print("=" * 78)
    print(f"{'dataset':8s} {'protocol':9s} {'n':>5s} {'PCC':>14s} {'rho':>7s} {'RMSE':>7s}"
          f"   ProtAttBA PCC/rho/RMSE")
    for dataset in ("S1131", "AB645", "AB1101"):
        n = int((frame.dataset == dataset).sum())
        for protocol in ("paper", "grouped"):
            s = summary[(summary.dataset == dataset) & (summary.protocol == protocol)]
            if s.empty:
                continue
            p, pr, sp, rm = s.pearson.mean(), s.pearson_sd.mean(), s.spearman.mean(), s.rmse.mean()
            pub = PUBLISHED[dataset]
            tail = f"   {pub[0]:.2f} / {pub[1]:.2f} / {pub[2]:.2f}" if protocol == "paper" else ""
            print(f"{dataset:8s} {protocol:9s} {n:5d} {p:8.3f} +/-{pr:.3f} {sp:7.3f} {rm:7.3f}{tail}")
    print(f"\nwrote {results.BENCHMARKS_CSV.relative_to(paths.ROOT)}")
    print("Feature set is chem + ProteinMPNN site only; geom/geomrev omitted, so this is a")
    print("floor on the documented forest rather than a like-for-like transfer of it.")


if __name__ == "__main__":
    main()
