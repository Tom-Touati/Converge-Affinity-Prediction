"""E6b — our random forest on ProtAttBA's three benchmarks, against their published numbers.

Answers one question: how does this project's structure-aware tree baseline do on the
benchmarks ProtAttBA reports, scored the way ProtAttBA scores them.

**The feature set is not identical to the documented forest, and that matters.** The
documented run is ``chem + geom + geomrev + mpnn``. Here it is:

* ``chem`` -- substitution chemistry, computed from the mutation string alone, so it needs no
  structure and is exact for every row.
* ``geom`` and ``geomrev`` -- the project's interface geometry, run over their complexes with
  the same extractors the documented forest uses. Cached to ``benchmarks_geom.parquet``.
* ``mpnn_site`` -- the ProteinMPNN encoder field at the mutated position, read out of the
  per-residue cache already built for these complexes. This is the same quantity
  ``mpnnrep.parquet`` stores for our dataset (``hc*``/``ha*``), recovered by indexing rather
  than by re-running the model.

So the feature set now matches the documented forest's ``chem + geom + geomrev + mpnn``,
except that the ProteinMPNN channel is the encoder field at the site rather than the
likelihood ratios -- both come from the same model, and the field is what was already cached.

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
from src.features import chem, geom_rev, geometry
from src.fusion import benchmark_data as BD
from src.fusion import metrics, results
from src.fusion.mpnn_per_residue import BENCH_CACHE, resolve_chains
from src.model import MODELS
from src.structures import parse_mutations, parse_pdb, verify

#: Interface geometry over their complexes is the slow step (minutes, not seconds), so it is
#: computed once and cached. Deleting this file forces a rebuild.
GEOM_CACHE = paths.FEATURES / "benchmarks_geom.parquet"

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

    # --- interface geometry, the slow part, cached so it is paid once -------------------
    # geometry.build and geom_rev.build both want SKEMPI's column names, and key complex-level
    # work on `#Pdb`. The key is built from the *resolved* chains rather than the raw Partners
    # string, so the H/L fallback cannot produce two keys for one chain assignment, and it
    # omits the dataset so a complex shared by AB645 and AB1101 is computed once.
    ok["#Pdb"] = ok.pdb + "_" + ok.ab_chains + "_" + ok.ag_chains
    geo_input = ok[["row_id", "#Pdb", "pdb", "ab_chains", "ag_chains", "mutations"]]

    if GEOM_CACHE.exists():
        G = pd.read_parquet(GEOM_CACHE)
        missing = set(ok.row_id) - set(G.index)
        if missing:
            print(f"geometry cache is missing {len(missing)} rows; rebuilding")
            G = None
        else:
            print(f"geometry: reusing cache ({G.shape[1]} columns)")
    else:
        G = None

    if G is None:
        t0 = time.time()
        print(f"geometry: building over {ok['#Pdb'].nunique()} complexes, {len(ok)} rows ...",
              flush=True)
        g1 = geometry.build(geo_input)
        print(f"  geom done, {g1.shape[1]} cols, {(time.time()-t0)/60:.1f} min", flush=True)
        g2 = geom_rev.build(geo_input)
        print(f"  geomrev done, {g2.shape[1]} cols, {(time.time()-t0)/60:.1f} min", flush=True)
        G = g1.join(g2, how="outer")
        paths.FEATURES.mkdir(parents=True, exist_ok=True)
        G.to_parquet(GEOM_CACHE)
        print(f"  cached to {GEOM_CACHE.name}")

    # A .loc[] with duplicate labels silently returns MORE rows than it was given, and the
    # positional concat below then shifts every feature out of alignment with its label. That
    # happened here and was only caught by the row count moving from 2465 to 2489.
    assert ok.row_id.is_unique, "row_id is not unique; the geometry join would misalign"
    X_geom = G.loc[ok.row_id].reset_index(drop=True)
    assert len(X_geom) == len(ok), f"geometry join returned {len(X_geom)} rows for {len(ok)}"

    X = pd.concat([X_chem.drop(columns=[c for c in ("wt_aa", "mut_aa") if c in X_chem]),
                   X_geom, X_mpnn], axis=1)
    assert len(X) == len(ok), f"feature matrix has {len(X)} rows for {len(ok)} labels"
    X = X.loc[:, X.nunique(dropna=False) > 1].astype(np.float32).fillna(0.0)
    print(f"feature matrix: {X.shape[1]} columns over {len(X)} rows "
          f"(chem + geom + geomrev + ProteinMPNN site)")
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

                # Keep the predictions, not just the metrics. Without them the benchmark
                # runs cannot be published as report directories, so the dashboard's "other
                # datasets" tick box has nothing to reveal -- which looks like a broken
                # toggle rather than an empty category.
                publish_run(f"{dataset}__{protocol}_seed{seed}", dataset, g, folds, y, oof)
                per_fold = []
                for i in range(k):
                    te = folds == i
                    s = metrics.score(y[te], oof[te])
                    s.update(exp=f"rf_full__{protocol}", dataset=dataset,
                             fold=i, seed=seed, n_train=int((~te).sum()),
                             n_test=int(te.sum()), params="", git_sha=sha,
                             train_minutes=round(mins / k, 3),
                             note=f"chem+geom+geomrev+mpnn_site/rf/{protocol}")
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


def publish_run(name: str, dataset: str, frame: pd.DataFrame, folds, y_true, y_pred) -> None:
    """Write one benchmark run into reports/<name>/, tagged with its dataset.

    Same layout src/fusion/export.py writes for our own runs, so src/error_analysis.py and
    the dashboard read a benchmark run with no special-casing. ``cluster`` falls back to the
    pdb id: these sets have no homology clustering of their own, and inventing one would be
    worse than saying so.
    """
    out = paths.REPORTS / name
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame({
        "row_id": frame.row_id.to_numpy(), "complex": frame.pdb.to_numpy(),
        "cluster": frame.pdb.to_numpy(), "fold": folds,
        "y_true": np.asarray(y_true, float), "y_pred": np.asarray(y_pred, float),
    }).dropna(subset=["y_pred"])
    preds.to_csv(out / "predictions.csv", index=False)

    m = metrics.score(preds.y_true, preds.y_pred, complexes=preds["complex"])
    (out / "metrics.json").write_text(json.dumps({"metrics": {
        "n": int(len(preds)), "n_complexes": int(preds["complex"].nunique()),
        "per_complex_spearman": m.get("per_complex_spearman"),
        "global_spearman": m["spearman"], "global_pearson": m["pearson"],
        "rmse": m["rmse"], "mae": m["mae"],
    }, "ci": None}, indent=2))
    (out / "run.json").write_text(json.dumps({
        "name": name, "dataset": dataset, "source": "src.fusion.run_benchmarks",
        "model": "rf chem+geom+geomrev+mpnn_site", "features": [],
        "note": "ProtAttBA benchmark, not our antibody-antigen SKEMPI subset; "
                "cluster column falls back to the pdb id",
        "git_sha": results.git_sha(),
    }, indent=2))


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
    print("our forest (chem+geom+geomrev+mpnn site) vs ProtAttBA (ESM2, sequence only)")
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
    print("Feature set: chem + geom + geomrev + ProteinMPNN site at the mutated residue.")
    print("411 of 2876 rows dropped: their residue numbering does not verify against the PDB.")


if __name__ == "__main__":
    main()
