"""Class separation without letting the class mix or the cut points do the work.

Accuracy is unusable here and so is anything built on a threshold. The three classes are
13% stabilising / 34% neutral / 53% destabilising, so predicting the majority everywhere
scores 53%, and the complex-mean baseline -- a model with no access to the mutation --
reaches 64% accuracy and 0.59 macro-F1, beating every model in this project.

Binning a score into classes is also a decision that changes the answer. The ordinal head
emits an ordered score on an arbitrary scale, so it has to be cut somewhere, and cutting at
the label edges versus at the score's own quantiles moves minority recall by a lot. That
choice should not be load-bearing.

So the metrics here are threshold-free and prevalence-independent:

``AUC(stab|dest)``
    the probability a randomly chosen destabilising mutation scores above a randomly chosen
    stabilising one. This is the question an engineer asks -- can it tell a good mutation
    from a bad one -- and it is unaffected by how many of each exist.
``AUC(stab|rest)``, ``AUC(dest|rest)``
    one class against the other two, so a model cannot profit from ignoring the minority.
``macro AUC``
    the mean of the three pairwise AUCs, each computed on a balanced comparison.
``Somers' D``
    rank association between the score and the ordered class, over all discordant pairs.
    A single number for "does the score order the classes", with no thresholds at all.
``balanced accuracy``
    mean per-class recall at the quantile cut, kept only so the threshold-based view can be
    compared against the threshold-free one.

0.5 is chance for every AUC here; 0.333 is chance for balanced accuracy.

    python scripts/class_separation.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

EDGES = (-0.5, 0.5)
NAMES = ("stabilising", "neutral", "destabilising")
MODELS = {
    "forest":       ("reports", "E0a_rf_handcrafted_seed0"),
    "film_chem":    ("oof", "struct_film_chem"),          # the submitted model
    "film_esmif":   ("oof", "struct_film_chem_esmif"),
    "gated_cg":     ("oof", "gated_cg_clusterscale"),
    "film_nochem":  ("oof", "struct_film_site"),
    "seq_only":     ("oof", "mut_pair_ffn_sub"),
    "delta_xattn":  ("oof", "delta_xattn2"),
    "l1_gated":     ("oof", "l1_gated"),
    "l1_concat":    ("oof", "cat128_reg2_l1"),            # the previous submission
}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive outranks a random negative), ties counted as half.

    Computed from ranks rather than by sweeping thresholds, so it is exact and needs no
    grid: the Mann-Whitney U statistic normalised by the number of pairs.
    """
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def somers_d(score: np.ndarray, cls: np.ndarray) -> float:
    """Rank association between an arbitrary score and an ORDERED class.

    Kendall's tau over pairs that differ in class, which is what makes it prevalence
    independent: pairs within a class contribute nothing either way.
    """
    conc = disc = 0
    for a in range(3):
        for b in range(a + 1, 3):
            x, y = score[cls == a], score[cls == b]
            if not len(x) or not len(y):
                continue
            d = np.sign(y[:, None] - x[None, :])     # b is the higher class
            conc += int((d > 0).sum())
            disc += int((d < 0).sum())
    return (conc - disc) / max(conc + disc, 1)


def within_complex(score: np.ndarray, cls: np.ndarray, cx: np.ndarray,
                   lo: int, hi: int) -> tuple[float, int, int]:
    """AUC(hi over lo) computed inside each complex, then averaged over complexes.

    Pooled AUC does not work here for the same reason pooled Pearson does not: complexes
    differ in their class mix, so ordering complexes correctly scores well without ordering
    a single mutation. Predicting each complex's own mean -- a model that cannot see the
    mutation at all -- reaches 0.93 pooled AUC on stabilising against destabilising.

    Restricting every comparison to pairs drawn from the SAME complex removes that channel
    entirely: the complex term cancels, and what is left is the question a designer asks.
    Complexes contributing no such pair are skipped rather than scored as 0.5.
    """
    per, pairs = [], 0
    for c in np.unique(cx):
        m = cx == c
        a, b = score[m & (cls == lo)], score[m & (cls == hi)]
        if not len(a) or not len(b):
            continue
        per.append(auc(b, a))
        pairs += len(a) * len(b)
    return (float(np.mean(per)) if per else np.nan), len(per), pairs


def load(kind: str, name: str) -> pd.Series:
    if kind == "reports":
        return pd.read_csv(paths.REPORTS / name / "predictions.csv").set_index("row_id").y_pred
    return pd.read_csv(f"results/oof/{name}.csv").groupby("row_id").ddg_pred.mean()


def main() -> None:
    ref = pd.read_csv(paths.REPORTS / "perturb_v3_base" / "predictions.csv").set_index("row_id")
    truth = ref.y_true
    tc = np.digitize(truth.values, EDGES)
    n = [int((tc == k).sum()) for k in range(3)]
    print(f"{len(truth)} rows — " + ", ".join(f"{a} {b} ({b/len(truth):.0%})"
                                              for a, b in zip(NAMES, n)))
    print("chance: 0.500 for every AUC and Somers' D of 0.000; 0.333 balanced accuracy\n")

    rows = []
    for tag, (kind, name) in MODELS.items():
        try:
            p = load(kind, name)
        except FileNotFoundError:
            continue
        i = p.index.intersection(truth.index)
        s, c = p.loc[i].to_numpy(), np.digitize(truth.loc[i].values, EDGES)
        # balanced accuracy at the quantile cut, for the threshold-based comparison only
        q = np.quantile(s, np.cumsum([np.mean(c == k) for k in range(2)]))
        pc = np.digitize(s, q)
        bal = float(np.mean([np.mean(pc[c == k] == k) for k in range(3) if (c == k).any()]))
        rows.append({
            "model": tag,
            "AUC stab|dest": auc(s[c == 2], s[c == 0]),
            "AUC stab|rest": auc(s[c != 0], s[c == 0]),
            "AUC dest|rest": auc(s[c == 2], s[c != 2]),
            "macro AUC": np.nanmean([auc(s[c == 1], s[c == 0]),
                                     auc(s[c == 2], s[c == 1]),
                                     auc(s[c == 2], s[c == 0])]),
            "Somers D": somers_d(s, c),
            "bal acc": bal,
        })
    t = pd.DataFrame(rows).set_index("model").sort_values("macro AUC", ascending=False)
    print(t.round(3).to_string())

    print()
    print("=== WITHIN COMPLEX - the same question, complex identity removed ===")
    print("Every comparison is between two mutations of the SAME complex, averaged")
    print("over complexes. Chance is 0.500.")
    print()
    wrows = []
    cxall = ref["complex"]
    for tag, (kind, name) in MODELS.items():
        try:
            p = load(kind, name)
        except FileNotFoundError:
            continue
        i = p.index.intersection(truth.index)
        s_, c_ = p.loc[i].to_numpy(), np.digitize(truth.loc[i].values, EDGES)
        cx_ = cxall.loc[i].to_numpy()
        sd, ncx_sd, _ = within_complex(s_, c_, cx_, 0, 2)
        nd, ncx_nd, _ = within_complex(s_, c_, cx_, 1, 2)
        sn, ncx_sn, _ = within_complex(s_, c_, cx_, 0, 1)
        wrows.append({"model": tag, "stab|dest": sd, "n_cx": ncx_sd,
                      "neut|dest": nd, "stab|neut": sn,
                      "macro": np.nanmean([sd, nd, sn])})
    wt = pd.DataFrame(wrows).set_index("model").sort_values("macro", ascending=False)
    print(wt.round(3).to_string())

    cmv = ref.groupby("complex").y_true.transform("mean").to_numpy()
    cc = np.digitize(truth.values, EDGES)
    fsd, fn, _ = within_complex(cmv, cc, cxall.to_numpy(), 0, 2)
    print()
    print(f"[complex mean only]  stab|dest {fsd:.3f} over {fn} complexes")
    print("It is constant within a complex, so it cannot order two mutations of the")
    print("same one: chance by construction. That is the point -- this metric gives it")
    print("nothing, where pooled AUC handed it 0.93.")


if __name__ == "__main__":
    main()
