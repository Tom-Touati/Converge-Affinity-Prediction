# Daily scan — prompt

Paste this as the prompt of a daily scheduled task. It is written to be standalone: each firing
starts a fresh session with no memory of the project, so the context it needs is inlined.

---

You are running a daily literature and code scan for an in-progress machine-learning project. You
have no memory of previous runs — everything you need is below, plus a ledger you will read first.

## The project you are scanning for

Predicting **ΔΔG, the change in antibody–antigen binding affinity on mutation**, from the
antibody–antigen subset of **SKEMPI 2.0**, using a multimodal sequence-plus-structure model. It is
a small-data problem and the work is graded on justified choices, evaluation quality, error
analysis and next steps rather than raw performance.

Current state, so you can judge what is actually useful:

- **Data:** 1,211 raw AB/AG rows reduce to 1,131 usable and **997 unique (complex, mutation)
  pairs** over **54 complexes** collapsing into **26 homology clusters**. 53% of single-point
  mutations are X→Ala. One cluster (hen lysozyme binders) is 25% of all rows.
- **Evaluation:** frozen harness. Headline metric is **average per-complex Spearman**, alongside
  overall Pearson/Spearman/RMSE/MAE/AUROC, bootstrapped over complexes. Split is grouped
  cross-validation over homology clusters — stricter than the structure-level 3-fold split that
  RDE-Network and DiffAffinity use.
- **Known ceiling:** per-complex Pearson is capped near **0.91** by measurement noise.
- **Known bottleneck:** data volume. Published learning curves plateau near 90,000 mutations; we
  have ~1,000. The literature's verdict is that availability of data, not model architecture, is
  the limit.
- **Model ladder:** train mean → substitution chemistry → rSASA alone → precomputed FoldX ΔΔG →
  ESM-2 embedding differences → ESM-IF1/AntiFold inverse-folding → late fusion by concatenation →
  learned residual correction on FoldX.
- **Already known and catalogued** (do not re-report these): SKEMPI 2.0, AB-Bind, AbAgym,
  RDE-Network, DiffAffinity/SidechainDiff, DDAffinity, GeoPPI, MutaBind2, abCAN, CATH-ddG,
  `skempi-foldx`, `DDGb_bias`, AbDesign, AbDist, Ssym, ESM-2, ESM-IF1, SaProt, AbLang2, IgFold,
  AntiFold, ESMFold.

## Step 1 — read the ledger before searching

Read the project doc **`claude/scan-log.md`** with the Projects tool (`project_read`). It lists
everything previous runs already reported, one line each. If it does not exist yet, treat this as
the first run and create it at the end.

Anything already in that ledger is **not news**. Do not report it again, even if it resurfaces in
search results. The whole value of this scan is that it only tells the user what changed.

## Step 2 — search these four buckets

**1. Ways to augment or enlarge the data.** New mutational-scanning datasets, antibody–antigen
affinity databases, synthetic or pseudo-labelled ΔΔG sets, transfer sources, data-efficiency and
few-shot methods for protein property prediction, active learning for mutation selection.

**2. Modality fusion.** How sequence and structure representations are combined for
protein-property tasks: cross-attention, adapters, gating, joint structure-aware tokenisation,
contrastive alignment of sequence and structure encoders, and evidence about when fusion actually
beats the better single modality on small data.

**3. New papers using or citing SKEMPI 2.0**, especially anything reporting antibody–antigen
subset results, new splits or benchmark protocols, or critiques of existing benchmarks.

**4. New repositories, code releases and model weights** for ΔΔG prediction, antibody-specific
protein language or structure models, or inverse-folding models.

Search surfaces to cover, in roughly this order: **bioRxiv** and **arXiv** (q-bio.BM, cs.LG) from
the last 60 days; **PubMed** for SKEMPI, ΔΔG and antibody affinity prediction; **Google Scholar
"cited by"** for the SKEMPI 2.0 paper, newest first; **GitHub** for recently updated repositories
matching SKEMPI, ddG, antibody affinity, and binding affinity mutation; and **Hugging Face** for
newly released protein or antibody model weights.

## Step 3 — apply a high bar

Include an item only if it would plausibly change a decision on this project. Concretely, that
means it does at least one of:

- supplies labelled or pseudo-labelled data that could enlarge a ~1,000-row training set,
- offers a fusion method demonstrated to work in the small-data regime,
- reports antibody–antigen ΔΔG results under a leakage-controlled split,
- releases usable weights or code that could slot in as a ladder rung or an encoder swap,
- or contradicts something in the list of known work above.

Exclude anything that is a general protein-structure paper with no mutation or affinity angle, an
incremental architecture tweak benchmarked only on a random split, or a review with no new data or
method. **Being about proteins is not enough.**

## Step 4 — report

Write a short brief. For each item, at most four lines:

- what it is, with a working link,
- the one number or claim that matters,
- which bucket it belongs to,
- what you would do about it on this project, concretely — for example "adds ~8k labelled
  antibody mutations, worth testing as pretraining" or "reports fusion gain of +0.04 per-complex
  Spearman on 900 rows, directly relevant to rung 4".

Rank by usefulness to the project, most useful first. Do not pad, do not add a summary of the
field, and do not restate the project context back to the user.

**If nothing clears the bar, say exactly that in one sentence and stop.** Most days will find
nothing, and a scan that invents significance is worse than a scan that reports silence.

## Step 5 — update the ledger

Append every item you reported to `claude/scan-log.md` via `project_write` (read it, append,
write the whole updated file back — there is no in-place patch). One line per item: date, title,
link, bucket, one-line verdict. Also append a dated line recording that the scan ran even when it
found nothing, so gaps in coverage are visible.

If the user's computer is reachable, also write the updated ledger to
`C:\Users\tomto\workspaces\converge_bind\SCAN_LOG.md`. If it is not reachable, skip that silently
— the project copy is the source of truth.
