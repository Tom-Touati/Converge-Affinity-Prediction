"""Make a partial results/OOF pair consistent, so a rebuild can resume at fold level.

The trainer writes both files after every fold, but the collector grabs them as two
separate transfers -- so a session reclaimed mid-run leaves a pair captured at different
moments, and the OOF table can be a fold AHEAD of the results table. Resuming from that
pair unchanged retrains the fold the results table is missing and appends its rows to an
OOF table that already has them, which publish then scores twice.

Trimming the OOF down to the folds the results table knows about is the safe direction:
it costs one fold of retraining and cannot produce a duplicated row.

    python reconcile_partial.py <results.csv> <oof.csv>
"""
import sys

import pandas as pd

res, oof = sys.argv[1], sys.argv[2]
r = pd.read_csv(res)
d = pd.read_csv(oof)
keep = {(int(a), int(b)) for a, b in zip(r.fold, r.seed)}
before = len(d)
d = d[[(int(a), int(b)) in keep for a, b in zip(d.fold, d.seed)]]
if len(d) != before:
    d.to_csv(oof, index=False)
print(f"reconcile: results folds {sorted(r.fold.unique())}, "
      f"oof {before} -> {len(d)} rows")
