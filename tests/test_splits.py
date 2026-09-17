"""Split integrity. These are the checks that stop a silent leak, so they run in CI, not by eye."""
from __future__ import annotations

import pandas as pd
import pytest

from src import paths, splits

pytestmark = pytest.mark.skipif(
    not paths.FOLDS.exists() or not paths.DATASET.exists(),
    reason="run `make data && make splits` first",
)


@pytest.fixture(scope="module")
def df():
    return splits.load()


def test_every_row_has_exactly_one_fold(df):
    assert df["fold"].notna().all()
    assert df["row_id"].is_unique


def test_no_cluster_spans_two_folds(df):
    """The core guarantee: a test complex never has a homologue in training."""
    assert (df.groupby("cluster")["fold"].nunique() == 1).all()


def test_no_complex_spans_two_folds(df):
    assert (df.groupby("#Pdb")["fold"].nunique() == 1).all()


def test_folds_are_not_degenerate(df):
    counts = df["fold"].value_counts()
    assert len(counts) >= 2
    # no fold may hold more than half the data, or the "average over folds" is meaningless
    assert counts.max() / len(df) < 0.5


def test_dataset_matches_the_frozen_split(df):
    """A changed dataset with an unchanged folds.csv would silently drop or mis-assign rows."""
    folds = pd.read_csv(paths.FOLDS)
    ds = pd.read_parquet(paths.DATASET)
    assert set(folds["row_id"]) == set(ds["row_id"])


def test_row_ids_are_complex_plus_mutation(df):
    rebuilt = df["#Pdb"] + "|" + df["mutations"]
    assert (rebuilt == df["row_id"]).all()
