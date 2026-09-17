"""Harness behaviour. The harness is frozen, so its contract is pinned by tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluate


def _preds(y_true, y_pred, n_complexes=3):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    cx = [f"CX{i % n_complexes}" for i in range(len(y_true))]  # round-robin, so sizes stay even
    return pd.DataFrame({
        "row_id": [f"r{i}" for i in range(len(y_true))],
        "complex": cx, "y_true": y_true, "y_pred": y_pred,
    })


def test_perfect_prediction_scores_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    m = evaluate.metrics(_preds(y, y), min_group=10)
    assert m["per_complex_spearman"] == pytest.approx(1.0)
    assert m["global_pearson"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_constant_prediction_has_undefined_correlation_not_zero():
    """Rung 0 predicts a constant; a silent 0.0 here would misreport the floor."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=60)
    m = evaluate.metrics(_preds(y, np.full_like(y, y.mean())), min_group=10)
    assert np.isnan(m["per_complex_spearman"])
    assert np.isnan(m["global_spearman"])
    assert np.isfinite(m["rmse"])


def test_min_group_drops_small_complexes():
    y = np.arange(30, dtype=float)
    p = _preds(y, y, n_complexes=3)          # 10 per complex
    assert evaluate.metrics(p, min_group=10)["n_complexes_counted"] == 3
    assert evaluate.metrics(p, min_group=11)["n_complexes_counted"] == 0


def test_sign_accuracy_reports_its_own_floor():
    """An all-destabilising set must expose the majority floor next to the score."""
    y = np.concatenate([np.full(90, 2.0), np.full(10, -2.0)])
    m = evaluate.metrics(_preds(y, np.full_like(y, 1.0)), min_group=1)
    assert m["sign_acc_big"] == pytest.approx(0.9)
    assert m["sign_acc_big_base"] == pytest.approx(0.9)      # the score is pure base rate
    assert m["sign_acc_big_balanced"] == pytest.approx(0.5)  # and balanced, it is chance


def test_classes_use_the_documented_edges():
    y = np.array([-1.0, 0.0, 1.0])
    assert list(evaluate.to_classes(y)) == [0, 1, 2]


def test_paired_bootstrap_detects_a_real_improvement():
    rng = np.random.default_rng(2)
    y = rng.normal(size=120)
    worse = _preds(y, rng.normal(size=120), n_complexes=4)
    better = _preds(y, y + rng.normal(scale=0.2, size=120), n_complexes=4)
    r = evaluate.paired_bootstrap(worse, better, min_group=10, n_boot=200)
    assert r["delta"] > 0 and r["clears_zero"]
