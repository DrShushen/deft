"""Offline tests for class-imbalance support (`class_weight`) in the DEFT tree.

Deterministic and LLM-free: uses ``VanillaFeatureFinder`` (per-column threshold search) so a
whole tree can be fit without any model calls. Run without pytest-xdist so the signal-based
``@timeout`` in ``optimize_threshold`` stays on the main thread:

    pytest tests/test_class_weight.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_finder import VanillaFeatureFinder
from src.trees.split_criterion import GiniCriterion, InformationGainCriterion
from src.trees.tree import AdaptiveDecisionTree


def _make_tree(class_weight=None, max_depth=1):
    """Build a small LLM-free tree (Vanilla finder + Gini) with the given weighting."""
    return AdaptiveDecisionTree(
        feature_finder=VanillaFeatureFinder(),
        splitting_criterion=GiniCriterion(),
        max_depth=max_depth,
        min_samples_leaf=1,
        save_locally=False,
        class_weight=class_weight,
    )


# --- T-A1: "balanced" resolves to sklearn inverse-frequency weights ---------------------------


def test_balanced_resolves_to_inverse_frequency():
    y = np.array([0] * 90 + [1] * 10)
    resolved = _make_tree(class_weight="balanced")._resolve_class_weight(y)
    # sklearn's compute_class_weight("balanced"): n_samples / (n_classes * count).
    assert resolved == pytest.approx({0: 100 / (2 * 90), 1: 100 / (2 * 10)})
    # Keys are normalized to Python ints (not numpy scalars), so dict lookups match.
    assert all(type(k) is int for k in resolved)


# --- T-A2: mapping passthrough + validation ---------------------------------------------------


def test_mapping_passthrough_and_validation():
    y = np.array([0] * 8 + [1] * 2)

    assert _make_tree(class_weight={0: 1.0, 1: 5.0})._resolve_class_weight(y) == {0: 1.0, 1: 5.0}

    with pytest.raises(ValueError):
        _make_tree(class_weight="not_balanced")._resolve_class_weight(y)

    with pytest.raises(ValueError):
        _make_tree(class_weight={0: 1.0, 1: -1.0})._resolve_class_weight(y)

    with pytest.raises(ValueError):
        _make_tree(class_weight={0: 0.0, 1: 0.0})._resolve_class_weight(y)


# --- T-A3: weighted criterion is correct and backward-compatible ------------------------------


def test_weighted_criterion_and_backward_compat():
    y_left = np.array([0, 0, 0, 1])
    y_right = np.array([0, 1, 1, 1])

    for crit in (GiniCriterion(), InformationGainCriterion()):
        unweighted = crit(y_left, y_right, class_weight=None)
        # Uniform weight 1.0 must reproduce the unweighted score exactly (backward compatibility).
        uniform = crit(y_left, y_right, class_weight={0: 1.0, 1: 1.0})
        assert unweighted == pytest.approx(uniform)
        # A non-uniform weighting must change the score.
        weighted = crit(y_left, y_right, class_weight={0: 1.0, 1: 5.0})
        assert weighted != pytest.approx(unweighted)

    # Weighted class counts are raw counts scaled by per-class weight.
    counts = GiniCriterion()._weighted_class_counts(np.array([0, 0, 1]), {0: 1.0, 1: 5.0})
    assert np.allclose(counts, [2.0, 5.0])  # counts [2, 1] * weights [1.0, 5.0]


# --- T-A4: leaf prediction is class-weighted (the completion) ---------------------------------


def test_leaf_prediction_is_class_weighted():
    y_node = np.array([0] * 6 + [1] * 4)  # 6 neg, 4 pos -> raw positive rate 0.4

    tree_none = _make_tree(class_weight=None)  # class_weight_ stays None
    assert tree_none._get_prediction(y_node) == pytest.approx(0.4)

    tree_w = _make_tree(class_weight={0: 1.0, 1: 5.0})
    tree_w.class_weight_ = {0: 1.0, 1: 5.0}  # normally set in fit(); set directly here
    # w_pos*n_pos / (w_pos*n_pos + w_neg*n_neg) = 20 / (20 + 6) = 0.769...
    assert tree_w._get_prediction(y_node) == pytest.approx(20 / 26)

    # The same leaf flips from negative (raw) to positive (weighted) under the 0.5 threshold.
    assert tree_none._get_prediction(y_node) < 0.5 < tree_w._get_prediction(y_node)


# --- T-A5: end-to-end offline fit shows the imbalance effect -----------------------------------


def _imbalanced_dataset():
    """Two f0 clusters; within each, positives are a raw minority but locally enriched.

    Cluster A (f0=0.25): 6 neg + 4 pos. Cluster B (f0=0.75): 12 neg + 2 pos. The clusters are
    constant-valued, so a depth-1 tree is forced to split at 0.5 (the only candidate threshold),
    yielding exactly those two leaves.
    """
    f0 = [0.25] * 10 + [0.75] * 14
    labels = [0] * 6 + [1] * 4 + [0] * 12 + [1] * 2
    return pd.DataFrame({"f0": f0}), pd.Series(labels)


def _recall_pos(pred, y):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y).astype(bool)
    return float((pred & y).sum()) / float(y.sum())


def test_end_to_end_vanilla_fit_recovers_minority():
    X, y = _imbalanced_dataset()

    tree_none = _make_tree(class_weight=None, max_depth=1)
    tree_none.fit(X, y)
    recall_none = _recall_pos(tree_none.predict(X, return_proba=False), y)

    tree_bal = _make_tree(class_weight="balanced", max_depth=1)
    tree_bal.fit(X, y)
    recall_bal = _recall_pos(tree_bal.predict(X, return_proba=False), y)

    # Unweighted: every leaf is raw-majority-negative -> predicts all 0 -> recovers no positives.
    assert recall_none == 0.0
    # Balanced weighting flips the enriched leaf to positive -> recovers minority-class samples.
    assert recall_bal > recall_none
