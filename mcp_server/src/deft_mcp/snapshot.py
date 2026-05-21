from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.generic_csv_plugin import GenericCsvDatasetPlugin
from src.utils.tree import _compute_split_masks

logger = logging.getLogger(__name__)


def _y_to_series(y: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0]
    return y


def _feature_dict(feature: Any) -> dict | None:
    if feature is None:
        return None
    return {
        "name": str(getattr(feature, "name", "")),
        "description": str(getattr(feature, "description", "") or ""),
        "rationale": str(getattr(feature, "rationale", "") or ""),
        "code": str(getattr(feature, "string", "") or ""),
        "threshold": (
            float(feature.threshold) if getattr(feature, "threshold", None) is not None else None
        ),
        "score": float(feature.score) if getattr(feature, "score", None) is not None else None,
    }


def _walk_with_counts(root: Any, X: pd.DataFrame, y: pd.DataFrame) -> list[dict]:
    """BFS the tree; at each node compute n / n_pos / n_neg from the routed subset."""
    rows: list[dict] = []
    queue: deque[tuple[str, str | None, str | None, Any, pd.DataFrame, pd.DataFrame, int]] = deque(
        [("root", None, None, root, X.reset_index(drop=True), y.reset_index(drop=True), 0)]
    )

    while queue:
        node_id, parent_id, side, node, X_node, y_node, depth = queue.popleft()
        y_series = _y_to_series(y_node)
        n = int(len(y_series))
        n_pos = int((y_series == 1).sum())
        n_neg = int((y_series == 0).sum())

        row: dict[str, Any] = {
            "id": node_id,
            "depth": depth,
            "parent_id": parent_id,
            "side": side,
            "is_leaf": bool(node._is_leaf()),
            "prediction": (float(node.prediction) if node.prediction is not None else None),
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "feature": _feature_dict(node.feature) if not node._is_leaf() else None,
        }

        if not node._is_leaf() and node.feature is not None and n > 0:
            try:
                left_mask, right_mask = _compute_split_masks(
                    node=node, X_subset=X_node, right_is_complement=True, as_numpy=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed to compute split at node %s (%s); subtree counts will be NaN",
                    node_id,
                    exc,
                )
                left_mask = right_mask = None

            row["left"] = f"{node_id}L"
            row["right"] = f"{node_id}R"

            if left_mask is not None and right_mask is not None:
                X_left = X_node.iloc[left_mask].reset_index(drop=True)
                X_right = X_node.iloc[right_mask].reset_index(drop=True)
                y_left = y_node.iloc[left_mask].reset_index(drop=True)
                y_right = y_node.iloc[right_mask].reset_index(drop=True)
            else:
                X_left = X_node.iloc[0:0]
                X_right = X_node.iloc[0:0]
                y_left = y_node.iloc[0:0]
                y_right = y_node.iloc[0:0]

            if node.left is not None:
                queue.append((f"{node_id}L", node_id, "L", node.left, X_left, y_left, depth + 1))
            if node.right is not None:
                queue.append((f"{node_id}R", node_id, "R", node.right, X_right, y_right, depth + 1))

        rows.append(row)

    return rows


def _test_metrics(tree: Any, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[float | None, float | None]:
    if len(y_test) == 0:
        return None, None

    try:
        proba = np.asarray(tree.predict(X_test, return_proba=True), dtype=float)
    except Exception as exc:
        logger.warning("Tree predict failed on test set: %s", exc)
        return None, None

    y_true = _y_to_series(y_test).to_numpy(dtype=int)
    y_hat = (proba >= 0.5).astype(int)

    try:
        acc = float(accuracy_score(y_true, y_hat))
    except Exception:
        acc = None

    auc = None
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, proba))
    except Exception:
        pass

    return acc, auc


def make_snapshot(
    *,
    tree_path: Path,
    csv_path: Path,
    sequence_column: str,
    target_column: str,
    test_size: float,
    random_state: int,
    output_dir: Path,
    input_meta: dict,
    fit_seconds: float,
) -> dict:
    """Walk fitted tree -> tree.json + _meta.yaml. Returns the JSON dict."""
    with tree_path.open("rb") as f:
        tree = dill.load(f)

    plugin = GenericCsvDatasetPlugin(
        csv_path=str(csv_path),
        sequence_column=sequence_column,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = plugin._load_split()

    nodes = _walk_with_counts(tree.root, X_train, y_train)
    test_acc, test_auc = _test_metrics(tree, X_test, y_test)

    y_train_s = _y_to_series(y_train)
    meta = {
        "dataset_csv": str(csv_path),
        "sequence_column": sequence_column,
        "target_column": target_column,
        "max_depth": int(getattr(tree, "max_depth", 0)),
        "n_train": int(len(y_train_s)),
        "n_test": int(len(_y_to_series(y_test))),
        "n_pos_train": int((y_train_s == 1).sum()),
        "n_neg_train": int((y_train_s == 0).sum()),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "fit_seconds": round(float(fit_seconds), 2),
        "input": input_meta,
    }

    snapshot = {"meta": meta, "nodes": nodes, "root_id": "root"}

    tree_json_path = output_dir / "tree.json"
    meta_yaml_path = output_dir / "_meta.yaml"
    tree_json_path.write_text(json.dumps(snapshot, indent=2))
    meta_yaml_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    logger.info(
        "Wrote snapshot: %d nodes (%d leaves) -> %s",
        len(nodes),
        sum(1 for n in nodes if n["is_leaf"]),
        tree_json_path,
    )

    return snapshot
