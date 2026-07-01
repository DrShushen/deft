import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional

import dill
import numpy as np
import pandas as pd

from src.trees.split_criterion import SplitCriterion
from src.utils.feature import FeatureInfo
from src.utils.tree import (
    compress_node,
    evaluate_feature_function,
    feature_output_to_series,
    predict_at_depth_vectorized,
)

logger = logging.getLogger(__name__)

ClassWeight = Mapping[Any, float] | str | None


@dataclass
class Node:
    feature: Optional[
        "FeatureInfo"
    ]  # Contains: fn, name, description, string, threshold, score
    history: Optional[List]
    left: Optional["Node"]  # This is the left node
    right: Optional["Node"]  # This is the right node
    prediction: Optional[float]
    artifacts: Optional[dict] = None  # Used for logging and post-hoc debugging purposes

    # Implement a function to check if the node is a leaf node
    def _is_leaf(self):
        """Check whether leaf."""
        return self.left is None and self.right is None


class AdaptiveDecisionTree:
    def __init__(
        self,
        feature_finder,
        splitting_criterion: SplitCriterion,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        store_node_artifacts=False,
        save_locally=True,
        folder_path=None,
        class_weight: ClassWeight = None,
    ):
        """Initialize the instance."""
        self.feature_finder = feature_finder
        self.splitting_criterion = splitting_criterion
        self.max_depth = max_depth
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.store_node_artifacts = store_node_artifacts
        self.n_leaves = 0
        self.save_locally = save_locally
        self.folder_path = folder_path
        self.class_weight = class_weight
        self.class_weight_: dict[Any, float] | None = None

    def fit(self, X, y, **kwargs):
        """Fit values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AdaptiveDecisionTree.fit expects X to be a pandas DataFrame. "
                f"Received {type(X).__name__}."
            )
        if self.save_locally and not self.folder_path:
            logger.warning(
                "save_locally=True but folder_path is not set; disabling local snapshots."
            )
        self.class_weight_ = self._resolve_class_weight(y)
        self.root = self._grow_tree(X, y, depth=0, history=[], **kwargs)

    def _resolve_class_weight(self, y) -> dict[Any, float] | None:
        """Resolve class_weight into a concrete class-to-weight mapping."""
        if self.class_weight is None:
            return None

        y_values = np.asarray(y)
        classes, counts = np.unique(y_values, return_counts=True)

        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(
                    "`class_weight` must be None, 'balanced', or a mapping from "
                    "class labels to weights."
                )
            n_samples = len(y_values)
            n_classes = len(classes)
            return {
                self._normalize_class_label(class_label): n_samples
                / (n_classes * count)
                for class_label, count in zip(classes, counts)
            }

        if not isinstance(self.class_weight, Mapping):
            raise TypeError(
                "`class_weight` must be None, 'balanced', or a mapping from "
                "class labels to weights."
            )

        resolved_class_weight = {
            self._normalize_class_label(class_label): float(weight)
            for class_label, weight in self.class_weight.items()
        }
        if any(weight < 0 for weight in resolved_class_weight.values()):
            raise ValueError("`class_weight` values must be non-negative.")

        has_positive_weight = any(
            resolved_class_weight.get(self._normalize_class_label(class_label), 1.0) > 0
            for class_label in classes
        )
        if not has_positive_weight:
            raise ValueError("At least one observed class must have positive weight.")

        return resolved_class_weight

    def _normalize_class_label(self, class_label: Any) -> Any:
        """Convert NumPy scalar labels to their Python scalar equivalents."""
        if isinstance(class_label, np.generic):
            return class_label.item()
        return class_label

    def _grow_tree(self, X, y, depth=0, history=None, **kwargs):
        """Handle grow tree."""
        if history is None:
            history = []

        # Create root node
        root = Node(
            feature=None,
            history=history,
            left=None,
            right=None,
            prediction=self._get_prediction(y),
        )
        # Root starts as a leaf; each successful split increases this count by 1.
        self.n_leaves = 1
        should_save_locally = self.save_locally and bool(self.folder_path)
        if should_save_locally:
            os.makedirs(self.folder_path, exist_ok=True)

        # Queue of nodes to process: (node, X subset, y subset, depth, history)
        queue = [(root, X, y, depth, history)]

        while queue:
            # save it locally
            if should_save_locally:
                # Save the root node
                compressed_root = compress_node(root)
                with open(f"{self.folder_path}/intermediate_tree.pkl", "wb") as f:
                    dill.dump(compressed_root, f)

            current_node, node_X, node_y, node_depth, node_history = queue.pop(0)

            # Check stopping criteria
            if node_depth >= self.max_depth or self._should_stop(node_y):
                empty_feature = FeatureInfo()
                current_node.feature = empty_feature
                current_node.prediction = self._get_prediction(node_y)
                continue

            # Find optimal split
            optimal_feature = self.feature_finder.get_optimal_feature(
                node_X,
                node_y,
                history=node_history,
                min_samples_leaf=self.min_samples_leaf,
                splitting_criterion=self.splitting_criterion,
                class_weight=self.class_weight_,
                **kwargs,
            )

            if self.store_node_artifacts:
                current_node.artifacts = self.feature_finder.logging_artifacts
            else:
                current_node.artifacts = None

            # If no valid split found, make current node a leaf
            if not optimal_feature:
                empty_feature = FeatureInfo()
                current_node.feature = empty_feature
                current_node.prediction = self._get_prediction(node_y)
                continue

            current_node.feature = optimal_feature
            feature_values = evaluate_feature_function(optimal_feature.fn, node_X)
            feature_values = feature_output_to_series(feature_values, node_X.index)

            # Split data: left = samples where feature value <= threshold, right = samples where feature value > threshold
            left_mask = feature_values <= optimal_feature.threshold
            right_mask = ~left_mask

            # Validate split
            if not self._is_split_valid(node_y[left_mask], node_y[right_mask]):
                empty_feature = FeatureInfo()
                current_node.feature = empty_feature
                current_node.prediction = self._get_prediction(node_y)
                continue

            # Create child nodes with updated history
            updated_history_left = node_history + [(optimal_feature, "smaller")]
            updated_history_right = node_history + [(optimal_feature, "greater")]

            current_node.left = Node(
                feature=None,
                history=updated_history_left,
                left=None,
                right=None,
                prediction=self._get_prediction(node_y[left_mask]),
            )

            current_node.right = Node(
                feature=None,
                history=updated_history_right,
                left=None,
                right=None,
                prediction=self._get_prediction(node_y[right_mask]),
            )
            # One leaf replaced by two leaves => net +1 leaf.
            self.n_leaves += 1

            # Add child nodes to queue
            queue.append(
                (
                    current_node.left,
                    node_X[left_mask],
                    node_y[left_mask],
                    node_depth + 1,
                    updated_history_left,
                )
            )

            queue.append(
                (
                    current_node.right,
                    node_X[right_mask],
                    node_y[right_mask],
                    node_depth + 1,
                    updated_history_right,
                )
            )

        return root

    def _is_split_valid(self, left_y, right_y):
        """Check whether split valid."""
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return False  
        return True

    def _get_prediction(self, y):
        """Get prediction."""
        return np.mean(y)

    def _should_stop(self, y):
        """Check whether stop."""
        n_samples = len(y)

        if n_samples < self.min_samples_split:
            return True

        if self.max_leaf_nodes and self.n_leaves >= self.max_leaf_nodes:
            return True

        if len(np.unique(y)) == 1:
            return True

        return False

    def predict(self, X, return_proba=False):
        """Predict values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AdaptiveDecisionTree.predict expects X to be a pandas DataFrame. "
                f"Received {type(X).__name__}."
            )
        return predict_at_depth_vectorized(
            root=self.root,
            X=X,
            depth=self.max_depth,
            return_proba=return_proba,
        )

    def pretty_print(self, node=None, indent="", prefix=""):
        """Handle pretty print."""
        if node is None:
            node = self.root
            print("Decision Tree Structure:")

        # If it's a leaf node
        if node._is_leaf():
            print(f"{indent}└── Prediction: {node.prediction:.4f}")
            return

        feature_name = node.feature.name
        feature_description = node.feature.description
        feature_string = node.feature.string
        threshold = node.feature.threshold
        rationale = node.feature.rationale

        # Print the current node's information
        print(f"{indent}{prefix}Feature: {feature_name}")
        print(f"{indent}    Threshold: {threshold:.4f}")
        if feature_description:
            print(f"{indent}    Description: {feature_description}")
        if feature_string:
            print(f"{indent}    String: {feature_string}")
        if rationale:
            print(f"{indent}    Rationale: {rationale}")

        # Print left subtree
        if node.left:
            self.pretty_print(node.left, indent + "    ", "≤ :")

        # Print right subtree
        if node.right:
            self.pretty_print(node.right, indent + "    ", "> :")
