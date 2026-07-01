from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

"""
Note that we want to maximize the scores (hence the negative sign)    
"""


class SplitCriterion(ABC):
    @abstractmethod
    def __call__(
        self,
        y_left,
        y_right,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Run the instance as a callable."""
        pass

    def _normalize_class_label(self, class_label: Any) -> Any:
        """Convert NumPy scalar labels to their Python scalar equivalents."""
        if isinstance(class_label, np.generic):
            return class_label.item()
        return class_label

    def _weighted_class_counts(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> np.ndarray:
        """Return class counts after applying optional class weights."""
        labels, counts = np.unique(y, return_counts=True)
        weighted_counts = counts.astype(float)
        if class_weight is None:
            return weighted_counts

        weights = np.array(
            [
                float(class_weight.get(self._normalize_class_label(label), 1.0))
                for label in labels
            ],
            dtype=float,
        )
        return weighted_counts * weights

    def _weighted_total(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Return the class-weighted size of a node."""
        return float(np.sum(self._weighted_class_counts(y, class_weight)))


class InformationGainCriterion(SplitCriterion):
    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)

    def _entropy(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Handle entropy."""
        counts = self._weighted_class_counts(y, class_weight)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        probabilities = counts / total
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def __call__(
        self,
        y_left,
        y_right,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Run the instance as a callable."""
        n_left = self._weighted_total(y_left, class_weight)
        n_right = self._weighted_total(y_right, class_weight)
        n_total = n_left + n_right
        if n_total <= 0:
            return -np.inf

        p_left = n_left / n_total
        p_right = n_right / n_total

        entropy_left = self._entropy(y_left, class_weight)
        entropy_right = self._entropy(y_right, class_weight)

        return -p_left * entropy_left - p_right * entropy_right + self.offset


class GiniCriterion(SplitCriterion):
    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)

    def _gini(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Return the class-weighted Gini impurity of a node."""
        counts = self._weighted_class_counts(y, class_weight)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        probabilities = counts / total
        return float(1 - np.sum(probabilities**2))

    def __call__(
        self,
        y_left,
        y_right,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Run the instance as a callable."""
        n_left = self._weighted_total(y_left, class_weight)
        n_right = self._weighted_total(y_right, class_weight)
        n_total = n_left + n_right
        if n_total <= 0:
            return -np.inf

        p_left = n_left / n_total
        p_right = n_right / n_total

        gini_left = self._gini(y_left, class_weight)
        gini_right = self._gini(y_right, class_weight)

        return -p_left * gini_left - p_right * gini_right + self.offset
