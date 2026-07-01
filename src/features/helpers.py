import logging
import signal
import threading
from collections.abc import Mapping
from concurrent.futures import TimeoutError
from copy import copy
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


import functools


def timeout(seconds=60, default=None):

    """Handle timeout."""
    def decorator(func):
        """Handle decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # SIGALRM only works in the main thread; fallback to direct execution
            # to avoid runtime crashes in worker threads.
            """Handle wrapper."""
            if threading.current_thread() is not threading.main_thread():
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.info(f"Got the error: {e}")
                    return default

            def handle_timeout(signum, frame):
                """Handle handle timeout."""
                raise TimeoutError()

            previous_handler = signal.getsignal(signal.SIGALRM)

            try:
                signal.signal(signal.SIGALRM, handle_timeout)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                except TimeoutError:
                    return default
                except Exception as e:
                    logger.info(f"Got the error: {e}")
                    return default
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)

        return wrapper

    return decorator


def filter_features(list_features, X):
    """Filter features."""
    list_curated_features = []
    for feature in list_features:

        @timeout(seconds=60, default=None)
        def compute_feature(fn, data):
            """Compute feature."""
            return fn(data)

        feature_values = compute_feature(feature.fn, copy(X))

        if feature_values is None:
            continue

        # Convert to numpy array if not already
        if not isinstance(feature_values, np.ndarray):
            try:
                feature_values = np.array(feature_values)
            except Exception as e:
                logger.info(f"Got the error: {e}")
                continue

        # try converting to float
        try:
            feature_values = feature_values.astype(float)
        except Exception as e:
            logger.info(f"Got the error: {e}")
            continue

        # Check if feature value is a scalar
        if np.isscalar(feature_values):
            continue

        try:
            # Flatten
            feature_values = feature_values.flatten()

            if len(feature_values) != len(X):
                continue

            # check that there is not nan
            if np.isnan(feature_values).any():
                continue
        except Exception as e:
            logger.info(f"Got the error: {e}")
            continue

        # check if there is one only unique value
        if len(np.unique(feature_values)) == 1:
            logger.info("Feature has only one unique value")
            continue

        logger.info("Successfully added feature")
        list_curated_features.append(feature)
    return list_curated_features


@timeout(seconds=90, default=None)
def optimize_threshold(
    splitting_criterion,
    feature,
    X,
    y,
    min_samples_leaf,
    class_weight: Mapping[Any, float] | None = None,
):
    """
    Given a single feature, find the optimal threshold to split the data for this feature,
    based on the splitting criterion, and update the fields of the feature accordingly.
    """
    n_samples = len(y)

    feature_fn = feature.fn
    # Compute feature values
    feature_values = feature_fn(
        copy(X)
    )  # Copy so that we don't modify the original data

    # Convert to numpy array if not already
    if not isinstance(feature_values, np.ndarray):
        feature_values = np.array(feature_values)

    # convert to float
    feature_values = feature_values.astype(float)

    # Sort values and get unique thresholds between different values
    sorted_idx = np.argsort(feature_values)

    sorted_values = feature_values[sorted_idx]

    # Find unique points to evaluate thresholds
    value_changes = np.where(np.abs(np.diff(sorted_values)) > 1e-7)[0]
    # Make sure we don't go beyond array bounds
    valid_changes = value_changes[value_changes + 1 < len(sorted_values)]

    thresholds = (sorted_values[valid_changes] + sorted_values[valid_changes + 1]) / 2

    best_score = -np.inf
    feature.score = best_score
    feature.threshold = 0

    for threshold in thresholds:
        try:
            left_mask = feature_values <= threshold
            n_left = np.sum(left_mask)
            n_right = n_samples - n_left

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            score = splitting_criterion(
                y[left_mask],
                y[~left_mask],
                class_weight=class_weight,
            )

            if score > best_score:
                best_score = score

                feature.threshold = threshold
                feature.score = score
        except Exception as e:
            logger.info(f"Got the error: {e}")
            continue

    return feature


def optimize_features(
    splitting_criterion,
    list_features,
    X,
    y,
    min_samples_leaf,
    class_weight: Mapping[Any, float] | None = None,
):

    """Handle optimize features."""
    for i in range(len(list_features)):
        feature = list_features[i]
        logging.info(f"Optimizing feature {feature.name}")
        try:
            result = optimize_threshold(
                splitting_criterion,
                feature,
                X,
                y,
                min_samples_leaf,
                class_weight=class_weight,
            )
            if result is None:
                logger.info(f"Feature {feature.name} timed out after 90 seconds")
                feature.score = -np.inf
                feature.threshold = 0
        except Exception as e:
            logger.info(f"Got the error: {e}")
            # Set the score to -inf if we encounter an error
            feature.score = -np.inf
            feature.threshold = 0
            continue
    # Now find the best feature based on the score once we have optimized all the feature thresholds

    return list_features
