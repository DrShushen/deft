from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset_plugins import BaseDatasetPlugin

logger = logging.getLogger(__name__)


class GenericCsvDatasetPlugin(BaseDatasetPlugin):
    """Generic CSV-backed dataset plugin.

    Reads a single CSV with at least:
    - a sequence column (string DNA)
    - a binary target column (0/1)
    Any other columns are ignored.

    Performs a stratified train/test split internally with configurable
    `test_size` and `random_state`. The split is recomputed on each call
    so the same instance returns consistent splits across load_for_deft
    and load_for_baseline.
    """

    dl_subsample_before_val_split = False

    def __init__(
        self,
        csv_path: str,
        sequence_column: str = "raw_sequence",
        target_column: str = "label",
        test_size: float = 0.2,
        random_state: int = 42,
        tree_filename: str = "tree_generic.dill",
    ):
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            csv_path_obj = (repo_root / csv_path_obj).resolve()
        self.csv_path = csv_path_obj
        self.sequence_column = str(sequence_column)
        self.target_column = str(target_column)
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.tree_filename = str(tree_filename)

    def _load_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        for col in (self.sequence_column, self.target_column):
            if col not in df.columns:
                raise KeyError(
                    f"Column `{col}` not found in {self.csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        X = pd.DataFrame(
            df[self.sequence_column].astype(str).values, columns=["raw_sequence"]
        )
        y = pd.DataFrame(df[self.target_column].astype(int).values, columns=["label"])

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y["label"],
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        return (
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
        )

    def load_for_deft(self) -> dict[str, Any]:
        X_train, _, y_train, _ = self._load_split()
        logger.info(
            "GenericCsv loaded %d train rows (test_size=%.2f, seed=%d) from %s",
            len(X_train),
            self.test_size,
            self.random_state,
            self.csv_path,
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "tree_filename": self.tree_filename,
            "prompt_context": {},
        }

    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        return self._load_split()

    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        return self._load_split()
