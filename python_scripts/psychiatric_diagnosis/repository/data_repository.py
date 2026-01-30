"""Data repository for loading and preparing datasets."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..domain import DataSplit


class DataRepository:
    """Handles data loading and preparation."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self._feature_names = None

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from CSV."""
        dataset_path = self.config.dataset_path

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                f"Please ensure Mental_Health_Pure_Numerical.csv is in {self.config.DATA_DIR}"
            )

        print(f"  Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"  Dataset shape: {df.shape}")

        return df

    def prepare_data(self, df: pd.DataFrame) -> DataSplit:
        """Prepare data: separate features/labels, split, and scale."""

        # Separate features and target
        X = df.drop(columns=[self.config.TARGET_COLUMN]).values
        y = df[self.config.TARGET_COLUMN].values

        # Store feature names for later use
        self._feature_names = df.drop(columns=[self.config.TARGET_COLUMN]).columns.tolist()

        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Class distribution: {np.bincount(y.astype(int))}")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        # Second split: train vs val
        val_ratio = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        data_split = DataSplit(
            X_train=X_train_scaled.astype(np.float32),
            X_val=X_val_scaled.astype(np.float32),
            X_test=X_test_scaled.astype(np.float32),
            y_train=y_train.astype(np.int64),
            y_val=y_val.astype(np.int64),
            y_test=y_test.astype(np.int64)
        )

        print(f"  Train samples: {data_split.num_samples['train']}")
        print(f"  Validation samples: {data_split.num_samples['val']}")
        print(f"  Test samples: {data_split.num_samples['test']}")

        return data_split

    @property
    def feature_names(self):
        """Get feature names after loading data."""
        return self._feature_names

    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler for later use."""
        return self.scaler
