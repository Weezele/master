"""Dataset repository for file operations."""

from pathlib import Path
from typing import List

import pandas as pd

from ..config import Config, LABEL_MAPPING
from ..domain import DatasetFile


class DatasetRepository:
    """Handles file discovery and data loading operations."""

    def __init__(self, config: Config):
        self.config = config

    def discover_files(self) -> List[DatasetFile]:
        """Discover all dataset files in the datasets directory."""
        files = []

        if not self.config.DATASETS_DIR.exists():
            raise FileNotFoundError(
                f"Datasets directory not found: {self.config.DATASETS_DIR}"
            )

        for filepath in self.config.DATASETS_DIR.glob("*.csv"):
            filename = filepath.name

            # Skip the output file if it exists
            if filename == self.config.OUTPUT_FILENAME:
                continue

            # Parse file type and topic
            if filename.endswith(self.config.FILE_PATTERN_SUFFIX_PRE):
                file_type = "pre"
                topic = filename.replace(self.config.FILE_PATTERN_SUFFIX_PRE, "")
            elif filename.endswith(self.config.FILE_PATTERN_SUFFIX_POST):
                file_type = "post"
                topic = filename.replace(self.config.FILE_PATTERN_SUFFIX_POST, "")
            else:
                print(f"  [SKIP] Unknown file pattern: {filename}")
                continue

            # Get label for topic
            if topic not in LABEL_MAPPING:
                print(f"  [SKIP] Unknown topic: {topic}")
                continue

            files.append(DatasetFile(
                filepath=filepath,
                topic=topic,
                label=LABEL_MAPPING[topic],
                file_type=file_type
            ))

        return sorted(files, key=lambda x: (x.topic, x.file_type))

    def load_file(self, dataset_file: DatasetFile) -> pd.DataFrame:
        """Load a single CSV file and prepare it for merging."""
        df = pd.read_csv(dataset_file.filepath, low_memory=False)

        # Drop non-numerical columns immediately to save memory
        cols_to_drop = [
            col for col in self.config.COLUMNS_TO_DROP
            if col in df.columns
        ]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Add target label
        df[self.config.TARGET_COLUMN] = dataset_file.label

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: Path) -> None:
        """Save the merged dataset to CSV."""
        df.to_csv(filepath, index=False)
