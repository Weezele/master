"""Dataset merger service - Core business logic."""

import pandas as pd

from ..config import Config, LABEL_NAMES
from ..repository import DatasetRepository


class DatasetMergerService:
    """Handles the core business logic of merging datasets."""

    def __init__(self, repository: DatasetRepository, config: Config):
        self.repository = repository
        self.config = config

    def merge_datasets(self) -> pd.DataFrame:
        """Main method to merge all datasets."""
        print("=" * 60)
        print("MENTAL HEALTH DATASET MERGER")
        print("=" * 60)

        # Step 1: Discover files
        print("\n[1/4] Discovering dataset files...")
        files = self.repository.discover_files()
        print(f"      Found {len(files)} files to process")

        # Step 2: Load and stack all files
        print("\n[2/4] Loading and merging files...")
        dataframes = []

        for file in files:
            df = self.repository.load_file(file)
            dataframes.append(df)
            print(f"      [OK] {file.filepath.name}: {len(df):,} rows | "
                  f"Label: {file.label} ({file.label_name})")

        # Stack all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n      Total rows after merge: {len(merged_df):,}")

        # Step 3: Handle missing data
        print("\n[3/4] Handling missing data...")
        missing_before = merged_df.isnull().sum().sum()

        # Get numerical columns (excluding target_label)
        numerical_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
        tfidf_cols = [col for col in numerical_cols if col != self.config.TARGET_COLUMN]

        # Fill missing TF-IDF values with 0
        merged_df[tfidf_cols] = merged_df[tfidf_cols].fillna(0)

        missing_after = merged_df.isnull().sum().sum()
        print(f"      Missing values: {missing_before:,} -> {missing_after:,}")

        # Step 4: Shuffle the dataset
        print("\n[4/4] Shuffling dataset...")
        merged_df = merged_df.sample(
            frac=1,
            random_state=self.config.RANDOM_STATE
        ).reset_index(drop=True)
        print("      [OK] Dataset shuffled")

        return merged_df

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics of the merged dataset."""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"\n  Final Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

        print("\n  Label Distribution:")
        label_counts = df[self.config.TARGET_COLUMN].value_counts().sort_index()
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"    {label} ({LABEL_NAMES[label]:12}): {count:>8,} samples ({pct:5.1f}%)")

        print("\n  Memory Usage:")
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"    {mem_mb:.2f} MB")

        print("\n  Column Types:")
        print(f"    Numerical features: {len(df.select_dtypes(include=['float64']).columns)}")
        print(f"    Target column: {self.config.TARGET_COLUMN}")
