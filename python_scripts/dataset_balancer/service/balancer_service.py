"""Dataset balancer service - Core business logic."""

import pandas as pd

from ..config import Config, LABEL_NAMES


class DatasetBalancerService:
    """Handles balancing, reducing, and shuffling the dataset."""

    def __init__(self, config: Config):
        self.config = config

    def load_dataset(self) -> pd.DataFrame:
        """Load the merged dataset."""
        input_path = self.config.DATASETS_DIR / self.config.INPUT_FILENAME
        print(f"Loading dataset from: {input_path}")
        df = pd.read_csv(input_path)
        return df

    def analyze_dataset(self, df: pd.DataFrame) -> None:
        """Print detailed analysis of the current dataset."""
        print("\n" + "=" * 70)
        print("ORIGINAL DATASET ANALYSIS")
        print("=" * 70)

        print(f"\n  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

        print("\n  Label Distribution (Before Balancing):")
        print("  " + "-" * 55)
        label_counts = df[self.config.TARGET_COLUMN].value_counts().sort_index()

        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            bar = "#" * int(pct / 2)
            label_name = LABEL_NAMES.get(label, f"Unknown-{label}")
            print(f"    {label} ({label_name:12}): {count:>8,} ({pct:5.1f}%) {bar}")

        print("\n  Statistics:")
        print(f"    Min samples in a class: {label_counts.min():,}")
        print(f"    Max samples in a class: {label_counts.max():,}")
        print(f"    Imbalance ratio: {label_counts.max() / label_counts.min():.2f}x")

        print("\n  Memory Usage: {:.2f} MB".format(
            df.memory_usage(deep=True).sum() / (1024 * 1024)
        ))

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset to have equal samples per class."""
        print("\n" + "=" * 70)
        print("BALANCING DATASET")
        print("=" * 70)

        samples_per_class = self.config.SAMPLES_PER_CLASS
        print(f"\n  Target: {self.config.TOTAL_SAMPLES:,} total samples")
        print(f"  Samples per class: {samples_per_class:,}")

        balanced_dfs = []

        for label in sorted(df[self.config.TARGET_COLUMN].unique()):
            class_df = df[df[self.config.TARGET_COLUMN] == label]
            available = len(class_df)
            label_name = LABEL_NAMES.get(label, f"Unknown-{label}")

            if available >= samples_per_class:
                # Undersample: randomly select samples_per_class
                sampled = class_df.sample(
                    n=samples_per_class,
                    random_state=self.config.RANDOM_STATE
                )
                status = "undersampled"
            else:
                # Oversample: sample with replacement to reach samples_per_class
                sampled = class_df.sample(
                    n=samples_per_class,
                    replace=True,
                    random_state=self.config.RANDOM_STATE
                )
                status = "oversampled"

            balanced_dfs.append(sampled)
            print(f"    {label} ({label_name:12}): {available:>8,} -> {samples_per_class:,} ({status})")

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"\n  Balanced dataset size: {len(balanced_df):,}")

        return balanced_df

    def shuffle_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle the dataset randomly."""
        print("\n" + "=" * 70)
        print("SHUFFLING DATASET")
        print("=" * 70)

        shuffled_df = df.sample(
            frac=1,
            random_state=self.config.RANDOM_STATE
        ).reset_index(drop=True)

        print("  [OK] Dataset shuffled with random_state={}".format(
            self.config.RANDOM_STATE
        ))

        return shuffled_df

    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save the balanced dataset."""
        output_path = self.config.DATASETS_DIR / self.config.OUTPUT_FILENAME
        print("\n" + "=" * 70)
        print("SAVING DATASET")
        print("=" * 70)

        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        print(f"  Size: {df.shape[0]:,} rows x {df.shape[1]} columns")

    def print_final_summary(self, df: pd.DataFrame) -> None:
        """Print summary of the final balanced dataset."""
        print("\n" + "=" * 70)
        print("FINAL BALANCED DATASET SUMMARY")
        print("=" * 70)

        print(f"\n  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

        print("\n  Label Distribution (After Balancing):")
        print("  " + "-" * 55)
        label_counts = df[self.config.TARGET_COLUMN].value_counts().sort_index()

        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            bar = "#" * int(pct / 2)
            label_name = LABEL_NAMES.get(label, f"Unknown-{label}")
            print(f"    {label} ({label_name:12}): {count:>8,} ({pct:5.1f}%) {bar}")

        print("\n  Verification:")
        print(f"    All classes equal: {label_counts.nunique() == 1}")
        print(f"    Total samples: {len(df):,}")

        print("\n  Memory Usage: {:.2f} MB".format(
            df.memory_usage(deep=True).sum() / (1024 * 1024)
        ))
