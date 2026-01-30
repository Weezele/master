"""Main application orchestrator."""

from typing import Optional

from ..config import Config
from ..repository import DatasetRepository
from ..service import DatasetMergerService


class DatasetMergerApp:
    """Main application class that orchestrates the merge process."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.repository = DatasetRepository(self.config)
        self.service = DatasetMergerService(self.repository, self.config)

    def run(self) -> None:
        """Execute the dataset merging pipeline."""
        try:
            # Merge datasets
            merged_df = self.service.merge_datasets()

            # Print summary
            self.service.print_summary(merged_df)

            # Save output
            output_path = self.config.OUTPUT_DIR / self.config.OUTPUT_FILENAME
            print(f"\n  Saving to: {output_path}")
            self.repository.save_dataset(merged_df, output_path)
            print("  [OK] Dataset saved successfully!")

            print("\n" + "=" * 60)
            print("MERGE COMPLETE")
            print("=" * 60)

        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            raise
