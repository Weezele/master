"""Main application entry point."""

from ..config import Config
from ..service import DatasetBalancerService


class DatasetBalancerApp:
    """Main application class for dataset balancing."""

    def __init__(self):
        self.config = Config()
        self.service = DatasetBalancerService(self.config)

    def run(self) -> None:
        """Execute the dataset balancing pipeline."""
        print("\n" + "#" * 70)
        print("#" + " " * 68 + "#")
        print("#" + "  MENTAL HEALTH DATASET BALANCER  ".center(68) + "#")
        print("#" + "  Reduce to 20,000 balanced samples  ".center(68) + "#")
        print("#" + " " * 68 + "#")
        print("#" * 70)

        # Step 1: Load dataset
        df = self.service.load_dataset()

        # Step 2: Analyze original dataset
        self.service.analyze_dataset(df)

        # Step 3: Balance dataset (equal samples per class)
        balanced_df = self.service.balance_dataset(df)

        # Step 4: Shuffle dataset
        shuffled_df = self.service.shuffle_dataset(balanced_df)

        # Step 5: Save balanced dataset
        self.service.save_dataset(shuffled_df)

        # Step 6: Print final summary
        self.service.print_final_summary(shuffled_df)

        print("\n" + "#" * 70)
        print("  DONE! Dataset balanced and saved successfully.")
        print("#" * 70 + "\n")
