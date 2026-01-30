"""Application configuration settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Application configuration constants."""

    # Paths
    DATA_DIR: Path = Path(__file__).parent.parent.parent.parent / "datasets"
    DATASET_FILENAME: str = "Mental_Health_Balanced_20k.csv"
    OUTPUT_DIR: Path = Path(__file__).parent.parent / "models"

    # Data settings
    TARGET_COLUMN: str = "target_label"
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1  # From remaining training data
    RANDOM_STATE: int = 42

    # Number of classes
    NUM_CLASSES: int = 7

    # Class labels for display
    CLASS_NAMES: tuple = (
        "Normal",
        "Depression",
        "Autism",
        "Anxiety",
        "PTSD",
        "Addiction",
        "Alcoholism"
    )

    @property
    def dataset_path(self) -> Path:
        return self.DATA_DIR / self.DATASET_FILENAME
