"""Application configuration settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Application configuration constants."""

    DATASETS_DIR: Path = Path(__file__).parent.parent.parent.parent / "datasets"
    INPUT_FILENAME: str = "Mental_Health_Pure_Numerical.csv"
    OUTPUT_FILENAME: str = "Mental_Health_Balanced_20k.csv"

    TARGET_COLUMN: str = "target_label"
    TOTAL_SAMPLES: int = 20000
    NUM_CLASSES: int = 7
    SAMPLES_PER_CLASS: int = TOTAL_SAMPLES // NUM_CLASSES  # ~2857 per class
    RANDOM_STATE: int = 42
