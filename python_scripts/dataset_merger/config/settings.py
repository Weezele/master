"""Application configuration settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Application configuration constants."""

    DATASETS_DIR: Path = Path(__file__).parent.parent.parent.parent / "datasets"
    OUTPUT_DIR: Path = Path(__file__).parent.parent.parent.parent / "datasets"
    OUTPUT_FILENAME: str = "Mental_Health_Pure_Numerical.csv"

    FILE_PATTERN_SUFFIX_PRE: str = "_pre_features_tfidf_256.csv"
    FILE_PATTERN_SUFFIX_POST: str = "_post_features_tfidf_256.csv"

    COLUMNS_TO_DROP: tuple = ("post", "author", "date", "subreddit")
    TARGET_COLUMN: str = "target_label"
    RANDOM_STATE: int = 42
