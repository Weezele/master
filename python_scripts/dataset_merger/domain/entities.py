"""Domain entities."""

from dataclasses import dataclass
from pathlib import Path

from ..config import LABEL_NAMES


@dataclass
class DatasetFile:
    """Represents a dataset file with its metadata."""

    filepath: Path
    topic: str
    label: int
    file_type: str  # 'pre' or 'post'

    @property
    def label_name(self) -> str:
        """Get human-readable label name."""
        return LABEL_NAMES.get(self.label, "Unknown")
