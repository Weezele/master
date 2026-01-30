"""Domain module - Core business entities."""

from .entities import (
    DataSplit,
    TrainingResult,
    EvaluationMetrics,
    OptimizationResult,
    WhalePosition
)

__all__ = [
    "DataSplit",
    "TrainingResult",
    "EvaluationMetrics",
    "OptimizationResult",
    "WhalePosition"
]
