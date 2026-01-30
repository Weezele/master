"""Domain entities for psychiatric diagnosis system."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class DataSplit:
    """Represents train/validation/test data splits."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    @property
    def input_dim(self) -> int:
        """Get input feature dimension."""
        return self.X_train.shape[1]

    @property
    def num_samples(self) -> Dict[str, int]:
        """Get number of samples in each split."""
        return {
            "train": len(self.X_train),
            "val": len(self.X_val),
            "test": len(self.X_test)
        }


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for model performance."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_per_class: List[float]
    recall_per_class: List[float]
    f1_per_class: List[float]
    confusion_matrix: np.ndarray
    roc_auc: Optional[float] = None
    roc_auc_per_class: Optional[List[float]] = None
    classification_report: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "roc_auc": self.roc_auc,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class
        }


@dataclass
class TrainingResult:
    """Results from neural network training."""

    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    training_time: float
    hyperparameters: Dict[str, Any]

    @property
    def final_train_loss(self) -> float:
        return self.history["loss"][-1]

    @property
    def final_val_loss(self) -> float:
        return self.history["val_loss"][-1]


@dataclass
class WhalePosition:
    """Represents a whale's position in the search space (hyperparameters)."""

    position: np.ndarray  # Continuous values
    fitness: float = float('inf')
    decoded_params: Dict[str, Any] = field(default_factory=dict)

    def decode(self, bounds_config) -> Dict[str, Any]:
        """Decode continuous position to actual hyperparameters."""
        self.decoded_params = {
            "num_hidden_layers": int(round(self.position[0])),
            "neurons_per_layer": int(round(self.position[1])),
            "dropout_rate": float(self.position[2]),
            "learning_rate": float(self.position[3]),
            "batch_size": int(round(self.position[4])),
            "l2_regularization": float(self.position[5])
        }
        return self.decoded_params


@dataclass
class OptimizationResult:
    """Results from AWOA optimization."""

    best_position: WhalePosition
    best_hyperparameters: Dict[str, Any]
    best_fitness: float
    convergence_history: List[float]
    all_evaluations: List[Dict[str, Any]]
    total_iterations: int
    optimization_time: float
