"""
Evaluation Service for Model Performance Assessment

Computes comprehensive metrics including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC (multi-class)
- Classification Report
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from ..config import Config
from ..domain import EvaluationMetrics


class EvaluationService:
    """Service for evaluating model performance."""

    def __init__(self, config: Config):
        self.config = config

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive model evaluation.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for ROC-AUC)

        Returns:
            EvaluationMetrics with all computed metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=list(self.config.CLASS_NAMES),
            zero_division=0
        )

        # ROC-AUC (if probabilities provided)
        roc_auc = None
        roc_auc_per_class = None

        if y_prob is not None:
            try:
                # Binarize labels for multi-class ROC-AUC
                y_true_bin = label_binarize(
                    y_true,
                    classes=list(range(self.config.NUM_CLASSES))
                )

                # One-vs-Rest ROC-AUC
                roc_auc = roc_auc_score(
                    y_true_bin, y_prob,
                    average='macro',
                    multi_class='ovr'
                )

                # Per-class ROC-AUC
                roc_auc_per_class = []
                for i in range(self.config.NUM_CLASSES):
                    if len(np.unique(y_true_bin[:, i])) > 1:
                        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                        roc_auc_per_class.append(auc)
                    else:
                        roc_auc_per_class.append(0.0)

            except Exception as e:
                print(f"  Warning: Could not compute ROC-AUC: {e}")

        return EvaluationMetrics(
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            f1_per_class=f1_per_class,
            confusion_matrix=cm,
            roc_auc=roc_auc,
            roc_auc_per_class=roc_auc_per_class,
            classification_report=report
        )

    def print_metrics(self, metrics: EvaluationMetrics) -> None:
        """Print formatted evaluation metrics."""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)

        print(f"\n  Overall Metrics:")
        print(f"    Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        print(f"    Precision: {metrics.precision_macro:.4f}")
        print(f"    Recall:    {metrics.recall_macro:.4f}")
        print(f"    F1-Score:  {metrics.f1_macro:.4f}")

        if metrics.roc_auc is not None:
            print(f"    ROC-AUC:   {metrics.roc_auc:.4f}")

        print(f"\n  Per-Class Metrics:")
        print(f"    {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}", end="")
        if metrics.roc_auc_per_class:
            print(f" {'ROC-AUC':>10}")
        else:
            print()

        for i, class_name in enumerate(self.config.CLASS_NAMES):
            print(f"    {class_name:<15} {metrics.precision_per_class[i]:>10.4f} "
                  f"{metrics.recall_per_class[i]:>10.4f} "
                  f"{metrics.f1_per_class[i]:>10.4f}", end="")
            if metrics.roc_auc_per_class:
                print(f" {metrics.roc_auc_per_class[i]:>10.4f}")
            else:
                print()

        print(f"\n  Confusion Matrix:")
        self._print_confusion_matrix(metrics.confusion_matrix)

    def _print_confusion_matrix(self, cm: np.ndarray) -> None:
        """Print formatted confusion matrix."""
        # Header
        print("    " + " " * 12, end="")
        print("Predicted")
        print("    " + " " * 12, end="")
        for i, name in enumerate(self.config.CLASS_NAMES):
            print(f"{name[:8]:>9}", end="")
        print()

        # Matrix rows
        for i, name in enumerate(self.config.CLASS_NAMES):
            if i == len(self.config.CLASS_NAMES) // 2:
                print(f"    Actual {name[:8]:>8}", end="")
            else:
                print(f"           {name[:8]:>8}", end="")
            for j in range(len(self.config.CLASS_NAMES)):
                print(f"{cm[i, j]:>9}", end="")
            print()

    def get_summary_dict(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Get metrics as a dictionary for saving."""
        return {
            "accuracy": metrics.accuracy,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
            "f1_macro": metrics.f1_macro,
            "roc_auc": metrics.roc_auc,
            "precision_per_class": {
                name: val for name, val in
                zip(self.config.CLASS_NAMES, metrics.precision_per_class)
            },
            "recall_per_class": {
                name: val for name, val in
                zip(self.config.CLASS_NAMES, metrics.recall_per_class)
            },
            "f1_per_class": {
                name: val for name, val in
                zip(self.config.CLASS_NAMES, metrics.f1_per_class)
            },
            "confusion_matrix": metrics.confusion_matrix.tolist(),
            "classification_report": metrics.classification_report
        }
