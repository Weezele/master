"""Model repository for saving and loading models."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pickle

import numpy as np


class ModelRepository:
    """Handles model persistence operations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str = "best_model") -> Path:
        """Save the trained model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"{model_name}_{timestamp}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"  Model saved to: {model_path}")
        return model_path

    def save_scaler(self, scaler, name: str = "scaler") -> Path:
        """Save the feature scaler."""
        scaler_path = self.output_dir / f"{name}.pkl"

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return scaler_path

    def save_metrics(self, metrics: Dict[str, Any], name: str = "metrics") -> Path:
        """Save evaluation metrics to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = self.output_dir / f"{name}_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = self._make_serializable(metrics)

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"  Metrics saved to: {metrics_path}")
        return metrics_path

    def save_optimization_results(self, results: Dict[str, Any]) -> Path:
        """Save AWOA optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"optimization_results_{timestamp}.json"

        serializable_results = self._make_serializable(results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"  Optimization results saved to: {results_path}")
        return results_path

    def save_training_history(self, history: Dict[str, list]) -> Path:
        """Save training history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.output_dir / f"training_history_{timestamp}.json"

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history_path

    def load_model(self, model_path: Path):
        """Load a saved model."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
