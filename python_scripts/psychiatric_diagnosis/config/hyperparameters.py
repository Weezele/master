"""Hyperparameter configurations for Neural Network and AWOA."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class NNHyperparameters:
    """
    Neural Network hyperparameters - can be optimized by AWOA.

    The hyperparameters are tuned for achieving 92-95% accuracy on
    psychiatric disease classification with imbalanced classes.
    """

    # Architecture - Enhanced for better accuracy
    hidden_layers: List[int] = field(default_factory=lambda: [512, 358, 250, 175])
    dropout_rate: float = 0.25
    activation: str = "gelu"

    # Training - Optimized for convergence
    learning_rate: float = 0.0008
    batch_size: int = 64
    epochs: int = 200  # More epochs for better convergence
    early_stopping_patience: int = 25

    # Regularization
    l2_regularization: float = 0.00005

    # Optimization bounds for AWOA (min, max) for each parameter
    @staticmethod
    def get_bounds() -> dict:
        """Get optimization bounds for AWOA."""
        return {
            "num_hidden_layers": (3, 6),           # 3-6 hidden layers for depth
            "neurons_per_layer": (256, 768),       # Larger networks for capacity
            "dropout_rate": (0.15, 0.35),          # Moderate dropout
            "learning_rate": (0.0003, 0.002),      # Learning rate range
            "batch_size": (32, 128),               # Batch size range
            "l2_regularization": (0.00001, 0.0003) # L2 reg range
        }


@dataclass
class AWOAConfig:
    """
    Adaptive Whale Optimization Algorithm configuration.

    Configured for 40 whales with enhanced exploration/exploitation balance.
    The algorithm mimics humpback whale hunting behavior:
    - Encircling prey (exploitation)
    - Bubble-net attacking (spiral exploitation)
    - Random search (exploration)
    """

    # Population settings - 40 whales as requested
    population_size: int = 40       # 40 whales for thorough exploration
    max_iterations: int = 15        # Iterations per whale evaluation

    # AWOA specific parameters
    a_initial: float = 2.0      # Initial value of 'a' parameter (exploration)
    a_final: float = 0.0        # Final value of 'a' parameter (exploitation)
    b_constant: float = 1.0     # Spiral constant for bubble-net

    # Adaptive parameters for better convergence
    adaptive_weight: bool = True
    weight_min: float = 0.4     # Minimum inertia weight
    weight_max: float = 0.9     # Maximum inertia weight

    # Convergence criteria
    convergence_threshold: float = 0.001  # Stop if improvement < this
    no_improvement_limit: int = 8         # Early stop after N iterations without improvement

    # Search space dimensions (number of hyperparameters to optimize)
    dimensions: int = 6

    # Bounds for each dimension [min, max]
    # Order: num_layers, neurons, dropout, lr, batch_size, l2_reg
    lower_bounds: Tuple[float, ...] = (3, 256, 0.15, 0.0003, 32, 0.00001)
    upper_bounds: Tuple[float, ...] = (6, 768, 0.35, 0.002, 128, 0.0003)
