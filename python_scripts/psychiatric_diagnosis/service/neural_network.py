"""
Modified Neural Network for Psychiatric Disease Diagnosis

This module implements a configurable neural network architecture
that can be optimized using AWOA (Adaptive Whale Optimization Algorithm).

Features:
- GPU acceleration with mixed precision training
- Class-weighted loss for handling imbalanced data
- Residual connections for better gradient flow
- Label smoothing for better generalization
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from ..config import Config, NNHyperparameters
from ..domain import DataSplit, TrainingResult


class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

        # Skip connection with projection if dimensions differ
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out = out + identity  # Residual connection
        out = self.activation(out)
        return out


class PsychiatricDiagnosisNN(nn.Module):
    """
    Modified Neural Network for multi-class psychiatric diagnosis.

    Architecture features:
    - Residual blocks for better gradient flow
    - Batch normalization for stable training
    - Dropout for regularization
    - GELU activation for smoother gradients
    - Attention-like feature weighting
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.3,
        activation: str = "gelu",
        use_residual: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_residual = use_residual

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # Feature attention (learns importance of features)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[0] // 4),
            nn.GELU(),
            nn.Linear(hidden_layers[0] // 4, hidden_layers[0]),
            nn.Sigmoid()
        )

        # Build hidden layers with residual connections
        self.hidden_blocks = nn.ModuleList()
        prev_dim = hidden_layers[0]

        for i, hidden_dim in enumerate(hidden_layers[1:]):
            if use_residual:
                self.hidden_blocks.append(
                    ResidualBlock(prev_dim, hidden_dim, dropout_rate)
                )
            else:
                self.hidden_blocks.append(nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout_rate)
                ))
            prev_dim = hidden_dim

        # Output head with extra capacity
        self.output_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.BatchNorm1d(prev_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_dim // 2, num_classes)
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU()
        }
        return activations.get(activation, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # Input projection
        x = self.input_proj(x)

        # Apply feature attention
        attention = self.feature_attention(x)
        x = x * attention

        # Hidden blocks
        for block in self.hidden_blocks:
            x = block(x)

        # Output
        return self.output_head(x)


class NeuralNetworkService:
    """
    Service for training and managing neural networks.

    Features:
    - GPU acceleration with CUDA
    - Mixed precision training (FP16) for faster GPU training
    - Class-weighted loss for imbalanced data
    - Label smoothing for better generalization
    """

    def __init__(self, config: Config):
        self.config = config

        # GPU Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()  # Mixed precision on GPU
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.model: Optional[PsychiatricDiagnosisNN] = None
        self.class_weights: Optional[torch.Tensor] = None

        # Print GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"  Mixed Precision Training: Enabled")
        else:
            print(f"  Using device: CPU (GPU not available)")
            print(f"  Warning: Training will be slower without GPU")

    def build_model(
        self,
        input_dim: int,
        hyperparameters: Dict[str, Any]
    ) -> PsychiatricDiagnosisNN:
        """Build neural network with given hyperparameters."""

        # Decode hyperparameters
        num_layers = hyperparameters.get("num_hidden_layers", 4)
        neurons = hyperparameters.get("neurons_per_layer", 512)
        dropout = hyperparameters.get("dropout_rate", 0.25)

        # Create hidden layer configuration with gradual decrease
        hidden_layers = []
        current_neurons = neurons
        for i in range(num_layers):
            hidden_layers.append(current_neurons)
            # Gradual decrease for better feature extraction
            current_neurons = max(64, int(current_neurons * 0.7))

        model = PsychiatricDiagnosisNN(
            input_dim=input_dim,
            num_classes=self.config.NUM_CLASSES,
            hidden_layers=hidden_layers,
            dropout_rate=dropout,
            activation="gelu",
            use_residual=True
        )

        self.model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {trainable_params:,} trainable / {total_params:,} total")

        return self.model

    def _compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        class_counts = np.bincount(y_train.astype(int))
        total_samples = len(y_train)

        # Inverse frequency weighting with smoothing
        weights = total_samples / (len(class_counts) * class_counts + 1e-6)

        # Normalize weights to prevent extreme values
        weights = weights / weights.sum() * len(class_counts)

        # Apply sqrt to reduce extreme weights
        weights = np.sqrt(weights)

        return torch.FloatTensor(weights).to(self.device)

    def _create_data_loaders(
        self,
        data_split: DataSplit,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders with GPU optimization."""

        # Use pin_memory for faster GPU transfer
        pin_memory = torch.cuda.is_available()

        # Training data
        train_dataset = TensorDataset(
            torch.FloatTensor(data_split.X_train),
            torch.LongTensor(data_split.y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory
        )

        # Validation data
        val_dataset = TensorDataset(
            torch.FloatTensor(data_split.X_val),
            torch.LongTensor(data_split.y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )

        return train_loader, val_loader

    def train(
        self,
        data_split: DataSplit,
        hyperparameters: Dict[str, Any],
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Train the neural network with GPU acceleration.

        Features:
        - Mixed precision training (FP16) for faster GPU training
        - Class-weighted loss for handling imbalanced data
        - Label smoothing for better generalization
        - Cosine annealing learning rate schedule

        Args:
            data_split: Train/val/test data splits
            hyperparameters: Model hyperparameters
            epochs: Number of training epochs (overrides hyperparameters)
            verbose: Print training progress

        Returns:
            TrainingResult with training history and metrics
        """
        start_time = time.time()

        # Extract hyperparameters
        lr = hyperparameters.get("learning_rate", 0.001)
        batch_size = int(hyperparameters.get("batch_size", 64))
        l2_reg = hyperparameters.get("l2_regularization", 0.0001)
        num_epochs = epochs or hyperparameters.get("epochs", 100)
        patience = hyperparameters.get("early_stopping_patience", 20)

        # Build model if not exists
        if self.model is None:
            self.build_model(data_split.input_dim, hyperparameters)

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            data_split, batch_size
        )

        # Compute class weights for imbalanced data
        class_weights = self._compute_class_weights(data_split.y_train)
        if verbose:
            print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

        # Loss with class weights and label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1  # Helps with generalization
        )

        # AdamW optimizer (better weight decay handling)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=l2_reg,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warm restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Training history
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

                # Mixed precision training
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                    self.scaler.scale(loss).backward()
                    # Gradient clipping for stability
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Update learning rate
            scheduler.step()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with autocast(device_type='cuda'):
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

            # Save history
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # Check for best model (use accuracy as primary metric)
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch+1}/{num_epochs} - "
                      f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
                      f"lr: {current_lr:.6f}")

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        training_time = time.time() - start_time

        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=history["val_accuracy"][best_epoch],
            training_time=training_time,
            hyperparameters=hyperparameters
        )

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Tuple of (predicted labels, prediction probabilities)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        return (
            predictions.cpu().numpy(),
            probabilities.cpu().numpy()
        )

    def evaluate_fitness(
        self,
        data_split: DataSplit,
        hyperparameters: Dict[str, Any],
        quick_epochs: int = 20,
        fitness_metric: str = "val_loss"
    ) -> float:
        """
        Quick evaluation for AWOA fitness function.

        Uses fewer epochs for faster optimization.

        Args:
            data_split: Data splits
            hyperparameters: Hyperparameters to evaluate
            quick_epochs: Number of epochs for quick evaluation
            fitness_metric: Metric to optimize ("val_loss" or "val_accuracy")

        Returns:
            Fitness value (lower is better)
        """
        metric = (fitness_metric or "val_loss").lower().strip()
        if metric in ("loss", "val_loss"):
            metric = "val_loss"
        elif metric in ("accuracy", "val_accuracy"):
            metric = "val_accuracy"
        else:
            raise ValueError(
                "fitness_metric must be 'val_loss' or 'val_accuracy'"
            )

        # Reset model for fresh evaluation
        self.model = None
        self.build_model(data_split.input_dim, hyperparameters)

        # Quick training
        fitness_params = hyperparameters.copy()
        fitness_params["early_stopping_patience"] = 15
        result = self.train(
            data_split,
            fitness_params,
            epochs=quick_epochs,
            verbose=False
        )

        if metric == "val_accuracy":
            val_accuracies = result.history.get("val_accuracy", [])
            best_accuracy = max(val_accuracies) if val_accuracies else 0.0
            return 1.0 - best_accuracy

        return result.best_val_loss

    def get_model(self) -> PsychiatricDiagnosisNN:
        """Get the current model."""
        return self.model
