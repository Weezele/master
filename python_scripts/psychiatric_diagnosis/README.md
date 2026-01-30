# Modified Neural Network for Psychiatric Disease Diagnosis
## Using Natural Language Processing and Adaptive Whale Optimization Algorithm (AWOA)

This project implements a modified neural network for multi-class psychiatric disease diagnosis. It trains on TF-IDF features derived from social media posts and uses the Adaptive Whale Optimization Algorithm (AWOA) to tune hyperparameters.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Outputs](#outputs)
6. [Results](#results)
7. [Code Walkthrough](#code-walkthrough)
8. [Algorithm Details](#algorithm-details)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)
11. [License](#license)
12. [Citation](#citation)
13. [Contact](#contact)

---

## Overview

### Problem Statement

Mental health disorders affect millions worldwide. Early detection through analysis of social media text can help identify individuals at risk. This project uses NLP features extracted from Reddit posts to classify mental health conditions.

### Classification Categories

| Label | Category | Description |
|-------|----------|-------------|
| 0 | Normal | Control group (jokes, fitness, relationships, teaching) |
| 1 | Depression | Depression-related posts |
| 2 | Autism | Autism spectrum disorder posts |
| 3 | Anxiety | Anxiety disorder posts |
| 4 | PTSD | Post-traumatic stress disorder posts |
| 5 | Addiction | Addiction-related posts |
| 6 | Alcoholism | Alcoholism-related posts |

### Key Features

- Clean Architecture with clear separation of concerns
- AWOA hyperparameter optimization with selectable fitness metric (loss or accuracy)
- PyTorch neural network with configurable depth and regularization
- Comprehensive evaluation: accuracy, F1, precision, recall, ROC-AUC, confusion matrix

---

## Architecture

The project follows Clean Architecture principles:

```
python_scripts/
|
|-- train_diagnosis_model.py         # CLI training script
|-- psychiatric_diagnosis/
|   |-- __main__.py                  # python -m psychiatric_diagnosis entry point
|   |-- __init__.py                  # Package metadata
|   |-- application/
|   |   |-- app.py                   # Pipeline orchestrator
|   |-- config/
|   |   |-- settings.py              # Paths and global settings
|   |   |-- hyperparameters.py       # NN defaults + AWOA bounds
|   |-- domain/
|   |   |-- entities.py              # Core dataclasses
|   |-- repository/
|   |   |-- data_repository.py       # Data loading and prep
|   |   |-- model_repository.py      # Model/metrics persistence
|   |-- service/
|   |   |-- neural_network.py        # Model + training loop
|   |   |-- awoa_optimizer.py        # AWOA implementation
|   |   |-- evaluation_service.py    # Metrics and reporting
|   |-- models/                      # Saved outputs
|   |-- README.md                    # This documentation
```

### Data Flow

```
[CSV Dataset] --> [DataRepository] --> [DataSplit]
                                        |
                                        v
                    [AWOAOptimizer] <--> [NeuralNetworkService]
                                        |
                                        v
                                [EvaluationService]
                                        |
                                        v
                                [ModelRepository] --> [Saved Models]
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Setup

```bash
# Navigate to project directory
cd python_scripts

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn torch
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0 | Data manipulation |
| numpy | >= 1.24 | Numerical operations |
| scikit-learn | >= 1.3 | Preprocessing, metrics |
| torch | >= 2.0 | Neural network framework |

---

## Usage

### Basic Usage

```bash
# Full pipeline with AWOA optimization (default fitness: validation loss)
python train_diagnosis_model.py

# Without AWOA (default hyperparameters)
python train_diagnosis_model.py --no-awoa

# Optimize AWOA for validation accuracy
python train_diagnosis_model.py --optimize-metric accuracy

# Quick test run
python train_diagnosis_model.py --quick
```

### Advanced Options

```bash
# Custom AWOA settings
python train_diagnosis_model.py --population 40 --iterations 60

# Higher-accuracy search (slower)
python train_diagnosis_model.py --population 40 --iterations 60 --optimize-metric accuracy

# Run as module
python -m psychiatric_diagnosis --optimize-metric accuracy
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--quick` | False | Quick test with reduced optimization |
| `--no-awoa` | False | Skip AWOA, use default hyperparameters |
| `--population` | 20 | AWOA population size |
| `--iterations` | 30 | AWOA max iterations |
| `--optimize-metric` | loss | Fitness metric for AWOA (loss or accuracy) |

---

## Outputs

All run artifacts are saved under `python_scripts/psychiatric_diagnosis/models/`:

- `psychiatric_diagnosis_AWOA_<timestamp>.pkl` (model with AWOA)
- `psychiatric_diagnosis_without_AWOA_<timestamp>.pkl` (model without AWOA)
- `evaluation_metrics_AWOA_<timestamp>.json`
- `evaluation_metrics_without_AWOA_<timestamp>.json`
- `optimization_results_<timestamp>.json` (AWOA only)
- `training_history_<timestamp>.json`
- `scaler.pkl`

---

## Results

### Model Performance Comparison

| Metric | Without AWOA (2025-12-24) | With AWOA (run required) |
|--------|----------------------------|---------------------------|
| Accuracy | 85.63% | TBD (goal: 98%) |
| Precision | 0.7775 | TBD |
| Recall | 0.7002 | TBD |
| F1-Score | 0.7327 | TBD |
| ROC-AUC | 0.9613 | TBD |

To target higher accuracy, run AWOA with `--optimize-metric accuracy` and increase `--population` and `--iterations`. Actual results depend on data split and hardware; fill the "With AWOA" column from the saved `evaluation_metrics_AWOA_<timestamp>.json`.

---

## Code Walkthrough

### Entry Points

- `python_scripts/train_diagnosis_model.py`
  - CLI wrapper that parses arguments, configures AWOA, and runs the pipeline.
  - Supports `--no-awoa`, `--quick`, and `--optimize-metric`.
  - Calls `PsychiatricDiagnosisApp.run()` or `run_quick()`.

- `python_scripts/psychiatric_diagnosis/__main__.py`
  - Module entry point for `python -m psychiatric_diagnosis`.
  - Mirrors the CLI arguments and launches the same pipeline.

### Application Layer

- `python_scripts/psychiatric_diagnosis/application/app.py`
  - Orchestrates the 5-step pipeline:
    1. Load and split data
    2. Optimize hyperparameters with AWOA (optional)
    3. Train final model
    4. Evaluate on test set
    5. Save outputs
  - `_run_optimization()` builds the fitness function and calls `AWOAOptimizer`.
  - Uses `fitness_metric` to choose loss-based or accuracy-based optimization.

### Config Layer

- `python_scripts/psychiatric_diagnosis/config/settings.py`
  - Paths (dataset location, output directory).
  - Train/val/test split sizes and random seed.
  - Class names and number of classes.

- `python_scripts/psychiatric_diagnosis/config/hyperparameters.py`
  - Default neural network hyperparameters.
  - AWOA search bounds (layers, neurons, dropout, learning rate, batch size, L2).

- `python_scripts/psychiatric_diagnosis/config/__init__.py`
  - Re-exports `Config`, `NNHyperparameters`, and `AWOAConfig`.

### Domain Layer

- `python_scripts/psychiatric_diagnosis/domain/entities.py`
  - `DataSplit`: train/val/test arrays and helper properties.
  - `TrainingResult`: training history and best validation metrics.
  - `EvaluationMetrics`: computed evaluation scores and reports.
  - `WhalePosition`: AWOA position with decode method for hyperparameters.
  - `OptimizationResult`: summary of the AWOA search.

- `python_scripts/psychiatric_diagnosis/domain/__init__.py`
  - Re-exports domain entities.

### Repository Layer

- `python_scripts/psychiatric_diagnosis/repository/data_repository.py`
  - Loads the CSV dataset.
  - Splits into train/val/test with stratification.
  - Scales features using `StandardScaler`.

- `python_scripts/psychiatric_diagnosis/repository/model_repository.py`
  - Saves models, scalers, metrics, optimization results, and history files.
  - Handles JSON serialization of numpy types.

- `python_scripts/psychiatric_diagnosis/repository/__init__.py`
  - Re-exports repository classes.

### Service Layer

- `python_scripts/psychiatric_diagnosis/service/neural_network.py`
  - Defines `PsychiatricDiagnosisNN` (linear layers, batch norm, dropout).
  - Implements training with Adam, ReduceLROnPlateau, and early stopping.
  - Provides `evaluate_fitness()` for AWOA using loss or accuracy.

- `python_scripts/psychiatric_diagnosis/service/awoa_optimizer.py`
  - Implements the AWOA loop: initialization, exploration, exploitation, spiral.
  - Tracks convergence history and best hyperparameters.

- `python_scripts/psychiatric_diagnosis/service/evaluation_service.py`
  - Computes accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
  - Prints a readable summary and returns serializable metrics.

- `python_scripts/psychiatric_diagnosis/service/__init__.py`
  - Re-exports service classes.

### Package Metadata

- `python_scripts/psychiatric_diagnosis/__init__.py`
  - Package metadata (`__version__`, `__author__`).

---

## Algorithm Details

### Neural Network Architecture

```
Input Layer: 346 neurons (TF-IDF features)
     |
     v
Hidden Layer 1: N neurons + BatchNorm + ReLU + Dropout
     |
     v
Hidden Layer 2: N/2 neurons + BatchNorm + ReLU + Dropout
     |
     v
Hidden Layer 3: N/4 neurons + BatchNorm + ReLU + Dropout
     |
     v
Output Layer: 7 neurons (Softmax for classification)
```

### Training Configuration

| Parameter | Default | AWOA Range |
|-----------|---------|------------|
| Hidden Layers | 3 | 2-5 |
| Neurons | 256 | 128-1024 |
| Dropout | 0.3 | 0.1-0.4 |
| Learning Rate | 0.001 | 0.0001-0.005 |
| Batch Size | 32 | 32-128 |
| L2 Regularization | 0.0001 | 0.00001-0.0005 |

### AWOA Overview

AWOA mimics humpback whale hunting behavior:

```
AWOA Phases:
1. EXPLORATION (|A| >= 1): Search for prey (global search)
2. EXPLOITATION (|A| < 1): Attack prey (local search)
   a. Encircling prey (p < 0.5)
   b. Spiral bubble-net (p >= 0.5)
```

#### Mathematical Formulation

```
Encircling Prey:
    D = |C * X_best - X|
    X_new = X_best - A * D

Spiral Update:
    D' = |X_best - X|
    X_new = D' * e^(b*l) * cos(2*pi*l) + X_best

Where:
    A = 2 * a * r - a  (decreases from 2 to 0)
    C = 2 * r          (random coefficient)
    b = 1              (spiral constant)
    l = random(-1, 1)  (spiral parameter)
```

#### Adaptive Enhancements

```python
def _get_adaptive_weight(self, iteration: int) -> float:
    w = w_max - (w_max - w_min) * (iteration / max_iterations)
    return w

def _get_adaptive_a(self, iteration: int) -> float:
    a = a_initial - (a_initial - a_final) * (iteration / max_iterations)
    return a
```

#### Fitness Metric

AWOA fitness can be optimized by:

- `val_loss` (default): lower validation loss is better.
- `val_accuracy`: higher validation accuracy is better (fitness = 1 - accuracy).

Use `--optimize-metric accuracy` to switch to accuracy-based fitness.

---

## Configuration

### Customizing Settings

Edit `python_scripts/psychiatric_diagnosis/config/settings.py`:

```python
@dataclass(frozen=True)
class Config:
    DATA_DIR: Path = Path("your/data/path")
    OUTPUT_DIR: Path = Path("your/output/path")
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1
```

### Customizing Hyperparameters

Edit `python_scripts/psychiatric_diagnosis/config/hyperparameters.py`:

```python
@dataclass
class AWOAConfig:
    population_size: int = 30
    max_iterations: int = 50
```

---

## API Reference

### Quick Start Code

```python
from psychiatric_diagnosis.application import PsychiatricDiagnosisApp
from psychiatric_diagnosis.config import Config, AWOAConfig

config = Config()
awoa_config = AWOAConfig(population_size=30, max_iterations=50)

app = PsychiatricDiagnosisApp(
    config=config,
    awoa_config=awoa_config,
    use_optimization=True,
    fitness_metric="val_accuracy"
)

results = app.run()
print(f"Accuracy: {results['evaluation']['accuracy']}")
```

### Loading Saved Model

```python
from psychiatric_diagnosis.repository import ModelRepository
from pathlib import Path
import pickle

model_path = Path("models/psychiatric_diagnosis_AWOA_20241224_000000.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

---

## License

This project is part of a Master's thesis research on psychiatric disease diagnosis using machine learning.

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{psychiatric_diagnosis_awoa,
  title={Modified Neural Network Model for Psychiatric Disease Diagnosis
         Using Natural Language Processing and Adaptive Whale Optimization Algorithm},
  author={[Your Name]},
  year={2024},
  school={[Your University]}
}
```

## Contact

For questions or issues, please contact the author.
