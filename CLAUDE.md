# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for multi-class psychiatric disease classification from social media text. Combines a modified PyTorch neural network (residual blocks + attention) with Adaptive Whale Optimization Algorithm (AWOA) for hyperparameter tuning. Classifies into 7 categories: Normal, Depression, Autism, Anxiety, PTSD, Addiction, Alcoholism.

## Commands

All commands run from `python_scripts/`:

```bash
# Full pipeline with AWOA optimization
python train_diagnosis_model.py

# Quick test run (reduced optimization)
python train_diagnosis_model.py --quick

# Skip AWOA, use default hyperparameters
python train_diagnosis_model.py --no-awoa

# Custom AWOA settings
python train_diagnosis_model.py --population 40 --iterations 30 --optimize-metric accuracy

# Alternative module entry point
python -m psychiatric_diagnosis --population 40 --iterations 15 --optimize-metric accuracy

# Utility tools (run from python_scripts/)
python -m dataset_merger
python -m dataset_balancer
```

### Dependencies

```bash
pip install pandas numpy scikit-learn torch
```

## Architecture

The main package is `python_scripts/psychiatric_diagnosis/` and follows clean architecture with layered separation:

- **`application/app.py`** — Pipeline orchestrator. Runs the 5-step process: load data → AWOA optimization (optional) → train final model → evaluate → save artifacts.
- **`service/neural_network.py`** — `PsychiatricDiagnosisNN` model definition (residual blocks, batch norm, GELU, attention weighting, dropout) and training loop with early stopping, mixed precision, and class-weighted loss.
- **`service/awoa_optimizer.py`** — Adaptive Whale Optimization Algorithm. Population-based search over 6 hyperparameters (layers, neurons, dropout, learning rate, batch size, L2 weight).
- **`service/evaluation_service.py`** — Computes accuracy, F1, precision, recall, ROC-AUC, confusion matrix.
- **`repository/data_repository.py`** — Loads CSV, stratified train/val/test split (80/10/10), StandardScaler normalization.
- **`repository/model_repository.py`** — Saves/loads models (pickle), metrics (JSON), scaler, training history.
- **`domain/entities.py`** — Data classes: `DataSplit`, `TrainingResult`, `EvaluationMetrics`, `WhalePosition`, `OptimizationResult`.
- **`config/settings.py`** — Paths, dataset filename, class names, split ratios.
- **`config/hyperparameters.py`** — Default NN params and AWOA search bounds.

Entry points: `train_diagnosis_model.py` (CLI script) and `psychiatric_diagnosis/__main__.py` (module).

## Key Details

- Dataset is a ~429MB CSV of TF-IDF features (346 columns) at `ready_dataset_to_train/`. CSV files are gitignored.
- Model outputs saved to `python_scripts/psychiatric_diagnosis/models/` with timestamps.
- `.pkl` and `.pt`/`.pth` files are gitignored; only code is tracked.
- Baseline accuracy without AWOA: ~85.6%. Target: 92-95%.
- Supports CPU and GPU (CUDA) training via PyTorch.
