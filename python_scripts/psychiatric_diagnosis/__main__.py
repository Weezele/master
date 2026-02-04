"""
Entry point for running as a module: python -m psychiatric_diagnosis

Usage:
    python -m psychiatric_diagnosis              # Full pipeline with AWOA (40 whales)
    python -m psychiatric_diagnosis --quick      # Quick test run
    python -m psychiatric_diagnosis --no-awoa    # Skip AWOA, use defaults
    python -m psychiatric_diagnosis --optimize-metric accuracy  # Optimize for accuracy

Examples:
    # Run with AWOA optimization (recommended for best accuracy)
    python -m psychiatric_diagnosis --population 40 --iterations 15 --optimize-metric accuracy

    # Run without AWOA (faster, uses default hyperparameters)
    python -m psychiatric_diagnosis --no-awoa
"""

import argparse
from .application import PsychiatricDiagnosisApp
from .config import AWOAConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Psychiatric Disease Diagnosis using Modified NN + AWOA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target Accuracy: 92-95%
Features:
  - GPU acceleration with mixed precision training
  - Adaptive Whale Optimization Algorithm (AWOA)
  - Class-weighted loss for imbalanced data
  - Residual neural network architecture
        """
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with reduced optimization (for testing)"
    )
    parser.add_argument(
        "--no-awoa",
        action="store_true",
        help="Skip AWOA optimization, use default hyperparameters"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=25,
        help="AWOA population size / number of whales (default: 25)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="AWOA max iterations (default: 10)"
    )
    parser.add_argument(
        "--optimize-metric",
        choices=["loss", "accuracy"],
        default="accuracy",
        help="Metric to optimize during AWOA: loss or accuracy (default: accuracy)"
    )

    args = parser.parse_args()

    # Configure AWOA
    awoa_config = AWOAConfig(
        population_size=args.population,
        max_iterations=args.iterations
    )
    fitness_metric = (
        "val_accuracy" if args.optimize_metric == "accuracy" else "val_loss"
    )

    # Create and run app
    app = PsychiatricDiagnosisApp(
        awoa_config=awoa_config,
        use_optimization=not args.no_awoa,
        fitness_metric=fitness_metric
    )

    if args.quick:
        results = app.run_quick()
    else:
        results = app.run()

    return results


if __name__ == "__main__":
    main()
