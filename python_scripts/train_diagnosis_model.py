"""
Psychiatric Disease Diagnosis - Training Script

Modified Neural Network Model for Psychiatric Disease Diagnosis
Using Natural Language Processing and Adaptive Whale Optimization Algorithm (AWOA)

Usage:
    python train_diagnosis_model.py              # Full pipeline with AWOA
    python train_diagnosis_model.py --quick      # Quick test run
    python train_diagnosis_model.py --no-awoa    # Skip AWOA, use defaults
    python train_diagnosis_model.py --optimize-metric accuracy  # Optimize for accuracy

Or run as module:
    python -m psychiatric_diagnosis
"""

import sys
import argparse

from psychiatric_diagnosis.application import PsychiatricDiagnosisApp
from psychiatric_diagnosis.config import AWOAConfig


def main():
    """Main entry point."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   MODIFIED NEURAL NETWORK FOR PSYCHIATRIC DISEASE DIAGNOSIS       #")
    print("#   Using Adaptive Whale Optimization Algorithm (AWOA)              #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    parser = argparse.ArgumentParser(
        description="Train psychiatric diagnosis model with AWOA-optimized NN"
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
        help="AWOA population size (default: 25)"
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

    # Create application
    app = PsychiatricDiagnosisApp(
        awoa_config=awoa_config,
        use_optimization=not args.no_awoa,
        fitness_metric=fitness_metric
    )

    try:
        if args.quick:
            print("\n  [MODE] Quick test run")
            results = app.run_quick()
        else:
            if args.no_awoa:
                print("\n  [MODE] Training without AWOA (default hyperparameters)")
            else:
                print(f"\n  [MODE] Full AWOA optimization")
                print(f"         Population: {args.population}")
                print(f"         Max iterations: {args.iterations}")
                print(f"         Optimize metric: {args.optimize_metric}")
            results = app.run()

        print("\n  [SUCCESS] Training completed successfully!")
        return 0

    except FileNotFoundError as e:
        print(f"\n  [ERROR] Dataset not found: {e}")
        print("\n  Please ensure the dataset file exists at:")
        print("    ready_dataset_to_train/Mental_Health_Pure_Numerical.csv")
        return 1

    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
