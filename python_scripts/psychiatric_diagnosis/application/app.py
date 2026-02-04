"""
Main Application Orchestrator

Coordinates the entire pipeline:
1. Data loading and preparation
2. AWOA hyperparameter optimization
3. Neural network training with optimal parameters
4. Evaluation and reporting
5. Model and results saving
"""

import time
from typing import Optional, Dict, Any

from ..config import Config, AWOAConfig, NNHyperparameters
from ..repository import DataRepository, ModelRepository
from ..service import NeuralNetworkService, AWOAOptimizer, EvaluationService
from ..domain import DataSplit, OptimizationResult


class PsychiatricDiagnosisApp:
    """
    Main application for psychiatric disease diagnosis.

    Implements the thesis pipeline:
    "Modified Neural Network Model for Psychiatric Disease Diagnosis
    Using Natural Language Processing and Adaptive Whale Optimization Algorithm"
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        awoa_config: Optional[AWOAConfig] = None,
        use_optimization: bool = True,
        fitness_metric: str = "val_accuracy"  # Default to accuracy for better results
    ):
        self.config = config or Config()
        self.awoa_config = awoa_config or AWOAConfig()
        self.use_optimization = use_optimization
        self.fitness_metric = self._normalize_fitness_metric(fitness_metric)

        # Initialize components
        self.data_repo = DataRepository(self.config)
        self.model_repo = ModelRepository(self.config.OUTPUT_DIR)
        self.nn_service = NeuralNetworkService(self.config)
        self.eval_service = EvaluationService(self.config)
        self.optimizer = AWOAOptimizer(self.awoa_config)

        # State
        self.data_split: Optional[DataSplit] = None
        self.optimization_result: Optional[OptimizationResult] = None

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.

        Returns:
            Dictionary with all results and metrics
        """
        total_start_time = time.time()

        print("\n" + "=" * 70)
        print("  MODIFIED NEURAL NETWORK FOR PSYCHIATRIC DISEASE DIAGNOSIS")
        print("  Using Adaptive Whale Optimization Algorithm (AWOA)")
        print("=" * 70)

        results = {}

        # Step 1: Load and prepare data
        print("\n" + "-" * 60)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("-" * 60)

        df = self.data_repo.load_dataset()
        self.data_split = self.data_repo.prepare_data(df)
        results["data_info"] = self.data_split.num_samples

        # Step 2: Hyperparameter optimization with AWOA (optional)
        if self.use_optimization:
            print("\n" + "-" * 60)
            print("STEP 2: HYPERPARAMETER OPTIMIZATION (AWOA)")
            print("-" * 60)
            metric_label = (
                "validation loss (lower is better)"
                if self.fitness_metric == "val_loss"
                else "validation accuracy (higher is better)"
            )
            print(f"  Optimization metric: {metric_label}")

            self.optimization_result = self._run_optimization()
            best_params = self.optimization_result.best_hyperparameters
            results["optimization"] = {
                "best_hyperparameters": best_params,
                "best_fitness": self.optimization_result.best_fitness,
                "convergence_history": self.optimization_result.convergence_history,
                "optimization_time": self.optimization_result.optimization_time
            }
        else:
            print("\n" + "-" * 60)
            print("STEP 2: USING DEFAULT HYPERPARAMETERS (AWOA skipped)")
            print("-" * 60)

            default_hp = NNHyperparameters()
            best_params = {
                "num_hidden_layers": len(default_hp.hidden_layers),
                "neurons_per_layer": default_hp.hidden_layers[0],
                "dropout_rate": default_hp.dropout_rate,
                "learning_rate": default_hp.learning_rate,
                "batch_size": default_hp.batch_size,
                "l2_regularization": default_hp.l2_regularization
            }
            print("  Using default parameters:")
            for k, v in best_params.items():
                print(f"    {k}: {v}")

        # Step 3: Train final model with optimal hyperparameters
        print("\n" + "-" * 60)
        print("STEP 3: TRAINING FINAL MODEL")
        print("-" * 60)

        # Reset model for final training
        self.nn_service.model = None
        self.nn_service.build_model(self.data_split.input_dim, best_params)

        # Add full training parameters for best accuracy
        final_params = best_params.copy()
        final_params["epochs"] = 300  # Extended epochs for optimal convergence
        final_params["early_stopping_patience"] = 30

        print(f"\n  Training with optimized hyperparameters...")
        training_result = self.nn_service.train(
            self.data_split,
            final_params,
            verbose=True
        )

        results["training"] = {
            "best_epoch": training_result.best_epoch,
            "best_val_loss": training_result.best_val_loss,
            "best_val_accuracy": training_result.best_val_accuracy,
            "training_time": training_result.training_time,
            "final_train_loss": training_result.final_train_loss,
            "final_val_loss": training_result.final_val_loss
        }

        print(f"\n  Training completed in {training_result.training_time:.2f}s")
        print(f"  Best epoch: {training_result.best_epoch + 1}")
        print(f"  Best validation loss: {training_result.best_val_loss:.4f}")
        print(f"  Best validation accuracy: {training_result.best_val_accuracy:.4f}")

        # Step 4: Evaluate on test set
        print("\n" + "-" * 60)
        print("STEP 4: FINAL EVALUATION ON TEST SET")
        print("-" * 60)

        y_pred, y_prob = self.nn_service.predict(self.data_split.X_test)
        metrics = self.eval_service.evaluate(
            self.data_split.y_test, y_pred, y_prob
        )

        self.eval_service.print_metrics(metrics)
        results["evaluation"] = self.eval_service.get_summary_dict(metrics)

        # Step 5: Save everything
        print("\n" + "-" * 60)
        print("STEP 5: SAVING RESULTS")
        print("-" * 60)

        # Save model with appropriate name
        model_name = "psychiatric_diagnosis_AWOA" if self.use_optimization else "psychiatric_diagnosis_without_AWOA"
        model_path = self.model_repo.save_model(
            self.nn_service.get_model(),
            model_name
        )

        # Save scaler
        self.model_repo.save_scaler(self.data_repo.get_scaler())

        # Save metrics with appropriate name
        metrics_name = "evaluation_metrics_AWOA" if self.use_optimization else "evaluation_metrics_without_AWOA"
        self.model_repo.save_metrics(results["evaluation"], metrics_name)

        # Save training history
        self.model_repo.save_training_history(training_result.history)

        # Save optimization results if available
        if self.optimization_result:
            self.model_repo.save_optimization_results(results["optimization"])

        # Final summary
        total_time = time.time() - total_start_time

        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n  Total execution time: {total_time:.2f}s")
        print(f"\n  Final Test Results:")
        print(f"    Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        print(f"    F1-Score:  {metrics.f1_macro:.4f}")
        print(f"    Precision: {metrics.precision_macro:.4f}")
        print(f"    Recall:    {metrics.recall_macro:.4f}")
        if metrics.roc_auc:
            print(f"    ROC-AUC:   {metrics.roc_auc:.4f}")

        print(f"\n  Model saved to: {model_path}")
        print("=" * 70)

        results["total_time"] = total_time
        return results

    def _run_optimization(self) -> OptimizationResult:
        """Run AWOA optimization for hyperparameters."""

        # Create fitness function
        def fitness_function(params: Dict[str, Any]) -> float:
            """Evaluate hyperparameters (lower is better)."""
            return self.nn_service.evaluate_fitness(
                self.data_split,
                params,
                quick_epochs=50,  # More epochs for accurate fitness evaluation
                fitness_metric=self.fitness_metric
            )

        # Run optimization
        return self.optimizer.optimize(fitness_function, verbose=True)

    def _normalize_fitness_metric(self, metric: str) -> str:
        """Normalize the fitness metric name."""
        normalized = (metric or "val_loss").lower().strip()
        if normalized in ("loss", "val_loss"):
            return "val_loss"
        if normalized in ("accuracy", "val_accuracy"):
            return "val_accuracy"
        raise ValueError("fitness_metric must be 'val_loss' or 'val_accuracy'")

    def run_quick(self) -> Dict[str, Any]:
        """
        Run a quick version with fewer optimization iterations.
        Useful for testing.
        """
        # Reduce optimization iterations
        self.awoa_config = AWOAConfig(
            population_size=5,
            max_iterations=5,
            no_improvement_limit=3
        )
        self.optimizer = AWOAOptimizer(self.awoa_config)

        return self.run()
