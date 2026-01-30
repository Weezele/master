"""
Adaptive Whale Optimization Algorithm (AWOA) Implementation

This module implements AWOA for optimizing neural network hyperparameters.
Based on the Whale Optimization Algorithm (WOA) with adaptive mechanisms.

Reference:
- Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm.
- Adaptive improvements for better exploration/exploitation balance.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional
import time

from ..config import AWOAConfig
from ..domain import WhalePosition, OptimizationResult


class AWOAOptimizer:
    """
    Adaptive Whale Optimization Algorithm for hyperparameter optimization.

    The algorithm mimics the hunting behavior of humpback whales:
    1. Encircling prey (exploitation)
    2. Bubble-net attacking (exploitation with spiral)
    3. Search for prey (exploration)
    """

    def __init__(self, config: AWOAConfig):
        self.config = config
        self.population: List[WhalePosition] = []
        self.best_whale: Optional[WhalePosition] = None
        self.convergence_history: List[float] = []
        self.all_evaluations: List[Dict[str, Any]] = []

    def initialize_population(self) -> None:
        """Initialize whale population with random positions."""
        self.population = []

        for _ in range(self.config.population_size):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high)
                for low, high in zip(
                    self.config.lower_bounds,
                    self.config.upper_bounds
                )
            ])
            self.population.append(WhalePosition(position=position))

        print(f"  Initialized {len(self.population)} whales")

    def _get_adaptive_weight(self, iteration: int) -> float:
        """Calculate adaptive inertia weight."""
        if not self.config.adaptive_weight:
            return 1.0

        # Linearly decreasing weight
        w = self.config.weight_max - (
            (self.config.weight_max - self.config.weight_min) *
            (iteration / self.config.max_iterations)
        )
        return w

    def _get_adaptive_a(self, iteration: int) -> float:
        """Calculate adaptive 'a' parameter (decreases from 2 to 0)."""
        return self.config.a_initial - (
            (self.config.a_initial - self.config.a_final) *
            (iteration / self.config.max_iterations)
        )

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        """Clip position to stay within bounds."""
        return np.clip(
            position,
            self.config.lower_bounds,
            self.config.upper_bounds
        )

    def _encircling_prey(
        self,
        whale: WhalePosition,
        best_position: np.ndarray,
        A: np.ndarray,
        C: np.ndarray
    ) -> np.ndarray:
        """
        Encircling prey mechanism (exploitation phase).

        Whales encircle the prey and update position towards the best solution.
        """
        D = np.abs(C * best_position - whale.position)
        new_position = best_position - A * D
        return self._clip_position(new_position)

    def _spiral_update(
        self,
        whale: WhalePosition,
        best_position: np.ndarray,
        iteration: int
    ) -> np.ndarray:
        """
        Bubble-net spiral update (exploitation phase).

        Mimics the spiral-shaped movement of whales in bubble-net attacking.
        """
        D_prime = np.abs(best_position - whale.position)
        l = np.random.uniform(-1, 1)
        new_position = (
            D_prime * np.exp(self.config.b_constant * l) *
            np.cos(2 * np.pi * l) + best_position
        )
        return self._clip_position(new_position)

    def _exploration_search(
        self,
        whale: WhalePosition,
        A: np.ndarray,
        C: np.ndarray
    ) -> np.ndarray:
        """
        Search for prey (exploration phase).

        Random whale is selected to update position (global search).
        """
        # Select random whale
        random_whale = self.population[
            np.random.randint(0, len(self.population))
        ]
        D = np.abs(C * random_whale.position - whale.position)
        new_position = random_whale.position - A * D
        return self._clip_position(new_position)

    def optimize(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run AWOA optimization.

        Args:
            fitness_function: Function that takes hyperparameters dict and returns fitness
                             (lower is better, e.g., validation loss)
            verbose: Print progress updates

        Returns:
            OptimizationResult with best hyperparameters and convergence history
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        print("ADAPTIVE WHALE OPTIMIZATION ALGORITHM")
        print("=" * 60)

        # Initialize population
        self.initialize_population()
        self.convergence_history = []
        self.all_evaluations = []

        # Evaluate initial population
        print("\n  Evaluating initial population...")
        for i, whale in enumerate(self.population):
            params = whale.decode(self.config)
            fitness = fitness_function(params)
            whale.fitness = fitness
            self.all_evaluations.append({
                "iteration": 0,
                "whale_id": i,
                "params": params.copy(),
                "fitness": fitness
            })
            if verbose:
                print(f"    Whale {i+1}/{self.config.population_size}: "
                      f"fitness = {fitness:.6f}")

        # Find best whale
        self.best_whale = min(self.population, key=lambda w: w.fitness)
        self.convergence_history.append(self.best_whale.fitness)

        print(f"\n  Initial best fitness: {self.best_whale.fitness:.6f}")

        no_improvement_count = 0
        previous_best = self.best_whale.fitness

        # Main optimization loop
        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n  Iteration {iteration}/{self.config.max_iterations}")

            a = self._get_adaptive_a(iteration)
            w = self._get_adaptive_weight(iteration)

            for i, whale in enumerate(self.population):
                # Random coefficients
                r1, r2 = np.random.random(2)
                A = 2 * a * r1 - a  # A decreases from 2 to 0
                C = 2 * r2

                # Make A a vector for element-wise operations
                A = np.full(self.config.dimensions, A)
                C = np.full(self.config.dimensions, C)

                p = np.random.random()  # Probability for spiral vs encircling

                if p < 0.5:
                    if np.abs(A[0]) < 1:
                        # Exploitation: Encircling prey
                        new_position = self._encircling_prey(
                            whale, self.best_whale.position, A, C
                        )
                    else:
                        # Exploration: Search for prey
                        new_position = self._exploration_search(whale, A, C)
                else:
                    # Exploitation: Spiral bubble-net attack
                    new_position = self._spiral_update(
                        whale, self.best_whale.position, iteration
                    )

                # Apply adaptive weight
                whale.position = w * new_position + (1 - w) * whale.position
                whale.position = self._clip_position(whale.position)

                # Evaluate new position
                params = whale.decode(self.config)
                fitness = fitness_function(params)
                whale.fitness = fitness

                self.all_evaluations.append({
                    "iteration": iteration,
                    "whale_id": i,
                    "params": params.copy(),
                    "fitness": fitness
                })

            # Update best whale
            current_best = min(self.population, key=lambda w: w.fitness)
            if current_best.fitness < self.best_whale.fitness:
                self.best_whale = WhalePosition(
                    position=current_best.position.copy(),
                    fitness=current_best.fitness,
                    decoded_params=current_best.decoded_params.copy()
                )

            self.convergence_history.append(self.best_whale.fitness)

            # Check for improvement
            if abs(previous_best - self.best_whale.fitness) < self.config.convergence_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            previous_best = self.best_whale.fitness

            if verbose:
                print(f"    Best fitness: {self.best_whale.fitness:.6f} | "
                      f"a = {a:.3f} | w = {w:.3f}")

            # Early stopping
            if no_improvement_count >= self.config.no_improvement_limit:
                print(f"\n  Early stopping: No improvement for "
                      f"{self.config.no_improvement_limit} iterations")
                break

        optimization_time = time.time() - start_time

        # Final results
        best_params = self.best_whale.decode(self.config)

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"\n  Best fitness: {self.best_whale.fitness:.6f}")
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Total evaluations: {len(self.all_evaluations)}")
        print("\n  Best hyperparameters:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")

        return OptimizationResult(
            best_position=self.best_whale,
            best_hyperparameters=best_params,
            best_fitness=self.best_whale.fitness,
            convergence_history=self.convergence_history,
            all_evaluations=self.all_evaluations,
            total_iterations=iteration,
            optimization_time=optimization_time
        )
