"""
Hyperparameter Optimization Module for Neuromorphic Fraud Detection

This module provides automated hyperparameter tuning capabilities for the
Spiking Neural Network fraud detection system using various optimization strategies.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
License: MIT License
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm.auto import tqdm


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    
    # Network architecture
    n_input: List[int] = field(default_factory=lambda: [128, 256, 512])
    n_hidden1: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_hidden2: List[int] = field(default_factory=lambda: [32, 64, 128])
    n_output: List[int] = field(default_factory=lambda: [2])
    
    # LIF neuron parameters
    tau_m: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])  # ms
    v_rest: List[float] = field(default_factory=lambda: [-70.0, -65.0])  # mV
    v_reset: List[float] = field(default_factory=lambda: [-70.0, -65.0])  # mV
    v_thresh: List[float] = field(default_factory=lambda: [-50.0, -45.0, -40.0])  # mV
    tau_ref: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])  # ms
    
    # STDP parameters
    A_pre: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])
    A_post: List[float] = field(default_factory=lambda: [-0.006, -0.012, -0.024])
    tau_pre: List[float] = field(default_factory=lambda: [10.0, 20.0])  # ms
    tau_post: List[float] = field(default_factory=lambda: [10.0, 20.0])  # ms
    
    # Encoding parameters
    encoding_window: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0])  # ms
    max_spike_rate: List[float] = field(default_factory=lambda: [100.0, 200.0, 300.0])  # Hz
    
    # Training parameters
    simulation_time: List[float] = field(default_factory=lambda: [100.0, 200.0, 300.0])  # ms
    learning_rate: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary format."""
        return {
            'n_input': self.n_input,
            'n_hidden1': self.n_hidden1,
            'n_hidden2': self.n_hidden2,
            'tau_m': self.tau_m,
            'v_thresh': self.v_thresh,
            'A_pre': self.A_pre,
            'A_post': self.A_post,
            'encoding_window': self.encoding_window,
            'max_spike_rate': self.max_spike_rate,
            'simulation_time': self.simulation_time
        }
    
    def count_combinations(self) -> int:
        """Count total number of hyperparameter combinations."""
        space_dict = self.to_dict()
        total = 1
        for values in space_dict.values():
            total *= len(values)
        return total


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    n_trials: int
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        data = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'all_results': self.all_results,
            'optimization_time': self.optimization_time,
            'n_trials': self.n_trials
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationResult':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class GridSearchOptimizer:
    """
    Grid search hyperparameter optimization.
    
    Exhaustively searches through all combinations of hyperparameters
    in the defined search space.
    """
    
    def __init__(self, search_space: HyperparameterSpace,
                 objective_function: Callable,
                 n_jobs: int = 1):
        """
        Initialize grid search optimizer.
        
        Args:
            search_space: Hyperparameter search space
            objective_function: Function to optimize (higher is better)
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.search_space = search_space
        self.objective_function = objective_function
        self.n_jobs = n_jobs
        
    def optimize(self, max_trials: Optional[int] = None) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            max_trials: Maximum number of trials (None = all combinations)
            
        Returns:
            OptimizationResult with best parameters
        """
        print("Starting Grid Search Optimization...")
        start_time = time.time()
        
        # Generate all parameter combinations
        space_dict = self.search_space.to_dict()
        param_names = list(space_dict.keys())
        param_values = [space_dict[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        total_combinations = len(all_combinations)
        
        print(f"Total combinations: {total_combinations}")
        
        # Limit trials if requested
        if max_trials is not None and max_trials < total_combinations:
            indices = np.random.choice(total_combinations, max_trials, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
            print(f"Randomly sampling {max_trials} combinations")
        
        # Evaluate all combinations
        results = []
        best_score = -np.inf
        best_params = None
        
        # Progress bar for grid search
        pbar = tqdm(enumerate(all_combinations), total=len(all_combinations), desc="Grid Search")
        
        for i, combination in pbar:
            params = dict(zip(param_names, combination))
            
            try:
                score = self.objective_function(params)
                
                results.append({
                    'params': params,
                    'score': float(score),
                    'trial': i
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                # Update progress bar
                pbar.set_postfix({
                    'best_score': f'{best_score:.4f}',
                    'current': f'{score:.4f}'
                })
                    
                if (i + 1) % 10 == 0:
                    tqdm.write(f"Trial {i+1}/{len(all_combinations)}: "
                          f"Current best = {best_score:.4f}")
                    
            except Exception as e:
                print(f"Trial {i} failed: {e}")
                results.append({
                    'params': params,
                    'score': -np.inf,
                    'trial': i,
                    'error': str(e)
                })
        
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.2f}s")
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_time=optimization_time,
            n_trials=len(all_combinations)
        )


class RandomSearchOptimizer:
    """
    Random search hyperparameter optimization.
    
    Randomly samples from the hyperparameter space.
    More efficient than grid search for large spaces.
    """
    
    def __init__(self, search_space: HyperparameterSpace,
                 objective_function: Callable):
        """
        Initialize random search optimizer.
        
        Args:
            search_space: Hyperparameter search space
            objective_function: Function to optimize (higher is better)
        """
        self.search_space = search_space
        self.objective_function = objective_function
        
    def optimize(self, n_trials: int = 50) -> OptimizationResult:
        """
        Run random search optimization.
        
        Args:
            n_trials: Number of random trials to evaluate
            
        Returns:
            OptimizationResult with best parameters
        """
        print(f"Starting Random Search Optimization ({n_trials} trials)...")
        start_time = time.time()
        
        space_dict = self.search_space.to_dict()
        
        results = []
        best_score = -np.inf
        best_params = None
        
        # Progress bar for random search
        pbar = tqdm(range(n_trials), desc="Random Search")
        
        for i in pbar:
            # Sample random parameters
            params = {
                name: np.random.choice(values)
                for name, values in space_dict.items()
            }
            
            try:
                score = self.objective_function(params)
                
                results.append({
                    'params': params,
                    'score': float(score),
                    'trial': i
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                # Update progress bar
                pbar.set_postfix({
                    'best_score': f'{best_score:.4f}',
                    'current': f'{score:.4f}'
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                if (i + 1) % 10 == 0:
                    print(f"Trial {i+1}/{n_trials}: "
                          f"Current best = {best_score:.4f}")
                    
            except Exception as e:
                print(f"Trial {i} failed: {e}")
                results.append({
                    'params': params,
                    'score': -np.inf,
                    'trial': i,
                    'error': str(e)
                })
        
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.2f}s")
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_time=optimization_time,
            n_trials=n_trials
        )


class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Processes.
    
    Intelligently explores the hyperparameter space by building a
    probabilistic model of the objective function.
    """
    
    def __init__(self, search_space: HyperparameterSpace,
                 objective_function: Callable):
        """
        Initialize Bayesian optimizer.
        
        Args:
            search_space: Hyperparameter search space
            objective_function: Function to optimize (higher is better)
        """
        self.search_space = search_space
        self.objective_function = objective_function
        
    def optimize(self, n_trials: int = 30) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            n_trials: Number of trials to evaluate
            
        Returns:
            OptimizationResult with best parameters
        """
        print(f"Starting Bayesian Optimization ({n_trials} trials)...")
        start_time = time.time()
        
        # Note: Full Bayesian optimization requires libraries like scikit-optimize
        # This is a simplified version using random search with exploitation
        
        space_dict = self.search_space.to_dict()
        
        results = []
        best_score = -np.inf
        best_params = None
        
        # Initial random exploration
        n_initial = min(10, n_trials // 3)
        
        for i in range(n_trials):
            if i < n_initial:
                # Random exploration phase
                params = {
                    name: np.random.choice(values)
                    for name, values in space_dict.items()
                }
            else:
                # Exploitation phase: sample near best parameters
                params = self._sample_near_best(best_params, space_dict)
            
            try:
                score = self.objective_function(params)
                
                results.append({
                    'params': params,
                    'score': float(score),
                    'trial': i,
                    'phase': 'exploration' if i < n_initial else 'exploitation'
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                if (i + 1) % 5 == 0:
                    print(f"Trial {i+1}/{n_trials}: "
                          f"Current best = {best_score:.4f}")
                    
            except Exception as e:
                print(f"Trial {i} failed: {e}")
                results.append({
                    'params': params,
                    'score': -np.inf,
                    'trial': i,
                    'error': str(e)
                })
        
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.2f}s")
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_time=optimization_time,
            n_trials=n_trials
        )
    
    def _sample_near_best(self, best_params: Dict, space_dict: Dict) -> Dict:
        """Sample parameters near the current best."""
        params = {}
        for name, values in space_dict.items():
            if np.random.random() < 0.7:  # 70% chance to use best value
                params[name] = best_params[name]
            else:
                # Sample neighbor
                current_idx = values.index(best_params[name])
                neighbor_indices = [
                    max(0, current_idx - 1),
                    current_idx,
                    min(len(values) - 1, current_idx + 1)
                ]
                params[name] = values[np.random.choice(neighbor_indices)]
        return params


class HyperparameterAnalyzer:
    """
    Analyze hyperparameter optimization results.
    
    Provides insights into parameter importance and relationships.
    """
    
    @staticmethod
    def analyze_results(result: OptimizationResult) -> Dict[str, Any]:
        """
        Analyze optimization results.
        
        Args:
            result: OptimizationResult to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Extract all successful trials
        successful_trials = [
            r for r in result.all_results 
            if r['score'] != -np.inf
        ]
        
        if not successful_trials:
            return {'error': 'No successful trials'}
        
        # Score statistics
        scores = [r['score'] for r in successful_trials]
        analysis['score_stats'] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }
        
        # Parameter importance (correlation with score)
        param_names = list(successful_trials[0]['params'].keys())
        correlations = {}
        
        for param_name in param_names:
            param_values = [r['params'][param_name] for r in successful_trials]
            
            # Convert to numeric if possible
            try:
                param_values_numeric = [float(v) for v in param_values]
                corr = np.corrcoef(param_values_numeric, scores)[0, 1]
                correlations[param_name] = float(corr)
            except (ValueError, TypeError):
                correlations[param_name] = None
        
        analysis['parameter_correlations'] = correlations
        
        # Top 5 parameter combinations
        top_trials = sorted(successful_trials, 
                          key=lambda x: x['score'], 
                          reverse=True)[:5]
        analysis['top_5_trials'] = top_trials
        
        return analysis
    
    @staticmethod
    def print_analysis(result: OptimizationResult):
        """Print formatted analysis."""
        analysis = HyperparameterAnalyzer.analyze_results(result)
        
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION ANALYSIS")
        print("="*60)
        
        print("\nScore Statistics:")
        for key, value in analysis['score_stats'].items():
            print(f"  {key:10s}: {value:.4f}")
        
        print("\nParameter Correlations with Score:")
        correlations = analysis['parameter_correlations']
        sorted_corr = sorted(
            [(k, v) for k, v in correlations.items() if v is not None],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for param, corr in sorted_corr:
            print(f"  {param:20s}: {corr:+.3f}")
        
        print("\nTop 5 Parameter Combinations:")
        for i, trial in enumerate(analysis['top_5_trials'], 1):
            print(f"\n  Rank {i} (Score: {trial['score']:.4f}):")
            for param, value in trial['params'].items():
                print(f"    {param:20s}: {value}")


# Example usage
if __name__ == "__main__":
    # Define a simple objective function
    def objective(params):
        """Dummy objective function for testing."""
        # Simulate training and evaluation
        time.sleep(0.1)
        
        # Return random score (in practice, this would train and evaluate the SNN)
        score = np.random.random()
        
        # Favor certain parameter values (for demo)
        if params['n_hidden1'] == 128:
            score += 0.1
        if params['tau_m'] == 10.0:
            score += 0.1
            
        return score
    
    # Create search space
    space = HyperparameterSpace()
    
    # Run random search
    optimizer = RandomSearchOptimizer(space, objective)
    result = optimizer.optimize(n_trials=20)
    
    # Analyze results
    HyperparameterAnalyzer.print_analysis(result)
    
    # Save results
    result.save('optimization_results.json')
    print("\nResults saved to optimization_results.json")
