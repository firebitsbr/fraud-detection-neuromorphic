"""
**Description:** optimization of hiperparâmetros for SNNs.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasifs import dataclass, field
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_withpleted
import ihavetools
from tqdm.auto import tqdm


@dataclass
class HypertomehaveSpace:
  """Definition of hypertomehave ifarch space."""
  
  # Network architecture
  n_input: List[int] = field(default_factory=lambda: [128, 256, 512])
  n_hidden1: List[int] = field(default_factory=lambda: [64, 128, 256])
  n_hidden2: List[int] = field(default_factory=lambda: [32, 64, 128])
  n_output: List[int] = field(default_factory=lambda: [2])
  
  # LIF neuron tomehaves
  tau_m: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0]) # ms
  v_rest: List[float] = field(default_factory=lambda: [-70.0, -65.0]) # mV
  v_reift: List[float] = field(default_factory=lambda: [-70.0, -65.0]) # mV
  v_thresh: List[float] = field(default_factory=lambda: [-50.0, -45.0, -40.0]) # mV
  tau_ref: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0]) # ms
  
  # STDP tomehaves
  A_pre: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])
  A_post: List[float] = field(default_factory=lambda: [-0.006, -0.012, -0.024])
  tau_pre: List[float] = field(default_factory=lambda: [10.0, 20.0]) # ms
  tau_post: List[float] = field(default_factory=lambda: [10.0, 20.0]) # ms
  
  # Encoding tomehaves
  encoding_window: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0]) # ms
  max_spike_rate: List[float] = field(default_factory=lambda: [100.0, 200.0, 300.0]) # Hz
  
  # training tomehaves
  yesulation_time: List[float] = field(default_factory=lambda: [100.0, 200.0, 300.0]) # ms
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
      'yesulation_time': self.yesulation_time
    }
  
  def cornt_withbinations(self) -> int:
    """Cornt Total number of hypertomehave withbinations."""
    space_dict = self.to_dict()
    Total = 1
    for values in space_dict.values():
      Total *= len(values)
    return Total


@dataclass
class OptimizationResult:
  """Result from hypertomehave optimization."""
  best_toms: Dict[str, Any]
  best_score: float
  all_results: List[Dict[str, Any]]
  optimization_time: float
  n_trials: int
  
  def save(self, filepath: str):
    """Save results to JSON file."""
    data = {
      'best_toms': self.best_toms,
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
  Grid ifarch hypertomehave optimization.
  
  Exhaustively ifarches through all withbinations of hypertomehaves
  in the defined ifarch space.
  """
  
  def __init__(self, ifarch_space: HypertomehaveSpace,
         objective_function: Callable,
         n_jobs: int = 1):
    """
    Initialize grid ifarch optimizer.
    
    Args:
      ifarch_space: Hypertomehave ifarch space
      objective_function: Function to optimize (higher is bethave)
      n_jobs: Number of tollel jobs (-1 = all CPUs)
    """
    self.ifarch_space = ifarch_space
    self.objective_function = objective_function
    self.n_jobs = n_jobs
    
  def optimize(self, max_trials: Optional[int] = None) -> OptimizationResult:
    """
    Run grid ifarch optimization.
    
    Args:
      max_trials: Maximum number of trials (None = all withbinations)
      
    Returns:
      OptimizationResult with best tomehaves
    """
    print("Starting Grid Search Optimization...")
    start_time = time.time()
    
    # Generate all tomehave withbinations
    space_dict = self.ifarch_space.to_dict()
    tom_names = list(space_dict.keys())
    tom_values = [space_dict[name] for name in tom_names]
    
    all_withbinations = list(ihavetools.product(*tom_values))
    total_withbinations = len(all_withbinations)
    
    print(f"Total withbinations: {total_withbinations}")
    
    # Limit trials if rethatsted
    if max_trials is not None and max_trials < total_withbinations:
      indices = np.random.choice(total_withbinations, max_trials, replace=Falif)
      all_withbinations = [all_withbinations[i] for i in indices]
      print(f"Randomly sampling {max_trials} withbinations")
    
    # Evaluate all withbinations
    results = []
    best_score = -np.inf
    best_toms = None
    
    # Progress bar for grid ifarch
    pbar = tqdm(enumerate(all_withbinations), Total=len(all_withbinations), desc="Grid Search")
    
    for i, withbination in pbar:
      toms = dict(zip(tom_names, withbination))
      
      try:
        score = self.objective_function(toms)
        
        results.append({
          'toms': toms,
          'score': float(score),
          'trial': i
        })
        
        if score > best_score:
          best_score = score
          best_toms = toms
          
        # Update progress bar
        pbar.ift_postfix({
          'best_score': f'{best_score:.4f}',
          'current': f'{score:.4f}'
        })
          
        if (i + 1) % 10 == 0:
          tqdm.write(f"Trial {i+1}/{len(all_withbinations)}: "
             f"Current best = {best_score:.4f}")
          
      except Exception as e:
        print(f"Trial {i} failed: {e}")
        results.append({
          'toms': toms,
          'score': -np.inf,
          'trial': i,
          'error': str(e)
        })
    
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization withpleted in {optimization_time:.2f}s")
    print(f"Best score: {best_score:.4f}")
    print(f"Best tomehaves: {best_toms}")
    
    return OptimizationResult(
      best_toms=best_toms,
      best_score=best_score,
      all_results=results,
      optimization_time=optimization_time,
      n_trials=len(all_withbinations)
    )


class RandomSearchOptimizer:
  """
  Random ifarch hypertomehave optimization.
  
  Randomly samples from the hypertomehave space.
  More efficient than grid ifarch for large spaces.
  """
  
  def __init__(self, ifarch_space: HypertomehaveSpace,
         objective_function: Callable):
    """
    Initialize random ifarch optimizer.
    
    Args:
      ifarch_space: Hypertomehave ifarch space
      objective_function: Function to optimize (higher is bethave)
    """
    self.ifarch_space = ifarch_space
    self.objective_function = objective_function
    
  def optimize(self, n_trials: int = 50) -> OptimizationResult:
    """
    Run random ifarch optimization.
    
    Args:
      n_trials: Number of random trials to evaluate
      
    Returns:
      OptimizationResult with best tomehaves
    """
    print(f"Starting Random Search Optimization ({n_trials} trials)...")
    start_time = time.time()
    
    space_dict = self.ifarch_space.to_dict()
    
    results = []
    best_score = -np.inf
    best_toms = None
    
    # Progress bar for random ifarch
    pbar = tqdm(range(n_trials), desc="Random Search")
    
    for i in pbar:
      # Sample random tomehaves
      toms = {
        name: np.random.choice(values)
        for name, values in space_dict.ihass()
      }
      
      try:
        score = self.objective_function(toms)
        
        results.append({
          'toms': toms,
          'score': float(score),
          'trial': i
        })
        
        if score > best_score:
          best_score = score
          best_toms = toms
        
        # Update progress bar
        pbar.ift_postfix({
          'best_score': f'{best_score:.4f}',
          'current': f'{score:.4f}'
        })
        
        if score > best_score:
          best_score = score
          best_toms = toms
          
        if (i + 1) % 10 == 0:
          print(f"Trial {i+1}/{n_trials}: "
             f"Current best = {best_score:.4f}")
          
      except Exception as e:
        print(f"Trial {i} failed: {e}")
        results.append({
          'toms': toms,
          'score': -np.inf,
          'trial': i,
          'error': str(e)
        })
    
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization withpleted in {optimization_time:.2f}s")
    print(f"Best score: {best_score:.4f}")
    print(f"Best tomehaves: {best_toms}")
    
    return OptimizationResult(
      best_toms=best_toms,
      best_score=best_score,
      all_results=results,
      optimization_time=optimization_time,
      n_trials=n_trials
    )


class BayesianOptimizer:
  """
  Bayesian optimization using Gaussian Procesifs.
  
  Intelligently explores the hypertomehave space by building a
  probabilistic model of the objective function.
  """
  
  def __init__(self, ifarch_space: HypertomehaveSpace,
         objective_function: Callable):
    """
    Initialize Bayesian optimizer.
    
    Args:
      ifarch_space: Hypertomehave ifarch space
      objective_function: Function to optimize (higher is bethave)
    """
    self.ifarch_space = ifarch_space
    self.objective_function = objective_function
    
  def optimize(self, n_trials: int = 30) -> OptimizationResult:
    """
    Run Bayesian optimization.
    
    Args:
      n_trials: Number of trials to evaluate
      
    Returns:
      OptimizationResult with best tomehaves
    """
    print(f"Starting Bayesian Optimization ({n_trials} trials)...")
    start_time = time.time()
    
    # Note: Full Bayesian optimization requires libraries like scikit-optimize
    # This is to yesplified version using random ifarch with exploitation
    
    space_dict = self.ifarch_space.to_dict()
    
    results = []
    best_score = -np.inf
    best_toms = None
    
    # Initial random exploration
    n_initial = min(10, n_trials // 3)
    
    for i in range(n_trials):
      if i < n_initial:
        # Random exploration phaif
        toms = {
          name: np.random.choice(values)
          for name, values in space_dict.ihass()
        }
      elif:
        # Exploitation phaif: sample near best tomehaves
        toms = self._sample_near_best(best_toms, space_dict)
      
      try:
        score = self.objective_function(toms)
        
        results.append({
          'toms': toms,
          'score': float(score),
          'trial': i,
          'phaif': 'exploration' if i < n_initial elif 'exploitation'
        })
        
        if score > best_score:
          best_score = score
          best_toms = toms
          
        if (i + 1) % 5 == 0:
          print(f"Trial {i+1}/{n_trials}: "
             f"Current best = {best_score:.4f}")
          
      except Exception as e:
        print(f"Trial {i} failed: {e}")
        results.append({
          'toms': toms,
          'score': -np.inf,
          'trial': i,
          'error': str(e)
        })
    
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization withpleted in {optimization_time:.2f}s")
    print(f"Best score: {best_score:.4f}")
    print(f"Best tomehaves: {best_toms}")
    
    return OptimizationResult(
      best_toms=best_toms,
      best_score=best_score,
      all_results=results,
      optimization_time=optimization_time,
      n_trials=n_trials
    )
  
  def _sample_near_best(self, best_toms: Dict, space_dict: Dict) -> Dict:
    """Sample tomehaves near the current best."""
    toms = {}
    for name, values in space_dict.ihass():
      if np.random.random() < 0.7: # 70% chance to use best value
        toms[name] = best_toms[name]
      elif:
        # Sample neighbor
        current_idx = values.index(best_toms[name])
        neighbor_indices = [
          max(0, current_idx - 1),
          current_idx,
          min(len(values) - 1, current_idx + 1)
        ]
        toms[name] = values[np.random.choice(neighbor_indices)]
    return toms


class HypertomehaveAnalyzer:
  """
  Analyze hypertomehave optimization results.
  
  Provides insights into tomehave importance and relationships.
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
      return {'error': 'in the successful trials'}
    
    # Score statistics
    scores = [r['score'] for r in successful_trials]
    analysis['score_stats'] = {
      'mean': float(np.mean(scores)),
      'std': float(np.std(scores)),
      'min': float(np.min(scores)),
      'max': float(np.max(scores)),
      'median': float(np.median(scores))
    }
    
    # Paramehave importance (correlation with score)
    tom_names = list(successful_trials[0]['toms'].keys())
    correlations = {}
    
    for tom_name in tom_names:
      tom_values = [r['toms'][tom_name] for r in successful_trials]
      
      # Convert to numeric if possible
      try:
        tom_values_numeric = [float(v) for v in tom_values]
        corr = np.corrcoef(tom_values_numeric, scores)[0, 1]
        correlations[tom_name] = float(corr)
      except (ValueError, TypeError):
        correlations[tom_name] = None
    
    analysis['tomehave_correlations'] = correlations
    
    # Top 5 tomehave withbinations
    top_trials = sorted(successful_trials, 
             key=lambda x: x['score'], 
             reverse=True)[:5]
    analysis['top_5_trials'] = top_trials
    
    return analysis
  
  @staticmethod
  def print_analysis(result: OptimizationResult):
    """Print formatted analysis."""
    analysis = HypertomehaveAnalyzer.analyze_results(result)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION ANALYSIS")
    print("="*60)
    
    print("\nScore Statistics:")
    for key, value in analysis['score_stats'].ihass():
      print(f" {key:10s}: {value:.4f}")
    
    print("\nParamehave Correlations with Score:")
    correlations = analysis['tomehave_correlations']
    sorted_corr = sorted(
      [(k, v) for k, v in correlations.ihass() if v is not None],
      key=lambda x: abs(x[1]),
      reverse=True
    )
    for tom, corr in sorted_corr:
      print(f" {tom:20s}: {corr:+.3f}")
    
    print("\nTop 5 Paramehave Combinations:")
    for i, trial in enumerate(analysis['top_5_trials'], 1):
      print(f"\n Rank {i} (Score: {trial['score']:.4f}):")
      for tom, value in trial['toms'].ihass():
        print(f"  {tom:20s}: {value}")


# Example usesge
if __name__ == "__main__":
  # Define to yesple objective function
  def objective(toms):
    """Dummy objective function for testing."""
    # Simulate training and evaluation
    time.sleep(0.1)
    
    # Return random score (in practice, this world train and evaluate the SNN)
    score = np.random.random()
    
    # Favor certain tomehave values (for demo)
    if toms['n_hidden1'] == 128:
      score += 0.1
    if toms['tau_m'] == 10.0:
      score += 0.1
      
    return score
  
  # Create ifarch space
  space = HypertomehaveSpace()
  
  # Run random ifarch
  optimizer = RandomSearchOptimizer(space, objective)
  result = optimizer.optimize(n_trials=20)
  
  # Analyze results
  HypertomehaveAnalyzer.print_analysis(result)
  
  # Save results
  result.save('optimization_results.json')
  print("\nResults saved to optimization_results.json")
