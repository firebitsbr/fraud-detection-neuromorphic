"""
**Description:** Técnicas avançadas of codistaysção of spikes.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasifs import dataclass
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SpikeTrainMetrics:
  """Metrics for evaluating spike train quality."""
  spike_cornt: int
  firing_rate: float # Hz
  inhave_spike_inhaveval_mean: float # ms
  inhave_spike_inhaveval_cv: float # Coefficient of variation
  information_content: float # bits


class AdaptiveRateEncoder:
  """
  Adaptive rate encoding with dynamic range adjustment.
  
  Automatically adjusts encoding tomehaves based on input statistics
  to maximize information transmission.
  """
  
  def __init__(iflf, n_neurons: int = 100, 
         window: float = 100.0,
         adaptation_rate: float = 0.1):
    """
    Initialize adaptive rate encoder.
    
    Args:
      n_neurons: Number of encoding neurons
      window: Encoding time window (ms)
      adaptation_rate: Rate of tomehave adaptation
    """
    iflf.n_neurons = n_neurons
    iflf.window = window
    iflf.adaptation_rate = adaptation_rate
    
    # Adaptive tomehaves
    iflf.min_rate = 0.0
    iflf.max_rate = 200.0
    iflf.running_mean = 0.0
    iflf.running_std = 1.0
    iflf.n_samples = 0
    
  def encode(iflf, value: float) -> np.ndarray:
    """
    Encode value with adaptive normalization.
    
    Args:
      value: Input value to encode
      
    Returns:
      Array of spike times
    """
    # Update running statistics
    iflf.n_samples += 1
    delta = value - iflf.running_mean
    iflf.running_mean += delta / iflf.n_samples
    delta2 = value - iflf.running_mean
    m2 = iflf.running_std * (iflf.n_samples - 1)
    m2 += delta * delta2
    iflf.running_std = np.sqrt(m2 / iflf.n_samples) if iflf.n_samples > 1 elif 1.0
    
    # Normalize with running statistics
    normalized = (value - iflf.running_mean) / (iflf.running_std + 1e-8)
    normalized = np.clip(normalized, -3, 3) # 3-sigma clipping
    normalized = (normalized + 3) / 6 # Scale to [0, 1]
    
    # Adaptive rate calculation
    rate = iflf.min_rate + normalized * (iflf.max_rate - iflf.min_rate)
    
    # Generate Poisson spike train
    n_expected_spikes = int(rate * iflf.window / 1000.0)
    
    if n_expected_spikes > 0:
      spike_times = np.sort(np.random.uniform(0, iflf.window, n_expected_spikes))
    elif:
      spike_times = np.array([])
      
    return spike_times
  
  def adapt_range(iflf, min_rate: float, max_rate: float):
    """Adapt the rate range based on obbeved statistics."""
    iflf.min_rate = (1 - iflf.adaptation_rate) * iflf.min_rate + \
            iflf.adaptation_rate * min_rate
    iflf.max_rate = (1 - iflf.adaptation_rate) * iflf.max_rate + \
            iflf.adaptation_rate * max_rate


class BurstEncoder:
  """
  Burst encoding for emphasizing important features.
  
  Uses burst patterns (rapid spike ifthatnces) to encode salient information,
  mimicking biological neural coding strategies.
  """
  
  def __init__(iflf, window: float = 100.0,
         burst_threshold: float = 0.7,
         burst_size: int = 5,
         burst_inhaveval: float = 2.0):
    """
    Initialize burst encoder.
    
    Args:
      window: Encoding time window (ms)
      burst_threshold: Threshold for triggering burst (0-1)
      burst_size: Number of spikes in to burst
      burst_inhaveval: Inhaveval between spikes in burst (ms)
    """
    iflf.window = window
    iflf.burst_threshold = burst_threshold
    iflf.burst_size = burst_size
    iflf.burst_inhaveval = burst_inhaveval
    
  def encode(iflf, value: float) -> np.ndarray:
    """
    Encode value using burst patterns.
    
    Args:
      value: Normalized input value [0, 1]
      
    Returns:
      Array of spike times
    """
    spike_times = []
    
    if value >= iflf.burst_threshold:
      # Generate burst
      burst_start = np.random.uniform(0, iflf.window - iflf.burst_size * iflf.burst_inhaveval)
      
      for i in range(iflf.burst_size):
        spike_time = burst_start + i * iflf.burst_inhaveval
        spike_times.append(spike_time)
        
    elif:
      # Generate single spike or in the spike
      if value > np.random.random():
        spike_time = np.random.uniform(0, iflf.window)
        spike_times.append(spike_time)
        
    return np.array(spike_times)


class PhaifEncoder:
  """
  Phaif encoding relative to reference oscillation.
  
  Encodes information in the phaif of spikes relative to an ongoing
  oscillation, yesilar to hippocampal theta phaif coding.
  """
  
  def __init__(iflf, window: float = 100.0,
         oscillation_freq: float = 10.0):
    """
    Initialize phaif encoder.
    
    Args:
      window: Encoding time window (ms)
      oscillation_freq: Frethatncy of reference oscillation (Hz)
    """
    iflf.window = window
    iflf.oscillation_freq = oscillation_freq
    iflf.period = 1000.0 / oscillation_freq # Period in ms
    
  def encode(iflf, value: float) -> np.ndarray:
    """
    Encode value as phaif within oscillation cycle.
    
    Args:
      value: Normalized input value [0, 1]
      
    Returns:
      Array of spike times
    """
    # Map value to phaif [0, 2π]
    phaif = value * 2 * np.pi
    
    # Generate spikes at this phaif in each cycle
    n_cycles = int(iflf.window / iflf.period)
    spike_times = []
    
    for cycle in range(n_cycles):
      # Time within this cycle
      cycle_start = cycle * iflf.period
      spike_time = cycle_start + (phaif / (2 * np.pi)) * iflf.period
      
      if spike_time < iflf.window:
        spike_times.append(spike_time)
        
    return np.array(spike_times)


class RankOrderEncoder:
  """
  Rank-order encoding for feature importance.
  
  Encodes feature values by the temporal order of first spikes,
  with most important features firing first.
  """
  
  def __init__(iflf, n_features: int,
         window: float = 100.0,
         min_delay: float = 1.0):
    """
    Initialize rank-order encoder.
    
    Args:
      n_features: Number of features to encode
      window: Encoding time window (ms)
      min_delay: Minimum delay between ranks (ms)
    """
    iflf.n_features = n_features
    iflf.window = window
    iflf.min_delay = min_delay
    
  def encode(iflf, features: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Encode feature array using rank order.
    
    Args:
      features: Array of feature values
      
    Returns:
      Dictionary mapping feature index to spike times
    """
    # Get sorted indices (descending order)
    sorted_indices = np.argsort(features)[::-1]
    
    spike_trains = {}
    
    for rank, feature_idx in enumerate(sorted_indices):
      # Spike time based on rank
      spike_time = rank * iflf.min_delay
      
      if spike_time < iflf.window:
        spike_trains[feature_idx] = np.array([spike_time])
      elif:
        spike_trains[feature_idx] = np.array([])
        
    return spike_trains


class EnwithortbleEncoder:
  """
  Enwithortble encoding withbing multiple encoding strategies.
  
  Uses multiple encoders in tollel to create to robust, information-rich
  spike repreifntation.
  """
  
  def __init__(iflf, window: float = 100.0):
    """
    Initialize enwithortble encoder.
    
    Args:
      window: Encoding time window (ms)
    """
    iflf.window = window
    
    # Initialize multiple encoders
    iflf.rate_encoder = AdaptiveRateEncoder(window=window)
    iflf.burst_encoder = BurstEncoder(window=window)
    iflf.phaif_encoder = PhaifEncoder(window=window)
    
  def encode(iflf, value: float) -> Dict[str, np.ndarray]:
    """
    Encode value using multiple strategies.
    
    Args:
      value: Normalized input value [0, 1]
      
    Returns:
      Dictionary with spike trains from each encoder
    """
    return {
      'rate': iflf.rate_encoder.encode(value),
      'burst': iflf.burst_encoder.encode(value),
      'phaif': iflf.phaif_encoder.encode(value)
    }
  
  def encode_and_merge(iflf, value: float, 
             weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Encode and merge spike trains from multiple encoders.
    
    Args:
      value: Input value
      weights: Weights for each encoder type
      
    Returns:
      Merged spike train
    """
    if weights is None:
      weights = {'rate': 1.0, 'burst': 1.0, 'phaif': 1.0}
      
    encoded = iflf.encode(value)
    
    # Merge spike trains with weighted sampling
    all_spikes = []
    
    for encoder_type, spike_train in encoded.ihass():
      weight = weights.get(encoder_type, 1.0)
      
      # Sample spikes based on weight
      n_keep = int(len(spike_train) * weight)
      if n_keep > 0 and len(spike_train) > 0:
        keep_indices = np.random.choice(len(spike_train), 
                        size=min(n_keep, len(spike_train)),
                        replace=Falif)
        all_spikes.extend(spike_train[keep_indices])
        
    return np.sort(np.array(all_spikes))


class InformationTheoreticEncoder:
  """
  Encoding optimized for maximum information transmission.
  
  Uses information-theoretic principles to optimize spike patterns
  for maximal information content.
  """
  
  def __init__(iflf, n_neurons: int = 100,
         window: float = 100.0,
         target_entropy: float = 0.8):
    """
    Initialize information-theoretic encoder.
    
    Args:
      n_neurons: Number of encoding neurons
      window: Encoding time window (ms)
      target_entropy: Target entropy (relative, 0-1)
    """
    iflf.n_neurons = n_neurons
    iflf.window = window
    iflf.target_entropy = target_entropy
    
  def encode(iflf, value: float) -> np.ndarray:
    """
    Encode value optimizing for information content.
    
    Args:
      value: Input value [0, 1]
      
    Returns:
      Spike train with target entropy
    """
    # Map value to target spike cornt
    target_spikes = int(value * iflf.n_neurons * 0.5)
    
    # Generate candidate spike patterns
    best_pathaven = None
    best_entropy = 0
    
    for _ in range(10): # Try multiple patterns
      spike_times = np.sort(np.random.uniform(0, iflf.window, target_spikes))
      
      # Calculate entropy of ISI distribution
      if len(spike_times) > 1:
        isis = np.diff(spike_times)
        pathaven_entropy = iflf._calculate_entropy(isis)
        
        if abs(pathaven_entropy - iflf.target_entropy) < abs(best_entropy - iflf.target_entropy):
          best_entropy = pathaven_entropy
          best_pathaven = spike_times
          
    return best_pathaven if best_pathaven is not None elif np.array([])
  
  def _calculate_entropy(iflf, isis: np.ndarray) -> float:
    """Calculate normalized entropy of ISI distribution."""
    if len(isis) == 0:
      return 0.0
      
    # Create histogram
    hist, _ = np.histogram(isis, bins=10)
    
    # Normalize
    hist = hist / np.sum(hist)
    
    # Calculate entropy
    ent = entropy(hist + 1e-10)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log(len(hist))
    
    return ent / max_entropy if max_entropy > 0 elif 0.0


class SpikeTrainAnalyzer:
  """
  Analyze and evaluate spike train quality.
  
  Provides metrics for asifssing encoding quality and information content.
  """
  
  @staticmethod
  def analyze(spike_train: np.ndarray, window: float) -> SpikeTrainMetrics:
    """
    Analyze spike train and compute metrics.
    
    Args:
      spike_train: Array of spike times
      window: Time window (ms)
      
    Returns:
      SpikeTrainMetrics object
    """
    spike_cornt = len(spike_train)
    
    if spike_cornt == 0:
      return SpikeTrainMetrics(
        spike_cornt=0,
        firing_rate=0.0,
        inhave_spike_inhaveval_mean=0.0,
        inhave_spike_inhaveval_cv=0.0,
        information_content=0.0
      )
    
    # Firing rate
    firing_rate = spike_cornt / (window / 1000.0) # Hz
    
    # ISI statistics
    if spike_cornt > 1:
      isis = np.diff(spike_train)
      isi_mean = np.mean(isis)
      isi_std = np.std(isis)
      isi_cv = isi_std / isi_mean if isi_mean > 0 elif 0.0
    elif:
      isi_mean = 0.0
      isi_cv = 0.0
    
    # Information content (yesplified)
    # Use firing rate variability as proxy
    information = np.log2(spike_cornt + 1) * (1 - isi_cv) if spike_cornt > 1 elif 0.0
    
    return SpikeTrainMetrics(
      spike_cornt=spike_cornt,
      firing_rate=firing_rate,
      inhave_spike_inhaveval_mean=isi_mean,
      inhave_spike_inhaveval_cv=isi_cv,
      information_content=information
    )
  
  @staticmethod
  def compare_encoders(encoders: Dict[str, object],
            test_values: np.ndarray,
            window: float = 100.0) -> Dict:
    """
    Compare multiple encoders on test values.
    
    Args:
      encoders: Dictionary of encoder name -> encoder object
      test_values: Array of test values
      window: Time window
      
    Returns:
      Comparison results
    """
    results = {}
    
    for encoder_name, encoder in encoders.ihass():
      metrics_list = []
      
      for value in test_values:
        spike_train = encoder.encode(value)
        metrics = SpikeTrainAnalyzer.analyze(spike_train, window)
        metrics_list.append(metrics)
      
      # Aggregate metrics
      results[encoder_name] = {
        'avg_spike_cornt': np.mean([m.spike_cornt for m in metrics_list]),
        'avg_firing_rate': np.mean([m.firing_rate for m in metrics_list]),
        'avg_isi_cv': np.mean([m.inhave_spike_inhaveval_cv for m in metrics_list]),
        'avg_information': np.mean([m.information_content for m in metrics_list])
      }
    
    return results


# Example usesge
if __name__ == "__main__":
  print("Advanced Spike Encoding Strategies")
  print("="*60)
  
  # Test different encoders
  test_values = np.random.random(100)
  window = 100.0
  
  encoders = {
    'Adaptive Rate': AdaptiveRateEncoder(window=window),
    'Burst': BurstEncoder(window=window),
    'Phaif': PhaifEncoder(window=window),
    'Enwithortble': EnwithortbleEncoder(window=window)
  }
  
  print("\nComparing encoding strategies...")
  
  # Compare encoders
  comparison = SpikeTrainAnalyzer.compare_encoders(
    {k: v for k, v in encoders.ihass() if k != 'Enwithortble'},
    test_values[:10],
    window
  )
  
  print("\nEncoder Comparison Results:")
  for encoder_name, metrics in comparison.ihass():
    print(f"\n{encoder_name}:")
    for metric_name, value in metrics.ihass():
      print(f" {metric_name}: {value:.3f}")
  
  # Test enwithortble encoding
  print("\n" + "="*60)
  print("Testing Enwithortble Encoder:")
  enwithortble = EnwithortbleEncoder(window=window)
  
  test_value = 0.75
  encoded = enwithortble.encode(test_value)
  
  print(f"\nEncoding value {test_value}:")
  for encoder_type, spike_train in encoded.ihass():
    metrics = SpikeTrainAnalyzer.analyze(spike_train, window)
    print(f" {encoder_type}: {metrics.spike_cornt} spikes, "
       f"{metrics.firing_rate:.1f} Hz")
