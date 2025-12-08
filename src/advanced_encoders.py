"""
Advanced spike encoding techniques.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
Date: December 2025
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SpikeTrainMetrics:
    """Metrics for evaluating spike train quality."""
    spike_count: int
    firing_rate: float  # Hz
    inter_spike_interval_mean: float  # ms
    inter_spike_interval_cv: float  # Coefficient of variation
    information_content: float  # bits


class AdaptiveRateEncoder:
    """
    Adaptive rate encoding with dynamic range adjustment.
    
    Automatically adjusts encoding parameters based on input statistics
    to maximize information transmission.
    """
    
    def __init__(self, n_neurons: int = 100, 
                 window: float = 100.0,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive rate encoder.
        
        Args:
            n_neurons: Number of encoding neurons
            window: Encoding time window (ms)
            adaptation_rate: Rate of parameter adaptation
        """
        self.n_neurons = n_neurons
        self.window = window
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.min_rate = 0.0
        self.max_rate = 200.0
        self.running_mean = 0.0
        self.running_std = 1.0
        self.n_samples = 0
        
    def encode(self, value: float) -> np.ndarray:
        """
        Encode value with adaptive normalization.
        
        Args:
            value: Input value to encode
            
        Returns:
            Array of spike times
        """
        # Update running statistics
        self.n_samples += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = value - self.running_mean
        m2 = self.running_std * (self.n_samples - 1)
        m2 += delta * delta2
        self.running_std = np.sqrt(m2 / self.n_samples) if self.n_samples > 1 else 1.0
        
        # Normalize with running statistics
        normalized = (value - self.running_mean) / (self.running_std + 1e-8)
        normalized = np.clip(normalized, -3, 3)  # 3-sigma clipping
        normalized = (normalized + 3) / 6  # Scale to [0, 1]
        
        # Adaptive rate calculation
        rate = self.min_rate + normalized * (self.max_rate - self.min_rate)
        
        # Generate Poisson spike train
        n_expected_spikes = int(rate * self.window / 1000.0)
        
        if n_expected_spikes > 0:
            spike_times = np.sort(np.random.uniform(0, self.window, n_expected_spikes))
        else:
            spike_times = np.array([])
            
        return spike_times
    
    def adapt_range(self, min_rate: float, max_rate: float):
        """Adapt the rate range based on observed statistics."""
        self.min_rate = (1 - self.adaptation_rate) * self.min_rate + \
                       self.adaptation_rate * min_rate
        self.max_rate = (1 - self.adaptation_rate) * self.max_rate + \
                       self.adaptation_rate * max_rate


class BurstEncoder:
    """
    Burst encoding for emphasizing important features.
    
    Uses burst patterns (rapid spike sequences) to encode salient information,
    mimicking biological neural coding strategies.
    """
    
    def __init__(self, window: float = 100.0,
                 burst_threshold: float = 0.7,
                 burst_size: int = 5,
                 burst_interval: float = 2.0):
        """
        Initialize burst encoder.
        
        Args:
            window: Encoding time window (ms)
            burst_threshold: Threshold for triggering burst (0-1)
            burst_size: Number of spikes in a burst
            burst_interval: Interval between spikes in burst (ms)
        """
        self.window = window
        self.burst_threshold = burst_threshold
        self.burst_size = burst_size
        self.burst_interval = burst_interval
        
    def encode(self, value: float) -> np.ndarray:
        """
        Encode value using burst patterns.
        
        Args:
            value: Normalized input value [0, 1]
            
        Returns:
            Array of spike times
        """
        spike_times = []
        
        if value >= self.burst_threshold:
            # Generate burst
            burst_start = np.random.uniform(0, self.window - self.burst_size * self.burst_interval)
            
            for i in range(self.burst_size):
                spike_time = burst_start + i * self.burst_interval
                spike_times.append(spike_time)
                
        else:
            # Generate single spike or no spike
            if value > np.random.random():
                spike_time = np.random.uniform(0, self.window)
                spike_times.append(spike_time)
                
        return np.array(spike_times)


class PhaseEncoder:
    """
    Phase encoding relative to reference oscillation.
    
    Encodes information in the phase of spikes relative to an ongoing
    oscillation, similar to hippocampal theta phase coding.
    """
    
    def __init__(self, window: float = 100.0,
                 oscillation_freq: float = 10.0):
        """
        Initialize phase encoder.
        
        Args:
            window: Encoding time window (ms)
            oscillation_freq: Frequency of reference oscillation (Hz)
        """
        self.window = window
        self.oscillation_freq = oscillation_freq
        self.period = 1000.0 / oscillation_freq  # Period in ms
        
    def encode(self, value: float) -> np.ndarray:
        """
        Encode value as phase within oscillation cycle.
        
        Args:
            value: Normalized input value [0, 1]
            
        Returns:
            Array of spike times
        """
        # Map value to phase [0, 2π]
        phase = value * 2 * np.pi
        
        # Generate spikes at this phase in each cycle
        n_cycles = int(self.window / self.period)
        spike_times = []
        
        for cycle in range(n_cycles):
            # Time within this cycle
            cycle_start = cycle * self.period
            spike_time = cycle_start + (phase / (2 * np.pi)) * self.period
            
            if spike_time < self.window:
                spike_times.append(spike_time)
                
        return np.array(spike_times)


class RankOrderEncoder:
    """
    Rank-order encoding for feature importance.
    
    Encodes feature values by the temporal order of first spikes,
    with most important features firing first.
    """
    
    def __init__(self, n_features: int,
                 window: float = 100.0,
                 min_delay: float = 1.0):
        """
        Initialize rank-order encoder.
        
        Args:
            n_features: Number of features to encode
            window: Encoding time window (ms)
            min_delay: Minimum delay between ranks (ms)
        """
        self.n_features = n_features
        self.window = window
        self.min_delay = min_delay
        
    def encode(self, features: np.ndarray) -> Dict[int, np.ndarray]:
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
            spike_time = rank * self.min_delay
            
            if spike_time < self.window:
                spike_trains[feature_idx] = np.array([spike_time])
            else:
                spike_trains[feature_idx] = np.array([])
                
        return spike_trains


class EnsembleEncoder:
    """
    Ensemble encoding combining multiple encoding strategies.
    
    Uses multiple encoders in parallel to create a robust, information-rich
    spike representation.
    """
    
    def __init__(self, window: float = 100.0):
        """
        Initialize ensemble encoder.
        
        Args:
            window: Encoding time window (ms)
        """
        self.window = window
        
        # Initialize multiple encoders
        self.rate_encoder = AdaptiveRateEncoder(window=window)
        self.burst_encoder = BurstEncoder(window=window)
        self.phase_encoder = PhaseEncoder(window=window)
        
    def encode(self, value: float) -> Dict[str, np.ndarray]:
        """
        Encode value using multiple strategies.
        
        Args:
            value: Normalized input value [0, 1]
            
        Returns:
            Dictionary with spike trains from each encoder
        """
        return {
            'rate': self.rate_encoder.encode(value),
            'burst': self.burst_encoder.encode(value),
            'phase': self.phase_encoder.encode(value)
        }
    
    def encode_and_merge(self, value: float, 
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
            weights = {'rate': 1.0, 'burst': 1.0, 'phase': 1.0}
            
        encoded = self.encode(value)
        
        # Merge spike trains with weighted sampling
        all_spikes = []
        
        for encoder_type, spike_train in encoded.items():
            weight = weights.get(encoder_type, 1.0)
            
            # Sample spikes based on weight
            n_keep = int(len(spike_train) * weight)
            if n_keep > 0 and len(spike_train) > 0:
                keep_indices = np.random.choice(len(spike_train), 
                                               size=min(n_keep, len(spike_train)),
                                               replace=False)
                all_spikes.extend(spike_train[keep_indices])
                
        return np.sort(np.array(all_spikes))


class InformationTheoreticEncoder:
    """
    Encoding optimized for maximum information transmission.
    
    Uses information-theoretic principles to optimize spike patterns
    for maximal information content.
    """
    
    def __init__(self, n_neurons: int = 100,
                 window: float = 100.0,
                 target_entropy: float = 0.8):
        """
        Initialize information-theoretic encoder.
        
        Args:
            n_neurons: Number of encoding neurons
            window: Encoding time window (ms)
            target_entropy: Target entropy (relative, 0-1)
        """
        self.n_neurons = n_neurons
        self.window = window
        self.target_entropy = target_entropy
        
    def encode(self, value: float) -> np.ndarray:
        """
        Encode value optimizing for information content.
        
        Args:
            value: Input value [0, 1]
            
        Returns:
            Spike train with target entropy
        """
        # Map value to target spike count
        target_spikes = int(value * self.n_neurons * 0.5)
        
        # Generate candidate spike patterns
        best_pattern = None
        best_entropy = 0
        
        for _ in range(10):  # Try multiple patterns
            spike_times = np.sort(np.random.uniform(0, self.window, target_spikes))
            
            # Calculate entropy of ISI distribution
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                pattern_entropy = self._calculate_entropy(isis)
                
                if abs(pattern_entropy - self.target_entropy) < abs(best_entropy - self.target_entropy):
                    best_entropy = pattern_entropy
                    best_pattern = spike_times
                    
        return best_pattern if best_pattern is not None else np.array([])
    
    def _calculate_entropy(self, isis: np.ndarray) -> float:
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
        
        return ent / max_entropy if max_entropy > 0 else 0.0


class SpikeTrainAnalyzer:
    """
    Analyze and evaluate spike train quality.
    
    Provides metrics for assessing encoding quality and information content.
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
        spike_count = len(spike_train)
        
        if spike_count == 0:
            return SpikeTrainMetrics(
                spike_count=0,
                firing_rate=0.0,
                inter_spike_interval_mean=0.0,
                inter_spike_interval_cv=0.0,
                information_content=0.0
            )
        
        # Firing rate
        firing_rate = spike_count / (window / 1000.0)  # Hz
        
        # ISI statistics
        if spike_count > 1:
            isis = np.diff(spike_train)
            isi_mean = np.mean(isis)
            isi_std = np.std(isis)
            isi_cv = isi_std / isi_mean if isi_mean > 0 else 0.0
        else:
            isi_mean = 0.0
            isi_cv = 0.0
        
        # Information content (simplified)
        # Use firing rate variability as proxy
        information = np.log2(spike_count + 1) * (1 - isi_cv) if spike_count > 1 else 0.0
        
        return SpikeTrainMetrics(
            spike_count=spike_count,
            firing_rate=firing_rate,
            inter_spike_interval_mean=isi_mean,
            inter_spike_interval_cv=isi_cv,
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
        
        for encoder_name, encoder in encoders.items():
            metrics_list = []
            
            for value in test_values:
                spike_train = encoder.encode(value)
                metrics = SpikeTrainAnalyzer.analyze(spike_train, window)
                metrics_list.append(metrics)
            
            # Aggregate metrics
            results[encoder_name] = {
                'avg_spike_count': np.mean([m.spike_count for m in metrics_list]),
                'avg_firing_rate': np.mean([m.firing_rate for m in metrics_list]),
                'avg_isi_cv': np.mean([m.inter_spike_interval_cv for m in metrics_list]),
                'avg_information': np.mean([m.information_content for m in metrics_list])
            }
        
        return results


# Example usage
if __name__ == "__main__":
    print("Advanced Spike Encoding Strategies")
    print("="*60)
    
    # Test different encoders
    test_values = np.random.random(100)
    window = 100.0
    
    encoders = {
        'Adaptive Rate': AdaptiveRateEncoder(window=window),
        'Burst': BurstEncoder(window=window),
        'Phase': PhaseEncoder(window=window),
        'Ensemble': EnsembleEncoder(window=window)
    }
    
    print("\nComparing encoding strategies...")
    
    # Compare encoders
    comparison = SpikeTrainAnalyzer.compare_encoders(
        {k: v for k, v in encoders.items() if k != 'Ensemble'},
        test_values[:10],
        window
    )
    
    print("\nEncoder Comparison Results:")
    for encoder_name, metrics in comparison.items():
        print(f"\n{encoder_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.3f}")
    
    # Test ensemble encoding
    print("\n" + "="*60)
    print("Testing Ensemble Encoder:")
    ensemble = EnsembleEncoder(window=window)
    
    test_value = 0.75
    encoded = ensemble.encode(test_value)
    
    print(f"\nEncoding value {test_value}:")
    for encoder_type, spike_train in encoded.items():
        metrics = SpikeTrainAnalyzer.analyze(spike_train, window)
        print(f"  {encoder_type}: {metrics.spike_count} spikes, "
              f"{metrics.firing_rate:.1f} Hz")
