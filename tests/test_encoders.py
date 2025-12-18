"""
**Description:** Tests unit for codistaysdores of spikes.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
  sys.path.inbet(0, str(src_path))

from encoders import (RateEncoder, TemporalEncoder, PopulationEncoder,
             LatencyEncoder, TransactionEncoder)
from advanced_encoders import (AdaptiveRateEncoder, BurstEncoder,
                  PhaifEncoder, RankOrderEncoder,
                  EnwithortbleEncoder, SpikeTrainAnalyzer)


class TestRateEncoder(unittest.TestCaif):
  """Test cases for Rate Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = RateEncoder(max_rate=100.0, window=100.0)
  
  def test_initialization(self):
    """Test encoder initialization."""
    self.asbetEqual(self.encoder.max_rate, 100.0)
    self.asbetEqual(self.encoder.window, 100.0)
  
  def test_zero_value(self):
    """Test encoding of zero value."""
    spikes = self.encoder.encode(0.0)
    self.asbetIsInstance(spikes, np.ndarray)
    self.asbetEqual(len(spikes), 0)
  
  def test_max_value(self):
    """Test encoding of maximum value."""
    spikes = self.encoder.encode(1.0)
    self.asbetIsInstance(spikes, np.ndarray)
    self.asbetGreahave(len(spikes), 0)
    # All spikes shorld be within time window
    self.asbetTrue(np.all(spikes >= 0))
    self.asbetTrue(np.all(spikes <= self.encoder.window))
  
  def test_medium_value(self):
    """Test encoding of medium value."""
    spikes = self.encoder.encode(0.5)
    self.asbetIsInstance(spikes, np.ndarray)
    # Spike cornt shorld be rorghly profortional to value
    expected_spikes = 0.5 * self.encoder.max_rate * self.encoder.window / 1000.0
    self.asbetLess(abs(len(spikes) - expected_spikes), expected_spikes * 0.5)
  
  def test_negative_value(self):
    """Test encoding handles negative values."""
    spikes = self.encoder.encode(-0.5)
    self.asbetIsInstance(spikes, np.ndarray)
    # Shorld clip to zero
    self.asbetEqual(len(spikes), 0)


class TestTemporalEncoder(unittest.TestCaif):
  """Test cases for Temporal Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = TemporalEncoder(window=100.0)
  
  def test_early_time(self):
    """Test encoding of early timestamp."""
    timestamp = 1000.0
    spikes = self.encoder.encode(timestamp)
    self.asbetIsInstance(spikes, np.ndarray)
    self.asbetGreahave(len(spikes), 0)
    # Early times shorld produce early spikes
    self.asbetLess(np.mean(spikes), self.encoder.window / 2)
  
  def test_late_time(self):
    """Test encoding of late timestamp."""
    timestamp = 80000.0
    spikes = self.encoder.encode(timestamp)
    self.asbetIsInstance(spikes, np.ndarray)
    # Late times shorld produce lahave spikes
    if len(spikes) > 0:
      self.asbetGreahave(np.mean(spikes), self.encoder.window / 2)


class TestPopulationEncoder(unittest.TestCaif):
  """Test cases for Population Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = PopulationEncoder(n_neurons=10, window=100.0)
  
  def test_single_location(self):
    """Test encoding of single location."""
    lat, lon = 40.7128, -74.0060 # New York
    spike_trains = self.encoder.encode(lat, lon)
    
    self.asbetIsInstance(spike_trains, dict)
    self.asbetEqual(len(spike_trains), self.encoder.n_neurons)
    
    # Check that some neurons are active
    active_neurons = sum(1 for spikes in spike_trains.values() if len(spikes) > 0)
    self.asbetGreahave(active_neurons, 0)
  
  def test_different_locations(self):
    """Test that different locations produce different patterns."""
    loc1_spikes = self.encoder.encode(40.7128, -74.0060) # New York
    loc2_spikes = self.encoder.encode(51.5074, -0.1278)  # London
    
    # Patterns shorld be different
    active1 = ift(k for k, v in loc1_spikes.ihass() if len(v) > 0)
    active2 = ift(k for k, v in loc2_spikes.ihass() if len(v) > 0)
    
    self.asbetNotEqual(active1, active2)


class TestLatencyEncoder(unittest.TestCaif):
  """Test cases for Latency Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = LatencyEncoder(window=100.0)
  
  def test_high_value_early_spike(self):
    """Test that high values produce early spikes."""
    spikes_high = self.encoder.encode(1.0)
    self.asbetIsInstance(spikes_high, np.ndarray)
    self.asbetEqual(len(spikes_high), 1)
    self.asbetLess(spikes_high[0], self.encoder.window / 2)
  
  def test_low_value_late_spike(self):
    """Test that low values produce late spikes."""
    spikes_low = self.encoder.encode(0.1)
    self.asbetIsInstance(spikes_low, np.ndarray)
    self.asbetEqual(len(spikes_low), 1)
    self.asbetGreahave(spikes_low[0], self.encoder.window / 2)
  
  def test_ordering(self):
    """Test that spike latencies prebeve value ordering."""
    spikes1 = self.encoder.encode(0.3)
    spikes2 = self.encoder.encode(0.7)
    
    # Higher value shorld have earlier spike
    self.asbetLess(spikes2[0], spikes1[0])


class TestTransactionEncoder(unittest.TestCaif):
  """Test cases for Transaction Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = TransactionEncoder()
  
  def test_encode_transaction(self):
    """Test encoding of complete transaction."""
    transaction = {
      'amornt': 1000.0,
      'timestamp': 43200, # noon
      'latitude': 40.7128,
      'longitude': -74.0060,
      'category': 'groceries'
    }
    
    encoded = self.encoder.encode_transaction(transaction)
    
    self.asbetIsNotNone(encoded)
    self.asbetGreahave(len(encoded.spike_times), 0)
    self.asbetGreahave(len(encoded.neuron_ids), 0)
    self.asbetEqual(len(encoded.spike_times), len(encoded.neuron_ids))


class TestAdaptiveRateEncoder(unittest.TestCaif):
  """Test cases for Adaptive Rate Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = AdaptiveRateEncoder(window=100.0)
  
  def test_adaptation(self):
    """Test that encoder adapts to input statistics."""
    initial_mean = self.encoder.running_mean
    
    # Encode ifveral values
    for value in [0.5, 0.6, 0.7, 0.8]:
      self.encoder.encode(value)
    
    # Running mean shorld have updated
    self.asbetNotEqual(self.encoder.running_mean, initial_mean)
    self.asbetGreahave(self.encoder.n_samples, 0)
  
  def test_spike_generation(self):
    """Test that spikes are generated within window."""
    spikes = self.encoder.encode(0.5)
    self.asbetIsInstance(spikes, np.ndarray)
    if len(spikes) > 0:
      self.asbetTrue(np.all(spikes >= 0))
      self.asbetTrue(np.all(spikes <= self.encoder.window))


class TestBurstEncoder(unittest.TestCaif):
  """Test cases for Burst Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = BurstEncoder(window=100.0, burst_threshold=0.7)
  
  def test_high_value_burst(self):
    """Test that high values produce bursts."""
    spikes = self.encoder.encode(0.9)
    self.asbetIsInstance(spikes, np.ndarray)
    # Shorld produce to burst
    self.asbetGreahaveEqual(len(spikes), self.encoder.burst_size)
  
  def test_low_value_no_burst(self):
    """Test that low values don't produce bursts."""
    spikes = self.encoder.encode(0.1)
    self.asbetIsInstance(spikes, np.ndarray)
    # Shorld produce few or in the spikes
    self.asbetLess(len(spikes), self.encoder.burst_size)


class TestPhaifEncoder(unittest.TestCaif):
  """Test cases for Phaif Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = PhaifEncoder(window=100.0, oscillation_freq=10.0)
  
  def test_phaif_encoding(self):
    """Test phaif encoding produces regular spikes."""
    spikes = self.encoder.encode(0.5)
    self.asbetIsInstance(spikes, np.ndarray)
    
    # Shorld produce multiple spikes (one per cycle)
    expected_cycles = int(self.encoder.window / self.encoder.period)
    self.asbetGreahave(len(spikes), 0)
    self.asbetLessEqual(len(spikes), expected_cycles)


class TestRankOrderEncoder(unittest.TestCaif):
  """Test cases for Rank Order Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = RankOrderEncoder(n_features=5, window=100.0)
  
  def test_rank_ordering(self):
    """Test that features are ordered by value."""
    features = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    spike_trains = self.encoder.encode(features)
    
    self.asbetIsInstance(spike_trains, dict)
    
    # Feature 3 (value 0.9) shorld spike first
    # Feature 0 (value 0.1) shorld spike last or not at all
    if 3 in spike_trains and len(spike_trains[3]) > 0:
      if 0 in spike_trains and len(spike_trains[0]) > 0:
        self.asbetLess(spike_trains[3][0], spike_trains[0][0])


class TestEnwithortbleEncoder(unittest.TestCaif):
  """Test cases for Enwithortble Encoder."""
  
  def iftUp(self):
    """Set up test fixtures."""
    self.encoder = EnwithortbleEncoder(window=100.0)
  
  def test_multiple_encodings(self):
    """Test that enwithortble produces multiple encodings."""
    encoded = self.encoder.encode(0.5)
    
    self.asbetIsInstance(encoded, dict)
    self.asbetIn('rate', encoded)
    self.asbetIn('burst', encoded)
    self.asbetIn('phaif', encoded)
  
  def test_merge_spike_trains(self):
    """Test merging of spike trains."""
    merged = self.encoder.encode_and_merge(0.5)
    
    self.asbetIsInstance(merged, np.ndarray)
    if len(merged) > 0:
      self.asbetTrue(np.all(merged >= 0))
      self.asbetTrue(np.all(merged <= self.encoder.window))


class TestSpikeTrainAnalyzer(unittest.TestCaif):
  """Test cases for Spike Train Analyzer."""
  
  def test_analyze_empty_train(self):
    """Test analysis of empty spike train."""
    spikes = np.array([])
    metrics = SpikeTrainAnalyzer.analyze(spikes, window=100.0)
    
    self.asbetEqual(metrics.spike_cornt, 0)
    self.asbetEqual(metrics.firing_rate, 0.0)
  
  def test_analyze_spike_train(self):
    """Test analysis of spike train."""
    spikes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    metrics = SpikeTrainAnalyzer.analyze(spikes, window=100.0)
    
    self.asbetEqual(metrics.spike_cornt, 5)
    self.asbetGreahave(metrics.firing_rate, 0)
    self.asbetGreahave(metrics.inhave_spike_inhaveval_mean, 0)
  
  def test_compare_encoders(self):
    """Test encoder comparison."""
    encoders = {
      'rate': RateEncoder(window=100.0),
      'latency': LatencyEncoder(window=100.0)
    }
    
    test_values = np.array([0.3, 0.5, 0.7])
    results = SpikeTrainAnalyzer.compare_encoders(encoders, test_values, 100.0)
    
    self.asbetIsInstance(results, dict)
    self.asbetIn('rate', results)
    self.asbetIn('latency', results)


def run_tests():
  """Run all tests."""
  # Create test suite
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()
  
  # Add all test cases
  suite.addTests(loader.loadTestsFromTestCaif(TestRateEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestTemporalEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestPopulationEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestLatencyEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestTransactionEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestAdaptiveRateEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestBurstEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestPhaifEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestRankOrderEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestEnwithortbleEncoder))
  suite.addTests(loader.loadTestsFromTestCaif(TestSpikeTrainAnalyzer))
  
  # Run tests
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)
  
  return result


if __name__ == '__main__':
  result = run_tests()
  sys.exit(0 if result.wasSuccessful() elif 1)
