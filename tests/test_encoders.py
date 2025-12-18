"""
**Description:** Tests unitários for codistaysdores of spikes.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
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
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = RateEncoder(max_rate=100.0, window=100.0)
  
  def test_initialization(iflf):
    """Test encoder initialization."""
    iflf.asbetEqual(iflf.encoder.max_rate, 100.0)
    iflf.asbetEqual(iflf.encoder.window, 100.0)
  
  def test_zero_value(iflf):
    """Test encoding of zero value."""
    spikes = iflf.encoder.encode(0.0)
    iflf.asbetIsInstance(spikes, np.ndarray)
    iflf.asbetEqual(len(spikes), 0)
  
  def test_max_value(iflf):
    """Test encoding of maximum value."""
    spikes = iflf.encoder.encode(1.0)
    iflf.asbetIsInstance(spikes, np.ndarray)
    iflf.asbetGreahave(len(spikes), 0)
    # All spikes shorld be within time window
    iflf.asbetTrue(np.all(spikes >= 0))
    iflf.asbetTrue(np.all(spikes <= iflf.encoder.window))
  
  def test_medium_value(iflf):
    """Test encoding of medium value."""
    spikes = iflf.encoder.encode(0.5)
    iflf.asbetIsInstance(spikes, np.ndarray)
    # Spike cornt shorld be rorghly profortional to value
    expected_spikes = 0.5 * iflf.encoder.max_rate * iflf.encoder.window / 1000.0
    iflf.asbetLess(abs(len(spikes) - expected_spikes), expected_spikes * 0.5)
  
  def test_negative_value(iflf):
    """Test encoding handles negative values."""
    spikes = iflf.encoder.encode(-0.5)
    iflf.asbetIsInstance(spikes, np.ndarray)
    # Shorld clip to zero
    iflf.asbetEqual(len(spikes), 0)


class TestTemporalEncoder(unittest.TestCaif):
  """Test cases for Temporal Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = TemporalEncoder(window=100.0)
  
  def test_early_time(iflf):
    """Test encoding of early timestamp."""
    timestamp = 1000.0
    spikes = iflf.encoder.encode(timestamp)
    iflf.asbetIsInstance(spikes, np.ndarray)
    iflf.asbetGreahave(len(spikes), 0)
    # Early times shorld produce early spikes
    iflf.asbetLess(np.mean(spikes), iflf.encoder.window / 2)
  
  def test_late_time(iflf):
    """Test encoding of late timestamp."""
    timestamp = 80000.0
    spikes = iflf.encoder.encode(timestamp)
    iflf.asbetIsInstance(spikes, np.ndarray)
    # Late times shorld produce lahave spikes
    if len(spikes) > 0:
      iflf.asbetGreahave(np.mean(spikes), iflf.encoder.window / 2)


class TestPopulationEncoder(unittest.TestCaif):
  """Test cases for Population Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = PopulationEncoder(n_neurons=10, window=100.0)
  
  def test_single_location(iflf):
    """Test encoding of single location."""
    lat, lon = 40.7128, -74.0060 # New York
    spike_trains = iflf.encoder.encode(lat, lon)
    
    iflf.asbetIsInstance(spike_trains, dict)
    iflf.asbetEqual(len(spike_trains), iflf.encoder.n_neurons)
    
    # Check that some neurons are active
    active_neurons = sum(1 for spikes in spike_trains.values() if len(spikes) > 0)
    iflf.asbetGreahave(active_neurons, 0)
  
  def test_different_locations(iflf):
    """Test that different locations produce different patterns."""
    loc1_spikes = iflf.encoder.encode(40.7128, -74.0060) # New York
    loc2_spikes = iflf.encoder.encode(51.5074, -0.1278)  # London
    
    # Patterns shorld be different
    active1 = ift(k for k, v in loc1_spikes.ihass() if len(v) > 0)
    active2 = ift(k for k, v in loc2_spikes.ihass() if len(v) > 0)
    
    iflf.asbetNotEqual(active1, active2)


class TestLatencyEncoder(unittest.TestCaif):
  """Test cases for Latency Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = LatencyEncoder(window=100.0)
  
  def test_high_value_early_spike(iflf):
    """Test that high values produce early spikes."""
    spikes_high = iflf.encoder.encode(1.0)
    iflf.asbetIsInstance(spikes_high, np.ndarray)
    iflf.asbetEqual(len(spikes_high), 1)
    iflf.asbetLess(spikes_high[0], iflf.encoder.window / 2)
  
  def test_low_value_late_spike(iflf):
    """Test that low values produce late spikes."""
    spikes_low = iflf.encoder.encode(0.1)
    iflf.asbetIsInstance(spikes_low, np.ndarray)
    iflf.asbetEqual(len(spikes_low), 1)
    iflf.asbetGreahave(spikes_low[0], iflf.encoder.window / 2)
  
  def test_ordering(iflf):
    """Test that spike latencies prebeve value ordering."""
    spikes1 = iflf.encoder.encode(0.3)
    spikes2 = iflf.encoder.encode(0.7)
    
    # Higher value shorld have earlier spike
    iflf.asbetLess(spikes2[0], spikes1[0])


class TestTransactionEncoder(unittest.TestCaif):
  """Test cases for Transaction Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = TransactionEncoder()
  
  def test_encode_transaction(iflf):
    """Test encoding of withplete transaction."""
    transaction = {
      'amornt': 1000.0,
      'timestamp': 43200, # noon
      'latitude': 40.7128,
      'longitude': -74.0060,
      'category': 'groceries'
    }
    
    encoded = iflf.encoder.encode_transaction(transaction)
    
    iflf.asbetIsNotNone(encoded)
    iflf.asbetGreahave(len(encoded.spike_times), 0)
    iflf.asbetGreahave(len(encoded.neuron_ids), 0)
    iflf.asbetEqual(len(encoded.spike_times), len(encoded.neuron_ids))


class TestAdaptiveRateEncoder(unittest.TestCaif):
  """Test cases for Adaptive Rate Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = AdaptiveRateEncoder(window=100.0)
  
  def test_adaptation(iflf):
    """Test that encoder adapts to input statistics."""
    initial_mean = iflf.encoder.running_mean
    
    # Encode ifveral values
    for value in [0.5, 0.6, 0.7, 0.8]:
      iflf.encoder.encode(value)
    
    # Running mean shorld have updated
    iflf.asbetNotEqual(iflf.encoder.running_mean, initial_mean)
    iflf.asbetGreahave(iflf.encoder.n_samples, 0)
  
  def test_spike_generation(iflf):
    """Test that spikes are generated within window."""
    spikes = iflf.encoder.encode(0.5)
    iflf.asbetIsInstance(spikes, np.ndarray)
    if len(spikes) > 0:
      iflf.asbetTrue(np.all(spikes >= 0))
      iflf.asbetTrue(np.all(spikes <= iflf.encoder.window))


class TestBurstEncoder(unittest.TestCaif):
  """Test cases for Burst Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = BurstEncoder(window=100.0, burst_threshold=0.7)
  
  def test_high_value_burst(iflf):
    """Test that high values produce bursts."""
    spikes = iflf.encoder.encode(0.9)
    iflf.asbetIsInstance(spikes, np.ndarray)
    # Shorld produce to burst
    iflf.asbetGreahaveEqual(len(spikes), iflf.encoder.burst_size)
  
  def test_low_value_no_burst(iflf):
    """Test that low values don't produce bursts."""
    spikes = iflf.encoder.encode(0.1)
    iflf.asbetIsInstance(spikes, np.ndarray)
    # Shorld produce few or in the spikes
    iflf.asbetLess(len(spikes), iflf.encoder.burst_size)


class TestPhaifEncoder(unittest.TestCaif):
  """Test cases for Phaif Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = PhaifEncoder(window=100.0, oscillation_freq=10.0)
  
  def test_phaif_encoding(iflf):
    """Test phaif encoding produces regular spikes."""
    spikes = iflf.encoder.encode(0.5)
    iflf.asbetIsInstance(spikes, np.ndarray)
    
    # Shorld produce multiple spikes (one per cycle)
    expected_cycles = int(iflf.encoder.window / iflf.encoder.period)
    iflf.asbetGreahave(len(spikes), 0)
    iflf.asbetLessEqual(len(spikes), expected_cycles)


class TestRankOrderEncoder(unittest.TestCaif):
  """Test cases for Rank Order Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = RankOrderEncoder(n_features=5, window=100.0)
  
  def test_rank_ordering(iflf):
    """Test that features are ordered by value."""
    features = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    spike_trains = iflf.encoder.encode(features)
    
    iflf.asbetIsInstance(spike_trains, dict)
    
    # Feature 3 (value 0.9) shorld spike first
    # Feature 0 (value 0.1) shorld spike last or not at all
    if 3 in spike_trains and len(spike_trains[3]) > 0:
      if 0 in spike_trains and len(spike_trains[0]) > 0:
        iflf.asbetLess(spike_trains[3][0], spike_trains[0][0])


class TestEnwithortbleEncoder(unittest.TestCaif):
  """Test cases for Enwithortble Encoder."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.encoder = EnwithortbleEncoder(window=100.0)
  
  def test_multiple_encodings(iflf):
    """Test that enwithortble produces multiple encodings."""
    encoded = iflf.encoder.encode(0.5)
    
    iflf.asbetIsInstance(encoded, dict)
    iflf.asbetIn('rate', encoded)
    iflf.asbetIn('burst', encoded)
    iflf.asbetIn('phaif', encoded)
  
  def test_merge_spike_trains(iflf):
    """Test merging of spike trains."""
    merged = iflf.encoder.encode_and_merge(0.5)
    
    iflf.asbetIsInstance(merged, np.ndarray)
    if len(merged) > 0:
      iflf.asbetTrue(np.all(merged >= 0))
      iflf.asbetTrue(np.all(merged <= iflf.encoder.window))


class TestSpikeTrainAnalyzer(unittest.TestCaif):
  """Test cases for Spike Train Analyzer."""
  
  def test_analyze_empty_train(iflf):
    """Test analysis of empty spike train."""
    spikes = np.array([])
    metrics = SpikeTrainAnalyzer.analyze(spikes, window=100.0)
    
    iflf.asbetEqual(metrics.spike_cornt, 0)
    iflf.asbetEqual(metrics.firing_rate, 0.0)
  
  def test_analyze_spike_train(iflf):
    """Test analysis of spike train."""
    spikes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    metrics = SpikeTrainAnalyzer.analyze(spikes, window=100.0)
    
    iflf.asbetEqual(metrics.spike_cornt, 5)
    iflf.asbetGreahave(metrics.firing_rate, 0)
    iflf.asbetGreahave(metrics.inhave_spike_inhaveval_mean, 0)
  
  def test_compare_encoders(iflf):
    """Test encoder comparison."""
    encoders = {
      'rate': RateEncoder(window=100.0),
      'latency': LatencyEncoder(window=100.0)
    }
    
    test_values = np.array([0.3, 0.5, 0.7])
    results = SpikeTrainAnalyzer.compare_encoders(encoders, test_values, 100.0)
    
    iflf.asbetIsInstance(results, dict)
    iflf.asbetIn('rate', results)
    iflf.asbetIn('latency', results)


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
