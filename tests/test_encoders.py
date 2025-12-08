"""
Unit tests for spike encoders.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
Date: December 2025
License: MIT
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
    sys.path.insert(0, str(src_path))

from encoders import (RateEncoder, TemporalEncoder, PopulationEncoder,
                          LatencyEncoder, TransactionEncoder)
from advanced_encoders import (AdaptiveRateEncoder, BurstEncoder,
                                   PhaseEncoder, RankOrderEncoder,
                                   EnsembleEncoder, SpikeTrainAnalyzer)


class TestRateEncoder(unittest.TestCase):
    """Test cases for Rate Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = RateEncoder(max_rate=100.0, window=100.0)
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.max_rate, 100.0)
        self.assertEqual(self.encoder.window, 100.0)
    
    def test_zero_value(self):
        """Test encoding of zero value."""
        spikes = self.encoder.encode(0.0)
        self.assertIsInstance(spikes, np.ndarray)
        self.assertEqual(len(spikes), 0)
    
    def test_max_value(self):
        """Test encoding of maximum value."""
        spikes = self.encoder.encode(1.0)
        self.assertIsInstance(spikes, np.ndarray)
        self.assertGreater(len(spikes), 0)
        # All spikes should be within time window
        self.assertTrue(np.all(spikes >= 0))
        self.assertTrue(np.all(spikes <= self.encoder.window))
    
    def test_medium_value(self):
        """Test encoding of medium value."""
        spikes = self.encoder.encode(0.5)
        self.assertIsInstance(spikes, np.ndarray)
        # Spike count should be roughly proportional to value
        expected_spikes = 0.5 * self.encoder.max_rate * self.encoder.window / 1000.0
        self.assertLess(abs(len(spikes) - expected_spikes), expected_spikes * 0.5)
    
    def test_negative_value(self):
        """Test encoding handles negative values."""
        spikes = self.encoder.encode(-0.5)
        self.assertIsInstance(spikes, np.ndarray)
        # Should clip to zero
        self.assertEqual(len(spikes), 0)


class TestTemporalEncoder(unittest.TestCase):
    """Test cases for Temporal Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = TemporalEncoder(window=100.0)
    
    def test_early_time(self):
        """Test encoding of early timestamp."""
        timestamp = 1000.0
        spikes = self.encoder.encode(timestamp)
        self.assertIsInstance(spikes, np.ndarray)
        self.assertGreater(len(spikes), 0)
        # Early times should produce early spikes
        self.assertLess(np.mean(spikes), self.encoder.window / 2)
    
    def test_late_time(self):
        """Test encoding of late timestamp."""
        timestamp = 80000.0
        spikes = self.encoder.encode(timestamp)
        self.assertIsInstance(spikes, np.ndarray)
        # Late times should produce later spikes
        if len(spikes) > 0:
            self.assertGreater(np.mean(spikes), self.encoder.window / 2)


class TestPopulationEncoder(unittest.TestCase):
    """Test cases for Population Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = PopulationEncoder(n_neurons=10, window=100.0)
    
    def test_single_location(self):
        """Test encoding of single location."""
        lat, lon = 40.7128, -74.0060  # New York
        spike_trains = self.encoder.encode(lat, lon)
        
        self.assertIsInstance(spike_trains, dict)
        self.assertEqual(len(spike_trains), self.encoder.n_neurons)
        
        # Check that some neurons are active
        active_neurons = sum(1 for spikes in spike_trains.values() if len(spikes) > 0)
        self.assertGreater(active_neurons, 0)
    
    def test_different_locations(self):
        """Test that different locations produce different patterns."""
        loc1_spikes = self.encoder.encode(40.7128, -74.0060)  # New York
        loc2_spikes = self.encoder.encode(51.5074, -0.1278)   # London
        
        # Patterns should be different
        active1 = set(k for k, v in loc1_spikes.items() if len(v) > 0)
        active2 = set(k for k, v in loc2_spikes.items() if len(v) > 0)
        
        self.assertNotEqual(active1, active2)


class TestLatencyEncoder(unittest.TestCase):
    """Test cases for Latency Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = LatencyEncoder(window=100.0)
    
    def test_high_value_early_spike(self):
        """Test that high values produce early spikes."""
        spikes_high = self.encoder.encode(1.0)
        self.assertIsInstance(spikes_high, np.ndarray)
        self.assertEqual(len(spikes_high), 1)
        self.assertLess(spikes_high[0], self.encoder.window / 2)
    
    def test_low_value_late_spike(self):
        """Test that low values produce late spikes."""
        spikes_low = self.encoder.encode(0.1)
        self.assertIsInstance(spikes_low, np.ndarray)
        self.assertEqual(len(spikes_low), 1)
        self.assertGreater(spikes_low[0], self.encoder.window / 2)
    
    def test_ordering(self):
        """Test that spike latencies preserve value ordering."""
        spikes1 = self.encoder.encode(0.3)
        spikes2 = self.encoder.encode(0.7)
        
        # Higher value should have earlier spike
        self.assertLess(spikes2[0], spikes1[0])


class TestTransactionEncoder(unittest.TestCase):
    """Test cases for Transaction Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = TransactionEncoder()
    
    def test_encode_transaction(self):
        """Test encoding of complete transaction."""
        transaction = {
            'amount': 1000.0,
            'timestamp': 43200,  # noon
            'latitude': 40.7128,
            'longitude': -74.0060,
            'category': 'groceries'
        }
        
        encoded = self.encoder.encode_transaction(transaction)
        
        self.assertIsNotNone(encoded)
        self.assertGreater(len(encoded.spike_times), 0)
        self.assertGreater(len(encoded.neuron_ids), 0)
        self.assertEqual(len(encoded.spike_times), len(encoded.neuron_ids))


class TestAdaptiveRateEncoder(unittest.TestCase):
    """Test cases for Adaptive Rate Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = AdaptiveRateEncoder(window=100.0)
    
    def test_adaptation(self):
        """Test that encoder adapts to input statistics."""
        initial_mean = self.encoder.running_mean
        
        # Encode several values
        for value in [0.5, 0.6, 0.7, 0.8]:
            self.encoder.encode(value)
        
        # Running mean should have updated
        self.assertNotEqual(self.encoder.running_mean, initial_mean)
        self.assertGreater(self.encoder.n_samples, 0)
    
    def test_spike_generation(self):
        """Test that spikes are generated within window."""
        spikes = self.encoder.encode(0.5)
        self.assertIsInstance(spikes, np.ndarray)
        if len(spikes) > 0:
            self.assertTrue(np.all(spikes >= 0))
            self.assertTrue(np.all(spikes <= self.encoder.window))


class TestBurstEncoder(unittest.TestCase):
    """Test cases for Burst Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = BurstEncoder(window=100.0, burst_threshold=0.7)
    
    def test_high_value_burst(self):
        """Test that high values produce bursts."""
        spikes = self.encoder.encode(0.9)
        self.assertIsInstance(spikes, np.ndarray)
        # Should produce a burst
        self.assertGreaterEqual(len(spikes), self.encoder.burst_size)
    
    def test_low_value_no_burst(self):
        """Test that low values don't produce bursts."""
        spikes = self.encoder.encode(0.1)
        self.assertIsInstance(spikes, np.ndarray)
        # Should produce few or no spikes
        self.assertLess(len(spikes), self.encoder.burst_size)


class TestPhaseEncoder(unittest.TestCase):
    """Test cases for Phase Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = PhaseEncoder(window=100.0, oscillation_freq=10.0)
    
    def test_phase_encoding(self):
        """Test phase encoding produces regular spikes."""
        spikes = self.encoder.encode(0.5)
        self.assertIsInstance(spikes, np.ndarray)
        
        # Should produce multiple spikes (one per cycle)
        expected_cycles = int(self.encoder.window / self.encoder.period)
        self.assertGreater(len(spikes), 0)
        self.assertLessEqual(len(spikes), expected_cycles)


class TestRankOrderEncoder(unittest.TestCase):
    """Test cases for Rank Order Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = RankOrderEncoder(n_features=5, window=100.0)
    
    def test_rank_ordering(self):
        """Test that features are ordered by value."""
        features = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        spike_trains = self.encoder.encode(features)
        
        self.assertIsInstance(spike_trains, dict)
        
        # Feature 3 (value 0.9) should spike first
        # Feature 0 (value 0.1) should spike last or not at all
        if 3 in spike_trains and len(spike_trains[3]) > 0:
            if 0 in spike_trains and len(spike_trains[0]) > 0:
                self.assertLess(spike_trains[3][0], spike_trains[0][0])


class TestEnsembleEncoder(unittest.TestCase):
    """Test cases for Ensemble Encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = EnsembleEncoder(window=100.0)
    
    def test_multiple_encodings(self):
        """Test that ensemble produces multiple encodings."""
        encoded = self.encoder.encode(0.5)
        
        self.assertIsInstance(encoded, dict)
        self.assertIn('rate', encoded)
        self.assertIn('burst', encoded)
        self.assertIn('phase', encoded)
    
    def test_merge_spike_trains(self):
        """Test merging of spike trains."""
        merged = self.encoder.encode_and_merge(0.5)
        
        self.assertIsInstance(merged, np.ndarray)
        if len(merged) > 0:
            self.assertTrue(np.all(merged >= 0))
            self.assertTrue(np.all(merged <= self.encoder.window))


class TestSpikeTrainAnalyzer(unittest.TestCase):
    """Test cases for Spike Train Analyzer."""
    
    def test_analyze_empty_train(self):
        """Test analysis of empty spike train."""
        spikes = np.array([])
        metrics = SpikeTrainAnalyzer.analyze(spikes, window=100.0)
        
        self.assertEqual(metrics.spike_count, 0)
        self.assertEqual(metrics.firing_rate, 0.0)
    
    def test_analyze_spike_train(self):
        """Test analysis of spike train."""
        spikes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        metrics = SpikeTrainAnalyzer.analyze(spikes, window=100.0)
        
        self.assertEqual(metrics.spike_count, 5)
        self.assertGreater(metrics.firing_rate, 0)
        self.assertGreater(metrics.inter_spike_interval_mean, 0)
    
    def test_compare_encoders(self):
        """Test encoder comparison."""
        encoders = {
            'rate': RateEncoder(window=100.0),
            'latency': LatencyEncoder(window=100.0)
        }
        
        test_values = np.array([0.3, 0.5, 0.7])
        results = SpikeTrainAnalyzer.compare_encoders(encoders, test_values, 100.0)
        
        self.assertIsInstance(results, dict)
        self.assertIn('rate', results)
        self.assertIn('latency', results)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRateEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestPopulationEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestLatencyEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestTransactionEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveRateEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestBurstEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestPhaseEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestRankOrderEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestSpikeTrainAnalyzer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
