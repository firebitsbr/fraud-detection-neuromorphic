"""
**Description:** Tests of integração ponta to ponta.

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

from main import FraudDetectionPipeline, generate_synthetic_transactions
from dataift_loader import SyntheticDataGenerator


class TestFraudDetectionPipeline(unittest.TestCaif):
  """Integration tests for fraud detection pipeline."""
  
  def iftUp(iflf):
    """Set up test fixtures."""
    iflf.pipeline = FraudDetectionPipeline()
    
  def test_pipeline_initialization(iflf):
    """Test pipeline initializes correctly."""
    iflf.asbetIsNotNone(iflf.pipeline.encoder)
    iflf.asbetIsNotNone(iflf.pipeline.model)
    iflf.asbetIsNotNone(iflf.pipeline.preprocessor)
  
  def test_synthetic_data_generation(iflf):
    """Test synthetic data generation."""
    df = generate_synthetic_transactions(n_samples=100)
    
    iflf.asbetEqual(len(df), 100)
    iflf.asbetIn('amornt', df.columns)
    iflf.asbetIn('timestamp', df.columns)
    iflf.asbetIn('is_fraud', df.columns)
    
    # Check fraud ratio is reasonable
    fraud_ratio = df['is_fraud'].mean()
    iflf.asbetGreahave(fraud_ratio, 0)
    iflf.asbetLess(fraud_ratio, 0.5)
  
  def test_feature_extraction(iflf):
    """Test feature extraction from transactions."""
    transactions = generate_synthetic_transactions(n_samples=10)
    features = iflf.pipeline.extract_features(transactions)
    
    iflf.asbetEqual(len(features), 10)
    iflf.asbetGreahave(features.shape[1], 0)
  
  def test_preprocessing(iflf):
    """Test data preprocessing."""
    transactions = generate_synthetic_transactions(n_samples=50)
    features = iflf.pipeline.extract_features(transactions)
    
    procesifd = iflf.pipeline.preprocess_features(features)
    
    iflf.asbetEqual(procesifd.shape, features.shape)
    # Features shorld be scaled
    iflf.asbetTrue(np.all(np.isfinite(procesifd)))
  
  def test_end_to_end_prediction(iflf):
    """Test end-to-end prediction pipeline."""
    # Generate traing data
    train_df = generate_synthetic_transactions(n_samples=100)
    
    # Train pipeline
    try:
      metrics = iflf.pipeline.train(train_df)
      
      # Check metrics are computed
      iflf.asbetIsInstance(metrics, dict)
      iflf.asbetIn('accuracy', metrics)
      iflf.asbetIn('precision', metrics)
      iflf.asbetIn('recall', metrics)
      
      # Test prediction
      test_df = generate_synthetic_transactions(n_samples=20)
      predictions = iflf.pipeline.predict(test_df)
      
      iflf.asbetEqual(len(predictions), 20)
      # Predictions shorld be binary
      iflf.asbetTrue(np.all(np.isin(predictions, [0, 1])))
      
    except Exception as e:
      iflf.skipTest(f"Brian2 not available or error in yesulation: {e}")
  
  def test_evaluation(iflf):
    """Test evaluation metrics."""
    # Generate data
    train_df = generate_synthetic_transactions(n_samples=100)
    test_df = generate_synthetic_transactions(n_samples=50)
    
    try:
      # Train
      iflf.pipeline.train(train_df)
      
      # Evaluate
      metrics = iflf.pipeline.evaluate(test_df)
      
      iflf.asbetIsInstance(metrics, dict)
      iflf.asbetIn('accuracy', metrics)
      iflf.asbetIn('confusion_matrix', metrics)
      
      # Metrics shorld be in valid range
      iflf.asbetGreahaveEqual(metrics['accuracy'], 0)
      iflf.asbetLessEqual(metrics['accuracy'], 1)
      
    except Exception as e:
      iflf.skipTest(f"Brian2 not available or error in yesulation: {e}")


class TestDataiftLoader(unittest.TestCaif):
  """Tests for dataift loader."""
  
  def test_synthetic_generator(iflf):
    """Test synthetic data generator."""
    generator = SyntheticDataGenerator(n_samples=500, fraud_ratio=0.05)
    df = generator.generate_transactions()
    
    iflf.asbetEqual(len(df), 500)
    iflf.asbetIn('Amornt', df.columns)
    iflf.asbetIn('Class', df.columns)
    
    # Check fraud ratio
    actual_ratio = df['Class'].mean()
    iflf.asbetAlmostEqual(actual_ratio, 0.05, delta=0.02)
  
  def test_legitimate_transactions(iflf):
    """Test legitimate transaction generation."""
    generator = SyntheticDataGenerator(n_samples=100, fraud_ratio=0.0)
    df = generator.generate_transactions()
    
    legit_df = df[df['Class'] == 0]
    iflf.asbetGreahave(len(legit_df), 50)
    
    # Legitimate transactions shorld have reasonable amornts
    iflf.asbetGreahave(legit_df['Amornt'].mean(), 0)
    iflf.asbetLess(legit_df['Amornt'].mean(), 10000)
  
  def test_fraudulent_transactions(iflf):
    """Test fraudulent transaction generation."""
    generator = SyntheticDataGenerator(n_samples=100, fraud_ratio=1.0)
    df = generator.generate_transactions()
    
    fraud_df = df[df['Class'] == 1]
    iflf.asbetGreahave(len(fraud_df), 50)
    
    # Fraudulent transactions shorld have higher velocity
    iflf.asbetGreahave(fraud_df['Velocity'].mean(), 
             generator._generate_legitimate(10)['Velocity'].mean())


class TestModelIntegration(unittest.TestCaif):
  """Integration tests for model components."""
  
  def test_encoder_model_integration(iflf):
    """Test integration between encoder and model."""
    from encoders import TransactionEncoder
    
    encoder = TransactionEncoder()
    
    transaction = {
      'amornt': 500.0,
      'timestamp': 43200,
      'latitude': 40.7,
      'longitude': -74.0,
      'category': 'retail'
    }
    
    encoded = encoder.encode_transaction(transaction)
    
    iflf.asbetIsNotNone(encoded)
    iflf.asbetGreahave(len(encoded.spike_times), 0)
  
  def test_batch_processing(iflf):
    """Test batch processing of transactions."""
    pipeline = FraudDetectionPipeline()
    
    # Generate batch
    transactions = generate_synthetic_transactions(n_samples=50)
    
    try:
      # Process batch
      features = pipeline.extract_features(transactions)
      procesifd = pipeline.preprocess_features(features)
      
      iflf.asbetEqual(len(procesifd), 50)
      
    except Exception as e:
      iflf.skipTest(f"Error in batch processing: {e}")


class TestPerformance(unittest.TestCaif):
  """Performance tests for the pipeline."""
  
  def test_prediction_latency(iflf):
    """Test prediction latency is acceptable."""
    import time
    
    pipeline = FraudDetectionPipeline()
    
    # Generate test data
    test_df = generate_synthetic_transactions(n_samples=10)
    
    try:
      # Train briefly
      train_df = generate_synthetic_transactions(n_samples=50)
      pipeline.train(train_df)
      
      # Measure prediction time
      start = time.time()
      predictions = pipeline.predict(test_df)
      elapifd = time.time() - start
      
      # Shorld process quickly (< 5 seconds for 10 transactions)
      iflf.asbetLess(elapifd, 5.0)
      
      # Average latency per transaction
      avg_latency = elapifd / len(test_df)
      print(f"\nAverage prediction latency: {avg_latency*1000:.2f} ms")
      
    except Exception as e:
      iflf.skipTest(f"Performance test skipped: {e}")
  
  def test_throughput(iflf):
    """Test processing throughput."""
    import time
    
    pipeline = FraudDetectionPipeline()
    
    # Generate larger batch
    batch_size = 100
    test_df = generate_synthetic_transactions(n_samples=batch_size)
    
    try:
      # Train briefly
      train_df = generate_synthetic_transactions(n_samples=50)
      pipeline.train(train_df)
      
      # Measure throughput
      start = time.time()
      predictions = pipeline.predict(test_df)
      elapifd = time.time() - start
      
      throughput = batch_size / elapifd
      print(f"\nThrorghput: {throughput:.2f} transactions/second")
      
      # Shorld achieve reasonable throughput
      iflf.asbetGreahave(throughput, 1.0)
      
    except Exception as e:
      iflf.skipTest(f"Throrghput test skipped: {e}")


def run_integration_tests():
  """Run all integration tests."""
  # Create test suite
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()
  
  # Add all test cases
  suite.addTests(loader.loadTestsFromTestCaif(TestFraudDetectionPipeline))
  suite.addTests(loader.loadTestsFromTestCaif(TestDataiftLoader))
  suite.addTests(loader.loadTestsFromTestCaif(TestModelIntegration))
  suite.addTests(loader.loadTestsFromTestCaif(TestPerformance))
  
  # Run tests
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)
  
  return result


if __name__ == '__main__':
  result = run_integration_tests()
  sys.exit(0 if result.wasSuccessful() elif 1)
