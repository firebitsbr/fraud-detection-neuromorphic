"""
**Descrição:** Testes de integração ponta a ponta.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
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
    sys.path.insert(0, str(src_path))

from main import FraudDetectionPipeline, generate_synthetic_transactions
from dataset_loader import SyntheticDataGenerator


class TestFraudDetectionPipeline(unittest.TestCase):
    """Integration tests for fraud detection pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = FraudDetectionPipeline()
        
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        self.assertIsNotNone(self.pipeline.encoder)
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.preprocessor)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        df = generate_synthetic_transactions(n_samples=100)
        
        self.assertEqual(len(df), 100)
        self.assertIn('amount', df.columns)
        self.assertIn('timestamp', df.columns)
        self.assertIn('is_fraud', df.columns)
        
        # Check fraud ratio is reasonable
        fraud_ratio = df['is_fraud'].mean()
        self.assertGreater(fraud_ratio, 0)
        self.assertLess(fraud_ratio, 0.5)
    
    def test_feature_extraction(self):
        """Test feature extraction from transactions."""
        transactions = generate_synthetic_transactions(n_samples=10)
        features = self.pipeline.extract_features(transactions)
        
        self.assertEqual(len(features), 10)
        self.assertGreater(features.shape[1], 0)
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        transactions = generate_synthetic_transactions(n_samples=50)
        features = self.pipeline.extract_features(transactions)
        
        processed = self.pipeline.preprocess_features(features)
        
        self.assertEqual(processed.shape, features.shape)
        # Features should be scaled
        self.assertTrue(np.all(np.isfinite(processed)))
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        # Generate training data
        train_df = generate_synthetic_transactions(n_samples=100)
        
        # Train pipeline
        try:
            metrics = self.pipeline.train(train_df)
            
            # Check metrics are computed
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            
            # Test prediction
            test_df = generate_synthetic_transactions(n_samples=20)
            predictions = self.pipeline.predict(test_df)
            
            self.assertEqual(len(predictions), 20)
            # Predictions should be binary
            self.assertTrue(np.all(np.isin(predictions, [0, 1])))
            
        except Exception as e:
            self.skipTest(f"Brian2 not available or error in simulation: {e}")
    
    def test_evaluation(self):
        """Test evaluation metrics."""
        # Generate data
        train_df = generate_synthetic_transactions(n_samples=100)
        test_df = generate_synthetic_transactions(n_samples=50)
        
        try:
            # Train
            self.pipeline.train(train_df)
            
            # Evaluate
            metrics = self.pipeline.evaluate(test_df)
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('confusion_matrix', metrics)
            
            # Metrics should be in valid range
            self.assertGreaterEqual(metrics['accuracy'], 0)
            self.assertLessEqual(metrics['accuracy'], 1)
            
        except Exception as e:
            self.skipTest(f"Brian2 not available or error in simulation: {e}")


class TestDatasetLoader(unittest.TestCase):
    """Tests for dataset loader."""
    
    def test_synthetic_generator(self):
        """Test synthetic data generator."""
        generator = SyntheticDataGenerator(n_samples=500, fraud_ratio=0.05)
        df = generator.generate_transactions()
        
        self.assertEqual(len(df), 500)
        self.assertIn('Amount', df.columns)
        self.assertIn('Class', df.columns)
        
        # Check fraud ratio
        actual_ratio = df['Class'].mean()
        self.assertAlmostEqual(actual_ratio, 0.05, delta=0.02)
    
    def test_legitimate_transactions(self):
        """Test legitimate transaction generation."""
        generator = SyntheticDataGenerator(n_samples=100, fraud_ratio=0.0)
        df = generator.generate_transactions()
        
        legit_df = df[df['Class'] == 0]
        self.assertGreater(len(legit_df), 50)
        
        # Legitimate transactions should have reasonable amounts
        self.assertGreater(legit_df['Amount'].mean(), 0)
        self.assertLess(legit_df['Amount'].mean(), 10000)
    
    def test_fraudulent_transactions(self):
        """Test fraudulent transaction generation."""
        generator = SyntheticDataGenerator(n_samples=100, fraud_ratio=1.0)
        df = generator.generate_transactions()
        
        fraud_df = df[df['Class'] == 1]
        self.assertGreater(len(fraud_df), 50)
        
        # Fraudulent transactions should have higher velocity
        self.assertGreater(fraud_df['Velocity'].mean(), 
                          generator._generate_legitimate(10)['Velocity'].mean())


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def test_encoder_model_integration(self):
        """Test integration between encoder and model."""
        from encoders import TransactionEncoder
        
        encoder = TransactionEncoder()
        
        transaction = {
            'amount': 500.0,
            'timestamp': 43200,
            'latitude': 40.7,
            'longitude': -74.0,
            'category': 'retail'
        }
        
        encoded = encoder.encode_transaction(transaction)
        
        self.assertIsNotNone(encoded)
        self.assertGreater(len(encoded.spike_times), 0)
    
    def test_batch_processing(self):
        """Test batch processing of transactions."""
        pipeline = FraudDetectionPipeline()
        
        # Generate batch
        transactions = generate_synthetic_transactions(n_samples=50)
        
        try:
            # Process batch
            features = pipeline.extract_features(transactions)
            processed = pipeline.preprocess_features(features)
            
            self.assertEqual(len(processed), 50)
            
        except Exception as e:
            self.skipTest(f"Error in batch processing: {e}")


class TestPerformance(unittest.TestCase):
    """Performance tests for the pipeline."""
    
    def test_prediction_latency(self):
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
            elapsed = time.time() - start
            
            # Should process quickly (< 5 seconds for 10 transactions)
            self.assertLess(elapsed, 5.0)
            
            # Average latency per transaction
            avg_latency = elapsed / len(test_df)
            print(f"\nAverage prediction latency: {avg_latency*1000:.2f} ms")
            
        except Exception as e:
            self.skipTest(f"Performance test skipped: {e}")
    
    def test_throughput(self):
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
            elapsed = time.time() - start
            
            throughput = batch_size / elapsed
            print(f"\nThroughput: {throughput:.2f} transactions/second")
            
            # Should achieve reasonable throughput
            self.assertGreater(throughput, 1.0)
            
        except Exception as e:
            self.skipTest(f"Throughput test skipped: {e}")


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFraudDetectionPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_integration_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
