"""
Pipeline completo de ponta a ponta para detecção de transações
bancárias fraudulentas usando Redes Neurais Spiking (SNNs).
Integra extração de features, codificação de spikes, inferência
SNN e motor de decisão.

Autor: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
Linkedin: https://www.linkedin.com/in/maurorisonho
github: https://github.com/maurorisonho
Data de criação: Dezembro 2025
LICENSE MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm.auto import tqdm

from encoders import TransactionEncoder, SpikeEncoding
from models_snn import FraudSNN


class FraudDetectionPipeline:
    """
    Complete neuromorphic fraud detection system.
    
    Pipeline stages:
        1. Feature extraction from transaction
        2. Spike encoding (temporal + rate + population)
        3. SNN inference
        4. Decision and alerting
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fraud detection pipeline.
        
        Args:
            model_path: Path to pre-trained SNN model (optional)
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Initialize encoder
        self.encoder = TransactionEncoder(self.config['encoding'])
        
        # Initialize SNN
        self.snn = FraudSNN(
            input_size=self.config['model']['input_size'],
            hidden_sizes=self.config['model']['hidden_sizes'],
            output_size=self.config['model']['output_size']
        )
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.snn.load(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("Using randomly initialized SNN (requires training)")
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'fraud_detected': 0,
            'legitimate': 0,
            'avg_latency_ms': 0.0
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'encoding': {
                'min_rate': 0.1,
                'max_rate': 100.0,
                'duration': 0.1,
                'time_window': 1.0,
                'pop_neurons': 32,
                'sigma': 0.1,
                'max_latency': 0.1
            },
            'model': {
                'input_size': 256,
                'hidden_sizes': [128, 64],
                'output_size': 2
            },
            'detection': {
                'fraud_threshold': 0.5,  # Confidence threshold for fraud
                'min_spikes': 1  # Minimum spikes in output to consider valid
            }
        }
    
    def extract_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize features from raw transaction.
        
        Args:
            transaction: Raw transaction data
        
        Returns:
            Normalized feature dictionary
        """
        features = {}
        
        # Amount
        features['amount'] = float(transaction.get('amount', 0.0))
        
        # Timestamp
        if 'timestamp' in transaction:
            if isinstance(transaction['timestamp'], str):
                dt = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
                features['timestamp'] = dt.timestamp()
            else:
                features['timestamp'] = float(transaction['timestamp'])
        else:
            features['timestamp'] = time.time()
        
        # Location
        if 'location' in transaction:
            features['location'] = tuple(transaction['location'])
        else:
            features['location'] = (0.0, 0.0)
        
        # Merchant category (map string to int)
        category_map = {
            'groceries': 0, 'restaurants': 1, 'gas': 2,
            'electronics': 3, 'clothing': 4, 'healthcare': 5,
            'entertainment': 6, 'travel': 7, 'online': 8,
            'other': 9
        }
        cat = transaction.get('merchant_category', 'other')
        features['merchant_category'] = category_map.get(cat, 9)
        
        # Device ID (hash to number)
        device_id = transaction.get('device_id', 'unknown')
        features['device_hash'] = hash(device_id) % 1000
        
        # Historical frequency (mock - in production, query database)
        features['daily_frequency'] = transaction.get('daily_frequency', 5.0)
        
        return features
    
    def preprocess(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess transaction for encoding.
        
        Args:
            transaction: Raw transaction
        
        Returns:
            Preprocessed features
        """
        features = self.extract_features(transaction)
        
        # Additional preprocessing
        # Log-scale for amount (to handle wide range)
        features['amount_log'] = np.log1p(features['amount'])
        
        # Time-of-day features
        dt = datetime.fromtimestamp(features['timestamp'])
        features['hour'] = dt.hour
        features['minute'] = dt.minute
        features['weekday'] = dt.weekday()
        
        # Velocity feature (mock - time since last transaction)
        features['velocity'] = transaction.get('seconds_since_last', 3600.0)
        
        return features
    
    def encode(self, features: Dict[str, Any]) -> SpikeEncoding:
        """
        Encode preprocessed features into spikes.
        
        Args:
            features: Preprocessed transaction features
        
        Returns:
            SpikeEncoding object
        """
        # Encode transaction
        encoded = self.encoder.encode_transaction(features)
        
        # Convert to unified format
        unified = self.encoder.to_unified_format(
            encoded, 
            n_input_neurons=self.config['model']['input_size']
        )
        
        return unified
    
    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete prediction pipeline for a single transaction.
        
        Args:
            transaction: Raw transaction dictionary
        
        Returns:
            Prediction result with fraud probability and metadata
        """
        start_time = time.time()
        
        # 1. Preprocess
        features = self.preprocess(transaction)
        
        # 2. Encode
        spikes = self.encode(features)
        
        # 3. SNN Inference
        if len(spikes.spike_times) == 0:
            # No spikes generated (e.g., zero transaction)
            result = {
                'is_fraud': False,
                'confidence': 0.0,
                'output_rates': [0.0, 0.0]
            }
        else:
            result = self.snn.predict(
                spikes.spike_times,
                spikes.neuron_indices,
                duration=self.config['encoding']['duration']
            )
        
        # 4. Post-process
        latency_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['total_predictions'] += 1
        if result['is_fraud']:
            self.stats['fraud_detected'] += 1
        else:
            self.stats['legitimate'] += 1
        
        # Update average latency
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (self.stats['total_predictions'] - 1) + latency_ms)
            / self.stats['total_predictions']
        )
        
        # Build response
        response = {
            'transaction_id': transaction.get('id', 'unknown'),
            'is_fraud': result['is_fraud'],
            'confidence': result['confidence'],
            'fraud_score': result['output_rates'][1] if len(result['output_rates']) > 1 else 0.0,
            'legitimate_score': result['output_rates'][0] if len(result['output_rates']) > 0 else 0.0,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat(),
            'features_used': list(features.keys()),
            'n_spikes_generated': len(spikes.spike_times)
        }
        
        return response
    
    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
        
        Returns:
            List of prediction results
        """
        results = []
        for transaction in tqdm(transactions, desc="Predicting Batch"):
            result = self.predict(transaction)
            results.append(result)
        return results
    
    def train(self, training_data: pd.DataFrame, epochs: int = 100):
        """
        Train the SNN on labeled transaction data.
        
        Args:
            training_data: DataFrame with columns:
                - All transaction features
                - 'is_fraud': Binary label (0=legitimate, 1=fraud)
            epochs: Number of training epochs
        """
        print(f"Training on {len(training_data)} transactions...")
        
        # Prepare spike data
        spike_data = []
        
        for idx, row in tqdm(training_data.iterrows(), total=len(training_data), desc="Preparing Data"):
            # Convert row to transaction dict
            transaction = row.to_dict()
            
            # Encode
            features = self.preprocess(transaction)
            spikes = self.encode(features)
            
            # Label
            label = int(row['is_fraud'])
            
            spike_data.append((spikes.spike_times, spikes.neuron_indices, label))
        
        # Train SNN with STDP
        self.snn.train_stdp(spike_data, epochs=epochs)
        
        print("Training complete!")
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            test_data: DataFrame with labeled transactions
        
        Returns:
            Dictionary with metrics (accuracy, precision, recall, F1)
        """
        print(f"Evaluating on {len(test_data)} transactions...")
        
        y_true = []
        y_pred = []
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
            transaction = row.to_dict()
            result = self.predict(transaction)
            
            y_true.append(int(row['is_fraud']))
            y_pred.append(int(result['is_fraud']))
        
        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        print("\n=== Evaluation Results ===")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1-Score:  {f1:.2%}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained SNN model."""
        self.snn.save(filepath)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        stats['snn_stats'] = self.snn.get_network_stats()
        return stats


def generate_synthetic_transactions(n: int = 1000, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic transaction data for testing.
    
    Args:
        n: Number of transactions to generate
        fraud_ratio: Fraction of fraudulent transactions
    
    Returns:
        DataFrame with synthetic transactions
    """
    np.random.seed(42)
    
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud
    
    transactions = []
    
    # Legitimate transactions
    for i in range(n_legit):
        transaction = {
            'id': f'txn_{i:06d}',
            'amount': np.random.lognormal(mean=3.5, sigma=1.2),  # ~$50 average
            'timestamp': time.time() - np.random.uniform(0, 30*24*3600),  # Last 30 days
            'merchant_category': np.random.choice([
                'groceries', 'restaurants', 'gas', 'healthcare', 'entertainment'
            ]),
            'location': (
                np.random.uniform(-30, 30),  # Latitude
                np.random.uniform(-60, 60)   # Longitude
            ),
            'device_id': f'device_{np.random.randint(0, 100):03d}',
            'daily_frequency': np.random.poisson(5),
            'is_fraud': 0
        }
        transactions.append(transaction)
    
    # Fraudulent transactions (anomalous patterns)
    for i in range(n_fraud):
        transaction = {
            'id': f'txn_{n_legit + i:06d}',
            'amount': np.random.uniform(1000, 10000),  # High amounts
            'timestamp': time.time() - np.random.uniform(0, 30*24*3600),
            'merchant_category': np.random.choice(['electronics', 'online', 'travel']),
            'location': (
                np.random.uniform(-90, 90),  # Random global location
                np.random.uniform(-180, 180)
            ),
            'device_id': f'device_new_{i:03d}',  # New devices
            'daily_frequency': np.random.poisson(20),  # High frequency
            'is_fraud': 1
        }
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle


def main():
    """Main demonstration of the pipeline."""
    print("=" * 60)
    print("NEUROMORPHIC FRAUD DETECTION SYSTEM")
    print("Author: Mauro Risonho de Paula Assumpção")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = FraudDetectionPipeline()
    print(f"SNN Architecture: {pipeline.snn.get_network_stats()}")
    print()
    
    # Generate synthetic data
    print("Generating synthetic transaction data...")
    data = generate_synthetic_transactions(n=1000, fraud_ratio=0.05)
    print(f"Generated {len(data)} transactions ({data['is_fraud'].sum()} fraudulent)")
    print()
    
    # Split train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Train
    print("Training SNN with STDP...")
    pipeline.train(train_data, epochs=50)
    print()
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = pipeline.evaluate(test_data)
    print()
    
    # Single prediction example
    print("=" * 60)
    print("SINGLE TRANSACTION PREDICTION EXAMPLE")
    print("=" * 60)
    
    test_transaction = {
        'id': 'test_001',
        'amount': 5000.00,
        'timestamp': time.time(),
        'merchant_category': 'electronics',
        'location': (-23.5505, -46.6333),  # São Paulo
        'device_id': 'new_device_123',
        'daily_frequency': 15
    }
    
    print(f"Transaction: ${test_transaction['amount']:.2f} at {test_transaction['merchant_category']}")
    result = pipeline.predict(test_transaction)
    print(f"\nResult:")
    print(f"  Fraud: {result['is_fraud']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")
    print(f"  Spikes generated: {result['n_spikes_generated']}")
    print()
    
    # Statistics
    print("=" * 60)
    print("PIPELINE STATISTICS")
    print("=" * 60)
    stats = pipeline.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Save model
    print("\nSaving model...")
    pipeline.save_model('models/fraud_snn_trained.pkl')
    print("Done!")


if __name__ == '__main__':
    main()
