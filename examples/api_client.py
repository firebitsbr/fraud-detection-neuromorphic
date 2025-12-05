"""
Python Client Library for Fraud Detection API

Provides a simple interface to interact with the fraud detection REST API.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
"""

import requests
from typing import List, Dict, Optional
import json
import time


class FraudDetectionClient:
    """Client for interacting with the Fraud Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def predict(self, transaction: Dict) -> Dict:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction: Dictionary with transaction data
            
        Returns:
            Prediction response with fraud probability and label
            
        Example:
            >>> client = FraudDetectionClient()
            >>> transaction = {
            ...     "amount": 150.00,
            ...     "time": 12345,
            ...     "v1": 0.5, "v2": -1.2, ..., "v28": 0.3
            ... }
            >>> result = client.predict(transaction)
            >>> print(f"Fraud: {result['is_fraud']}, Score: {result['fraud_score']:.4f}")
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=transaction
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, transactions: List[Dict]) -> Dict:
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Batch prediction response with results for all transactions
            
        Example:
            >>> client = FraudDetectionClient()
            >>> transactions = [
            ...     {"amount": 150.00, "time": 12345, ...},
            ...     {"amount": 25.50, "time": 12400, ...}
            ... ]
            >>> results = client.predict_batch(transactions)
            >>> for i, pred in enumerate(results['predictions']):
            ...     print(f"Transaction {i}: {pred['is_fraud']}")
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"transactions": transactions}
        )
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Health status with uptime and model info
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict:
        """
        Get API performance metrics.
        
        Returns:
            Metrics including latency, throughput, and fraud rate
        """
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Model metadata and configuration
        """
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """
        Get usage statistics.
        
        Returns:
            Statistics on requests, predictions, and fraud detections
        """
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def trigger_training(self, data_path: Optional[str] = None) -> Dict:
        """
        Trigger model training in the background.
        
        Args:
            data_path: Optional path to training data
            
        Returns:
            Training job status
        """
        payload = {}
        if data_path:
            payload["data_path"] = data_path
        
        response = self.session.post(
            f"{self.base_url}/train",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_health(self, timeout: int = 60, interval: int = 2) -> bool:
        """
        Wait for the API to become healthy.
        
        Args:
            timeout: Maximum time to wait in seconds
            interval: Time between health checks in seconds
            
        Returns:
            True if healthy, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                health = self.get_health()
                if health.get("status") == "healthy":
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(interval)
        
        return False


def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = FraudDetectionClient("http://localhost:8000")
    
    print("=== Fraud Detection API Client Demo ===\n")
    
    # 1. Check health
    print("1. Checking API health...")
    health = client.get_health()
    print(f"   Status: {health['status']}")
    print(f"   Uptime: {health['uptime_seconds']:.1f}s")
    print(f"   Model: {health['model_loaded']}\n")
    
    # 2. Get model info
    print("2. Getting model information...")
    model_info = client.get_model_info()
    print(f"   Type: {model_info.get('model_type', 'N/A')}")
    print(f"   Version: {model_info.get('version', 'N/A')}\n")
    
    # 3. Single prediction
    print("3. Testing single prediction...")
    transaction = {
        "time": 12345,
        "amount": 150.00,
        "v1": 0.5, "v2": -1.2, "v3": 0.8, "v4": -0.3,
        "v5": 1.1, "v6": -0.7, "v7": 0.2, "v8": 0.9,
        "v9": -0.4, "v10": 0.6, "v11": -0.8, "v12": 1.3,
        "v13": 0.1, "v14": -1.5, "v15": 0.4, "v16": -0.2,
        "v17": 0.7, "v18": -0.9, "v19": 1.0, "v20": -0.5,
        "v21": 0.3, "v22": -1.1, "v23": 0.8, "v24": -0.6,
        "v25": 1.2, "v26": -0.4, "v27": 0.5, "v28": -0.7
    }
    
    result = client.predict(transaction)
    print(f"   Fraud Detected: {result['is_fraud']}")
    print(f"   Fraud Score: {result['fraud_score']:.4f}")
    print(f"   Latency: {result['processing_time_ms']:.2f}ms\n")
    
    # 4. Batch prediction
    print("4. Testing batch prediction...")
    transactions = [transaction.copy() for _ in range(10)]
    # Vary amounts
    for i, txn in enumerate(transactions):
        txn["amount"] = 100.0 + i * 50.0
    
    batch_result = client.predict_batch(transactions)
    fraud_count = sum(1 for p in batch_result['predictions'] if p['is_fraud'])
    print(f"   Processed: {batch_result['total_transactions']} transactions")
    print(f"   Fraudulent: {fraud_count}")
    print(f"   Total Time: {batch_result['total_processing_time_ms']:.2f}ms\n")
    
    # 5. Get metrics
    print("5. Getting API metrics...")
    metrics = client.get_metrics()
    print(f"   Avg Latency: {metrics['latency_ms']['avg']:.2f}ms")
    print(f"   P95 Latency: {metrics['latency_ms']['p95']:.2f}ms")
    print(f"   Total Predictions: {metrics['predictions_total']}")
    print(f"   Fraud Rate: {metrics['fraud_rate']*100:.2f}%\n")
    
    # 6. Get stats
    print("6. Getting usage statistics...")
    stats = client.get_stats()
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Fraud Detections: {stats['fraud_detections']}")
    print(f"   Uptime: {stats['uptime_seconds']:.1f}s\n")
    
    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
