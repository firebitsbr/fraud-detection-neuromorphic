"""
Example Kafka Producer for Fraud Detection

Simulates transaction stream for real-time fraud detection.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
License: MIT License
"""

import json
import time
import random
from kafka import KafkaProducer
from typing import Dict
import argparse


class TransactionGenerator:
    """Generates realistic transaction data."""
    
    def __init__(self, fraud_rate: float = 0.05):
        """
        Initialize generator.
        
        Args:
            fraud_rate: Probability of generating fraudulent transaction
        """
        self.fraud_rate = fraud_rate
        self.transaction_id = 0
    
    def generate_transaction(self) -> Dict:
        """
        Generate a random transaction.
        
        Returns:
            Dictionary with transaction data
        """
        self.transaction_id += 1
        is_fraud = random.random() < self.fraud_rate
        
        # Base transaction
        transaction = {
            "transaction_id": f"TXN{self.transaction_id:06d}",
            "time": int(time.time()),
            "amount": self._generate_amount(is_fraud)
        }
        
        # Generate PCA components
        for i in range(1, 29):
            if is_fraud:
                # Fraudulent transactions have different patterns
                transaction[f"v{i}"] = round(random.gauss(1.5, 2.0), 6)
            else:
                # Normal transactions
                transaction[f"v{i}"] = round(random.gauss(0, 1), 6)
        
        return transaction
    
    def _generate_amount(self, is_fraud: bool) -> float:
        """Generate transaction amount."""
        if is_fraud:
            # Fraudulent transactions tend to be larger
            return round(random.uniform(500.0, 5000.0), 2)
        else:
            # Normal transactions are typically smaller
            return round(random.lognormvariate(3.5, 1.5), 2)


class FraudTransactionProducer:
    """Kafka producer for transaction stream."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "transactions"
    ):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to publish to
        """
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topic = topic
        self.generator = TransactionGenerator()
        self.stats = {
            "total_sent": 0,
            "errors": 0
        }
    
    def send_transaction(self) -> bool:
        """
        Generate and send a single transaction.
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            transaction = self.generator.generate_transaction()
            
            # Send to Kafka
            future = self.producer.send(
                self.topic,
                value=transaction,
                key=transaction["transaction_id"]
            )
            
            # Wait for confirmation
            future.get(timeout=10)
            
            self.stats["total_sent"] += 1
            return True
            
        except Exception as e:
            print(f"Error sending transaction: {e}")
            self.stats["errors"] += 1
            return False
    
    def run_stream(
        self,
        duration_s: int = 60,
        rate: float = 1.0
    ):
        """
        Run continuous transaction stream.
        
        Args:
            duration_s: Duration to run in seconds
            rate: Transactions per second
        """
        interval = 1.0 / rate
        start_time = time.time()
        
        print(f"Starting transaction stream...")
        print(f"  Topic: {self.topic}")
        print(f"  Rate: {rate} txn/s")
        print(f"  Duration: {duration_s}s")
        print(f"  Press Ctrl+C to stop\n")
        
        try:
            while time.time() - start_time < duration_s:
                batch_start = time.time()
                
                # Send transaction
                if self.send_transaction():
                    if self.stats["total_sent"] % 10 == 0:
                        print(f"Sent {self.stats['total_sent']} transactions...")
                
                # Maintain target rate
                elapsed = time.time() - batch_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n\nStopping producer...")
        
        finally:
            self.producer.flush()
            self.producer.close()
            
            # Print stats
            duration = time.time() - start_time
            print(f"\n{'='*50}")
            print("Producer Statistics:")
            print(f"{'='*50}")
            print(f"Total Sent:       {self.stats['total_sent']}")
            print(f"Errors:           {self.stats['errors']}")
            print(f"Duration:         {duration:.2f}s")
            print(f"Actual Rate:      {self.stats['total_sent']/duration:.2f} txn/s")
            print(f"{'='*50}\n")
    
    def send_batch(self, count: int):
        """
        Send a batch of transactions.
        
        Args:
            count: Number of transactions to send
        """
        print(f"Sending batch of {count} transactions...")
        
        start_time = time.time()
        success_count = 0
        
        for i in range(count):
            if self.send_transaction():
                success_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{count}")
        
        duration = time.time() - start_time
        
        print(f"\nBatch complete:")
        print(f"  Sent: {success_count}/{count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Rate: {success_count/duration:.2f} txn/s\n")


def main():
    """Run transaction producer."""
    parser = argparse.ArgumentParser(
        description="Kafka Transaction Producer for Fraud Detection"
    )
    parser.add_argument(
        "--broker",
        default="localhost:9092",
        help="Kafka broker address (default: localhost:9092)"
    )
    parser.add_argument(
        "--topic",
        default="transactions",
        help="Kafka topic (default: transactions)"
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "batch"],
        default="stream",
        help="Producer mode (default: stream)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Stream duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Transactions per second (default: 1.0)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of transactions for batch mode (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Create producer
    producer = FraudTransactionProducer(
        bootstrap_servers=args.broker,
        topic=args.topic
    )
    
    # Run based on mode
    if args.mode == "stream":
        producer.run_stream(
            duration_s=args.duration,
            rate=args.rate
        )
    else:
        producer.send_batch(args.count)


if __name__ == "__main__":
    main()
