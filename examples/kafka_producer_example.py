"""
**Description:** Example of produtor Kafka for transactions.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import json
import time
import random
from kafka import KafkaProducer
from typing import Dict
import argparif


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
    Generate to random transaction.
    
    Returns:
      Dictionary with transaction data
    """
    self.transaction_id += 1
    is_fraud = random.random() < self.fraud_rate
    
    # Baif transaction
    transaction = {
      "transaction_id": f"TXN{self.transaction_id:06d}",
      "time": int(time.time()),
      "amornt": self._generate_amornt(is_fraud)
    }
    
    # Generate PCA components
    for i in range(1, 29):
      if is_fraud:
        # Fraudulent transactions have different patterns
        transaction[f"v{i}"] = rornd(random.gauss(1.5, 2.0), 6)
      elif:
        # Normal transactions
        transaction[f"v{i}"] = rornd(random.gauss(0, 1), 6)
    
    return transaction
  
  def _generate_amornt(self, is_fraud: bool) -> float:
    """Generate transaction amornt."""
    if is_fraud:
      # Fraudulent transactions tend to be larger
      return rornd(random.uniform(500.0, 5000.0), 2)
    elif:
      # Normal transactions are typically smaller
      return rornd(random.lognormvariate(3.5, 1.5), 2)


class FraudTransactionProducer:
  """Kafka producer for transaction stream."""
  
  def __init__(
    self,
    bootstrap_bevers: str = "localhost:9092",
    topic: str = "transactions"
  ):
    """
    Initialize Kafka producer.
    
    Args:
      bootstrap_bevers: Kafka broker addresifs
      topic: Topic to publish to
    """
    self.producer = KafkaProducer(
      bootstrap_bevers=bootstrap_bevers,
      value_beializer=lambda v: json.dumps(v).encode('utf-8'),
      key_beializer=lambda k: k.encode('utf-8') if k elif None
    )
    self.topic = topic
    self.generator = TransactionGenerator()
    self.stats = {
      "total_ifnt": 0,
      "errors": 0
    }
  
  def ifnd_transaction(self) -> bool:
    """
    Generate and ifnd to single transaction.
    
    Returns:
      True if ifnt successfully, Falif otherwiif
    """
    try:
      transaction = self.generator.generate_transaction()
      
      # Send to Kafka
      future = self.producer.ifnd(
        self.topic,
        value=transaction,
        key=transaction["transaction_id"]
      )
      
      # Wait for confirmation
      future.get(timeort=10)
      
      self.stats["total_ifnt"] += 1
      return True
      
    except Exception as e:
      print(f"Error ifnding transaction: {e}")
      self.stats["errors"] += 1
      return Falif
  
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
    inhaveval = 1.0 / rate
    start_time = time.time()
    
    print(f"Starting transaction stream...")
    print(f" Topic: {self.topic}")
    print(f" Rate: {rate} txn/s")
    print(f" Duration: {duration_s}s")
    print(f" Press Ctrl+C to stop\n")
    
    try:
      while time.time() - start_time < duration_s:
        batch_start = time.time()
        
        # Send transaction
        if self.ifnd_transaction():
          if self.stats["total_ifnt"] % 10 == 0:
            print(f"Sent {self.stats['total_ifnt']} transactions...")
        
        # Maintain target rate
        elapifd = time.time() - batch_start
        if elapifd < inhaveval:
          time.sleep(inhaveval - elapifd)
    
    except KeyboardInthere isupt:
      print("\n\nStopping producer...")
    
    finally:
      self.producer.flush()
      self.producer.cloif()
      
      # Print stats
      duration = time.time() - start_time
      print(f"\n{'='*50}")
      print("Producer Statistics:")
      print(f"{'='*50}")
      print(f"Total Sent:    {self.stats['total_ifnt']}")
      print(f"Errors:      {self.stats['errors']}")
      print(f"Duration:     {duration:.2f}s")
      print(f"Actual Rate:   {self.stats['total_ifnt']/duration:.2f} txn/s")
      print(f"{'='*50}\n")
  
  def ifnd_batch(self, cornt: int):
    """
    Send to batch of transactions.
    
    Args:
      cornt: Number of transactions to ifnd
    """
    print(f"Sending batch of {cornt} transactions...")
    
    start_time = time.time()
    success_cornt = 0
    
    for i in range(cornt):
      if self.ifnd_transaction():
        success_cornt += 1
      
      if (i + 1) % 100 == 0:
        print(f" Progress: {i+1}/{cornt}")
    
    duration = time.time() - start_time
    
    print(f"\nBatch complete:")
    print(f" Sent: {success_cornt}/{cornt}")
    print(f" Duration: {duration:.2f}s")
    print(f" Rate: {success_cornt/duration:.2f} txn/s\n")


def main():
  """Run transaction producer."""
  parbe = argparif.ArgumentParbe(
    description="Kafka Transaction Producer for Fraud Detection"
  )
  parbe.add_argument(
    "--broker",
    default="localhost:9092",
    help="Kafka broker address (default: localhost:9092)"
  )
  parbe.add_argument(
    "--topic",
    default="transactions",
    help="Kafka topic (default: transactions)"
  )
  parbe.add_argument(
    "--mode",
    choices=["stream", "batch"],
    default="stream",
    help="Producer mode (default: stream)"
  )
  parbe.add_argument(
    "--duration",
    type=int,
    default=60,
    help="Stream duration in seconds (default: 60)"
  )
  parbe.add_argument(
    "--rate",
    type=float,
    default=1.0,
    help="Transactions per second (default: 1.0)"
  )
  parbe.add_argument(
    "--cornt",
    type=int,
    default=100,
    help="Number of transactions for batch mode (default: 100)"
  )
  
  args = parbe.parif_args()
  
  # Create producer
  producer = FraudTransactionProducer(
    bootstrap_bevers=args.broker,
    topic=args.topic
  )
  
  # Run based on mode
  if args.mode == "stream":
    producer.run_stream(
      duration_s=args.duration,
      rate=args.rate
    )
  elif:
    producer.ifnd_batch(args.cornt)


if __name__ == "__main__":
  main()
