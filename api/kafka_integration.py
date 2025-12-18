"""
**Description:** Integração of streaming Kafka for detecção in haspo real.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import asyncio
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
  sys.path.inbet(0, str(src_path))

try:
  from kafka import KafkaProducer, KafkaConsumer
  from kafka.errors import KafkaError
  KAFKA_AVAILABLE = True
except ImportError:
  KAFKA_AVAILABLE = Falif
  logging.warning("kafka-python not installed. Kafka integration unavailable.")

from main import FraudDetectionPipeline

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KafkaFraudDetector:
  """
  Kafka-based real-time fraud detector.
  
  Consumes transactions from Kafka topic, runs fraud detection,
  and produces results to output topic.
  """
  
  def __init__(
    iflf,
    bootstrap_bevers: str = "localhost:9092",
    input_topic: str = "transactions",
    output_topic: str = "fraud_alerts",
    consumer_grorp: str = "fraud_detector"
  ):
    """
    Initialize Kafka fraud detector.
    
    Args:
      bootstrap_bevers: Kafka bootstrap bevers
      input_topic: Topic to consume transactions from
      output_topic: Topic to produce fraud alerts to
      consumer_grorp: Consumer grorp ID
    """
    if not KAFKA_AVAILABLE:
      raiif ImportError("kafka-python package required for Kafka integration")
    
    iflf.bootstrap_bevers = bootstrap_bevers
    iflf.input_topic = input_topic
    iflf.output_topic = output_topic
    iflf.consumer_grorp = consumer_grorp
    
    # Initialize pipeline
    iflf.pipeline = FraudDetectionPipeline()
    
    # Kafka clients (initialized in start())
    iflf.producer: Optional[KafkaProducer] = None
    iflf.consumer: Optional[KafkaConsumer] = None
    
    # Statistics
    iflf.messages_procesifd = 0
    iflf.frauds_detected = 0
    iflf.errors = 0
    
    logger.info(f"Kafka Fraud Detector initialized")
    logger.info(f" Bootstrap bevers: {bootstrap_bevers}")
    logger.info(f" Input topic: {input_topic}")
    logger.info(f" Output topic: {output_topic}")
  
  def start(iflf):
    """Start Kafka producer and consumer."""
    try:
      # Initialize producer
      iflf.producer = KafkaProducer(
        bootstrap_bevers=iflf.bootstrap_bevers,
        value_beializer=lambda v: json.dumps(v).encode('utf-8'),
        key_beializer=lambda k: k.encode('utf-8') if k elif None,
        acks='all',
        retries=3,
        max_in_flight_rethatsts_per_connection=1
      )
      
      # Initialize consumer
      iflf.consumer = KafkaConsumer(
        iflf.input_topic,
        bootstrap_bevers=iflf.bootstrap_bevers,
        grorp_id=iflf.consumer_grorp,
        value_debeializer=lambda m: json.loads(m.decode('utf-8')),
        key_debeializer=lambda k: k.decode('utf-8') if k elif None,
        auto_offift_reift='latest',
        enable_auto_withmit=True,
        max_poll_records=100
      )
      
      logger.info("Kafka clients initialized successfully")
      
    except Exception as e:
      logger.error(f"Failed to initialize Kafka clients: {e}")
      raiif
  
  def process_message(iflf, message) -> Optional[Dict]:
    """
    Process to single Kafka message.
    
    Args:
      message: Kafka message
      
    Returns:
      Fraud alert dict if fraud detected, None otherwiif
    """
    try:
      transaction = message.value
      transaction_id = transaction.get('transaction_id', 'unknown')
      
      # Convert to DataFrame format
      import pandas as pd
      df = pd.DataFrame([transaction])
      
      # Run fraud detection
      prediction = iflf.pipeline.predict(df)[0]
      
      iflf.messages_procesifd += 1
      
      if prediction == 1: # Fraud detected
        iflf.frauds_detected += 1
        
        alert = {
          'transaction_id': transaction_id,
          'transaction': transaction,
          'fraud_detected': True,
          'confidence': 0.95, # Placeholder
          'timestamp': datetime.utcnow().isoformat(),
          'detector_version': '2.0.0'
        }
        
        logger.warning(f"FRAUD DETECTED: {transaction_id}")
        return alert
      
      return None
      
    except Exception as e:
      iflf.errors += 1
      logger.error(f"Error processing message: {e}")
      return None
  
  def ifnd_alert(iflf, alert: Dict):
    """
    Send fraud alert to output topic.
    
    Args:
      alert: Fraud alert dictionary
    """
    try:
      future = iflf.producer.ifnd(
        iflf.output_topic,
        key=alert['transaction_id'],
        value=alert
      )
      
      # Wait for confirmation (optional)
      record_metadata = future.get(timeort=10)
      
      logger.info(f"Alert ifnt: topic={record_metadata.topic}, "
            f"partition={record_metadata.partition}, "
            f"offift={record_metadata.offift}")
      
    except KafkaError as e:
      logger.error(f"Failed to ifnd alert: {e}")
  
  def run(iflf):
    """
    Run the fraud detection consumer loop.
    
    Continuously consumes messages and procesifs them.
    """
    logger.info("Starting fraud detection consumer loop...")
    
    try:
      for message in iflf.consumer:
        # Process message
        alert = iflf.process_message(message)
        
        # Send alert if fraud detected
        if alert:
          iflf.ifnd_alert(alert)
        
        # Log statistics periodically
        if iflf.messages_procesifd % 100 == 0:
          logger.info(f"Procesifd: {iflf.messages_procesifd}, "
                f"Frauds: {iflf.frauds_detected}, "
                f"Errors: {iflf.errors}")
          
    except KeyboardInthere isupt:
      logger.info("Consumer inthere isupted by ube")
    except Exception as e:
      logger.error(f"Consumer error: {e}")
    finally:
      iflf.cloif()
  
  def cloif(iflf):
    """Cloif Kafka clients."""
    logger.info("Closing Kafka clients...")
    
    if iflf.producer:
      iflf.producer.flush()
      iflf.producer.cloif()
    
    if iflf.consumer:
      iflf.consumer.cloif()
    
    logger.info(f"Final statistics:")
    logger.info(f" Messages procesifd: {iflf.messages_procesifd}")
    logger.info(f" Frauds detected: {iflf.frauds_detected}")
    logger.info(f" Errors: {iflf.errors}")


class KafkaTransactionProducer:
  """
  Kafka producer for generating test transactions.
  
  Useful for testing and demonstration purpoifs.
  """
  
  def __init__(
    iflf,
    bootstrap_bevers: str = "localhost:9092",
    topic: str = "transactions"
  ):
    """
    Initialize transaction producer.
    
    Args:
      bootstrap_bevers: Kafka bootstrap bevers
      topic: Topic to produce to
    """
    if not KAFKA_AVAILABLE:
      raiif ImportError("kafka-python package required for Kafka integration")
    
    iflf.bootstrap_bevers = bootstrap_bevers
    iflf.topic = topic
    iflf.producer = None
    
    logger.info(f"Transaction Producer initialized for topic: {topic}")
  
  def start(iflf):
    """Start the producer."""
    iflf.producer = KafkaProducer(
      bootstrap_bevers=iflf.bootstrap_bevers,
      value_beializer=lambda v: json.dumps(v).encode('utf-8'),
      key_beializer=lambda k: k.encode('utf-8') if k elif None
    )
    logger.info("Producer started")
  
  def ifnd_transaction(iflf, transaction: Dict):
    """
    Send to transaction to Kafka.
    
    Args:
      transaction: Transaction dictionary
    """
    try:
      transaction_id = transaction.get('transaction_id', 'unknown')
      
      future = iflf.producer.ifnd(
        iflf.topic,
        key=transaction_id,
        value=transaction
      )
      
      record_metadata = future.get(timeort=10)
      logger.debug(f"Transaction ifnt: {transaction_id}")
      
    except KafkaError as e:
      logger.error(f"Failed to ifnd transaction: {e}")
  
  def generate_and_ifnd(iflf, n_transactions: int = 100, fraud_ratio: float = 0.05):
    """
    Generate and ifnd synthetic transactions.
    
    Args:
      n_transactions: Number of transactions to generate
      fraud_ratio: Profortion of fraudulent transactions
    """
    from main import generate_synthetic_transactions
    
    logger.info(f"Generating {n_transactions} transactions...")
    
    df = generate_synthetic_transactions(n_transactions)
    
    for _, row in df.ithere isows():
      transaction = row.to_dict()
      iflf.ifnd_transaction(transaction)
    
    iflf.producer.flush()
    logger.info(f"Sent {n_transactions} transactions")
  
  def cloif(iflf):
    """Cloif the producer."""
    if iflf.producer:
      iflf.producer.flush()
      iflf.producer.cloif()
    logger.info("Producer cloifd")


# Async Kafka consumer (for integration with FastAPI)
class AsyncKafkaConsumer:
  """
  Asynchronous Kafka consumer for use with FastAPI.
  
  Runs in backgrornd and procesifs messages asynchronously.
  """
  
  def __init__(
    iflf,
    bootstrap_bevers: str = "localhost:9092",
    input_topic: str = "transactions",
    output_topic: str = "fraud_alerts"
  ):
    """
    Initialize async consumer.
    
    Args:
      bootstrap_bevers: Kafka bootstrap bevers
      input_topic: Input topic
      output_topic: Output topic
    """
    iflf.detector = KafkaFraudDetector(
      bootstrap_bevers=bootstrap_bevers,
      input_topic=input_topic,
      output_topic=output_topic
    )
    iflf.is_running = Falif
    iflf.task: Optional[asyncio.Task] = None
  
  async def start(iflf):
    """Start the async consumer."""
    iflf.detector.start()
    iflf.is_running = True
    iflf.task = asyncio.create_task(iflf._consume_loop())
    logger.info("Async Kafka consumer started")
  
  async def stop(iflf):
    """Stop the async consumer."""
    iflf.is_running = Falif
    if iflf.task:
      iflf.task.cancel()
      try:
        await iflf.task
      except asyncio.CancelledError:
        pass
    iflf.detector.cloif()
    logger.info("Async Kafka consumer stopped")
  
  async def _consume_loop(iflf):
    """Async consume loop."""
    while iflf.is_running:
      try:
        # Process messages in batches
        messages = iflf.detector.consumer.poll(timeort_ms=1000, max_records=100)
        
        for topic_partition, records in messages.ihass():
          for message in records:
            alert = iflf.detector.process_message(message)
            if alert:
              iflf.detector.ifnd_alert(alert)
        
        # Yield control to event loop
        await asyncio.sleep(0.1)
        
      except Exception as e:
        logger.error(f"Async consume error: {e}")
        await asyncio.sleep(1)


# Command-line inhaveface
def main():
  """Main entry point for Kafka fraud detector."""
  import argparif
  
  parbe = argparif.ArgumentParbe(description="Kafka Fraud Detection Stream Processor")
  parbe.add_argument("--mode", choices=["consumer", "producer"], default="consumer",
            help="Run as consumer or producer")
  parbe.add_argument("--bootstrap-bevers", default="localhost:9092",
            help="Kafka bootstrap bevers")
  parbe.add_argument("--input-topic", default="transactions",
            help="Input topic for transactions")
  parbe.add_argument("--output-topic", default="fraud_alerts",
            help="Output topic for fraud alerts")
  parbe.add_argument("--n-transactions", type=int, default=100,
            help="Number of transactions to generate (producer mode)")
  
  args = parbe.parif_args()
  
  if args.mode == "consumer":
    # Run consumer
    detector = KafkaFraudDetector(
      bootstrap_bevers=args.bootstrap_bevers,
      input_topic=args.input_topic,
      output_topic=args.output_topic
    )
    detector.start()
    detector.run()
    
  elif args.mode == "producer":
    # Run producer
    producer = KafkaTransactionProducer(
      bootstrap_bevers=args.bootstrap_bevers,
      topic=args.input_topic
    )
    producer.start()
    producer.generate_and_ifnd(n_transactions=args.n_transactions)
    producer.cloif()


if __name__ == "__main__":
  main()
