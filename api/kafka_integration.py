"""
**Descrição:** Integração de streaming Kafka para detecção em tempo real.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
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
    sys.path.insert(0, str(src_path))

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
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
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "transactions",
        output_topic: str = "fraud_alerts",
        consumer_group: str = "fraud_detector"
    ):
        """
        Initialize Kafka fraud detector.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic to consume transactions from
            output_topic: Topic to produce fraud alerts to
            consumer_group: Consumer group ID
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python package required for Kafka integration")
        
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer_group = consumer_group
        
        # Initialize pipeline
        self.pipeline = FraudDetectionPipeline()
        
        # Kafka clients (initialized in start())
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        
        # Statistics
        self.messages_processed = 0
        self.frauds_detected = 0
        self.errors = 0
        
        logger.info(f"Kafka Fraud Detector initialized")
        logger.info(f"  Bootstrap servers: {bootstrap_servers}")
        logger.info(f"  Input topic: {input_topic}")
        logger.info(f"  Output topic: {output_topic}")
    
    def start(self):
        """Start Kafka producer and consumer."""
        try:
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            
            # Initialize consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=100
            )
            
            logger.info("Kafka clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka clients: {e}")
            raise
    
    def process_message(self, message) -> Optional[Dict]:
        """
        Process a single Kafka message.
        
        Args:
            message: Kafka message
            
        Returns:
            Fraud alert dict if fraud detected, None otherwise
        """
        try:
            transaction = message.value
            transaction_id = transaction.get('transaction_id', 'unknown')
            
            # Convert to DataFrame format
            import pandas as pd
            df = pd.DataFrame([transaction])
            
            # Run fraud detection
            prediction = self.pipeline.predict(df)[0]
            
            self.messages_processed += 1
            
            if prediction == 1:  # Fraud detected
                self.frauds_detected += 1
                
                alert = {
                    'transaction_id': transaction_id,
                    'transaction': transaction,
                    'fraud_detected': True,
                    'confidence': 0.95,  # Placeholder
                    'timestamp': datetime.utcnow().isoformat(),
                    'detector_version': '2.0.0'
                }
                
                logger.warning(f"FRAUD DETECTED: {transaction_id}")
                return alert
            
            return None
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Error processing message: {e}")
            return None
    
    def send_alert(self, alert: Dict):
        """
        Send fraud alert to output topic.
        
        Args:
            alert: Fraud alert dictionary
        """
        try:
            future = self.producer.send(
                self.output_topic,
                key=alert['transaction_id'],
                value=alert
            )
            
            # Wait for confirmation (optional)
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Alert sent: topic={record_metadata.topic}, "
                       f"partition={record_metadata.partition}, "
                       f"offset={record_metadata.offset}")
            
        except KafkaError as e:
            logger.error(f"Failed to send alert: {e}")
    
    def run(self):
        """
        Run the fraud detection consumer loop.
        
        Continuously consumes messages and processes them.
        """
        logger.info("Starting fraud detection consumer loop...")
        
        try:
            for message in self.consumer:
                # Process message
                alert = self.process_message(message)
                
                # Send alert if fraud detected
                if alert:
                    self.send_alert(alert)
                
                # Log statistics periodically
                if self.messages_processed % 100 == 0:
                    logger.info(f"Processed: {self.messages_processed}, "
                               f"Frauds: {self.frauds_detected}, "
                               f"Errors: {self.errors}")
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close Kafka clients."""
        logger.info("Closing Kafka clients...")
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        logger.info(f"Final statistics:")
        logger.info(f"  Messages processed: {self.messages_processed}")
        logger.info(f"  Frauds detected: {self.frauds_detected}")
        logger.info(f"  Errors: {self.errors}")


class KafkaTransactionProducer:
    """
    Kafka producer for generating test transactions.
    
    Useful for testing and demonstration purposes.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "transactions"
    ):
        """
        Initialize transaction producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to produce to
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python package required for Kafka integration")
        
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        
        logger.info(f"Transaction Producer initialized for topic: {topic}")
    
    def start(self):
        """Start the producer."""
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        logger.info("Producer started")
    
    def send_transaction(self, transaction: Dict):
        """
        Send a transaction to Kafka.
        
        Args:
            transaction: Transaction dictionary
        """
        try:
            transaction_id = transaction.get('transaction_id', 'unknown')
            
            future = self.producer.send(
                self.topic,
                key=transaction_id,
                value=transaction
            )
            
            record_metadata = future.get(timeout=10)
            logger.debug(f"Transaction sent: {transaction_id}")
            
        except KafkaError as e:
            logger.error(f"Failed to send transaction: {e}")
    
    def generate_and_send(self, n_transactions: int = 100, fraud_ratio: float = 0.05):
        """
        Generate and send synthetic transactions.
        
        Args:
            n_transactions: Number of transactions to generate
            fraud_ratio: Proportion of fraudulent transactions
        """
        from main import generate_synthetic_transactions
        
        logger.info(f"Generating {n_transactions} transactions...")
        
        df = generate_synthetic_transactions(n_transactions)
        
        for _, row in df.iterrows():
            transaction = row.to_dict()
            self.send_transaction(transaction)
        
        self.producer.flush()
        logger.info(f"Sent {n_transactions} transactions")
    
    def close(self):
        """Close the producer."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
        logger.info("Producer closed")


# Async Kafka consumer (for integration with FastAPI)
class AsyncKafkaConsumer:
    """
    Asynchronous Kafka consumer for use with FastAPI.
    
    Runs in background and processes messages asynchronously.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "transactions",
        output_topic: str = "fraud_alerts"
    ):
        """
        Initialize async consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            input_topic: Input topic
            output_topic: Output topic
        """
        self.detector = KafkaFraudDetector(
            bootstrap_servers=bootstrap_servers,
            input_topic=input_topic,
            output_topic=output_topic
        )
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the async consumer."""
        self.detector.start()
        self.is_running = True
        self.task = asyncio.create_task(self._consume_loop())
        logger.info("Async Kafka consumer started")
    
    async def stop(self):
        """Stop the async consumer."""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.detector.close()
        logger.info("Async Kafka consumer stopped")
    
    async def _consume_loop(self):
        """Async consume loop."""
        while self.is_running:
            try:
                # Process messages in batches
                messages = self.detector.consumer.poll(timeout_ms=1000, max_records=100)
                
                for topic_partition, records in messages.items():
                    for message in records:
                        alert = self.detector.process_message(message)
                        if alert:
                            self.detector.send_alert(alert)
                
                # Yield control to event loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Async consume error: {e}")
                await asyncio.sleep(1)


# Command-line interface
def main():
    """Main entry point for Kafka fraud detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kafka Fraud Detection Stream Processor")
    parser.add_argument("--mode", choices=["consumer", "producer"], default="consumer",
                       help="Run as consumer or producer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                       help="Kafka bootstrap servers")
    parser.add_argument("--input-topic", default="transactions",
                       help="Input topic for transactions")
    parser.add_argument("--output-topic", default="fraud_alerts",
                       help="Output topic for fraud alerts")
    parser.add_argument("--n-transactions", type=int, default=100,
                       help="Number of transactions to generate (producer mode)")
    
    args = parser.parse_args()
    
    if args.mode == "consumer":
        # Run consumer
        detector = KafkaFraudDetector(
            bootstrap_servers=args.bootstrap_servers,
            input_topic=args.input_topic,
            output_topic=args.output_topic
        )
        detector.start()
        detector.run()
        
    elif args.mode == "producer":
        # Run producer
        producer = KafkaTransactionProducer(
            bootstrap_servers=args.bootstrap_servers,
            topic=args.input_topic
        )
        producer.start()
        producer.generate_and_send(n_transactions=args.n_transactions)
        producer.close()


if __name__ == "__main__":
    main()
