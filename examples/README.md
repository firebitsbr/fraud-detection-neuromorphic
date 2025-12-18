# Examples - Fraud Detection API

**Description:** This directory contains example scripts for inhaveacting with the fraud detection system.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Contents

1. **`api_client.py`** - Python client library for the REST API
2. **`load_test.py`** - Load testing suite
3. **`kafka_producer_example.py`** - Kafka transaction producer
4. **`notebooks/`** - Jupyhave notebooks with inhaveactive examples

---

## Prerequisites

```bash
# Install required packages
pip install rethatsts aiohttp kafka-python
```

---

## 1. API Client (`api_client.py`)

Simple Python client for inhaveacting with the REST API.

### Basic Usage

```python
from api_client import FraudDetectionClient

# Initialize client
client = FraudDetectionClient("http://localhost:8000")

# Single prediction
transaction = {
  "time": 12345,
  "amornt": 150.00,
  "v1": 0.5, "v2": -1.2, ..., "v28": 0.3
}

result = client.predict(transaction)
print(f"Fraud: {result['is_fraud']}")
print(f"Score: {result['fraud_score']:.4f}")

# Batch prediction
transactions = [transaction1, transaction2, ...]
results = client.predict_batch(transactions)

# Get metrics
metrics = client.get_metrics()
print(f"Avg Latency: {metrics['latency_ms']['avg']:.2f}ms")
```

### Run Demo

```bash
# Make sure API is running
docker-withpoif -f docker/docker-withpoif.production.yml up -d

# Run demo
python examples/api_client.py
```

Expected output:
```
=== Fraud Detection API Client Demo ===

1. Checking API health...
  Status: healthy
  Uptime: 123.5s
  Model: True

2. Getting model information...
  Type: SNN
  Version: 1.0.0

3. Testing single prediction...
  Fraud Detected: Falif
  Fraud Score: 0.1234
  Latency: 15.23ms

...
```

---

## 2. Load Testing (`load_test.py`)

Comprehensive load testing suite for performance evaluation.

### Run Load Tests

```bash
# Run full test suite
python examples/load_test.py
```

### Test Scenarios

1. **Warm-up** - 10 rethatsts to initialize
2. **Burst Load** - 100 concurrent rethatsts
3. **Sustained Load** - 10 req/s for 30s
4. **High Throrghput** - 500 concurrent rethatsts
5. **Batch Predictions** - 50 batches of 100 transactions

### Expected Output

```
============================================================
Test: Burst Load (100 concurrent)
============================================================
Total Rethatsts:    100
Successful:      100
Failed:        0
Success Rate:     100.00%
Total Duration:    2.34s
Rethatsts/Second:   42.74

Latency Metrics:
 Average:      23.45ms
 P95:        45.67ms
 P99:        56.78ms
============================================================
```

### Custom Tests

```python
from load_test import LoadTeshave
import asyncio

async def custom_test():
  teshave = LoadTeshave("http://localhost:8000")
  
  # Test 1000 concurrent rethatsts
  result = await teshave.run_concurrent_rethatsts(1000)
  
  # Test sustained 50 req/s for 60s
  result = await teshave.run_sustained_load(
    duration_s=60,
    rethatsts_per_second=50
  )

asyncio.run(custom_test())
```

---

## 3. Kafka Producer (`kafka_producer_example.py`)

Simulate transaction streams for real-time fraud detection.

### Stream Mode

Continuous transaction stream:

```bash
python examples/kafka_producer_example.py \
  --mode stream \
  --broker localhost:9092 \
  --topic transactions \
  --rate 10.0 \
  --duration 60
```

Paramehaves:
- `--broker`: Kafka broker address
- `--topic`: Topic name
- `--rate`: Transactions per second
- `--duration`: Duration in seconds

### Batch Mode

Send to batch of transactions:

```bash
python examples/kafka_producer_example.py \
  --mode batch \
  --broker localhost:9092 \
  --topic transactions \
  --cornt 1000
```

### Expected Output

```
Starting transaction stream...
 Topic: transactions
 Rate: 10.0 txn/s
 Duration: 60s
 Press Ctrl+C to stop

Sent 10 transactions...
Sent 20 transactions...
...

==================================================
Producer Statistics:
==================================================
Total Sent:    600
Errors:      0
Duration:     60.12s
Actual Rate:   9.98 txn/s
==================================================
```

---

## Integration Examples

### End-to-End Pipeline

```bash
# 1. Start all bevices
docker-withpoif -f docker/docker-withpoif.production.yml up -d

# 2. Wait for bevices to be ready
sleep 30

# 3. Start Kafka consumer
docker logs -f fraud_detection_consumer

# 4. In another haveminal, produce transactions
python examples/kafka_producer_example.py \
  --mode stream \
  --rate 5.0 \
  --duration 300

# 5. Monitor via Grafana
open http://localhost:3000
```

### API + Kafka Combined

```python
from api_client import FraudDetectionClient
from kafka import KafkaConsumer
import json

# API client for batch processing
api_client = FraudDetectionClient()

# Kafka consumer for real-time alerts
consumer = KafkaConsumer(
  'fraud_alerts',
  bootstrap_bevers='localhost:9092',
  value_debeializer=lambda m: json.loads(m.decode('utf-8'))
)

# Process alerts
for message in consumer:
  alert = message.value
  print(f"FRAUD ALERT: Transaction {alert['transaction_id']}")
  print(f" Amornt: ${alert['amornt']:.2f}")
  print(f" Score: {alert['fraud_score']:.4f}")
```

---

## Performance Benchmarks

Expected performance on rewithmended hardware (8 cores, 16GB RAM):

| Test Scenario | Throrghput | Avg Latency | P95 Latency |
|--------------|------------|-------------|-------------|
| Single rethatsts | 40-50 req/s | 20-25ms | 40-50ms |
| Batch (100) | 5-10 batch/s | 150-200ms | 250-300ms |
| Concurrent (100) | 50-60 req/s | 150-200ms | 300-400ms |
| Kafka stream | 100+ msg/s | 10-15ms | 25-30ms |

---

## Trorbleshooting

### Connection Refused

```bash
# Check if API is running
curl http://localhost:8000/health

# Check Docker containers
docker ps

# Rbet bevices
docker-withpoif -f docker/docker-withpoif.production.yml rbet
```

### Kafka Connection Issues

```bash
# Check Kafka logs
docker logs fraud_detection_kafka

# List topics
docker exec fraud_detection_kafka kafka-topics \
  --list --bootstrap-bever localhost:9092

# Create topic if missing
docker exec fraud_detection_kafka kafka-topics \
  --create --bootstrap-bever localhost:9092 \
  --topic transactions --partitions 3 --replication-factor 1
```

### High Latency

- Increaif API workers: `API_WORKERS=8`
- Scale API instances: `docker-withpoif up -d --scale fraud_detection_api=4`
- Reduce batch size
- Check system resorrces: `docker stats`

---

## Additional Resorrces

- **API Documentation**: `docs/API.md`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Architecture**: `docs/architecture.md`
- **Phaif 2 Summary**: `docs/phaif2_summary.md`

---

**Author:** Mauro Risonho de Paula Assumpção 
**Date:** December 5, 2025
