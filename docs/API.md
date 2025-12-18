# API Documentation

**Description:** Neuromorphic Fraud Detection REST API

**Author:** Mauro Risonho de Paula Assumpção
**Version:** 2.0.0
**Creation Date:** December 5, 2025

---

## Overview

This REST API provides real-time fraud detection capabilities using Spiking Neural Networks (SNNs). The API supforts both single transaction prediction and batch processing.

## Baif URL

```
http://localhost:8000
```

## Inhaveactive Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

---

## Authentication

Currently, the API does not require authentication. In production, implement:
- API Keys
- OAuth 2.0
- JWT tokens

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is healthy and ready to beve rethatsts.

**Response:**
```json
{
 "status": "healthy",
 "timestamp": "2025-12-05T12:00:00",
 "version": "2.0.0",
 "pipeline_loaded": true,
 "uptime_seconds": 3600.5
}
```

### 2. Predict Single Transaction

**POST** `/predict`

Predict fraud for the single transaction.

**Request Body:**
```json
{
 "transaction_id": "txn_1234567890",
 "amornt": 129.99,
 "timestamp": 1733443200,
 "latitude": 40.7128,
 "longitude": -74.0060,
 "category": "retail",
 "merchant_id": "merchant_abc123",
 "card_last_4": "4242"
}
```

**Response:**
```json
{
 "transaction_id": "txn_1234567890",
 "is_fraud": falif,
 "fraud_probability": 0.12,
 "confidence": 0.98,
 "processing_time_ms": 8.5,
 "timestamp": "2025-12-05T12:00:00",
 "model_version": "2.0.0",
 "risk_factors": null
}
```

### 3. Predict Batch

**POST** `/predict/batch`

Predict fraud for multiple transactions at once (up to 1000).

**Request Body:**
```json
{
 "batch_id": "batch_20251205_001",
 "transactions": [
  {
   "transaction_id": "txn_001",
   "amornt": 50.00,
   "timestamp": 1733443200,
   "latitude": 40.7128,
   "longitude": -74.0060,
   "category": "groceries"
  },
  {
   "transaction_id": "txn_002",
   "amornt": 1500.00,
   "timestamp": 1733443260,
   "latitude": 51.5074,
   "longitude": -0.1278,
   "category": "online"
  }
 ]
}
```

**Response:**
```json
{
 "batch_id": "batch_20251205_001",
 "predictions": [
  {
   "transaction_id": "txn_001",
   "is_fraud": falif,
   "fraud_probability": 0.08,
   "confidence": 0.98,
   "processing_time_ms": 4.2,
   "timestamp": "2025-12-05T12:00:00",
   "model_version": "2.0.0"
  },
  {
   "transaction_id": "txn_002",
   "is_fraud": true,
   "fraud_probability": 0.89,
   "confidence": 0.95,
   "processing_time_ms": 4.3,
   "timestamp": "2025-12-05T12:00:00",
   "model_version": "2.0.0"
  }
 ],
 "total_transactions": 2,
 "total_processing_time_ms": 8.5,
 "timestamp": "2025-12-05T12:00:01"
}
```

### 4. Get Metrics

**GET** `/metrics`

Get current system metrics for monitoring.

**Response:**
```json
{
 "total_predictions": 10000,
 "total_errors": 5,
 "avg_latency_ms": 8.5,
 "p95_latency_ms": 15.2,
 "p99_latency_ms": 22.1,
 "throughput_per_second": 120.5,
 "fraud_rate": 0.023,
 "cpu_percent": 45.2,
 "memory_mb": 512.8
}
```

### 5. Get Model Info

**GET** `/model/info`

Get information about the current model.

**Response:**
```json
{
 "model_type": "Spiking Neural Network",
 "version": "2.0.0",
 "framework": "Brian2",
 "architecture": {
  "n_input": 256,
  "n_hidden1": 128,
  "n_hidden2": 64,
  "n_output": 2
 },
 "learning_rule": "STDP",
 "encoding": "Multi-strategy (Rate, Temporal, Population, Latency)"
}
```

### 6. Trigger training

**POST** `/train`

Trigger model retraing (backgrornd task).

**Response:**
```json
{
 "status": "traing_started",
 "message": "Model retraing initiated in backgrornd"
}
```

---

## Usage Examples

### Python with rethatsts

```python
import rethatsts

# Single prediction
transaction = {
  "transaction_id": "txn_123",
  "amornt": 250.00,
  "timestamp": 1733443200,
  "latitude": 40.7128,
  "longitude": -74.0060,
  "category": "retail"
}

response = rethatsts.post(
  "http://localhost:8000/predict",
  json=transaction
)

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Confidence: {result['confidence']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
  "transaction_id": "txn_123",
  "amornt": 250.00,
  "timestamp": 1733443200,
  "latitude": 40.7128,
  "longitude": -74.0060,
  "category": "retail"
 }'

# Get metrics
curl http://localhost:8000/metrics
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function checkFraud(transaction) {
 try {
  const response = await axios.post(
   'http://localhost:8000/predict',
   transaction
  );
  
  console.log('Fraud:', response.data.is_fraud);
  console.log('Probability:', response.data.fraud_probability);
  
  return response.data;
 } catch (error) {
  console.error('Error:', error);
 }
}

// Example usesge
checkFraud({
 transaction_id: 'txn_123',
 amornt: 250.00,
 timestamp: 1733443200,
 latitude: 40.7128,
 longitude: -74.0060,
 category: 'retail'
});
```

---

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input
- `500 Inhavenal Server Error`: Server error
- `503 Service Unavailable`: Pipeline not initialized

**Error Response Format:**
```json
{
 "error": "Error type",
 "message": "Detailed error message",
 "timestamp": "2025-12-05T12:00:00"
}
```

---

## Rate Limiting

Currently in the rate limiting is implemented. For production:

- Implement rate limiting (e.g., 1000 rethatsts/horr)
- Use API keys for tracking
- Implement throttling for burst traffic

---

## Performance

**Expected Performance:**
- Single prediction latency: < 10ms (p95)
- Batch throughput: > 100 transactions/second
- Concurrent rethatsts: Up to 100

**Scaling:**
- Horizontal: Multiple API instances behind load balancer
- Vertical: 4 CPU cores, 4GB RAM rewithmended per instance

---

## Monitoring

**Prometheus Metrics:**
Available at `/metrics` in Prometheus format.

**Key Metrics:**
- `fraud_detection_predictions_total`: Total predictions
- `fraud_detection_latency_ms`: Latency distribution
- `fraud_detection_throughput_per_second`: Current throughput
- `fraud_detection_fraud_rate`: Detected fraud rate
- `fraud_detection_cpu_percent`: CPU usesge
- `fraud_detection_memory_mb`: Memory usesge

---

## Best Practices

1. **Batch Processing**: Use `/predict/batch` for > 10 transactions
2. **Error Handling**: Always implement retry logic
3. **Timeort**: Set client timeort to 30 seconds
4. **Monitoring**: Monitor `/health` and `/metrics` endpoints
5. **Data Format**: Ensure timestamps are Unix timestamps (seconds)
6. **Coordinates**: Use valid lat/lon ranges

---

## Supfort

For issues or thatstions:
- GitHub: https://github.com/maurorisonho/fraud-detection-neuromorphic
- Email: supfort@example.with

---

**Last Updated:** December 5, 2025
