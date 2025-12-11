# API Documentation

**Descrição:** Neuromorphic Fraud Detection REST API

**Autor:** Mauro Risonho de Paula Assumpção
**Versão:** 2.0.0
**Data de Criação:** 5 de Dezembro de 2025

---

## Overview

This REST API provides real-time fraud detection capabilities using Spiking Neural Networks (SNNs). The API supports both single transaction prediction and batch processing.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

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

Check if the API is healthy and ready to serve requests.

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

Predict fraud for a single transaction.

**Request Body:**
```json
{
  "transaction_id": "txn_1234567890",
  "amount": 129.99,
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
  "is_fraud": false,
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
      "amount": 50.00,
      "timestamp": 1733443200,
      "latitude": 40.7128,
      "longitude": -74.0060,
      "category": "groceries"
    },
    {
      "transaction_id": "txn_002",
      "amount": 1500.00,
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
      "is_fraud": false,
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

### 6. Trigger Training

**POST** `/train`

Trigger model retraining (background task).

**Response:**
```json
{
  "status": "training_started",
  "message": "Model retraining initiated in background"
}
```

---

## Usage Examples

### Python with requests

```python
import requests

# Single prediction
transaction = {
    "transaction_id": "txn_123",
    "amount": 250.00,
    "timestamp": 1733443200,
    "latitude": 40.7128,
    "longitude": -74.0060,
    "category": "retail"
}

response = requests.post(
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
    "amount": 250.00,
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

// Example usage
checkFraud({
  transaction_id: 'txn_123',
  amount: 250.00,
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
- `500 Internal Server Error`: Server error
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

Currently no rate limiting is implemented. For production:

- Implement rate limiting (e.g., 1000 requests/hour)
- Use API keys for tracking
- Implement throttling for burst traffic

---

## Performance

**Expected Performance:**
- Single prediction latency: < 10ms (p95)
- Batch throughput: > 100 transactions/second
- Concurrent requests: Up to 100

**Scaling:**
- Horizontal: Multiple API instances behind load balancer
- Vertical: 4 CPU cores, 4GB RAM recommended per instance

---

## Monitoring

**Prometheus Metrics:**
Available at `/metrics` in Prometheus format.

**Key Metrics:**
- `fraud_detection_predictions_total`: Total predictions
- `fraud_detection_latency_ms`: Latency distribution
- `fraud_detection_throughput_per_second`: Current throughput
- `fraud_detection_fraud_rate`: Detected fraud rate
- `fraud_detection_cpu_percent`: CPU usage
- `fraud_detection_memory_mb`: Memory usage

---

## Best Practices

1. **Batch Processing**: Use `/predict/batch` for > 10 transactions
2. **Error Handling**: Always implement retry logic
3. **Timeout**: Set client timeout to 30 seconds
4. **Monitoring**: Monitor `/health` and `/metrics` endpoints
5. **Data Format**: Ensure timestamps are Unix timestamps (seconds)
6. **Coordinates**: Use valid lat/lon ranges

---

## Support

For issues or questions:
- GitHub: https://github.com/maurorisonho/fraud-detection-neuromorphic
- Email: support@example.com

---

**Last Updated:** December 5, 2025
