# Deployment Guide - Production

**Neuromorphic Fraud Detection System**

Author: Mauro Risonho de Paula Assumpção  
Version: 2.0.0  
Date: December 5, 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Scaling](#scaling)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB
- OS: Linux (Ubuntu 20.04+, CentOS 8+)

**Recommended:**
- CPU: 8 cores
- RAM: 16 GB
- Disk: 50 GB SSD
- OS: Ubuntu 22.04 LTS

### Software Requirements

- Docker 24.0+
- Docker Compose 2.20+
- Git 2.30+
- Python 3.10+ (for local development)

### Network Requirements

- Open ports:
  - 8000: API
  - 9092: Kafka
  - 9090: Prometheus
  - 3000: Grafana
  - 2181: Zookeeper

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic
```

### 2. Deploy with Script

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

This will:
- Pull/build Docker images
- Start all services
- Run health checks
- Display service URLs

### 3. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/api/docs

# Access monitoring dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

---

## Production Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Navigate to project directory
cd fraud-detection-neuromorphic

# Start all services
docker-compose -f docker/docker-compose.production.yml up -d

# Check status
docker-compose -f docker/docker-compose.production.yml ps

# View logs
docker-compose -f docker/docker-compose.production.yml logs -f
```

### Option 2: Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment
kubectl get pods -n fraud-detection
kubectl logs -f deployment/fraud-detection-api -n fraud-detection
```

### Option 3: Manual Docker

```bash
# Build image
docker build -t fraud-detection:latest -f docker/Dockerfile.production .

# Run container
docker run -d \
  --name fraud-detection-api \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/fraud_snn.pkl \
  -v $(pwd)/models:/app/models \
  fraud-detection:latest
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=/app/models/fraud_snn.pkl

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
KAFKA_INPUT_TOPIC=transactions
KAFKA_OUTPUT_TOPIC=fraud_alerts

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090

# Security
API_KEY_ENABLED=false
API_KEY=your-secret-key-here
```

### Model Configuration

Place trained model in `models/` directory:

```bash
# Train model
python -m src.main --train --output models/fraud_snn.pkl

# Or use pre-trained model
cp path/to/pretrained/fraud_snn.pkl models/
```

### Kafka Topics

Create Kafka topics:

```bash
# Enter Kafka container
docker exec -it fraud_detection_kafka bash

# Create topics
kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic transactions \
  --partitions 3 \
  --replication-factor 1

kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic fraud_alerts \
  --partitions 3 \
  --replication-factor 1
```

---

## Monitoring

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Import dashboards:
   - Fraud Detection Overview
   - System Metrics
   - Kafka Metrics

### Prometheus Alerts

Configure alerts in `docker/prometheus.yml`:

```yaml
groups:
  - name: fraud_detection_alerts
    rules:
      - alert: HighLatency
        expr: fraud_detection_latency_ms{quantile="0.95"} > 50
        for: 5m
        annotations:
          summary: "High prediction latency detected"
      
      - alert: HighErrorRate
        expr: rate(fraud_detection_errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Kafka health
docker exec fraud_detection_kafka kafka-broker-api-versions \
  --bootstrap-server localhost:9092

# Prometheus health
curl http://localhost:9090/-/healthy

# Grafana health
curl http://localhost:3000/api/health
```

---

## Troubleshooting

### Common Issues

#### 1. API Not Starting

```bash
# Check logs
docker logs fraud_detection_api

# Common causes:
# - Model file not found
# - Port already in use
# - Insufficient memory

# Solutions:
docker restart fraud_detection_api
docker-compose -f docker/docker-compose.production.yml restart
```

#### 2. Kafka Connection Issues

```bash
# Check Kafka logs
docker logs fraud_detection_kafka

# Check topics
docker exec fraud_detection_kafka kafka-topics \
  --list --bootstrap-server localhost:9092

# Recreate topics if needed
```

#### 3. High Memory Usage

```bash
# Check memory
docker stats

# Reduce workers
docker-compose -f docker/docker-compose.production.yml up -d \
  --scale fraud_detection_api=2

# Or adjust in docker-compose.yml:
environment:
  - API_WORKERS=2
```

#### 4. Slow Predictions

```bash
# Check CPU usage
docker stats

# Scale horizontally
docker-compose -f docker/docker-compose.production.yml up -d \
  --scale fraud_detection_api=4

# Check logs for bottlenecks
docker logs fraud_detection_api --tail 100
```

### Debug Mode

Enable debug logging:

```bash
docker-compose -f docker/docker-compose.production.yml up -d \
  -e LOG_LEVEL=DEBUG
```

---

## Scaling

### Horizontal Scaling

#### With Docker Compose

```bash
# Scale API instances
docker-compose -f docker/docker-compose.production.yml up -d \
  --scale fraud_detection_api=4
```

#### With Kubernetes

```bash
# Scale deployment
kubectl scale deployment fraud-detection-api \
  --replicas=4 \
  -n fraud-detection

# Auto-scaling
kubectl autoscale deployment fraud-detection-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n fraud-detection
```

### Load Balancing

#### NGINX Configuration

```nginx
upstream fraud_detection {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
    server api4:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://fraud_detection;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Database Optimization

For production with real datasets:

```bash
# Use PostgreSQL for transaction storage
docker run -d \
  --name fraud_detection_db \
  -e POSTGRES_DB=fraud_detection \
  -e POSTGRES_USER=fraud_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:15
```

---

## Backup and Recovery

### Backup Models

```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Upload to S3
aws s3 cp models_backup_*.tar.gz s3://your-bucket/backups/
```

### Backup Configuration

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  docker/ .env prometheus.yml

# Version control
git add -A
git commit -m "Production config $(date +%Y%m%d)"
git push
```

---

## Security Checklist

- [ ] Change default passwords (Grafana, etc.)
- [ ] Enable API authentication
- [ ] Use HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Enable rate limiting
- [ ] Implement API keys
- [ ] Regular security updates
- [ ] Audit logging enabled
- [ ] Secrets in environment variables
- [ ] Network isolation (VPC)

---

## Maintenance

### Regular Tasks

**Daily:**
- Check health endpoints
- Review error logs
- Monitor metrics dashboards

**Weekly:**
- Review performance metrics
- Check disk usage
- Update dependencies

**Monthly:**
- Security updates
- Model retraining
- Backup verification
- Capacity planning

---

## Support

**Documentation:**
- API Reference: `docs/API.md`
- Architecture: `docs/architecture.md`
- Phase 2 Summary: `docs/phase2_summary.md`

**Contact:**
- GitHub Issues: https://github.com/maurorisonho/fraud-detection-neuromorphic/issues
- Email: maurorisonho@example.com

---

**Last Updated:** December 5, 2025
