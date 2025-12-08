# Docker Deployment Guide - Ubuntu 24.04 LTS

Complete production deployment guide for the Neuromorphic Fraud Detection System.

**Author:** Mauro Risonho de Paula Assumpção  
**Email:** mauro.risonho@gmail.com  
**LinkedIn:** [linkedin.com/in/maurorisonho](https://linkedin.com/in/maurorisonho)  
**GitHub:** [github.com/maurorisonho](https://github.com/maurorisonho)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Manual Deployment](#manual-deployment)
4. [Architecture Overview](#architecture-overview)
5. [Service Configuration](#service-configuration)
6. [Monitoring & Health Checks](#monitoring--health-checks)
7. [Troubleshooting](#troubleshooting)
8. [Production Checklist](#production-checklist)

---

## Prerequisites

### System Requirements

- **OS:** Ubuntu 24.04 LTS Server (Minimal installation recommended)
- **CPU:** 4+ cores (8+ recommended for optimal performance)
- **RAM:** 8GB minimum (16GB+ recommended)
- **Disk:** 50GB free space
- **Network:** Internet connection for pulling images

### Software Requirements

#### Install Docker Engine

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

#### Post-installation Setup

```bash
# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER

# Apply group changes
newgrp docker

# Verify docker works without sudo
docker ps
```

---

## Quick Start

### One-Command Deployment

```bash
# Clone repository
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic

# Run automated deployment
./scripts/deploy-production.sh
```

This script will:
1. ✅ Check system requirements
2. ✅ Create necessary directories
3. ✅ Generate `.env` file with default credentials
4. ✅ Build Docker images (base + API)
5. ✅ Pull external images (Redis, Prometheus, Grafana)
6. ✅ Start all services
7. ✅ Run health checks
8. ✅ Display service endpoints

**Expected output:**
```
═══════════════════════════════════════════════════════════
✅ Deployment Completed Successfully
═══════════════════════════════════════════════════════════

Services are available at:
  • API:        http://localhost:8000
  • API Docs:   http://localhost:8000/docs
  • Jupyter:    http://localhost:8888
  • Streamlit:  http://localhost:8501
  • Grafana:    http://localhost:3000
  • Prometheus: http://localhost:9090
```

---

## Manual Deployment

### Step 1: Build Images

```bash
# Build with progress tracking
./scripts/docker-build.sh

# Or manually
docker build -t fraud-detection-api:ubuntu24.04 -f Dockerfile .
```

**Build time:** ~10-15 minutes (first build)  
**Image sizes:**
- Base (builder): ~6GB
- API (runtime): ~2GB

### Step 2: Configure Environment

Create `.env` file:

```bash
# Jupyter
JUPYTER_TOKEN=your-secure-token-here

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=your-secure-password-here

# API Configuration
API_WORKERS=4
API_MAX_REQUESTS=1000
LOG_LEVEL=INFO

# Redis
REDIS_MAX_MEMORY=512mb
```

### Step 3: Start Services

```bash
# Start all services in detached mode
docker compose -f docker-compose.production.yml up -d

# View logs
docker compose -f docker-compose.production.yml logs -f

# Check status
docker compose -f docker-compose.production.yml ps
```

### Step 4: Verify Deployment

```bash
# Run monitoring dashboard
./scripts/monitor.sh

# Or check individual services
curl http://localhost:8000/health
curl http://localhost:9090/-/healthy
```

---

## Architecture Overview

### Microservices Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│                  (neuromorphic-net)                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Fraud API   │  │  Jupyter Lab │  │  Streamlit   │ │
│  │   (8000)     │  │   (8888)     │  │   (8501)     │ │
│  └──────┬───────┘  └──────────────┘  └──────────────┘ │
│         │                                               │
│  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Redis      │  │  Prometheus  │  │   Grafana    │ │
│  │   (6379)     │  │   (9090)     │  │   (3000)     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Services

| Service | Port | Description | Resources |
|---------|------|-------------|-----------|
| **fraud-api** | 8000 | FastAPI backend with SNN inference | 2 CPU, 4GB RAM |
| **jupyter-lab** | 8888 | Interactive notebooks for research | 2 CPU, 8GB RAM |
| **web-interface** | 8501 | Streamlit dashboard | 1 CPU, 2GB RAM |
| **redis** | 6379 | Caching layer for predictions | 0.5 CPU, 512MB |
| **prometheus** | 9090 | Metrics collection | 0.5 CPU, 1GB RAM |
| **grafana** | 3000 | Metrics visualization | 0.5 CPU, 512MB |

### Persistent Volumes

- **fraud-models:** `/app/models` - Trained SNN models
- **jupyter-notebooks:** `/home/jovyan/notebooks` - Research notebooks
- **fraud-data:** `/app/data` - Datasets and cache
- **redis-data:** `/data` - Redis persistence
- **prometheus-data:** `/prometheus` - Metrics storage
- **grafana-data:** `/var/lib/grafana` - Dashboards and configs

---

## Service Configuration

### API Service (fraud-api)

**Dockerfile:** `Dockerfile` (multi-stage build)

**Key features:**
- Python 3.12 on Ubuntu 24.04
- Non-root user (`appuser`)
- Health checks every 30s
- Graceful shutdown (SIGTERM handling)

**Environment variables:**
```yaml
API_WORKERS: 4
API_MAX_REQUESTS: 1000
LOG_LEVEL: INFO
REDIS_URL: redis://redis:6379
```

**Health check:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "1.0.0"}
```

### Jupyter Lab

**Port:** 8888  
**Token:** Set in `.env` as `JUPYTER_TOKEN`

**Access:**
```bash
# Open browser
http://localhost:8888?token=<JUPYTER_TOKEN>

# Or use auto-login URL from logs
docker compose logs jupyter-lab | grep token=
```

**Pre-installed packages:**
- Brian2 2.10.1 (SNN simulator)
- snnTorch 0.9.1 (PyTorch SNN library)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn, PyTorch

### Streamlit Dashboard

**Port:** 8501

**Features:**
- Real-time fraud prediction interface
- Model performance metrics
- Transaction visualization

**Configuration:**
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
```

### Redis Cache

**Port:** 6379  
**Max Memory:** 512MB  
**Eviction Policy:** `allkeys-lru`

**Test connection:**
```bash
docker exec -it neuromorphic-fraud-detection-redis-1 redis-cli ping
# Expected: PONG
```

### Prometheus Monitoring

**Port:** 9090  
**Scrape Interval:** 15s

**Targets:**
- API: `http://fraud-api:8000/metrics`
- Redis: `http://redis:6379`

**Example query:**
```promql
rate(http_requests_total[5m])
```

### Grafana Dashboards

**Port:** 3000  
**Credentials:** `admin / neuromorphic2025` (change in `.env`)

**Pre-configured dashboards:**
- API Performance (request rate, latency, errors)
- Redis Metrics (hit rate, memory usage)
- System Resources (CPU, memory, disk)

---

## Monitoring & Health Checks

### Real-time Dashboard

```bash
# Launch interactive monitoring
./scripts/monitor.sh

# Single status check
./scripts/monitor.sh once

# Service-specific monitoring
./scripts/monitor.sh services
./scripts/monitor.sh metrics
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║     Neuromorphic Fraud Detection - Real-time Monitoring         ║
╚══════════════════════════════════════════════════════════════════╝

════════════════════════ Services Status ════════════════════════
SERVICE              STATUS          HEALTH          CPU         MEMORY
────────────────────────────────────────────────────────────────────
fraud-api            ● Running       ✓ Healthy       15.23%      1.2GiB / 4GiB
jupyter-lab          ● Running       − No check      8.45%       2.8GiB / 8GiB
web-interface        ● Running       ✓ Healthy       3.12%       512MiB / 2GiB
redis                ● Running       ✓ Healthy       0.45%       128MiB / 512MiB
prometheus           ● Running       ✓ Healthy       1.23%       256MiB / 1GiB
grafana              ● Running       ✓ Healthy       0.89%       128MiB / 512MiB
```

### Health Check Endpoints

| Service | Endpoint | Expected Response |
|---------|----------|-------------------|
| API | `GET /health` | `{"status": "healthy"}` |
| Prometheus | `GET /-/healthy` | 200 OK |
| Grafana | `GET /api/health` | `{"database": "ok"}` |
| Redis | `redis-cli PING` | `PONG` |

### Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f fraud-api

# Last 100 lines
docker compose logs --tail=100 fraud-api

# With timestamps
docker compose logs -f -t fraud-api
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**Solution:**
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Or change port in docker-compose.production.yml
ports:
  - "8001:8000"  # External:Internal
```

#### 2. Out of Memory

**Error:**
```
Container killed due to memory limit
```

**Solution:**
```yaml
# Increase memory limit in docker-compose.production.yml
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G
```

#### 3. Permission Denied

**Error:**
```
docker: Got permission denied while trying to connect to Docker daemon
```

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 4. Image Build Fails

**Error:**
```
ERROR: failed to solve: process "/bin/sh -c pip install ..." did not complete successfully
```

**Solution:**
```bash
# Clean build cache
docker builder prune -a -f

# Rebuild with no cache
docker build --no-cache -t fraud-detection-api:ubuntu24.04 .
```

#### 5. Service Unhealthy

**Check logs:**
```bash
docker compose logs fraud-api
docker inspect neuromorphic-fraud-detection-fraud-api-1
```

**Restart service:**
```bash
docker compose restart fraud-api
```

### Debug Commands

```bash
# Inspect container
docker inspect neuromorphic-fraud-detection-fraud-api-1

# Execute commands inside container
docker exec -it neuromorphic-fraud-detection-fraud-api-1 bash

# Check container processes
docker top neuromorphic-fraud-detection-fraud-api-1

# View resource usage
docker stats

# Check network
docker network inspect neuromorphic-fraud-detection_neuromorphic-net
```

---

## Production Checklist

### Security

- [ ] Change default passwords in `.env`
- [ ] Use secrets management (Docker Secrets, HashiCorp Vault)
- [ ] Enable TLS/SSL for API endpoints
- [ ] Configure firewall rules (ufw, iptables)
- [ ] Run containers as non-root users
- [ ] Enable Docker Content Trust
- [ ] Regularly update base images

### Performance

- [ ] Tune API worker count (`API_WORKERS`)
- [ ] Optimize Redis memory limit
- [ ] Configure kernel parameters (`sysctl`)
- [ ] Enable Docker BuildKit
- [ ] Use multi-stage builds
- [ ] Implement request rate limiting

### Monitoring

- [ ] Configure Grafana alerts
- [ ] Set up email/Slack notifications
- [ ] Monitor disk usage (Prometheus)
- [ ] Track API latency (p50, p95, p99)
- [ ] Log aggregation (ELK stack, Loki)

### Backup & Recovery

- [ ] Automate volume backups
- [ ] Test restore procedures
- [ ] Document disaster recovery plan
- [ ] Use persistent volumes for critical data
- [ ] Implement database replication (if applicable)

### Scalability

- [ ] Use Docker Swarm or Kubernetes for orchestration
- [ ] Implement horizontal scaling (load balancer)
- [ ] Use external Redis cluster
- [ ] Configure auto-scaling policies
- [ ] Optimize container resource limits

### Maintenance

- [ ] Schedule regular updates
- [ ] Implement CI/CD pipeline (GitHub Actions)
- [ ] Monitor container security vulnerabilities (Trivy, Snyk)
- [ ] Clean up old images/containers
- [ ] Review and rotate logs

---

## Management Commands

### Lifecycle Management

```bash
# Start services
./scripts/deploy-production.sh deploy

# Stop services
./scripts/deploy-production.sh stop

# Restart services
./scripts/deploy-production.sh restart

# View status
./scripts/deploy-production.sh status

# View logs
./scripts/deploy-production.sh logs

# Backup data
./scripts/deploy-production.sh backup
```

### Cleanup

```bash
# Interactive cleanup menu
./scripts/docker-cleanup.sh

# Clean specific resources
./scripts/docker-cleanup.sh containers
./scripts/docker-cleanup.sh images
./scripts/docker-cleanup.sh volumes  # DANGEROUS
./scripts/docker-cleanup.sh cache

# Full cleanup (DANGEROUS)
./scripts/docker-cleanup.sh full
```

---

## Performance Tuning

### Docker Daemon Configuration

Edit `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
```

Restart Docker:
```bash
sudo systemctl restart docker
```

### Kernel Parameters

Edit `/etc/sysctl.conf`:

```bash
# Network tuning
net.core.somaxconn = 1024
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.ip_local_port_range = 10000 65535

# File descriptors
fs.file-max = 100000

# Apply changes
sudo sysctl -p
```

---

## Support & Resources

- **Documentation:** [docs/](docs/)
- **API Reference:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **GitHub Issues:** [github.com/maurorisonho/fraud-detection-neuromorphic/issues](https://github.com/maurorisonho/fraud-detection-neuromorphic/issues)
- **Email:** mauro.risonho@gmail.com

---

## License

MIT License - See [LICENSE](../LICENSE) file for details.

---

**Last Updated:** December 2025  
**Version:** 1.0.0
