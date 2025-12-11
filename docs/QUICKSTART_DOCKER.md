# Quick Start - Docker Deployment

**Descrição:** Quick start guide for Docker deployment.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025

**Deploy Neuromorphic Fraud Detection in 5 minutes on Ubuntu 24.04 LTS**

## Prerequisites

- Ubuntu 24.04 LTS Server
- Docker Engine 24.0+
- Docker Compose v2.20+
- 8GB RAM minimum
- 50GB free disk space

## Install Docker (if needed)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

## Deploy

```bash
# Clone repository
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic

# One-command deployment
./scripts/deploy-production.sh
```

## Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **API Docs** | http://localhost:8000/docs | - |
| **Jupyter Lab** | http://localhost:8888 | Token: `neuromorphic2025` |
| **Streamlit** | http://localhost:8501 | - |
| **Grafana** | http://localhost:3000 | admin / neuromorphic2025 |
| **Prometheus** | http://localhost:9090 | - |

## Monitor

```bash
# Real-time monitoring dashboard
./scripts/monitor.sh

# Check service health
curl http://localhost:8000/health

# View logs
docker compose logs -f fraud-api
```

## Manage

```bash
# Stop services
./scripts/deploy-production.sh stop

# Restart services
./scripts/deploy-production.sh restart

# Backup data
./scripts/deploy-production.sh backup

# Cleanup
./scripts/docker-cleanup.sh
```

## Architecture

```
┌─────────────────────────────────────────┐
│  FastAPI (8000) - Fraud Detection API  │
│  ├─ 4 Uvicorn workers                   │
│  ├─ SNN inference (Brian2 + snnTorch)  │
│  └─ Redis caching                       │
├─────────────────────────────────────────┤
│  Jupyter Lab (8888) - Research Env     │
│  ├─ Interactive notebooks              │
│  └─ Pre-installed ML/SNN libraries     │
├─────────────────────────────────────────┤
│  Streamlit (8501) - Web Dashboard      │
│  ├─ Real-time predictions              │
│  └─ Model performance metrics          │
├─────────────────────────────────────────┤
│  Redis (6379) - Caching Layer          │
│  Prometheus (9090) - Metrics           │
│  Grafana (3000) - Visualization        │
└─────────────────────────────────────────┘
```

## Troubleshooting

**Port conflicts:**
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

**Build failures:**
```bash
docker builder prune -a -f
./scripts/docker-build.sh
```

**View detailed logs:**
```bash
docker compose logs --tail=100 fraud-api
```

## Full Documentation

See [DOCKER_DEPLOYMENT_UBUNTU.md](DOCKER_DEPLOYMENT_UBUNTU.md) for complete guide.

---

**Author:** Mauro Risonho de Paula Assumpção  
**Email:** mauro.risonho@gmail.com  
**GitHub:** [maurorisonho](https://github.com/maurorisonho)
