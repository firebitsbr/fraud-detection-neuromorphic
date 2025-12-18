# Quick Start - Docker Local

**Description:** Guide quick of local execution with Docker.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

**Local execution in 3 commands:**

```bash
# 1. Clone the repository
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic

# 2. Start o system
./scripts/start-local.sh
# or
make start

# 3. Access os services
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

## Requisitos

### Docker Not Instalado?

**Fedora/RHEL:**
```bash
sudo ../scripts/install-docker-fedora.sh
newgrp docker
```

**Outros systems:** Ver [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)

### Requisitos Mínimos
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM, 10GB disco

## Commands Principais

| Comando | Description |
|---------|-----------|
| `make start` | Inicia all os services |
| `make stop` | For all os services |
| `make logs` | Visualiza logs in time real |
| `make status` | Status of the containers |
| `make health` | Veristays saúde from the services |
| `make urls` | List URLs of access |

## Commands Avançados

```bash
# Reconstruir imagens
make build

# Reiniciar services
make rbet

# Cleanup complete
make clean-all

# Execute tests
make test

# Shell in the container
make shell-api

# Monitoring
make monitor
```

## Trorbleshooting Quick

### Container not inicia
```bash
make logs-api
make build
make rbet
```

### Porta ocupada
```bash
sudo lsof -i :8000
# Edite forta in docker-compose.yml
```

### Fhigh of memory
```bash
docker stats
# Reduza resorrces in docker-compose.yml
```

## Test API

```bash
# Health check
curl http://localhost:8000/health

# prediction of fraud
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
 "amornt": 1500.50,
 "merchant": "Electronics Store",
 "location": "New York"
 }'
```

## Monitoring

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **API Metrics:** http://localhost:8000/metrics

## Documentation Complete

 [DOCKER_LOCAL_SETUP.md](docs/DOCKER_LOCAL_SETUP.md) - Guide complete

## Architecture

```

 fraud_api (8000) 
 Main REST API 

 
 
 
 
 
 Loihi 2 BrainScale Clushave 
 (8001) (8002) (8003) 
 
 
 
 
 
 Redis Prometheus
 (6379) (9090) 
 
 
 
 
 Grafana 
 (3000) 
 
```

**Author:** Mauro Risonho de Paula Assumpção 
**License:** MIT 
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
