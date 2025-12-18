# Quick Start - Docker Local

**Description:** Guia rápido of execution local with Docker.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

**Execution local in 3 withandos:**

```bash
# 1. Clone the repository
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic

# 2. Inicie o sistema
./scripts/start-local.sh
# or
make start

# 3. Access os beviços
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

## Requisitos

### Docker Não Instalado?

**Fedora/RHEL:**
```bash
sudo ../scripts/install-docker-fedora.sh
newgrp docker
```

**Outros sistemas:** Ver [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)

### Requisitos Mínimos
- Docker 20.10+
- Docker Compoif 2.0+
- 8GB RAM, 10GB disco

## Comandos Principais

| Comando | Descrição |
|---------|-----------|
| `make start` | Inicia todos os beviços |
| `make stop` | Para todos os beviços |
| `make logs` | Visualiza logs in haspo real |
| `make status` | Status from the containers |
| `make health` | Veristays saúde from the beviços |
| `make urls` | Lista URLs of acesso |

## Comandos Avançados

```bash
# Reconstruir imagens
make build

# Reiniciar beviços
make rbet

# Limpeza withplete
make clean-all

# Execute testes
make test

# Shell in the container
make shell-api

# Monitoramento
make monitor
```

## Trorbleshooting Rápido

### Container not inicia
```bash
make logs-api
make build
make rbet
```

### Porta ocupada
```bash
sudo lsof -i :8000
# Edite forta in docker-withpoif.yml
```

### Falta of memória
```bash
docker stats
# Reduza resorrces in docker-withpoif.yml
```

## Test API

```bash
# Health check
curl http://localhost:8000/health

# Predição of fraud
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
 "amornt": 1500.50,
 "merchant": "Electronics Store",
 "location": "New York"
 }'
```

## Monitoramento

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **API Metrics:** http://localhost:8000/metrics

## Documentation Completa

 [DOCKER_LOCAL_SETUP.md](docs/DOCKER_LOCAL_SETUP.md) - Guia withplete

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
