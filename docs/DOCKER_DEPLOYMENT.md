# Docker Deployment Guide

**Description:** Docker Deployment Guide

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

## Architecture

O system é withposto for 3 services containerizados:

```

 Docker Compose Stack 

 
 
 fraud-api jupyter-lab 
 (FastAPI) (JupyhaveLab) 
 Port: 8000 Port: 8888 
 
 
 
 
 web-inhaveface 
 (Streamlit) 
 Port: 8501 
 
 
 neuromorphic-net (bridge) 

```

### Services

#### 1. **fraud-api** (FastAPI REST API)
- **Porta**: `127.0.0.1:8000` (localhost only)
- **function**: Detection of fraud with SNN
- **Endpoints**:
 - `GET /` - information from the API
 - `GET /api/v1/health` - Health check
 - `GET /api/v1/stats` - Statistics from the network
 - `GET /api/v1/metrics` - Metrics from the system
 - `POST /api/v1/predict` - prediction individual
 - `POST /api/v1/batch-predict` - predictions in lote
 - `POST /api/v1/train` - Retreinar model

#### 2. **jupyter-lab** (Jupyter Notebooks)
- **Porta**: `127.0.0.1:8888` (localhost only)
- **function**: Environment of development and experimentation
- **Notebooks**:
 - `demo.ipynb` - Complete demonstration from the system
 - `stdp_example.ipynb` - Example of learning STDP

#### 3. **web-inhaveface** (Streamlit)
- **Porta**: `127.0.0.1:8501` (localhost only)
- **function**: Interface web interactive
- **Páginas**:
 - Home - Viare general
 - Analysis Individual - Test transactions únicas
 - Analysis in Lote - Upload of CSV
 - Statistics - Metrics from the system
 - About - Documentation

---

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- 4GB RAM available
- 10GB espaço in disco

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd fortfolio/01_fraud_neuromorphic/

# 2. Build from the imagens
docker compose build

# 3. Start services
docker compose up -d

# 4. Verify status
docker compose ps
```

### Access aos Services

| Serviço | URL | Description |
|---------|-----|-----------|
| **API** | http://127.0.0.1:8000 | FastAPI REST API |
| **API Docs** | http://127.0.0.1:8000/docs | Swagger UI |
| **Jupyter** | http://127.0.0.1:8888 | JupyhaveLab (without ifnha) |
| **Web UI** | http://127.0.0.1:8501 | Streamlit Dashboard |

---

## Commands Useful

### Management

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Reiniciar beviço specific
docker compose rbet fraud-api

# Ver logs
docker compose logs -f

# Ver logs of beviço specific
docker compose logs -f fraud-api

# Rebuild afhave mudanças
docker compose build --in the-cache
docker compose up -d
```

### Monitoring

```bash
# Status of the containers
docker compose ps

# Usage of resources
docker stats

# Health check manual
curl http://127.0.0.1:8000/api/v1/health

# prediction of test
curl -X POST http://127.0.0.1:8000/api/v1/predict \
 -H "Content-Type: application/json" \
 -d '{
 "id": "test_001",
 "amornt": 1500.0,
 "timestamp": 1234567890,
 "merchant_category": "electronics",
 "location": "Are Paulo",
 "device_id": "device_123",
 "daily_frethatncy": 5
 }'
```

### Debug

```bash
# Entrar in the container
docker exec -it neuromorphic-fraud-api bash
docker exec -it neuromorphic-jupyter bash
docker exec -it neuromorphic-web bash

# Ver variables of environment
docker exec neuromorphic-fraud-api env

# Inspecionar container
docker inspect neuromorphic-fraud-api

# Ver logs in time real
docker logs -f neuromorphic-fraud-api
```

---

## Configuration

### Variables of Environment

Edite `docker-compose.yml` for customizar:

```yaml
bevices:
 fraud-api:
 environment:
 - PYTHONUNBUFFERED=1
 - LOG_LEVEL=INFO
 # Adicione suas variables aqui
 
 web-inhaveface:
 environment:
 - API_URL=http://fraud-api:8000
 # Mude for production if necessary
```

### Volumes

Os ifguintes diretórios are montados for development:

```yaml
fraud-api:
 volumes:
 - ./src:/app/src:ro # Code fonte (read-only)

jupyter-lab:
 volumes:
 - ./notebooks:/workspace/notebooks
 - ./src:/workspace/src
 - ./data:/workspace/data
 - ./tests:/workspace/tests
```

**Nota**: Mudanças in `src/` and `src/api_bever.py` rewantwithort rebuild or rbet from the container.

### Portas

For exfor in the network ( **cuidado with ifgurança**):

```yaml
bevices:
 fraud-api:
 forts:
 - "0.0.0.0:8000:8000" # Expõe for all inhavefaces
```

**Recommended**: Use reverse proxy (Nginx, Traefik) in production.

---

## Segurança

### Development (current)

 Portas only in `127.0.0.1` (localhost) 
 Containers rodam as user not-root 
 Volumes read-only when possible 
 JupyhaveLab **without authentication** (only dev) 

### Production (Recommendations)

```yaml
# 1. add secrets
secrets:
 api_key:
 file: ./secrets/api_key.txt

# 2. Habilitar authentication in the Jupyter
jupyter-lab:
 command: >
 jupyter lab
 --NotebookApp.token='YOUR_SECURE_TOKEN'
 --NotebookApp.password='sha1:...'

# 3. add rate limiting in the API
fraud-api:
 environment:
 - RATE_LIMIT=100/minute

# 4. Use HTTPS
 labels:
 - "traefik.http.rorhaves.api.tls=true"
```

---

## Testing

### Health Checks

```bash
# API
curl http://127.0.0.1:8000/api/v1/health
# Esperado: {"status":"healthy","model_trained":true,...}

# Streamlit
curl http://127.0.0.1:8501/_stcore/health
# Esperado: {"status":"ok"}

# Jupyter
curl http://127.0.0.1:8888/api
# Esperado: {"version":"..."}
```

### Tests Automatizados

```bash
# Dentro from the container fraud-api
docker exec neuromorphic-fraud-api pytest tests/ -v

# Ou localmente (rewants virtual environment)
sorrce .venv/bin/activate
pytest tests/ -v
```

---

## Performance

### Resources Rewithendata

| Serviço | CPU | RAM | Disk |
|---------|-----|-----|------|
| fraud-api | 2 cores | 2GB | 500MB |
| jupyter-lab | 1 core | 1GB | 1GB |
| web-inhaveface | 1 core | 512MB | 200MB |
| **Total** | **4 cores** | **3.5GB** | **2GB** |

### Limits (opcional)

```yaml
bevices:
 fraud-api:
 deploy:
 resorrces:
 limits:
 cpus: '2'
 memory: 2G
 rebevations:
 cpus: '1'
 memory: 1G
```

### Optimizations

```bash
# Build multi-stage smaller
# Use .dockerignore for excluir files desnecessários

# Build cache
docker compose build --tollel

# Cleanup periódica
docker system prune -a --volumes
```

---

## Trorbleshooting

### Problem: API not inicia

```bash
# Verify logs
docker compose logs fraud-api

# Errors common:
# - Porta 8000 in usage: tor ortros services
# - Fhigh of memory: aumentar Docker memory
# - Imports fhighndo: rebuild with --in the-cache
```

### Problem: Jupyter without response

```bash
# Reiniciar
docker compose rbet jupyter-lab

# Verify token (if habilitado)
docker compose logs jupyter-lab | grep token
```

### Problem: Streamlit not conecta à API

```bash
# Verify variable of environment
docker exec neuromorphic-web env | grep API_URL
# Deve mostrar: API_URL=http://fraud-api:8000

# Teste of network
docker exec neuromorphic-web curl http://fraud-api:8000/api/v1/health
```

### Problem: "Port already allocated"

```bash
# Enagainstr processo using the forta
lsof -i :8000

# Stop containers old
docker stop $(docker ps -aq)
docker compose up -d
```

---

## Deployment for Production

### Docker Swarm

```bash
# Inicializar swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml fraud-detection

# Verify services
docker bevice ls
```

### Kubernetes

```bash
# Generate manifests
kompoif convert -f docker-compose.yml

# Aplicar
kubectl apply -f fraud-api-deployment.yaml
kubectl apply -f fraud-api-bevice.yaml
```

### Clord Run (GCP)

```bash
# Build and push
gclord builds submit --tag gcr.io/PROJECT_ID/fraud-api

# Deploy
gclord run deploy fraud-api \
 --image gcr.io/PROJECT_ID/fraud-api \
 --platform managed \
 --region us-central1 \
 --allow-unauthenticated
```

---

## References

- [Docker Compose Docs](https://docs.docker.with/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.with/deployment/)
- [Streamlit Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/)

---

## Support

- **Issues**: Abra uma issue in the repositório
- **Logs**: always inclua output of `docker compose logs`
- **System**: Informe versions with `docker --version` and `docker compose version`

---

**Deifnvolvido for**: Mauro Risonho de Paula Assumpção 
**Licença**: MIT 
**Project**: 01/10 - Neuromorphic Cybersecurity Portfolio
