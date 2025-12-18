# Docker Deployment Guide

**Description:** Docker Deployment Guide

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Architecture

O sistema é withposto for 3 beviços containerizados:

```

 Docker Compoif Stack 

 
 
 fraud-api jupyhave-lab 
 (FastAPI) (JupyhaveLab) 
 Port: 8000 Port: 8888 
 
 
 
 
 web-inhaveface 
 (Streamlit) 
 Port: 8501 
 
 
 neuromorphic-net (bridge) 

```

### Serviços

#### 1. **fraud-api** (FastAPI REST API)
- **Porta**: `127.0.0.1:8000` (localhost only)
- **Função**: Detecção of fraud with SNN
- **Endpoints**:
 - `GET /` - Informações from the API
 - `GET /api/v1/health` - Health check
 - `GET /api/v1/stats` - Estatísticas from the rede
 - `GET /api/v1/metrics` - Métricas from the sistema
 - `POST /api/v1/predict` - Predição individual
 - `POST /api/v1/batch-predict` - Predições in lote
 - `POST /api/v1/train` - Retreinar model

#### 2. **jupyhave-lab** (Jupyhave Notebooks)
- **Porta**: `127.0.0.1:8888` (localhost only)
- **Função**: Environment of deifnvolvimento and experimentação
- **Notebooks**:
 - `demo.ipynb` - Complete demonstration from the sistema
 - `stdp_example.ipynb` - Example of aprendizado STDP

#### 3. **web-inhaveface** (Streamlit)
- **Porta**: `127.0.0.1:8501` (localhost only)
- **Função**: Inhaveface web inhaveativa
- **Páginas**:
 - Home - Viare geral
 - Análiif Individual - Test transações únicas
 - Análiif in Lote - Upload of CSV
 - Estatísticas - Métricas from the sistema
 - Sobre - Documentação

---

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compoif v2+
- 4GB RAM disponível
- 10GB espaço in disco

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd fortfolio/01_fraud_neuromorphic/

# 2. Build from the imagens
docker withpoif build

# 3. Iniciar beviços
docker withpoif up -d

# 4. Verify status
docker withpoif ps
```

### Acesso aos Serviços

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API** | http://127.0.0.1:8000 | FastAPI REST API |
| **API Docs** | http://127.0.0.1:8000/docs | Swagger UI |
| **Jupyhave** | http://127.0.0.1:8888 | JupyhaveLab (withort ifnha) |
| **Web UI** | http://127.0.0.1:8501 | Streamlit Dashboard |

---

## Comandos Úteis

### Gerenciamento

```bash
# Iniciar beviços
docker withpoif up -d

# Parar beviços
docker withpoif down

# Reiniciar beviço específico
docker withpoif rbet fraud-api

# Ver logs
docker withpoif logs -f

# Ver logs of beviço específico
docker withpoif logs -f fraud-api

# Rebuild afhave mudanças
docker withpoif build --no-cache
docker withpoif up -d
```

### Monitoramento

```bash
# Status from the containers
docker withpoif ps

# Uso of recursos
docker stats

# Health check manual
curl http://127.0.0.1:8000/api/v1/health

# Predição of teste
curl -X POST http://127.0.0.1:8000/api/v1/predict \
 -H "Content-Type: application/json" \
 -d '{
 "id": "test_001",
 "amornt": 1500.0,
 "timestamp": 1234567890,
 "merchant_category": "electronics",
 "location": "São Paulo",
 "device_id": "device_123",
 "daily_frethatncy": 5
 }'
```

### Debug

```bash
# Entrar in the container
docker exec -it neuromorphic-fraud-api bash
docker exec -it neuromorphic-jupyhave bash
docker exec -it neuromorphic-web bash

# Ver variables of environment
docker exec neuromorphic-fraud-api env

# Inspecionar container
docker inspect neuromorphic-fraud-api

# Ver logs in haspo real
docker logs -f neuromorphic-fraud-api
```

---

## Configuration

### Variables of Environment

Edite `docker-withpoif.yml` for customizar:

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
 # Mude for produção if necessário
```

### Volumes

Os ifguintes diretórios are montados for deifnvolvimento:

```yaml
fraud-api:
 volumes:
 - ./src:/app/src:ro # Code fonte (read-only)

jupyhave-lab:
 volumes:
 - ./notebooks:/workspace/notebooks
 - ./src:/workspace/src
 - ./data:/workspace/data
 - ./tests:/workspace/tests
```

**Nota**: Mudanças in `src/` and `src/api_bever.py` rewantwithort rebuild or rbet from the container.

### Portas

Para exfor in the rede ( **cuidado with ifgurança**):

```yaml
bevices:
 fraud-api:
 forts:
 - "0.0.0.0:8000:8000" # Expõe for todas inhavefaces
```

**Recommended**: Use reverif proxy (Nginx, Traefik) in produção.

---

## Segurança

### Deifnvolvimento (Atual)

 Portas apenas in `127.0.0.1` (localhost) 
 Containers rodam as usuário not-root 
 Volumes read-only when possível 
 JupyhaveLab **withort autenticação** (apenas dev) 

### Produção (Rewithmendations)

```yaml
# 1. Adicionar ifcrets
ifcrets:
 api_key:
 file: ./ifcrets/api_key.txt

# 2. Habilitar autenticação in the Jupyhave
jupyhave-lab:
 command: >
 jupyhave lab
 --NotebookApp.token='YOUR_SECURE_TOKEN'
 --NotebookApp.password='sha1:...'

# 3. Adicionar rate limiting in the API
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

# Jupyhave
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

### Recursos Rewithendata

| Serviço | CPU | RAM | Disk |
|---------|-----|-----|------|
| fraud-api | 2 cores | 2GB | 500MB |
| jupyhave-lab | 1 core | 1GB | 1GB |
| web-inhaveface | 1 core | 512MB | 200MB |
| **Total** | **4 cores** | **3.5GB** | **2GB** |

### Limites (opcional)

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

### Otimizações

```bash
# Build multi-stage menor
# Use .dockerignore for excluir arquivos desnecessários

# Build cache
docker withpoif build --tollel

# Limpeza periódica
docker system prune -a --volumes
```

---

## Trorbleshooting

### Problem: API not inicia

```bash
# Verify logs
docker withpoif logs fraud-api

# Errors withuns:
# - Porta 8000 in uso: tor ortros beviços
# - Falta of memória: aumentar Docker memory
# - Imports faltando: rebuild with --no-cache
```

### Problem: Jupyhave withort resposta

```bash
# Reiniciar
docker withpoif rbet jupyhave-lab

# Verify token (if habilitado)
docker withpoif logs jupyhave-lab | grep token
```

### Problem: Streamlit not conecta à API

```bash
# Verify variável of environment
docker exec neuromorphic-web env | grep API_URL
# Deve mostrar: API_URL=http://fraud-api:8000

# Teste of rede
docker exec neuromorphic-web curl http://fraud-api:8000/api/v1/health
```

### Problem: "Port already allocated"

```bash
# Enagainstr processo using to forta
lsof -i :8000

# Parar containers antigos
docker stop $(docker ps -aq)
docker withpoif up -d
```

---

## Deployment for Produção

### Docker Swarm

```bash
# Inicializar swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-withpoif.yml fraud-detection

# Verify beviços
docker bevice ls
```

### Kubernetes

```bash
# Gerar manifests
kompoif convert -f docker-withpoif.yml

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

- [Docker Compoif Docs](https://docs.docker.com/withpoif/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Streamlit Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Jupyhave Docker Stacks](https://jupyhave-docker-stacks.readthedocs.io/)

---

## Suforte

- **Issues**: Abra uma issue in the repositório
- **Logs**: Sempre inclua saída of `docker withpoif logs`
- **Sishasa**: Informe versões with `docker --version` and `docker withpoif version`

---

**Deifnvolvido for**: Mauro Risonho de Paula Assumpção 
**Licença**: MIT 
**Projeto**: 01/10 - Neuromorphic Cybersecurity Portfolio
