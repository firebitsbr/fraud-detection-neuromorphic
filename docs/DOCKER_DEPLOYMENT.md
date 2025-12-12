# Docker Deployment Guide

**Descrição:** Docker Deployment Guide

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025

## Arquitetura

O sistema é composto por 3 serviços containerizados:

```

 Docker Compose Stack 

 
 
 fraud-api jupyter-lab 
 (FastAPI) (JupyterLab) 
 Port: 8000 Port: 8888 
 
 
 
 
 web-interface 
 (Streamlit) 
 Port: 8501 
 
 
 neuromorphic-net (bridge) 

```

### Serviços

#### 1. **fraud-api** (FastAPI REST API)
- **Porta**: `127.0.0.1:8000` (localhost only)
- **Função**: Detecção de fraude com SNN
- **Endpoints**:
 - `GET /` - Informações da API
 - `GET /api/v1/health` - Health check
 - `GET /api/v1/stats` - Estatísticas da rede
 - `GET /api/v1/metrics` - Métricas do sistema
 - `POST /api/v1/predict` - Predição individual
 - `POST /api/v1/batch-predict` - Predições em lote
 - `POST /api/v1/train` - Retreinar modelo

#### 2. **jupyter-lab** (Jupyter Notebooks)
- **Porta**: `127.0.0.1:8888` (localhost only)
- **Função**: Ambiente de desenvolvimento e experimentação
- **Notebooks**:
 - `demo.ipynb` - Demonstração completa do sistema
 - `stdp_example.ipynb` - Exemplo de aprendizado STDP

#### 3. **web-interface** (Streamlit)
- **Porta**: `127.0.0.1:8501` (localhost only)
- **Função**: Interface web interativa
- **Páginas**:
 - Home - Visão geral
 - Análise Individual - Testar transações únicas
 - Análise em Lote - Upload de CSV
 - Estatísticas - Métricas do sistema
 - Sobre - Documentação

---

## Quick Start

### Pré-requisitos

- Docker Engine 20.10+
- Docker Compose v2+
- 4GB RAM disponível
- 10GB espaço em disco

### Instalação

```bash
# 1. Clone o repositório
git clone <repo-url>
cd portfolio/01_fraud_neuromorphic/

# 2. Build das imagens
docker compose build

# 3. Iniciar serviços
docker compose up -d

# 4. Verificar status
docker compose ps
```

### Acesso aos Serviços

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API** | http://127.0.0.1:8000 | FastAPI REST API |
| **API Docs** | http://127.0.0.1:8000/docs | Swagger UI |
| **Jupyter** | http://127.0.0.1:8888 | JupyterLab (sem senha) |
| **Web UI** | http://127.0.0.1:8501 | Streamlit Dashboard |

---

## Comandos Úteis

### Gerenciamento

```bash
# Iniciar serviços
docker compose up -d

# Parar serviços
docker compose down

# Reiniciar serviço específico
docker compose restart fraud-api

# Ver logs
docker compose logs -f

# Ver logs de serviço específico
docker compose logs -f fraud-api

# Rebuild após mudanças
docker compose build --no-cache
docker compose up -d
```

### Monitoramento

```bash
# Status dos containers
docker compose ps

# Uso de recursos
docker stats

# Health check manual
curl http://127.0.0.1:8000/api/v1/health

# Predição de teste
curl -X POST http://127.0.0.1:8000/api/v1/predict \
 -H "Content-Type: application/json" \
 -d '{
 "id": "test_001",
 "amount": 1500.0,
 "timestamp": 1234567890,
 "merchant_category": "electronics",
 "location": "São Paulo",
 "device_id": "device_123",
 "daily_frequency": 5
 }'
```

### Debug

```bash
# Entrar no container
docker exec -it neuromorphic-fraud-api bash
docker exec -it neuromorphic-jupyter bash
docker exec -it neuromorphic-web bash

# Ver variáveis de ambiente
docker exec neuromorphic-fraud-api env

# Inspecionar container
docker inspect neuromorphic-fraud-api

# Ver logs em tempo real
docker logs -f neuromorphic-fraud-api
```

---

## Configuração

### Variáveis de Ambiente

Edite `docker-compose.yml` para customizar:

```yaml
services:
 fraud-api:
 environment:
 - PYTHONUNBUFFERED=1
 - LOG_LEVEL=INFO
 # Adicione suas variáveis aqui
 
 web-interface:
 environment:
 - API_URL=http://fraud-api:8000
 # Mude para produção se necessário
```

### Volumes

Os seguintes diretórios são montados para desenvolvimento:

```yaml
fraud-api:
 volumes:
 - ./src:/app/src:ro # Código fonte (read-only)

jupyter-lab:
 volumes:
 - ./notebooks:/workspace/notebooks
 - ./src:/workspace/src
 - ./data:/workspace/data
 - ./tests:/workspace/tests
```

**Nota**: Mudanças em `src/` e `src/api_server.py` requerem rebuild ou restart do container.

### Portas

Para expor na rede ( **cuidado com segurança**):

```yaml
services:
 fraud-api:
 ports:
 - "0.0.0.0:8000:8000" # Expõe para todas interfaces
```

**Recomendado**: Use reverse proxy (Nginx, Traefik) em produção.

---

## Segurança

### Desenvolvimento (Atual)

 Portas apenas em `127.0.0.1` (localhost) 
 Containers rodam como usuário não-root 
 Volumes read-only quando possível 
 JupyterLab **sem autenticação** (apenas dev) 

### Produção (Recomendações)

```yaml
# 1. Adicionar secrets
secrets:
 api_key:
 file: ./secrets/api_key.txt

# 2. Habilitar autenticação no Jupyter
jupyter-lab:
 command: >
 jupyter lab
 --NotebookApp.token='YOUR_SECURE_TOKEN'
 --NotebookApp.password='sha1:...'

# 3. Adicionar rate limiting no API
fraud-api:
 environment:
 - RATE_LIMIT=100/minute

# 4. Usar HTTPS
 labels:
 - "traefik.http.routers.api.tls=true"
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

### Testes Automatizados

```bash
# Dentro do container fraud-api
docker exec neuromorphic-fraud-api pytest tests/ -v

# Ou localmente (requer virtual environment)
source .venv/bin/activate
pytest tests/ -v
```

---

## Performance

### Recursos Recomendados

| Serviço | CPU | RAM | Disk |
|---------|-----|-----|------|
| fraud-api | 2 cores | 2GB | 500MB |
| jupyter-lab | 1 core | 1GB | 1GB |
| web-interface | 1 core | 512MB | 200MB |
| **Total** | **4 cores** | **3.5GB** | **2GB** |

### Limites (opcional)

```yaml
services:
 fraud-api:
 deploy:
 resources:
 limits:
 cpus: '2'
 memory: 2G
 reservations:
 cpus: '1'
 memory: 1G
```

### Otimizações

```bash
# Build multi-stage menor
# Use .dockerignore para excluir arquivos desnecessários

# Build cache
docker compose build --parallel

# Limpeza periódica
docker system prune -a --volumes
```

---

## Troubleshooting

### Problema: API não inicia

```bash
# Verificar logs
docker compose logs fraud-api

# Erros comuns:
# - Porta 8000 em uso: parar outros serviços
# - Falta de memória: aumentar Docker memory
# - Imports faltando: rebuild com --no-cache
```

### Problema: Jupyter sem resposta

```bash
# Reiniciar
docker compose restart jupyter-lab

# Verificar token (se habilitado)
docker compose logs jupyter-lab | grep token
```

### Problema: Streamlit não conecta à API

```bash
# Verificar variável de ambiente
docker exec neuromorphic-web env | grep API_URL
# Deve mostrar: API_URL=http://fraud-api:8000

# Teste de rede
docker exec neuromorphic-web curl http://fraud-api:8000/api/v1/health
```

### Problema: "Port already allocated"

```bash
# Encontrar processo usando a porta
lsof -i :8000

# Parar containers antigos
docker stop $(docker ps -aq)
docker compose up -d
```

---

## Deployment para Produção

### Docker Swarm

```bash
# Inicializar swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml fraud-detection

# Verificar serviços
docker service ls
```

### Kubernetes

```bash
# Gerar manifests
kompose convert -f docker-compose.yml

# Aplicar
kubectl apply -f fraud-api-deployment.yaml
kubectl apply -f fraud-api-service.yaml
```

### Cloud Run (GCP)

```bash
# Build e push
gcloud builds submit --tag gcr.io/PROJECT_ID/fraud-api

# Deploy
gcloud run deploy fraud-api \
 --image gcr.io/PROJECT_ID/fraud-api \
 --platform managed \
 --region us-central1 \
 --allow-unauthenticated
```

---

## Referências

- [Docker Compose Docs](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Streamlit Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/)

---

## Suporte

- **Issues**: Abra uma issue no repositório
- **Logs**: Sempre inclua saída de `docker compose logs`
- **Sistema**: Informe versões com `docker --version` e `docker compose version`

---

**Desenvolvido por**: Mauro Risonho de Paula Assumpção 
**Licença**: MIT 
**Projeto**: 01/10 - Neuromorphic Cybersecurity Portfolio
