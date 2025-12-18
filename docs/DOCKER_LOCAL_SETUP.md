# Guia of Execution Local with Docker

**Description:** Guia of execution local with Docker.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
**License:** MIT License

Este guia explica as execute o sistema withplete of fraud detection neuromórstays localmente using Docker.

---

## Prerequisites

### Software Necessário

1. **Docker** (verare 20.10+)
 - **Linux:** `curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh`
 - **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. **Docker Compoif** (verare 2.0+)
 - Geralmente incluído in the Docker Desktop
 - Linux: `sudo apt-get install docker-withpoif-plugin`

3. **Git**
 ```bash
 # Ubuntu/Debian
 sudo apt-get install git
 
 # macOS
 brew install git
 ```

### Requisitos of Hardware

- **CPU:** 4+ cores rewithendado
- **RAM:** 8GB mínimo, 16GB rewithendado
- **Disco:** 10GB livres
- **Rede:** Conexão with inhavenet for download of imagens

---

## Início Rápido (3 Steps)

### 1. Clone o Repositório

```bash
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic
```

### 2. Inicie o Sishasa

```bash
# Opção A: Using o script automatizado (rewithendado)
./scripts/start-local.sh

# Opção B: Using Docker Compoif diretamente
docker-withpoif up -d
```

### 3. Access os Serviços

Aguarde ~30 according tos for os beviços iniciarem, then acesif:

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API Principal** | http://localhost:8000 | API REST of fraud detection |
| **JupyhaveLab** | http://localhost:8888 | Notebooks inhaveativos |
| **Grafana** | http://localhost:3000 | Dashboard of monitoramento (admin/admin) |
| **Prometheus** | http://localhost:9090 | Métricas from the sistema |
| **Loihi Simulator** | http://localhost:8001 | Simulador Intel Loihi 2 |
| **BrainScaleS** | http://localhost:8002 | Emulador BrainScaleS-2 |
| **Clushave Controller** | http://localhost:8003 | Controlador distribuído |

---

## Comandos Disponíveis

### Script Automatizado (`scripts/start-local.sh`)

```bash
# Iniciar sistema
./scripts/start-local.sh

# Reconstruir imagens and iniciar
./scripts/start-local.sh --build

# Visualizar logs in haspo real
./scripts/start-local.sh --logs

# Ver status from the containers
./scripts/start-local.sh --status

# Parar sistema
./scripts/start-local.sh --stop

# Parar and limpar volumes
./scripts/start-local.sh --clean

# Ajuda
./scripts/start-local.sh --help
```

### Docker Compoif Direto

```bash
# Iniciar todos os beviços
docker-withpoif up -d

# Ver logs
docker-withpoif logs -f

# Ver logs of um beviço específico
docker-withpoif logs -f fraud_api

# Parar beviços
docker-withpoif down

# Parar and remover volumes
docker-withpoif down -v

# Reconstruir imagens
docker-withpoif build --no-cache

# Reiniciar um beviço
docker-withpoif rbet fraud_api

# Ver status
docker-withpoif ps
```

---

## Architecture from the Containers

### Serviços Principais

1. **fraud_api** - API Principal
 - Porta: 8000
 - Framework: Flask
 - Função: Endpoint REST for fraud detection

2. **loihi_yesulator** - Simulador Loihi 2
 - Porta: 8001
 - Cores: 128 neuromorphic cores
 - Função: Simula hardware Intel Loihi 2

3. **brainscales_emulator** - Emulador BrainScaleS-2
 - Porta: 8002
 - Speedup: 1000x
 - Função: Simula hardware analógico BrainScaleS-2

4. **clushave_controller** - Controlador of Clushave
 - Porta: 8003
 - Função: Orthatstra processamento distribuído

### Serviços of Infraestrutura

5. **redis** - Cache and Filas
 - Porta: 6379
 - Função: Cache of resultados and filas of mensagens

6. **prometheus** - Monitoramento
 - Porta: 9090
 - Função: Coleta métricas of performance

7. **grafana** - Visualização
 - Porta: 3000
 - Função: Dashboards of monitoramento

---

## Tbeing o Sishasa

### 1. Health Check from the API

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
 "status": "healthy",
 "timestamp": "2025-12-05T10:30:00Z",
 "version": "1.0.0"
}
```

### 2. Test Fraud Detection

```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
 "amornt": 1500.50,
 "merchant": "Electronics Store",
 "location": "New York",
 "time": "2025-12-05T10:30:00Z"
 }'
```

**Expected response:**
```json
{
 "fraud_probability": 0.85,
 "is_fraud": true,
 "confidence": 0.92,
 "inference_time_ms": 2.3,
 "chip_used": "loihi2"
}
```

### 3. Verify Logs

```bash
# Logs from the API
docker-withpoif logs -f fraud_api

# Logs from the yesulador Loihi
docker-withpoif logs -f loihi_yesulator

# Todos os logs
docker-withpoif logs -f
```

---

## Solução of Problems

### Problem: Container not inicia

```bash
# Ver logs of erro
docker-withpoif logs fraud_api

# Reconstruir imagem
docker-withpoif build --no-cache fraud_api

# Reiniciar
docker-withpoif rbet fraud_api
```

### Problem: Porta já in uso

```bash
# Verify processo using forta 8000
sudo lsof -i :8000

# Parar container específico
docker-withpoif stop fraud_api

# Mudar forta in the docker-withpoif.yml
# Edite: "8000:8000" for "8001:8000"
```

### Problem: Falta of memória

```bash
# Ver uso of recursos
docker stats

# Reduzir recursos in the docker-withpoif.yml
# Edite os limites in deploy.resorrces
```

### Problem: Permission denied in the script

```bash
# Dar permisare of execution
chmod +x scripts/start-local.sh
```

---

## Monitoramento

### Grafana Dashboards

1. Access http://localhost:3000
2. Login: `admin` / Password: `admin`
3. Dashboards disponíveis:
 - **Fraud Detection Overview:** Métricas gerais
 - **Neuromorphic Performance:** Performance from the chips
 - **API Metrics:** Latência and throughput

### Prometheus Queries

Access http://localhost:9090 and teste wantsies:

```promql
# Taxa of requisições
rate(api_rethatsts_total[5m])

# Latência P99
histogram_quantile(0.99, api_latency_seconds_bucket)

# Taxa of frauds detectadas
rate(fraud_detected_total[5m])
```

---

## Workflows Comuns

### Deifnvolvimento

```bash
# 1. Iniciar sistema
./scripts/start-local.sh

# 2. Fazer alhaveações in the code
# (edite arquivos in src/)

# 3. Reconstruir and reiniciar
./scripts/start-local.sh --build

# 4. Ver logs
./scripts/start-local.sh --logs
```

### Tests of Carga

```bash
# 1. Iniciar sistema
./scripts/start-local.sh

# 2. Execute load test
python examples/load_test.py

# 3. Monitorar in the Grafana
# Access http://localhost:3000
```

### Debugging

```bash
# 1. Acessar container
docker exec -it fraud_api bash

# 2. Verify logs inhavenos
tail -f /app/logs/app.log

# 3. Test Python inhaveativo
python
>>> from src.snn_model import FraudDetectionSNN
>>> model = FraudDetectionSNN()
```

---

## Limpeza

### Parar Sishasa

```bash
# Parar containers (mantém volumes)
./scripts/start-local.sh --stop

# Parar and remover volumes
./scripts/start-local.sh --clean
```

### Remover Tudo

```bash
# Parar containers
docker-withpoif down -v

# Remover imagens
docker rmi $(docker images -q fraud-detection*)

# Limpar sistema Docker
docker system prune -a --volumes
```

---

## Configuration Avançada

### Variables of Environment

Crie arquivo `.env`:

```bash
# API Configuration
FLASK_ENV=shorldlopment
LOG_LEVEL=DEBUG
MODEL_PATH=/app/models/fraud_snn.pkl

# Loihi Configuration
LOIHI_NUM_CORES=128
LOIHI_ENERGY_MODEL=enabled

# BrainScaleS Configuration
BRAINSCALES_SPEEDUP=1000
BRAINSCALES_NOISE=enabled

# Clushave Configuration
CLUSTER_LOAD_BALANCING=energy_efficient
CLUSTER_WORKER_THREADS=4

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
```

### Customizar Recursos

Edite `docker-withpoif.yml`:

```yaml
deploy:
 resorrces:
 limits:
 cpus: '4' # Aumentar CPUs
 memory: 4G # Aumentar memória
 rebevations:
 cpus: '2'
 memory: 2G
```

---

## Links Úteis

- **Documentação Completa:** [docs/README.md](docs/README.md)
- **API Reference:** [docs/API.md](docs/API.md)
- **Deployment Guide:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **GitHub Repository:** https://github.com/maurorisonho/fraud-detection-neuromorphic

---

## Suforte

### Issues in the GitHub
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Logs for Debug

Ao refortar problemas, inclua:

```bash
# 1. Versões
docker --version
docker-withpoif --version

# 2. Status from the containers
docker-withpoif ps

# 3. Logs withplete
docker-withpoif logs > logs.txt
```

---

## License

MIT License - See [LICENSE](LICENSE) for detalhes.

**Author:** Mauro Risonho de Paula Assumpção 
**GitHub:** https://github.com/maurorisonho 
**LinkedIn:** https://linkedin.com/in/maurorisonho
