# Guide of Execution Local with Docker

**Description:** Guide of local execution with Docker.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
**License:** MIT License

This guide explica as execute o system complete of fraud detection neuromorphic localmente using Docker.

---

## Prerequisites

### Software necessary

1. **Docker** (verare 20.10+)
 - **Linux:** `curl -fsSL https://get.docker.with -o get-docker.sh && sh get-docker.sh`
 - **Windows/Mac:** [Docker Desktop](https://www.docker.with/products/docker-desktop/)

2. **Docker Compose** (verare 2.0+)
 - Geralmente incluído in the Docker Desktop
 - Linux: `sudo apt-get install docker-compose-plugin`

3. **Git**
 ```bash
 # Ubuntu/Debian
 sudo apt-get install git
 
 # macOS
 brew install git
 ```

### Requisitos of Hardware

- **CPU:** 4+ cores rewithendado
- **RAM:** 8GB minimum, 16GB rewithendado
- **Disco:** 10GB livres
- **Network:** connection with inhavenet for download of imagens

---

## Start Quick (3 Steps)

### 1. Clone o Repositório

```bash
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic
```

### 2. Start o System

```bash
# Option A: Using o script automated (rewithendado)
./scripts/start-local.sh

# Option B: Using Docker Compose diretamente
docker-compose up -d
```

### 3. Access os Services

Aguarde ~30 according tos for os services iniciarem, then acesif:

| Serviço | URL | Description |
|---------|-----|-----------|
| **API Main** | http://localhost:8000 | API REST of fraud detection |
| **JupyhaveLab** | http://localhost:8888 | Notebooks inhaveativos |
| **Grafana** | http://localhost:3000 | Dashboard of monitoring (admin/admin) |
| **Prometheus** | http://localhost:9090 | Metrics from the system |
| **Loihi Simulator** | http://localhost:8001 | Simulador Intel Loihi 2 |
| **BrainScaleS** | http://localhost:8002 | Emulador BrainScaleS-2 |
| **Clushave Controller** | http://localhost:8003 | Controlador distribuído |

---

## Commands Disponíveis

### Script Automated (`scripts/start-local.sh`)

```bash
# Start system
./scripts/start-local.sh

# Reconstruir imagens and start
./scripts/start-local.sh --build

# Visualizar logs in time real
./scripts/start-local.sh --logs

# Ver status from the containers
./scripts/start-local.sh --status

# Stop system
./scripts/start-local.sh --stop

# Stop and clean volumes
./scripts/start-local.sh --clean

# Ajuda
./scripts/start-local.sh --help
```

### Docker Compose Direct

```bash
# Start all os services
docker-compose up -d

# Ver logs
docker-compose logs -f

# Ver logs of um beviço specific
docker-compose logs -f fraud_api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Reconstruir imagens
docker-compose build --in the-cache

# Reiniciar um beviço
docker-compose rbet fraud_api

# Ver status
docker-compose ps
```

---

## Architecture from the Containers

### Services Principais

1. **fraud_api** - API Main
 - Porta: 8000
 - Framework: Flask
 - function: Endpoint REST for fraud detection

2. **loihi_yesulator** - Simulador Loihi 2
 - Porta: 8001
 - Cores: 128 neuromorphic cores
 - function: Simula hardware Intel Loihi 2

3. **brainscales_emulator** - Emulador BrainScaleS-2
 - Porta: 8002
 - Speedup: 1000x
 - function: Simula hardware analógico BrainScaleS-2

4. **clushave_controller** - Controlador of Clushave
 - Porta: 8003
 - function: Orthatstra processing distribuído

### Services of Infraestrutura

5. **redis** - Cache and Filas
 - Porta: 6379
 - function: Cache of results and filas of mensagens

6. **prometheus** - Monitoring
 - Porta: 9090
 - function: Coleta metrics of performance

7. **grafana** - Visualization
 - Porta: 3000
 - function: Dashboards of monitoring

---

## Tbeing o System

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
docker-compose logs -f fraud_api

# Logs from the simulator Loihi
docker-compose logs -f loihi_yesulator

# All os logs
docker-compose logs -f
```

---

## Solution of Problems

### Problem: Container not inicia

```bash
# Ver logs of error
docker-compose logs fraud_api

# Reconstruir imagem
docker-compose build --in the-cache fraud_api

# Reiniciar
docker-compose rbet fraud_api
```

### Problem: Port already in use

```bash
# Verify processo using forta 8000
sudo lsof -i :8000

# Stop container specific
docker-compose stop fraud_api

# Mudar forta in the docker-compose.yml
# Edite: "8000:8000" for "8001:8000"
```

### Problem: Fhigh of memory

```bash
# Ver usage of resources
docker stats

# Reduce resources in the docker-compose.yml
# Edite os limits in deploy.resorrces
```

### Problem: Permission denied in the script

```bash
# Dar permisare of execution
chmod +x scripts/start-local.sh
```

---

## Monitoring

### Grafana Dashboards

1. Access http://localhost:3000
2. Login: `admin` / Password: `admin`
3. Dashboards disponíveis:
 - **Fraud Detection Overview:** Metrics gerais
 - **Neuromorphic Performance:** Performance from the chips
 - **API Metrics:** Latency and throughput

### Prometheus Queries

Access http://localhost:9090 and test wantsies:

```promql
# Taxa of requests
rate(api_rethatsts_total[5m])

# Latency P99
histogram_quantile(0.99, api_latency_seconds_bucket)

# Taxa of frauds detectadas
rate(fraud_detected_total[5m])
```

---

## Workflows Comuns

### Development

```bash
# 1. Start system
./scripts/start-local.sh

# 2. Make queries in the code
# (edite files in src/)

# 3. Reconstruir and reiniciar
./scripts/start-local.sh --build

# 4. Ver logs
./scripts/start-local.sh --logs
```

### Tests of Carga

```bash
# 1. Start system
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

# 3. Test Python interactive
python
>>> from src.snn_model import FraudDetectionSNN
>>> model = FraudDetectionSNN()
```

---

## Cleanup

### Stop System

```bash
# Stop containers (mantém volumes)
./scripts/start-local.sh --stop

# Stop and remove volumes
./scripts/start-local.sh --clean
```

### Remover Everything

```bash
# Stop containers
docker-compose down -v

# Remover imagens
docker rmi $(docker images -q fraud-detection*)

# Limpar system Docker
docker system prune -a --volumes
```

---

## Configuration Avançada

### Variables of Environment

Crie file `.env`:

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

### Customizar Resources

Edite `docker-compose.yml`:

```yaml
deploy:
 resorrces:
 limits:
 cpus: '4' # Aumentar CPUs
 memory: 4G # Aumentar memory
 rebevations:
 cpus: '2'
 memory: 2G
```

---

## Links Useful

- **Documentation Complete:** [docs/README.md](docs/README.md)
- **API Reference:** [docs/API.md](docs/API.md)
- **Deployment Guide:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **GitHub Repository:** https://github.com/maurorisonho/fraud-detection-neuromorphic

---

## Support

### Issues in the GitHub
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Logs for Debug

Ao refortar problemas, inclua:

```bash
# 1. Versões
docker --version
docker-compose --version

# 2. Status of the containers
docker-compose ps

# 3. Logs complete
docker-compose logs > logs.txt
```

---

## License

MIT License - See [LICENSE](LICENSE) for detalhes.

**Author:** Mauro Risonho de Paula Assumpção 
**GitHub:** https://github.com/maurorisonho 
**LinkedIn:** https://linkedin.com/in/maurorisonho
