# 🚀 Guia de Execução Local com Docker

**Descrição:** Guia de execução local com Docker.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
**Licença:** MIT License

Este guia explica como executar o sistema completo de detecção de fraude neuromórfica localmente usando Docker.

---

## 📋 Pré-requisitos

### Software Necessário

1. **Docker** (versão 20.10+)
   - **Linux:** `curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh`
   - **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. **Docker Compose** (versão 2.0+)
   - Geralmente incluído no Docker Desktop
   - Linux: `sudo apt-get install docker-compose-plugin`

3. **Git**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git
   
   # macOS
   brew install git
   ```

### Requisitos de Hardware

- **CPU:** 4+ cores recomendado
- **RAM:** 8GB mínimo, 16GB recomendado
- **Disco:** 10GB livres
- **Rede:** Conexão com internet para download de imagens

---

## 🎯 Início Rápido (3 Passos)

### 1. Clone o Repositório

```bash
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic
```

### 2. Inicie o Sistema

```bash
# Opção A: Usando o script automatizado (recomendado)
./scripts/start-local.sh

# Opção B: Usando Docker Compose diretamente
docker-compose up -d
```

### 3. Acesse os Serviços

Aguarde ~30 segundos para os serviços iniciarem, então acesse:

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API Principal** | http://localhost:8000 | API REST de detecção de fraude |
| **JupyterLab** | http://localhost:8888 | Notebooks interativos |
| **Grafana** | http://localhost:3000 | Dashboard de monitoramento (admin/admin) |
| **Prometheus** | http://localhost:9090 | Métricas do sistema |
| **Loihi Simulator** | http://localhost:8001 | Simulador Intel Loihi 2 |
| **BrainScaleS** | http://localhost:8002 | Emulador BrainScaleS-2 |
| **Cluster Controller** | http://localhost:8003 | Controlador distribuído |

---

## 🔧 Comandos Disponíveis

### Script Automatizado (`scripts/start-local.sh`)

```bash
# Iniciar sistema
./scripts/start-local.sh

# Reconstruir imagens e iniciar
./scripts/start-local.sh --build

# Visualizar logs em tempo real
./scripts/start-local.sh --logs

# Ver status dos containers
./scripts/start-local.sh --status

# Parar sistema
./scripts/start-local.sh --stop

# Parar e limpar volumes
./scripts/start-local.sh --clean

# Ajuda
./scripts/start-local.sh --help
```

### Docker Compose Direto

```bash
# Iniciar todos os serviços
docker-compose up -d

# Ver logs
docker-compose logs -f

# Ver logs de um serviço específico
docker-compose logs -f fraud_api

# Parar serviços
docker-compose down

# Parar e remover volumes
docker-compose down -v

# Reconstruir imagens
docker-compose build --no-cache

# Reiniciar um serviço
docker-compose restart fraud_api

# Ver status
docker-compose ps
```

---

## 📦 Arquitetura dos Containers

### Serviços Principais

1. **fraud_api** - API Principal
   - Porta: 8000
   - Framework: Flask
   - Função: Endpoint REST para detecção de fraude

2. **loihi_simulator** - Simulador Loihi 2
   - Porta: 8001
   - Cores: 128 neuromorphic cores
   - Função: Simula hardware Intel Loihi 2

3. **brainscales_emulator** - Emulador BrainScaleS-2
   - Porta: 8002
   - Speedup: 1000x
   - Função: Simula hardware analógico BrainScaleS-2

4. **cluster_controller** - Controlador de Cluster
   - Porta: 8003
   - Função: Orquestra processamento distribuído

### Serviços de Infraestrutura

5. **redis** - Cache e Filas
   - Porta: 6379
   - Função: Cache de resultados e filas de mensagens

6. **prometheus** - Monitoramento
   - Porta: 9090
   - Função: Coleta métricas de performance

7. **grafana** - Visualização
   - Porta: 3000
   - Função: Dashboards de monitoramento

---

## 🧪 Testando o Sistema

### 1. Health Check da API

```bash
curl http://localhost:8000/health
```

**Resposta esperada:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-05T10:30:00Z",
  "version": "1.0.0"
}
```

### 2. Testar Detecção de Fraude

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500.50,
    "merchant": "Electronics Store",
    "location": "New York",
    "time": "2025-12-05T10:30:00Z"
  }'
```

**Resposta esperada:**
```json
{
  "fraud_probability": 0.85,
  "is_fraud": true,
  "confidence": 0.92,
  "inference_time_ms": 2.3,
  "chip_used": "loihi2"
}
```

### 3. Verificar Logs

```bash
# Logs da API
docker-compose logs -f fraud_api

# Logs do simulador Loihi
docker-compose logs -f loihi_simulator

# Todos os logs
docker-compose logs -f
```

---

## 🐛 Solução de Problemas

### Problema: Container não inicia

```bash
# Ver logs de erro
docker-compose logs fraud_api

# Reconstruir imagem
docker-compose build --no-cache fraud_api

# Reiniciar
docker-compose restart fraud_api
```

### Problema: Porta já em uso

```bash
# Verificar processo usando porta 8000
sudo lsof -i :8000

# Parar container específico
docker-compose stop fraud_api

# Mudar porta no docker-compose.yml
# Edite: "8000:8000" para "8001:8000"
```

### Problema: Falta de memória

```bash
# Ver uso de recursos
docker stats

# Reduzir recursos no docker-compose.yml
# Edite os limites em deploy.resources
```

### Problema: Permission denied no script

```bash
# Dar permissão de execução
chmod +x scripts/start-local.sh
```

---

## 📊 Monitoramento

### Grafana Dashboards

1. Acesse http://localhost:3000
2. Login: `admin` / Password: `admin`
3. Dashboards disponíveis:
   - **Fraud Detection Overview:** Métricas gerais
   - **Neuromorphic Performance:** Performance dos chips
   - **API Metrics:** Latência e throughput

### Prometheus Queries

Acesse http://localhost:9090 e teste queries:

```promql
# Taxa de requisições
rate(api_requests_total[5m])

# Latência P99
histogram_quantile(0.99, api_latency_seconds_bucket)

# Taxa de fraudes detectadas
rate(fraud_detected_total[5m])
```

---

## 🔄 Workflows Comuns

### Desenvolvimento

```bash
# 1. Iniciar sistema
./scripts/start-local.sh

# 2. Fazer alterações no código
# (edite arquivos em src/)

# 3. Reconstruir e reiniciar
./scripts/start-local.sh --build

# 4. Ver logs
./scripts/start-local.sh --logs
```

### Testes de Carga

```bash
# 1. Iniciar sistema
./scripts/start-local.sh

# 2. Executar load test
python examples/load_test.py

# 3. Monitorar no Grafana
# Acesse http://localhost:3000
```

### Debugging

```bash
# 1. Acessar container
docker exec -it fraud_api bash

# 2. Verificar logs internos
tail -f /app/logs/app.log

# 3. Testar Python interativo
python
>>> from src.snn_model import FraudDetectionSNN
>>> model = FraudDetectionSNN()
```

---

## 🧹 Limpeza

### Parar Sistema

```bash
# Parar containers (mantém volumes)
./scripts/start-local.sh --stop

# Parar e remover volumes
./scripts/start-local.sh --clean
```

### Remover Tudo

```bash
# Parar containers
docker-compose down -v

# Remover imagens
docker rmi $(docker images -q fraud-detection*)

# Limpar sistema Docker
docker system prune -a --volumes
```

---

## 📝 Configuração Avançada

### Variáveis de Ambiente

Crie arquivo `.env`:

```bash
# API Configuration
FLASK_ENV=development
LOG_LEVEL=DEBUG
MODEL_PATH=/app/models/fraud_snn.pkl

# Loihi Configuration
LOIHI_NUM_CORES=128
LOIHI_ENERGY_MODEL=enabled

# BrainScaleS Configuration
BRAINSCALES_SPEEDUP=1000
BRAINSCALES_NOISE=enabled

# Cluster Configuration
CLUSTER_LOAD_BALANCING=energy_efficient
CLUSTER_WORKER_THREADS=4

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
```

### Customizar Recursos

Edite `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'      # Aumentar CPUs
      memory: 4G     # Aumentar memória
    reservations:
      cpus: '2'
      memory: 2G
```

---

## 🔗 Links Úteis

- **Documentação Completa:** [docs/README.md](docs/README.md)
- **API Reference:** [docs/API.md](docs/API.md)
- **Deployment Guide:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **GitHub Repository:** https://github.com/maurorisonho/fraud-detection-neuromorphic

---

## 🆘 Suporte

### Issues no GitHub
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Logs para Debug

Ao reportar problemas, inclua:

```bash
# 1. Versões
docker --version
docker-compose --version

# 2. Status dos containers
docker-compose ps

# 3. Logs completos
docker-compose logs > logs.txt
```

---

## 📄 Licença

MIT License - Veja [LICENSE](LICENSE) para detalhes.

**Autor:** Mauro Risonho de Paula Assumpção  
**GitHub:** https://github.com/maurorisonho  
**LinkedIn:** https://linkedin.com/in/maurorisonho
