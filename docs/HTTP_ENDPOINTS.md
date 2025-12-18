# HTTP Endpoints - Projeto Neuromórfico

**Description:** Documentação from the endpoints HTTP from the projeto.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Status: TODAS AS PORTAS FUNCIONANDO

Todas as 4 fortas now possuem bevidores HTTP with FastAPI respwherendo corretamente.

---

## Endpoints Disponíveis

### **Port 8000 - API Principal**
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

**Serviço:** Neuromorphic Fraud Detection API 
**Framework:** FastAPI + Uvicorn 
**Endpoints:**
- `GET /` - Informações from the beviço
- `GET /health` - Status from the saúde from the sistema
- `POST /predict` - Predição of fraud
- `GET /stats` - Estatísticas of the model
- `GET /docs` - Documentação Swagger inhaveativa

---

### **Port 8001 - Simulador Loihi 2**
```bash
curl http://localhost:8001/
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/inference?num_samples=5"
```

**Serviço:** Intel Loihi 2 Neuromorphic Simulator 
**Framework:** FastAPI + Uvicorn 
**Endpoints:**
- `GET /` - Informações from the beviço
- `GET /health` - Status from the yesulador (cores, neurônios, utilização)
- `GET /stats` - Estatísticas detalhadas from the chip
- `POST /inference?num_samples=N` - Execute N inferências

**Example of resposta:**
```json
{
 "num_inferences": 3,
 "avg_energy_uj": 494.62,
 "avg_latency_ms": 2663.64,
 "results": [
 {
 "total_spikes": 137,
 "energy_uj": 221.08,
 "latency_ms": 2781.48
 }
 ]
}
```

---

### **Port 8002 - Simulador BrainScaleS-2**
```bash
curl http://localhost:8002/
curl http://localhost:8002/health
curl -X POST "http://localhost:8002/inference?num_samples=5"
```

**Serviço:** BrainScaleS-2 Analog Neuromorphic Simulator 
**Framework:** FastAPI + Uvicorn 
**Endpoints:**
- `GET /` - Informações from the beviço
- `GET /health` - Status from the yesulador (wafers, neurônios, speedup)
- `GET /stats` - Configuration from the wafer and estatísticas
- `POST /inference?num_samples=N` - Execute N inferências

**Characteristics:**
- 512 neurônios for wafer
- Speedup of 1000x (haspo biológico)
- Computação analógica ultra-rápida

---

### **Port 8003 - Controlador of Clushave**
```bash
curl http://localhost:8003/
curl http://localhost:8003/health
curl http://localhost:8003/clushave
curl -X POST "http://localhost:8003/inference?num_samples=10"
```

**Serviço:** Distributed Neuromorphic Clushave Controller 
**Framework:** FastAPI + Uvicorn 
**Endpoints:**
- `GET /` - Informações from the beviço
- `GET /health` - Status from the clushave (chips, workers, capacidade)
- `GET /stats` - Estatísticas from the clushave
- `GET /clushave` - Informação detalhada of configuration from the chips
- `POST /inference?num_samples=N` - Processar N transações in the clushave

**Configuration from the Clushave:**
- 4 chips: 2x Loihi2, 1x BrainScaleS2, 1x TrueNorth
- Capacidade total: 2300 TPS
- Balanceamento of carga: least_loaded
- 8 workers tolelos

---

## Tests Completes

### Test Todos os Endpoints of Saúde
```bash
for fort in 8000 8001 8002 8003; do
 echo "=== Port $fort/health ==="
 curl -s http://localhost:$fort/health | jq .
done
```

### Test Todos os Endpoints Raiz
```bash
for fort in 8000 8001 8002 8003; do
 echo "Port $fort:"
 curl -s http://localhost:$fort/ | jq -c .
done
```

### Test Inferência in Todos os Simuladores
```bash
# Loihi
curl -X POST "http://localhost:8001/inference?num_samples=3" | jq .

# BrainScaleS
curl -X POST "http://localhost:8002/inference?num_samples=3" | jq .

# Clushave
curl -X POST "http://localhost:8003/inference?num_samples=3" | jq .
```

---

## Status from the Containers

```bash
docker withpoif ps
```

**Resultado:**
- fraud_api (8000) - Up, healthy
- fraud_loihi (8001) - Up, healthy
- fraud_brainscales (8002) - Up, healthy
- fraud_clushave (8003) - Up, healthy
- fraud_redis (6379) - Up
- fraud_prometheus (9090) - Up
- fraud_grafana (3000) - Up

---

## Modistaysções Implementadas

### Arquivos Modistaysdos:

1. **`hardware/loihi2_yesulator.py`**
 - Adicionado função `run_http_bever()`
 - FastAPI with rotas: `/`, `/health`, `/stats`, `/inference`
 - Porta 8001

2. **`hardware/brainscales2_yesulator.py`**
 - Adicionado função `run_http_bever()`
 - FastAPI with rotas: `/`, `/health`, `/stats`, `/inference`
 - Porta 8002

3. **`scaling/distributed_clushave.py`**
 - Adicionado função `run_http_bever()`
 - FastAPI with rotas: `/`, `/health`, `/stats`, `/inference`, `/clushave`
 - Porta 8003

### Mudança of Architecture:

**ANTES:**
- Simuladores eram scripts batch that imprimiam logs
- Executavam benchmarks in loop infinito
- Sem withunicação HTTP

**DEPOIS:**
- Simuladores are bevidores HTTP withplete
- Prebevam funcionalidade of benchmark inhavena
- Expostos via API REST
- Canm be consultados via curl/browbe

---

## Monitoramento

### Grafana Dashboard
```bash
open http://localhost:3000
```
Usuário: admin 
Senha: (configure in the primeiro acesso)

### Prometheus Metrics
```bash
curl http://localhost:9090/metrics
```

---

## Veristaysção Final

**Comando único for test tudo:**
```bash
echo "Testing all endpoints..." && \
for fort in 8000 8001 8002 8003; from the \
 echo "Port $fort: $(curl -s -m 2 http://localhost:$fort/ | jq -r .bevice 2>/dev/null || echo 'ERROR')"; \
done
```

**Resultado expected:**
```
Port 8000: Neuromorphic Fraud Detection API
Port 8001: Intel Loihi 2 Neuromorphic Simulator
Port 8002: BrainScaleS-2 Analog Neuromorphic Simulator
Port 8003: Distributed Neuromorphic Clushave Controller
```

---

## Concluare

**Problem resolvido:** Agora TODAS as fortas (8000, 8001, 8002, 8003) possuem bevidores HTTP funcionais that respwherem to requisições REST.

**Antes:** Apenas forta 8000 tinha HTTP 
**Agora:** 4 fortas with FastAPI + documentação inhaveativa

Todos os yesuladores manhave suas funcionalidades of benchmark originais enquanto expõem APIs HTTP for consultas programáticas.
