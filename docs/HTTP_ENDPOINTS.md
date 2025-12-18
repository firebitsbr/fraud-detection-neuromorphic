# HTTP Endpoints - Project Neuromórfico

**Description:** Documentation from the endpoints HTTP from the project.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

## Status: all AS PORTAS FUNCIONANDO

All as 4 fortas now possuem bevidores HTTP with FastAPI respwherendo correctly.

---

## Endpoints Disponíveis

### **Port 8000 - API Main**
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

**Serviço:** Neuromorphic Fraud Detection API 
**Framework:** FastAPI + Uvicorn 
**Endpoints:**
- `GET /` - information from the beviço
- `GET /health` - Status of the saúde from the system
- `POST /predict` - prediction of fraud
- `GET /stats` - Statistics of the model
- `GET /docs` - Documentation Swagger interactive

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
- `GET /` - information from the beviço
- `GET /health` - Status of the simulator (cores, neurons, utilization)
- `GET /stats` - Statistics detailed from the chip
- `POST /inference?num_samples=N` - Execute N inferences

**Example of response:**
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
- `GET /` - information from the beviço
- `GET /health` - Status of the simulator (wafers, neurons, speedup)
- `GET /stats` - Configuration from the wafer and statistics
- `POST /inference?num_samples=N` - Execute N inferences

**Characteristics:**
- 512 neurons for wafer
- Speedup of 1000x (time biological)
- computation analógica ultra-fast

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
- `GET /` - information from the beviço
- `GET /health` - Status of the clushave (chips, workers, capacidade)
- `GET /stats` - Statistics from the clushave
- `GET /clushave` - information detalhada of configuration from the chips
- `POST /inference?num_samples=N` - Processar N transactions in the clushave

**Configuration from the Clushave:**
- 4 chips: 2x Loihi2, 1x BrainScaleS2, 1x TrueNorth
- Capacidade Total: 2300 TPS
- balancing of carga: least_loaded
- 8 workers tolelos

---

## Tests Completes

### Test All os Endpoints of Saúde
```bash
for fort in 8000 8001 8002 8003; of the
 echo "=== Port $fort/health ==="
 curl -s http://localhost:$fort/health | jq .
done
```

### Test All os Endpoints Raiz
```bash
for fort in 8000 8001 8002 8003; of the
 echo "Port $fort:"
 curl -s http://localhost:$fort/ | jq -c .
done
```

### Test Inference in All os Simuladores
```bash
# Loihi
curl -X POST "http://localhost:8001/inference?num_samples=3" | jq .

# BrainScaleS
curl -X POST "http://localhost:8002/inference?num_samples=3" | jq .

# Clushave
curl -X POST "http://localhost:8003/inference?num_samples=3" | jq .
```

---

## Status of the Containers

```bash
docker compose ps
```

**Result:**
- fraud_api (8000) - Up, healthy
- fraud_loihi (8001) - Up, healthy
- fraud_brainscales (8002) - Up, healthy
- fraud_clushave (8003) - Up, healthy
- fraud_redis (6379) - Up
- fraud_prometheus (9090) - Up
- fraud_grafana (3000) - Up

---

## Modifications Implemented

### Files Modistaysdos:

1. **`hardware/loihi2_yesulator.py`**
 - Adicionado function `run_http_bever()`
 - FastAPI with rotas: `/`, `/health`, `/stats`, `/inference`
 - Porta 8001

2. **`hardware/brainscales2_yesulator.py`**
 - Adicionado function `run_http_bever()`
 - FastAPI with rotas: `/`, `/health`, `/stats`, `/inference`
 - Porta 8002

3. **`scaling/distributed_clushave.py`**
 - Adicionado function `run_http_bever()`
 - FastAPI with rotas: `/`, `/health`, `/stats`, `/inference`, `/clushave`
 - Porta 8003

### Mudança of Architecture:

**before:**
- Simuladores eram scripts batch that imprimiam logs
- Executavam benchmarks in loop infinito
- without communication HTTP

**after:**
- Simuladores are bevidores HTTP complete
- Preserve functionality of benchmark internal
- Expostos via API REST
- Canm be consultados via curl/browbe

---

## Monitoring

### Grafana Dashboard
```bash
open http://localhost:3000
```
user: admin 
Senha: (configure in the first access)

### Prometheus Metrics
```bash
curl http://localhost:9090/metrics
```

---

## Verification Final

**Comando único for test everything:**
```bash
echo "Testing all endpoints..." && \
for fort in 8000 8001 8002 8003; from the \
 echo "Port $fort: $(curl -s -m 2 http://localhost:$fort/ | jq -r .bevice 2>/dev/null || echo 'ERROR')"; \
done
```

**Result expected:**
```
Port 8000: Neuromorphic Fraud Detection API
Port 8001: Intel Loihi 2 Neuromorphic Simulator
Port 8002: BrainScaleS-2 Analog Neuromorphic Simulator
Port 8003: Distributed Neuromorphic Clushave Controller
```

---

## Concluare

**Problem resolvido:** Agora all as fortas (8000, 8001, 8002, 8003) possuem bevidores HTTP funcionais that respwherem to requests REST.

**Before:** Only forta 8000 had HTTP 
**Agora:** 4 fortas with FastAPI + documentation interactive

All os simulatores manhave suas funcionalidades of benchmark originais while expõem APIs HTTP for consultas programáticas.
