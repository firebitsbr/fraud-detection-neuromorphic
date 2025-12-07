# HTTP Endpoints - Projeto Neuromórfico

## ✅ Status: TODAS AS PORTAS FUNCIONANDO

Todas as 4 portas agora possuem servidores HTTP com FastAPI respondendo corretamente.

---

## 📡 Endpoints Disponíveis

### 🔵 **Port 8000 - API Principal**
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

**Serviço:** Neuromorphic Fraud Detection API  
**Framework:** FastAPI + Uvicorn  
**Endpoints:**
- `GET /` - Informações do serviço
- `GET /health` - Status da saúde do sistema
- `POST /predict` - Predição de fraude
- `GET /stats` - Estatísticas do modelo
- `GET /docs` - Documentação Swagger interativa

---

### 🟢 **Port 8001 - Simulador Loihi 2**
```bash
curl http://localhost:8001/
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/inference?num_samples=5"
```

**Serviço:** Intel Loihi 2 Neuromorphic Simulator  
**Framework:** FastAPI + Uvicorn  
**Endpoints:**
- `GET /` - Informações do serviço
- `GET /health` - Status do simulador (cores, neurônios, utilização)
- `GET /stats` - Estatísticas detalhadas do chip
- `POST /inference?num_samples=N` - Executar N inferências

**Exemplo de resposta:**
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

### 🟡 **Port 8002 - Simulador BrainScaleS-2**
```bash
curl http://localhost:8002/
curl http://localhost:8002/health
curl -X POST "http://localhost:8002/inference?num_samples=5"
```

**Serviço:** BrainScaleS-2 Analog Neuromorphic Simulator  
**Framework:** FastAPI + Uvicorn  
**Endpoints:**
- `GET /` - Informações do serviço
- `GET /health` - Status do simulador (wafers, neurônios, speedup)
- `GET /stats` - Configuração do wafer e estatísticas
- `POST /inference?num_samples=N` - Executar N inferências

**Características:**
- 512 neurônios por wafer
- Speedup de 1000x (tempo biológico)
- Computação analógica ultra-rápida

---

### 🟣 **Port 8003 - Controlador de Cluster**
```bash
curl http://localhost:8003/
curl http://localhost:8003/health
curl http://localhost:8003/cluster
curl -X POST "http://localhost:8003/inference?num_samples=10"
```

**Serviço:** Distributed Neuromorphic Cluster Controller  
**Framework:** FastAPI + Uvicorn  
**Endpoints:**
- `GET /` - Informações do serviço
- `GET /health` - Status do cluster (chips, workers, capacidade)
- `GET /stats` - Estatísticas do cluster
- `GET /cluster` - Informação detalhada de configuração dos chips
- `POST /inference?num_samples=N` - Processar N transações no cluster

**Configuração do Cluster:**
- 4 chips: 2x Loihi2, 1x BrainScaleS2, 1x TrueNorth
- Capacidade total: 2300 TPS
- Balanceamento de carga: least_loaded
- 8 workers paralelos

---

## 🧪 Testes Completos

### Testar Todos os Endpoints de Saúde
```bash
for port in 8000 8001 8002 8003; do
  echo "=== Port $port/health ==="
  curl -s http://localhost:$port/health | jq .
done
```

### Testar Todos os Endpoints Raiz
```bash
for port in 8000 8001 8002 8003; do
  echo "Port $port:"
  curl -s http://localhost:$port/ | jq -c .
done
```

### Testar Inferência em Todos os Simuladores
```bash
# Loihi
curl -X POST "http://localhost:8001/inference?num_samples=3" | jq .

# BrainScaleS
curl -X POST "http://localhost:8002/inference?num_samples=3" | jq .

# Cluster
curl -X POST "http://localhost:8003/inference?num_samples=3" | jq .
```

---

## 🐳 Status dos Containers

```bash
docker compose ps
```

**Resultado:**
- ✅ fraud_api (8000) - Up, healthy
- ✅ fraud_loihi (8001) - Up, healthy
- ✅ fraud_brainscales (8002) - Up, healthy
- ✅ fraud_cluster (8003) - Up, healthy
- ✅ fraud_redis (6379) - Up
- ✅ fraud_prometheus (9090) - Up
- ✅ fraud_grafana (3000) - Up

---

## 🔧 Modificações Implementadas

### Arquivos Modificados:

1. **`hardware/loihi2_simulator.py`**
   - Adicionado função `run_http_server()`
   - FastAPI com rotas: `/`, `/health`, `/stats`, `/inference`
   - Porta 8001

2. **`hardware/brainscales2_simulator.py`**
   - Adicionado função `run_http_server()`
   - FastAPI com rotas: `/`, `/health`, `/stats`, `/inference`
   - Porta 8002

3. **`scaling/distributed_cluster.py`**
   - Adicionado função `run_http_server()`
   - FastAPI com rotas: `/`, `/health`, `/stats`, `/inference`, `/cluster`
   - Porta 8003

### Mudança de Arquitetura:

**ANTES:**
- Simuladores eram scripts batch que imprimiam logs
- Executavam benchmarks em loop infinito
- Sem comunicação HTTP

**DEPOIS:**
- Simuladores são servidores HTTP completos
- Preservam funcionalidade de benchmark interna
- Expostos via API REST
- Podem ser consultados via curl/browser

---

## 📊 Monitoramento

### Grafana Dashboard
```bash
open http://localhost:3000
```
Usuário: admin  
Senha: (configurar no primeiro acesso)

### Prometheus Metrics
```bash
curl http://localhost:9090/metrics
```

---

## ✅ Verificação Final

**Comando único para testar tudo:**
```bash
echo "Testing all endpoints..." && \
for port in 8000 8001 8002 8003; do \
  echo "Port $port: $(curl -s -m 2 http://localhost:$port/ | jq -r .service 2>/dev/null || echo 'ERROR')"; \
done
```

**Resultado esperado:**
```
Port 8000: Neuromorphic Fraud Detection API
Port 8001: Intel Loihi 2 Neuromorphic Simulator
Port 8002: BrainScaleS-2 Analog Neuromorphic Simulator
Port 8003: Distributed Neuromorphic Cluster Controller
```

---

## 🎉 Conclusão

**Problema resolvido:** Agora TODAS as portas (8000, 8001, 8002, 8003) possuem servidores HTTP funcionais que respondem a requisições REST.

**Antes:** Apenas porta 8000 tinha HTTP  
**Agora:** 4 portas com FastAPI + documentação interativa

Todos os simuladores mantêm suas funcionalidades de benchmark originais enquanto expõem APIs HTTP para consultas programáticas.
