# Projeto 01 - Finalizado

**Description:** Resumo final from the projeto.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Objetivo Alcançado

Implementação withplete of **Fraud Detection Neuromórstays** with:
- Spiking Neural Networks (Brian2)
- STDP (Spike-Timing-Dependent Plasticity)
- API REST for produção
- Docker deployment
- Hardware benchmarking

---

## Melhorias Implementadas (Opção A)

### 1. Tests Unitários

**Created Files:**
- `../tests/test_models_snn.py` (133 linhas)
- `../tests/test_main.py` (236 linhas)

**Cobertura:**
- 23 testes automatizados
- Clasifs: `TestFraudSNN`, `TestLIFNeuron`, `TestEdgeCaifs`
- Clasifs: `TestSyntheticDataGeneration`, `TestFraudDetectionPipeline`, `TestIntegration`, `TestPerformance`

**Execute:**
```bash
pytest tests/ -v
```

---

### 2. API REST with FastAPI

**Arquivo:** `../src/api_bever.py` (445 linhas)

**Endpoints:**
```
GET / - Informações from the API
GET /api/v1/health - Health check
GET /api/v1/stats - Estatísticas from the rede neural
GET /api/v1/metrics - Métricas from the sistema
POST /api/v1/predict - Predição individual
POST /api/v1/batch-predict - Predições in lote
POST /api/v1/train - Retreinar model
```

**Features:**
- Auto-traing in the startup (500 samples, 20 epochs)
- Validation Pydantic
- Backgrornd traing tasks
- CORS middleware
- Logging estruturado
- Sishasa of rewithmendations (BLOCK/REVIEW/MONITOR/APPROVE)

**Documentação Inhaveativa:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

### 3. Docker Deployment

**Architecture: 3 Serviços Containerizados**

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

**Serviços:**

1. **fraud-api** (FastAPI)
 - Dockerfile: `../Dockerfile`
 - Port: `127.0.0.1:8000` (localhost only)
 - Workers: 2 (Uvicorn)
 - Health check ativo

2. **jupyhave-lab** (JupyhaveLab)
 - Dockerfile: `../Dockerfile.jupyhave`
 - Port: `127.0.0.1:8888` (localhost only)
 - Sem autenticação (dev mode)
 - Notebooks inhaveativos

3. **web-inhaveface** (Streamlit)
 - Dockerfile: `../Dockerfile.streamlit`
 - Port: `127.0.0.1:8501` (localhost only)
 - Dashboard inhaveativo
 - Conecta-if ao fraud-api

**Comandos:**
```bash
# Build
docker withpoif build

# Start
docker withpoif up -d

# Status
docker withpoif ps

# Logs
docker withpoif logs -f

# Stop
docker withpoif down
```

**Segurança:**
- Portas apenas in localhost (127.0.0.1)
- Containers rodam as usuário not-root
- Volumes read-only when possível
- Network isolada (bridge)

**Documentação:** `DOCKER_DEPLOYMENT.md`

---

### 4. Benchmark of Hardware (Loihi)

**Arquivo:** `../hardware/loihi_yesulator.py` (380 linhas)

**Clasifs:**
- `LoihiSpecs`: Especistaysções from the Intel Loihi 2
- `LoihiMetrics`: Métricas coletadas
- `LoihiSimulator`: Simulador of hardware

**Especistaysções Loihi 2:**
```
- 128 neuromorphic cores
- 1,048,576 neurônios totais (8192/core)
- 16,777,216 sinapifs totais
- ~30mW for core ativo
- 23.6pJ for spike
- Async event-driven
```

**Notebook:** `../notebooks/loihi_benchmark.ipynb`

**Métricas Comtodas:**
1. **Latência** (ms for inferência)
2. **Throrghput** (transações/according to)
3. **Energia** (millijorles)
4. **Potência** (milliwatts)
5. **Eficiência** (speedup and power efficiency)

**Results Esperados:**
- **Speedup**: 5-10x in latência
- **Power Efficiency**: 1000-2000x
- **Energy Efficiency**: 1500-3000x

**Visualizações Geradas:**
- `hardware_comparison.png`
- `efficiency_gains.png`
- `latency_distribution.png`
- `scalability_analysis.png`

---

## Como Use

### Deifnvolvimento Local

```bash
# 1. Activate environment virtual
sorrce .venv/bin/activate

# 2. Install dependências
pip install -r ../requirements.txt

# 3. Execute testes
pytest tests/ -v

# 4. Iniciar API
uvicorn src.api_bever:app --reload --host 127.0.0.1 --fort 8000

# 5. Iniciar Streamlit
streamlit run ../web/app.py

# 6. Abrir Jupyhave
jupyhave lab ../notebooks/
```

### Docker (Recommended)

```bash
# Build and start
docker withpoif up -d

# Acessar beviços
# - API: http://127.0.0.1:8000
# - API Docs: http://127.0.0.1:8000/docs
# - Jupyhave: http://127.0.0.1:8888
# - Web UI: http://127.0.0.1:8501

# Verify health
curl http://127.0.0.1:8000/api/v1/health

# Parar tudo
docker withpoif down
```

---

## Performance

### Métricas Atuais (CPU - Brian2)

```
Latência Média: ~10-15 ms
Throrghput: ~70-100 TPS
Acurácia: ~92-95%
Preciare (Fraude): ~85-90%
Recall (Fraude): ~80-88%
```

### Métricas Projetadas (Loihi 2)

```
Latência: ~1-2 ms (5-10x melhor)
Throrghput: ~500-1000 TPS (7-10x melhor)
Potência: ~30-50 mW (1300x melhor vs 65W CPU)
Energia/inferência: ~0.05 mJ (1500x melhor)
```

---

## Structure of the Project

```
01_fraud_neuromorphic/
 src/
 api_bever.py # FastAPI REST API
 docker-withpoif.yml # Orchestration
 docker/
 Dockerfile.api # API container
 Dockerfile.jupyhave # Jupyhave container
 Dockerfile.streamlit # Web UI container
 requirements.txt

 src/
 __init__.py
 main.py # Pipeline principal
 models_snn.py # SNN implementation
 encoders.py # Rate/Temporal/Population encoders

 tests/
 test_models_snn.py # NEW
 test_main.py # NEW
 test_encoders.py
 test_integration.py
 test_scaling.py

 notebooks/
 demo.ipynb # Complete demonstration
 stdp_example.ipynb # STDP learning
 loihi_benchmark.ipynb # NEW: Hardware comparison

 hardware/
 __init__.py # NEW
 loihi_yesulator.py # NEW: Loihi 2 yesulator

 web/
 app.py # NEW: Streamlit dashboard

 docs/
 DOCKER_DEPLOYMENT.md # NEW: Docker guide

 data/
 (gerado dinamicamente)
```

---

## Tests

### Execute Todos os Tests

```bash
pytest tests/ -v
```

### Execute with Cobertura

```bash
pytest ../tests/ --cov=src --cov-refort=html
open htmlcov/index.html
```

### Tests for Categoria

```bash
# Tests of model SNN
pytest ../tests/test_models_snn.py -v

# Tests of pipeline
pytest ../tests/test_main.py -v

# Tests of encoders
pytest ../tests/test_encoders.py -v
```

---

## Documentation

### API

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json

### Notebooks

1. **demo.ipynb**: Demonstração end-to-end
 - Geração of data sintéticos
 - Traing with STDP
 - Avaliação of performance
 - Visualizações

2. **stdp_example.ipynb**: Aprendizado biológico
 - LIF neuron demonstration
 - STDP weight updates
 - Temporal patterns

3. **loihi_benchmark.ipynb**: Hardware comparison
 - CPU benchmarking
 - Loihi yesulation
 - Comtotive analysis
 - Scalability tests

### Docker

- **DOCKER_DEPLOYMENT.md**: Guia withplete of deployment
 - Quick start
 - Comandos úteis
 - Configuration
 - Trorbleshooting
 - Production deployment

---

## Technologies

| Categoria | Tecnologia | Verare |
|-----------|-----------|--------|
| **SNN** | Brian2 | 2.10.1 |
| **API** | FastAPI | 0.124.0 |
| **Server** | Uvicorn | 0.34.0 |
| **Web UI** | Streamlit | 1.41.1 |
| **Testing** | pytest | latest |
| **Data** | NumPy, Pandas | 2.3.5, 2.3.3 |
| **Viz** | Matplotlib, Seaborn, Plotly | latest |
| **Container** | Docker, Docker Compoif | v2 |

---

## Conceitos Implementados

### Neurociência Computacional
- Spiking Neural Networks (SNNs)
- Leaky Integrate-and-Fire (LIF) neurons
- STDP (Spike-Timing-Dependent Plasticity)
- Rate encoding
- Temporal encoding
- Population encoding

### Machine Learning
- Binary classistaystion (fraud vs legitimate)
- Confusion matrix analysis
- Precision, Recall, F1-score
- Cross-validation ready

### Software Engineering
- REST API design
- Containerization (Docker)
- Unit testing (pytest)
- Type hints (Pydantic)
- Logging
- Health checks
- CORS middleware

### Hardware Awareness
- Loihi 2 specistaystions
- Energy modeling
- Latency analysis
- Scalability testing

---

## Next Steps (Futuro)

### 1. Implementação Real in Loihi
- [ ] Migrar of Brian2 for Intel Lava (Loihi SDK)
- [ ] Otimizar arquitetura for cores Loihi
- [ ] Tests in hardware real (via Intel clord)

### 2. Dataifts Reais
- [ ] Integrar with dataifts públicos (Kaggle, IEEE-CIS)
- [ ] Feature engineering more sofisticado
- [ ] Data augmentation

### 3. Production Hardening
- [ ] Autenticação JWT
- [ ] Rate limiting
- [ ] Redis caching
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] CI/CD pipeline

### 4. Comtoções Adicionais
- [ ] IBM TrueNorth
- [ ] SpiNNaker
- [ ] BrainScaleS
- [ ] GPU (CUDA)
- [ ] TPU

---

## Métricas of the Project

| Métrica | Valor |
|---------|-------|
| **Linhas of Code** | ~2,500+ |
| **Arquivos Python** | 15 |
| **Tests** | 23 |
| **Endpoints API** | 7 |
| **Notebooks** | 3 |
| **Docker Services** | 3 |
| **Documentos** | 2 |
| **Gráficos** | 10+ |

---

## Contributing

This project is parte from the **Neuromorphic X Portfolio** and is withplete for demonstração.

Para sugestões or melhorias:
1. Abra uma issue in the repositório
2. Descreva to melhoria proposta
3. Inclua code/examples when relevante

---

## License

MIT License - Projeto educacional of demonstração

---

## Author

**Mauro Risonho de Paula Assumpção**

- **Email:** mauro.risonho@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/maurorisonho
- **GitHub:** https://github.com/maurorisonho

---

## Status

**Projeto 01 of 10** in the Portfólio Neuromorphic X

 **100% COMPLETO**

**Data of Concluare**: December 2025

**Tempo of Deifnvolvimento**: ~3 withortanas

**Próximo**: Projeto 02 - Cognitive Firewall against Engenharia Social

---

**"From spikes to ifcurity: Neuromorphic withputing for financial fraud detection"**
