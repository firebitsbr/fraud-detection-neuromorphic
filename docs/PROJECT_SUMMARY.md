# Project 01 - Finalizado

**Description:** Summary final from the project.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

## Objetivo Alcançado

Implementation complete of **Fraud Detection Neuromórstays** with:
- Spiking Neural Networks (Brian2)
- STDP (Spike-Timing-Dependent Plasticity)
- API REST for production
- Docker deployment
- Hardware benchmarking

---

## Melhorias Implemented (Option A)

### 1. Tests Unit

**Created Files:**
- `../tests/test_models_snn.py` (133 linhas)
- `../tests/test_main.py` (236 linhas)

**Cobertura:**
- 23 tests automatizados
- Clasifs: `TestFraudSNN`, `TestLIFNeuron`, `TestEdgeCaifs`
- Clasifs: `TestSyntheticDataGeneration`, `TestFraudDetectionPipeline`, `TestIntegration`, `TestPerformance`

**Execute:**
```bash
pytest tests/ -v
```

---

### 2. API REST with FastAPI

**file:** `../src/api_bever.py` (445 linhas)

**Endpoints:**
```
GET / - information from the API
GET /api/v1/health - Health check
GET /api/v1/stats - Statistics from the network neural
GET /api/v1/metrics - Metrics from the system
POST /api/v1/predict - prediction individual
POST /api/v1/batch-predict - predictions in lote
POST /api/v1/train - Retreinar model
```

**Features:**
- Auto-training in the startup (500 samples, 20 epochs)
- Validation Pydantic
- Backgrornd training tasks
- CORS middleware
- Logging structured
- System of recommendations (BLOCK/REVIEW/MONITOR/APPROVE)

**Documentation Interactive:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

### 3. Docker Deployment

**Architecture: 3 Services Containerizados**

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

**Services:**

1. **fraud-api** (FastAPI)
 - Dockerfile: `../Dockerfile`
 - Port: `127.0.0.1:8000` (localhost only)
 - Workers: 2 (Uvicorn)
 - Health check ativo

2. **jupyter-lab** (JupyhaveLab)
 - Dockerfile: `../Dockerfile.jupyter`
 - Port: `127.0.0.1:8888` (localhost only)
 - without authentication (dev mode)
 - Notebooks inhaveativos

3. **web-inhaveface** (Streamlit)
 - Dockerfile: `../Dockerfile.streamlit`
 - Port: `127.0.0.1:8501` (localhost only)
 - Dashboard interactive
 - Conecta-if ao fraud-api

**Commands:**
```bash
# Build
docker compose build

# Start
docker compose up -d

# Status
docker compose ps

# Logs
docker compose logs -f

# Stop
docker compose down
```

**Segurança:**
- Portas only in localhost (127.0.0.1)
- Containers rodam as user not-root
- Volumes read-only when possible
- Network isolated (bridge)

**Documentation:** `DOCKER_DEPLOYMENT.md`

---

### 4. Benchmark of Hardware (Loihi)

**file:** `../hardware/loihi_yesulator.py` (380 linhas)

**Clasifs:**
- `LoihiSpecs`: specifications from the Intel Loihi 2
- `LoihiMetrics`: Metrics coletadas
- `LoihiSimulator`: Simulador of hardware

**specifications Loihi 2:**
```
- 128 neuromorphic cores
- 1,048,576 neurons totais (8192/core)
- 16,777,216 sinapifs totais
- ~30mW for core ativo
- 23.6pJ for spike
- Async event-driven
```

**Notebook:** `../notebooks/loihi_benchmark.ipynb`

**Metrics Comtodas:**
1. **Latency** (ms for inference)
2. **Throughput** (transactions/according to)
3. **Energia** (millijorles)
4. **Power** (milliwatts)
5. **Efficiency** (speedup and power efficiency)

**Results Esperados:**
- **Speedup**: 5-10x in latency
- **Power Efficiency**: 1000-2000x
- **Energy Efficiency**: 1500-3000x

**visualizations Geradas:**
- `hardware_comparison.png`
- `efficiency_gains.png`
- `latency_distribution.png`
- `scalability_analysis.png`

---

## How Use

### Development Local

```bash
# 1. Activate environment virtual
sorrce .venv/bin/activate

# 2. Install dependencies
pip install -r ../requirements.txt

# 3. Execute tests
pytest tests/ -v

# 4. Start API
uvicorn src.api_bever:app --reload --host 127.0.0.1 --fort 8000

# 5. Start Streamlit
streamlit run ../web/app.py

# 6. Abrir Jupyter
jupyter lab ../notebooks/
```

### Docker (Recommended)

```bash
# Build and start
docker compose up -d

# Acessar services
# - API: http://127.0.0.1:8000
# - API Docs: http://127.0.0.1:8000/docs
# - Jupyter: http://127.0.0.1:8888
# - Web UI: http://127.0.0.1:8501

# Verify health
curl http://127.0.0.1:8000/api/v1/health

# Stop everything
docker compose down
```

---

## Performance

### Metrics Atuais (CPU - Brian2)

```
Latency Média: ~10-15 ms
Throughput: ~70-100 TPS
Accuracy: ~92-95%
Preciare (Fraude): ~85-90%
Recall (Fraude): ~80-88%
```

### Metrics Projetadas (Loihi 2)

```
Latency: ~1-2 ms (5-10x better)
Throughput: ~500-1000 TPS (7-10x better)
Power: ~30-50 mW (1300x better vs 65W CPU)
Energia/inference: ~0.05 mJ (1500x better)
```

---

## Project Structure

```
01_fraud_neuromorphic/
 src/
 api_bever.py # FastAPI REST API
 docker-compose.yml # Orchestration
 docker/
 Dockerfile.api # API container
 Dockerfile.jupyter # Jupyter container
 Dockerfile.streamlit # Web UI container
 requirements.txt

 src/
 __init__.py
 main.py # Pipeline main
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
 (generated dinamicamente)
```

---

## Tests

### Execute All os Tests

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

1. **demo.ipynb**: demonstration end-to-end
 - generation of data sintéticos
 - training with STDP
 - evaluation of performance
 - visualizations

2. **stdp_example.ipynb**: Learning biological
 - LIF neuron demonstration
 - STDP weight updates
 - Temporal patterns

3. **loihi_benchmark.ipynb**: Hardware comparison
 - CPU benchmarking
 - Loihi yesulation
 - Comtotive analysis
 - Scalability tests

### Docker

- **DOCKER_DEPLOYMENT.md**: Guide complete of deployment
 - Quick start
 - Commands useful
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
| **Container** | Docker, Docker Compose | v2 |

---

## Conceitos Implementados

### Computational Neuroscience
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

### 1. Implementation Real in Loihi
- [ ] Migrar of Brian2 for Intel Lava (Loihi SDK)
- [ ] Otimizar architecture for cores Loihi
- [ ] Tests in hardware real (via Intel clord)

### 2. datasets Reais
- [ ] Integrar with datasets públicos (Kaggle, IEEE-CIS)
- [ ] Feature engineering more sofisticado
- [ ] Data augmentation

### 3. Production Hardening
- [ ] authentication JWT
- [ ] Rate limiting
- [ ] Redis caching
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] CI/CD pipeline

### 4. Compositions Adicionais
- [ ] IBM TrueNorth
- [ ] SpiNNaker
- [ ] BrainScaleS
- [ ] GPU (CUDA)
- [ ] TPU

---

## Metrics of the Project

| Métrica | Value |
|---------|-------|
| **Linhas of Code** | ~2,500+ |
| **Files Python** | 15 |
| **Tests** | 23 |
| **Endpoints API** | 7 |
| **Notebooks** | 3 |
| **Docker Services** | 3 |
| **Documentos** | 2 |
| **Gráficos** | 10+ |

---

## Contributing

This project is parte from the **Neuromorphic X Portfolio** and is complete for demonstration.

For sugestões or melhorias:
1. Abra uma issue in the repositório
2. Descreva to improvement proposta
3. Inclua code/examples when relevante

---

## License

MIT License - Project educacional of demonstration

---

## Author

**Mauro Risonho de Paula Assumpção**

- **Email:** mauro.risonho@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/maurorisonho
- **GitHub:** https://github.com/maurorisonho

---

## Status

**Project 01 of 10** in the Portfólio Neuromorphic X

 **100% COMPLETO**

**Data of Concluare**: December 2025

**time of Development**: ~3 withortanas

**next**: Project 02 - Cognitive Firewall against Engenharia Social

---

**"From spikes to ifcurity: Neuromorphic withputing for financial fraud detection"**
