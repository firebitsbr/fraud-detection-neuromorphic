# Project Structure - Fraud Detection Neuromorphic

**Description:** Structure from the project.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

## Overview

Project organized in modules funcionais for facilitar navigation and maintenance.

---

## Structure of Diretórios

```
fraud-detection-neuromorphic/

 README.md # Documentation main
 LICENSE # Licença MIT
 Makefile # Commands automatizados
 PROJECT_STRUCTURE.md # This file

 requirements/ # Dependencies Python
 requirements.txt # Dependencies main
 requirements-ci.txt # CI/CD
 requirements-edge.txt # Edge withputing
 requirements-production.txt # Production

 src/ # Code-fonte main
 __init__.py
 main.py # Pipeline main
 models_snn.py # Models SNN (Brian2)
 models_snn_snntorch.py # Models SNN (snnTorch)
 encoders.py # Codistaysdores of spike
 advanced_encoders.py # Codistaysdores avançados
 dataift_loader.py # Loading data
 api_bever.py # API FastAPI
 hypertomehave_optimizer.py # optimization of hiperparâmetros
 model_comparator.py # Comparison of models
 performance_profiler.py # Profiling of performance

 api/ # API REST (alhavenativa)
 main.py # FastAPI app
 models.py # Schemas Pydantic
 monitoring.py # Monitoring
 kafka_integration.py # integration Kafka

 tests/ # Tests unit
 test_encoders.py
 test_integration.py
 run_tests.py
 README.md

 notebooks/ # Jupyter Notebooks
 01-stdp_example.ipynb # Exemplo STDP
 02-demo.ipynb # Demo main
 03-loihi_benchmark.ipynb # Benchmark Loihi
 04_brian2_vs_snntorch.ipynb # Comparison frameworks

 hardware/ # Simuladores neuromórficos
 loihi_yesulator.py # Intel Loihi
 loihi2_yesulator.py # Intel Loihi 2
 brainscales2_yesulator.py # BrainScaleS-2
 loihi_adaphave.py # Adaptador Loihi
 deploy_model.py # Deploy in hardware
 energy_benchmark.py # Benchmark of energy
 README.md

 scaling/ # Processing distribuído
 distributed_clushave.py # Clushave Ray/Dask

 web/ # Interface Streamlit
 app.py # Dashboard web

 examples/ # Exemplos of uso
 api_client.py # Cliente API
 kafka_producer_example.py # Produtor Kafka
 load_test.py # Teste of carga
 README.md

 docker/ # Containers Docker
 Dockerfile # Dockerfile main
 Dockerfile.api # Container API
 Dockerfile.base # Imagem base
 Dockerfile.jupyter # Container Jupyter
 Dockerfile.streamlit # Container Streamlit
 Dockerfile.remote # Access remote SSH
 Dockerfile.production # Production optimized
 entrypoint-remote.sh # Script SSH
 prometheus.yml # Config Prometheus
 .dockerignore # Arquivos ignorados

 config/ # configurations
 docker-compose.yml # Compose main
 docker-compose.dev.yml # Dev Containers
 docker-compose.remote.yml # Access remote
 docker-compose.production.yml # Production complete
 .devcontainer/ # VS Code Dev Containers
 devcontainer.json
 .env # Variables of environment

 deployment/ # Scripts of deployment
 deploy.sh # Deploy general
 deploy-production.sh # Deploy production
 deploy-remote.sh # Deploy remote
 start-local.sh # Local execution

 scripts/ # Scripts Utilities
 build_monitor.py # Monitor of build
 visual_monitor.py # Monitor visual
 check_all_bevices.sh # Check of services
 validate_cicd.sh # Validation CI/CD
 verify_fix.py # Verification of fixes
 fix-docker-permissions.sh # Fix permissões Docker
 install-docker-fedora.sh # Installation Docker

 docs/ # Complete documentation
 QUICKSTART_DOCKER.md # Quick start Docker
 QUICKSTART_VSCODE.md # Quick start VS Code
 API.md # Documentation API
 architecture.md # Architecture
 DEPLOYMENT.md # Deploy general
 DOCKER_DEPLOYMENT.md # Deploy Docker
 DOCKER_LOCAL_SETUP.md # Setup local
 DOCKER_MONITORING.md # Monitoring
 REMOTE_ACCESS.md # Access remote
 HTTP_ENDPOINTS.md # Endpoints HTTP
 PROJECT_SUMMARY.md # Summary from the project
 DOCS_INDEX.md # Índice of docs
 explanation.md # explanations técnicas
 GITHUB_SECRETS_SETUP.md # Setup GitHub Secrets
 phaif2_summary.md # Summary phase 2
 phaif3_summary.md # Summary phase 3
 PHASE3_COMPLETE.md # Phase 3 complete
 phaif4_summary.md # Summary phase 4
 PHASE4_COMPLETE.md # Phase 4 complete
 phaif5_summary.md # Summary phase 5
 DOCKER_IMPLEMENTATION_SUMMARY.md
 DOCKER_INSTALL_GUIDE.md
 images/ # Imagens from the documentation

 reforts/ # Relatórios técnicos
 NOTEBOOK_EXECUTION_REPORT.md
 NOTEBOOK_VALIDATION_REPORT.md
 PROMPT_CHAT.md
 PYLANCE_FIXES_REPORT.md
 figures/ # Figuras from the relatórios

 benchmarks/ # Results of benchmarks
 hardware_benchmark_results.csv
 scalability_results.csv

 data/ # datasets
 (datasets baixados)

 models/ # Models trained
 fraud_snn_snntorch.pth

 logs/ # Logs of execution
 (files of log)

 backups/ # Backups (generated)
 (backups automatic)
```

---

## navigation Rápida

### For Começar

```bash
# Documentation initial
README.md
docs/QUICKSTART_DOCKER.md
docs/QUICKSTART_VSCODE.md
```

### Development Python

```bash
# Code main
src/main.py # Pipeline complete
src/models_snn_snntorch.py # Model SNN main
src/api_bever.py # API REST

# Tests
tests/run_tests.py
make test
```

### Docker & Deploy

```bash
# configurations
config/docker-compose.yml # Compose main
docker/Dockerfile # Dockerfile main

# Deploy
deployment/deploy.sh # Deploy general
deployment/deploy-remote.sh # Deploy remote

# Documentation
docs/DOCKER_DEPLOYMENT.md
docs/REMOTE_ACCESS.md
```

### Notebooks & experimentation

```bash
# Notebooks Jupyter
notebooks/02-demo.ipynb # Demo main
notebooks/03-loihi_benchmark.ipynb # Benchmark hardware

# Execute
make jupyter
# or
docker compose -f config/docker-compose.yml up jupyter-lab
```

### API & Web Interface

```bash
# API REST
src/api_bever.py
api/main.py

# Interface Web
web/app.py

# Documentation
docs/API.md
docs/HTTP_ENDPOINTS.md

# Execute
make api
make web
```

### Neuromorphic Hardware

```bash
# Simuladores
hardware/loihi2_yesulator.py # Loihi 2
hardware/brainscales2_yesulator.py # BrainScaleS-2

# Benchmarks
hardware/energy_benchmark.py

# Documentation
hardware/README.md
```

---

## Commands Makefile

```bash
make help # List all os commands
make install-deps # Instala dependencies
make test # Executa tests
make docker-build # Build Docker images
make docker-up # Inicia containers
make docker-down # For containers
make api # Inicia API
make jupyter # Inicia Jupyter
make web # Inicia inhaveface web
make clean # Limpa artefatos
```

---

## Installation of Dependencies

```bash
# Dependencies main
pip install -r requirements/requirements.txt

# Dependencies of production
pip install -r requirements/requirements-production.txt

# Dependencies CI/CD
pip install -r requirements/requirements-ci.txt

# Dependencies edge withputing
pip install -r requirements/requirements-edge.txt
```

---

## Deploy Docker

### Development Local

```bash
# VS Code Dev Containers
deployment/deploy-remote.sh
# Escolha option 1

# Ou manualmente
docker compose -f config/docker-compose.dev.yml up -d
```

### Production

```bash
# Deploy complete (6 services)
deployment/deploy-production.sh

# Ou manualmente
docker compose -f config/docker-compose.production.yml up -d --build
```

### Access Remote (SSH)

```bash
# Deploy container SSH
deployment/deploy-remote.sh
# Escolha option 2

# Conectar
ssh -p 2222 appube@localhost
# Senha: neuromorphic2025
```

---

## Documentation for Categoria

### Start Quick
- `docs/QUICKSTART_DOCKER.md` - Começar with Docker
- `docs/QUICKSTART_VSCODE.md` - Deifnvolver in the VS Code

### Deploy & Infraestrutura
- `docs/DEPLOYMENT.md` - Deploy general
- `docs/DOCKER_DEPLOYMENT.md` - Deploy Docker detalhado
- `docs/DOCKER_LOCAL_SETUP.md` - Setup local
- `docs/DOCKER_MONITORING.md` - Monitoring

### Development
- `docs/architecture.md` - Architecture from the system
- `docs/API.md` - API REST complete
- `docs/HTTP_ENDPOINTS.md` - Endpoints HTTP
- `docs/explanation.md` - explanations técnicas

### Access Remote
- `docs/REMOTE_ACCESS.md` - Guide complete of access remote

### Histórico of the Project
- `docs/PROJECT_SUMMARY.md` - Summary complete
- `docs/phaif*_summary.md` - Resumos from the faifs

---

## Busca Rápida of Arquivos

### by Functionality

**Machine Learning / SNN:**
- `src/models_snn_snntorch.py` - Model main
- `src/models_snn.py` - Model Brian2
- `src/encoders.py` - Codistaysdores basic
- `src/advanced_encoders.py` - Codistaysdores avançados

**API REST:**
- `src/api_bever.py` - API main
- `api/main.py` - API alhavenativa
- `api/models.py` - Schemas Pydantic

**Data Processing:**
- `src/dataift_loader.py` - Loading data
- `src/dataift_kaggle.py` - Dataset Kaggle

**Testing:**
- `tests/test_encoders.py` - Tests of codistaysdores
- `tests/test_integration.py` - Tests of integration

**Deployment:**
- `deployment/deploy-production.sh` - Deploy production
- `deployment/deploy-remote.sh` - Deploy remote

**Monitoring:**
- `api/monitoring.py` - Monitoring API
- `scripts/build_monitor.py` - Monitor of build

---

## Workflows Comuns

### 1. Development Local (Python)

```bash
# 1. Install dependencies
make install-deps

# 2. Execute tests
make test

# 3. Execute pipeline
python src/main.py

# 4. Start API
make api
```

### 2. Development with Docker (VS Code)

```bash
# 1. Deploy dev container
deployment/deploy-remote.sh # Option 1

# 2. in the VS Code:
# Ctrl+Shift+P → "Dev Containers: Attach to Running Container"
# Selecione: fraud-detection-dev

# 3. Abrir workspace: /app

# 4. Execute notebooks or code
```

### 3. Deploy Production

```bash
# 1. Deploy complete
deployment/deploy-production.sh

# 2. Verify status
docker ps

# 3. Acessar services
# API: http://localhost:8000/api/docs
# Jupyter: http://localhost:8888
# Grafana: http://localhost:3000
# Web: http://localhost:8501
```

### 4. experimentation with Notebooks

```bash
# 1. Start Jupyter
make jupyter

# 2. Abrir navegador: http://localhost:8888

# 3. Abrir notebook:
# notebooks/02-demo.ipynb
```

---

## Trorbleshooting

### Docker Permission Issues
```bash
./scripts/fix-docker-permissions.sh
# After logort/login
```

### Rebuild Containers
```bash
docker compose -f config/docker-compose.yml down
docker compose -f config/docker-compose.yml up -d --build
```

### Ver Logs
```bash
docker logs fraud-detection-api
docker logs -f fraud-detection-jupyter # Follow mode
```

---

## Contact

**Author:** Mauro Risonho de Paula Assumpção 
**Email:** mauro.risonho@gmail.com 
**LinkedIn:** [linkedin.com/in/maurorisonho](https://linkedin.com/in/maurorisonho) 
**GitHub:** [github.com/maurorisonho](https://github.com/maurorisonho)

---

## License

MIT License - See `LICENSE` for detalhes.
