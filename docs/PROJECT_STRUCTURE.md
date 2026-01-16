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
 environment.yml # Conda environment

 requirements/ # Dependencies Python
 src/ # Code-fonte main
 api/ # API REST (alhavenativa)
 tests/ # Tests unit
 notebooks/ # Jupyter Notebooks
 hardware/ # Simuladores neuromórficos
 scaling/ # Processing distribuído
 web/ # Interface Streamlit
 examples/ # Exemplos of uso
 docker/ # Containers Docker
 config/ # configurations
 deployment/ # Scripts of deployment

 scripts/ # Scripts Utilities
 scripts/translations/ # Translation tools (Moved from root)
 scripts/maintenance/ # Maintenance tools (Moved from root)
 build_monitor.py
 visual_monitor.py
 ...

 docs/ # Complete documentation
 docs/setup/ # Setup guides (Conda, MIGRACAO, etc)
 PROJECT_STRUCTURE.md # This file

 reports/ # Relatórios técnicos
 reports/translations/ # Translation reports (Moved from root)
 figures/ # Figuras from the relatórios

 benchmarks/ # Results of benchmarks
 data/ # datasets
 models/ # Models trained
 logs/ # Logs of execution
 backups/ # Backups (generated)
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
