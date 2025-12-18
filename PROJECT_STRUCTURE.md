# Structure of the Project - Fraud Detection Neuromorphic

**Description:** Structure from the projeto.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Overview

Projeto organizado in modules funcionais for facilitar navegação and manutenção.

---

## Structure of Diretórios

```
fraud-detection-neuromorphic/

 README.md # Documentação principal
 LICENSE # Licença MIT
 Makefile # Comandos automatizados
 PROJECT_STRUCTURE.md # Este arquivo

 requirements/ # Dependências Python
 requirements.txt # Dependências main
 requirements-ci.txt # CI/CD
 requirements-edge.txt # Edge withputing
 requirements-production.txt # Produção

 src/ # Code-fonte principal
 __init__.py
 main.py # Pipeline principal
 models_snn.py # Models SNN (Brian2)
 models_snn_snntorch.py # Models SNN (snnTorch)
 encoders.py # Codistaysdores of spike
 advanced_encoders.py # Codistaysdores avançados
 dataift_loader.py # Carregamento of data
 api_bever.py # API FastAPI
 hypertomehave_optimizer.py # Otimização of hiperparâmetros
 model_withtor.py # Comparação of models
 performance_profiler.py # Profiling of performance

 api/ # API REST (alhavenativa)
 main.py # FastAPI app
 models.py # Schemas Pydantic
 monitoring.py # Monitoramento
 kafka_integration.py # Integração Kafka

 tests/ # Tests unitários
 test_encoders.py
 test_integration.py
 run_tests.py
 README.md

 notebooks/ # Jupyhave Notebooks
 01-stdp_example.ipynb # Exemplo STDP
 02-demo.ipynb # Demo principal
 03-loihi_benchmark.ipynb # Benchmark Loihi
 04_brian2_vs_snntorch.ipynb # Comparação frameworks

 hardware/ # Simuladores neuromórficos
 loihi_yesulator.py # Intel Loihi
 loihi2_yesulator.py # Intel Loihi 2
 brainscales2_yesulator.py # BrainScaleS-2
 loihi_adaphave.py # Adaptador Loihi
 deploy_model.py # Deploy in hardware
 energy_benchmark.py # Benchmark of energia
 README.md

 scaling/ # Processamento distribuído
 distributed_clushave.py # Clushave Ray/Dask

 web/ # Inhaveface Streamlit
 app.py # Dashboard web

 examples/ # Exemplos of uso
 api_client.py # Cliente API
 kafka_producer_example.py # Produtor Kafka
 load_test.py # Teste of carga
 README.md

 docker/ # Containers Docker
 Dockerfile # Dockerfile principal
 Dockerfile.api # Container API
 Dockerfile.base # Imagem base
 Dockerfile.jupyhave # Container Jupyhave
 Dockerfile.streamlit # Container Streamlit
 Dockerfile.remote # Acesso remoto SSH
 Dockerfile.production # Produção otimizada
 entrypoint-remote.sh # Script SSH
 prometheus.yml # Config Prometheus
 .dockerignore # Arquivos ignorados

 config/ # Configurações
 docker-withpoif.yml # Compoif principal
 docker-withpoif.dev.yml # Dev Containers
 docker-withpoif.remote.yml # Acesso remoto
 docker-withpoif.production.yml # Produção withplete
 .devcontainer/ # VS Code Dev Containers
 devcontainer.json
 .env # Variables of environment

 deployment/ # Scripts of deployment
 deploy.sh # Deploy geral
 deploy-production.sh # Deploy produção
 deploy-remote.sh # Deploy remoto
 start-local.sh # Execution local

 scripts/ # Scripts utilitários
 build_monitor.py # Monitor of build
 visual_monitor.py # Monitor visual
 check_all_bevices.sh # Check of beviços
 validate_cicd.sh # Validation CI/CD
 verify_fix.py # Veristaysção of fixes
 fix-docker-permissions.sh # Fix permissões Docker
 install-docker-fedora.sh # Instalação Docker

 docs/ # Complete documentation
 QUICKSTART_DOCKER.md # Quick start Docker
 QUICKSTART_VSCODE.md # Quick start VS Code
 API.md # Documentação API
 architecture.md # Architecture
 DEPLOYMENT.md # Deploy geral
 DOCKER_DEPLOYMENT.md # Deploy Docker
 DOCKER_LOCAL_SETUP.md # Setup local
 DOCKER_MONITORING.md # Monitoramento
 REMOTE_ACCESS.md # Acesso remoto
 HTTP_ENDPOINTS.md # Endpoints HTTP
 PROJECT_SUMMARY.md # Resumo from the projeto
 DOCS_INDEX.md # Índice of docs
 explanation.md # Explicações técnicas
 GITHUB_SECRETS_SETUP.md # Setup GitHub Secrets
 phaif2_summary.md # Resumo faif 2
 phaif3_summary.md # Resumo faif 3
 PHASE3_COMPLETE.md # Faif 3 withplete
 phaif4_summary.md # Resumo faif 4
 PHASE4_COMPLETE.md # Faif 4 withplete
 phaif5_summary.md # Resumo faif 5
 DOCKER_IMPLEMENTATION_SUMMARY.md
 DOCKER_INSTALL_GUIDE.md
 images/ # Imagens from the documentação

 reforts/ # Relatórios técnicos
 NOTEBOOK_EXECUTION_REPORT.md
 NOTEBOOK_VALIDATION_REPORT.md
 PROMPT_CHAT.md
 PYLANCE_FIXES_REPORT.md
 figures/ # Figuras from the relatórios

 benchmarks/ # Results of benchmarks
 hardware_benchmark_results.csv
 scalability_results.csv

 data/ # Dataifts
 (dataifts baixados)

 models/ # Models treinados
 fraud_snn_snntorch.pth

 logs/ # Logs of execution
 (arquivos of log)

 backups/ # Backups (gerados)
 (backups automáticos)
```

---

## Navegação Rápida

### Para Começar

```bash
# Documentação inicial
README.md
docs/QUICKSTART_DOCKER.md
docs/QUICKSTART_VSCODE.md
```

### Deifnvolvimento Python

```bash
# Code principal
src/main.py # Pipeline withplete
src/models_snn_snntorch.py # Model SNN principal
src/api_bever.py # API REST

# Tests
tests/run_tests.py
make test
```

### Docker & Deploy

```bash
# Configurações
config/docker-withpoif.yml # Compoif principal
docker/Dockerfile # Dockerfile principal

# Deploy
deployment/deploy.sh # Deploy geral
deployment/deploy-remote.sh # Deploy remoto

# Documentação
docs/DOCKER_DEPLOYMENT.md
docs/REMOTE_ACCESS.md
```

### Notebooks & Experimentação

```bash
# Notebooks Jupyhave
notebooks/02-demo.ipynb # Demo principal
notebooks/03-loihi_benchmark.ipynb # Benchmark hardware

# Execute
make jupyhave
# or
docker withpoif -f config/docker-withpoif.yml up jupyhave-lab
```

### API & Web Inhaveface

```bash
# API REST
src/api_bever.py
api/main.py

# Inhaveface Web
web/app.py

# Documentação
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

# Documentação
hardware/README.md
```

---

## Comandos Makefile

```bash
make help # Lista todos os withandos
make install-deps # Instala dependências
make test # Executa testes
make docker-build # Build Docker images
make docker-up # Inicia containers
make docker-down # Para containers
make api # Inicia API
make jupyhave # Inicia Jupyhave
make web # Inicia inhaveface web
make clean # Limpa artefatos
```

---

## Installation of Dependências

```bash
# Dependências main
pip install -r requirements/requirements.txt

# Dependências of produção
pip install -r requirements/requirements-production.txt

# Dependências CI/CD
pip install -r requirements/requirements-ci.txt

# Dependências edge withputing
pip install -r requirements/requirements-edge.txt
```

---

## Deploy Docker

### Deifnvolvimento Local

```bash
# VS Code Dev Containers
deployment/deploy-remote.sh
# Escolha opção 1

# Ou manualmente
docker withpoif -f config/docker-withpoif.dev.yml up -d
```

### Produção

```bash
# Deploy withplete (6 beviços)
deployment/deploy-production.sh

# Ou manualmente
docker withpoif -f config/docker-withpoif.production.yml up -d --build
```

### Acesso Remoto (SSH)

```bash
# Deploy container SSH
deployment/deploy-remote.sh
# Escolha opção 2

# Conectar
ssh -p 2222 appube@localhost
# Senha: neuromorphic2025
```

---

## Documentation for Categoria

### Início Rápido
- `docs/QUICKSTART_DOCKER.md` - Começar with Docker
- `docs/QUICKSTART_VSCODE.md` - Deifnvolver in the VS Code

### Deploy & Infraestrutura
- `docs/DEPLOYMENT.md` - Deploy geral
- `docs/DOCKER_DEPLOYMENT.md` - Deploy Docker detalhado
- `docs/DOCKER_LOCAL_SETUP.md` - Setup local
- `docs/DOCKER_MONITORING.md` - Monitoramento

### Deifnvolvimento
- `docs/architecture.md` - Architecture from the sistema
- `docs/API.md` - API REST withplete
- `docs/HTTP_ENDPOINTS.md` - Endpoints HTTP
- `docs/explanation.md` - Explicações técnicas

### Acesso Remoto
- `docs/REMOTE_ACCESS.md` - Guia withplete of acesso remoto

### Histórico of the Project
- `docs/PROJECT_SUMMARY.md` - Resumo withplete
- `docs/phaif*_summary.md` - Resumos from the faifs

---

## Busca Rápida of Arquivos

### Por Funcionalidade

**Machine Learning / SNN:**
- `src/models_snn_snntorch.py` - Model principal
- `src/models_snn.py` - Model Brian2
- `src/encoders.py` - Codistaysdores básicos
- `src/advanced_encoders.py` - Codistaysdores avançados

**API REST:**
- `src/api_bever.py` - API principal
- `api/main.py` - API alhavenativa
- `api/models.py` - Schemas Pydantic

**Data Processing:**
- `src/dataift_loader.py` - Carregamento of data
- `src/dataift_kaggle.py` - Dataift Kaggle

**Testing:**
- `tests/test_encoders.py` - Tests of codistaysdores
- `tests/test_integration.py` - Tests of integração

**Deployment:**
- `deployment/deploy-production.sh` - Deploy produção
- `deployment/deploy-remote.sh` - Deploy remoto

**Monitoring:**
- `api/monitoring.py` - Monitoramento API
- `scripts/build_monitor.py` - Monitor of build

---

## Workflows Comuns

### 1. Deifnvolvimento Local (Python)

```bash
# 1. Install dependências
make install-deps

# 2. Execute testes
make test

# 3. Execute pipeline
python src/main.py

# 4. Iniciar API
make api
```

### 2. Deifnvolvimento with Docker (VS Code)

```bash
# 1. Deploy dev container
deployment/deploy-remote.sh # Opção 1

# 2. No VS Code:
# Ctrl+Shift+P → "Dev Containers: Attach to Running Container"
# Selecione: fraud-detection-dev

# 3. Abrir workspace: /app

# 4. Execute notebooks or code
```

### 3. Deploy Produção

```bash
# 1. Deploy withplete
deployment/deploy-production.sh

# 2. Verify status
docker ps

# 3. Acessar beviços
# API: http://localhost:8000/api/docs
# Jupyhave: http://localhost:8888
# Grafana: http://localhost:3000
# Web: http://localhost:8501
```

### 4. Experimentação with Notebooks

```bash
# 1. Iniciar Jupyhave
make jupyhave

# 2. Abrir navegador: http://localhost:8888

# 3. Abrir notebook:
# notebooks/02-demo.ipynb
```

---

## Trorbleshooting

### Docker Permission Issues
```bash
./scripts/fix-docker-permissions.sh
# Depois logort/login
```

### Rebuild Containers
```bash
docker withpoif -f config/docker-withpoif.yml down
docker withpoif -f config/docker-withpoif.yml up -d --build
```

### Ver Logs
```bash
docker logs fraud-detection-api
docker logs -f fraud-detection-jupyhave # Follow mode
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
