# 📁 Estrutura do Projeto - Fraud Detection Neuromorphic

**Descrição:** Estrutura do projeto.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025

## 🎯 Visão Geral

Projeto organizado em módulos funcionais para facilitar navegação e manutenção.

---

## 📂 Estrutura de Diretórios

```
fraud-detection-neuromorphic/
│
├── 📋 README.md                    # Documentação principal
├── 📋 LICENSE                      # Licença MIT
├── 📋 Makefile                     # Comandos automatizados
├── 📋 PROJECT_STRUCTURE.md         # Este arquivo
│
├── 📦 requirements/                # Dependências Python
│   ├── requirements.txt            # Dependências principais
│   ├── requirements-ci.txt         # CI/CD
│   ├── requirements-edge.txt       # Edge computing
│   └── requirements-production.txt # Produção
│
├── 🐍 src/                         # Código-fonte principal
│   ├── __init__.py
│   ├── main.py                     # Pipeline principal
│   ├── models_snn.py               # Modelos SNN (Brian2)
│   ├── models_snn_snntorch.py      # Modelos SNN (snnTorch)
│   ├── encoders.py                 # Codificadores de spike
│   ├── advanced_encoders.py        # Codificadores avançados
│   ├── dataset_loader.py           # Carregamento de dados
│   ├── api_server.py               # API FastAPI
│   ├── hyperparameter_optimizer.py # Otimização de hiperparâmetros
│   ├── model_comparator.py         # Comparação de modelos
│   └── performance_profiler.py     # Profiling de performance
│
├── 🌐 api/                         # API REST (alternativa)
│   ├── main.py                     # FastAPI app
│   ├── models.py                   # Schemas Pydantic
│   ├── monitoring.py               # Monitoramento
│   └── kafka_integration.py        # Integração Kafka
│
├── 🧪 tests/                       # Testes unitários
│   ├── test_encoders.py
│   ├── test_integration.py
│   ├── run_tests.py
│   └── README.md
│
├── 📓 notebooks/                   # Jupyter Notebooks
│   ├── 01-stdp_example.ipynb       # Exemplo STDP
│   ├── 02-demo.ipynb               # Demo principal
│   ├── 03-loihi_benchmark.ipynb    # Benchmark Loihi
│   └── 04_brian2_vs_snntorch.ipynb # Comparação frameworks
│
├── 💻 hardware/                    # Simuladores neuromórficos
│   ├── loihi_simulator.py          # Intel Loihi
│   ├── loihi2_simulator.py         # Intel Loihi 2
│   ├── brainscales2_simulator.py   # BrainScaleS-2
│   ├── loihi_adapter.py            # Adaptador Loihi
│   ├── deploy_model.py             # Deploy em hardware
│   ├── energy_benchmark.py         # Benchmark de energia
│   └── README.md
│
├── 🔄 scaling/                     # Processamento distribuído
│   └── distributed_cluster.py      # Cluster Ray/Dask
│
├── 🌍 web/                         # Interface Streamlit
│   └── app.py                      # Dashboard web
│
├── 📝 examples/                    # Exemplos de uso
│   ├── api_client.py               # Cliente API
│   ├── kafka_producer_example.py   # Produtor Kafka
│   ├── load_test.py                # Teste de carga
│   └── README.md
│
├── 🐳 docker/                      # Containers Docker
│   ├── Dockerfile                  # Dockerfile principal
│   ├── Dockerfile.api              # Container API
│   ├── Dockerfile.base             # Imagem base
│   ├── Dockerfile.jupyter          # Container Jupyter
│   ├── Dockerfile.streamlit        # Container Streamlit
│   ├── Dockerfile.remote           # Acesso remoto SSH
│   ├── Dockerfile.production       # Produção otimizada
│   ├── entrypoint-remote.sh        # Script SSH
│   ├── prometheus.yml              # Config Prometheus
│   └── .dockerignore               # Arquivos ignorados
│
├── ⚙️ config/                      # Configurações
│   ├── docker-compose.yml          # Compose principal
│   ├── docker-compose.dev.yml      # Dev Containers
│   ├── docker-compose.remote.yml   # Acesso remoto
│   ├── docker-compose.production.yml # Produção completa
│   ├── .devcontainer/              # VS Code Dev Containers
│   │   └── devcontainer.json
│   └── .env                        # Variáveis de ambiente
│
├── 🚀 deployment/                  # Scripts de deployment
│   ├── deploy.sh                   # Deploy geral
│   ├── deploy-production.sh        # Deploy produção
│   ├── deploy-remote.sh            # Deploy remoto
│   └── start-local.sh              # Execução local
│
├── 🛠️ scripts/                     # Scripts utilitários
│   ├── build_monitor.py            # Monitor de build
│   ├── visual_monitor.py           # Monitor visual
│   ├── check_all_services.sh       # Check de serviços
│   ├── validate_cicd.sh            # Validação CI/CD
│   ├── verify_fix.py               # Verificação de fixes
│   ├── fix-docker-permissions.sh   # Fix permissões Docker
│   └── install-docker-fedora.sh    # Instalação Docker
│
├── 📚 docs/                        # Documentação completa
│   ├── QUICKSTART_DOCKER.md        # Quick start Docker
│   ├── QUICKSTART_VSCODE.md        # Quick start VS Code
│   ├── API.md                      # Documentação API
│   ├── architecture.md             # Arquitetura
│   ├── DEPLOYMENT.md               # Deploy geral
│   ├── DOCKER_DEPLOYMENT.md        # Deploy Docker
│   ├── DOCKER_LOCAL_SETUP.md       # Setup local
│   ├── DOCKER_MONITORING.md        # Monitoramento
│   ├── REMOTE_ACCESS.md            # Acesso remoto
│   ├── HTTP_ENDPOINTS.md           # Endpoints HTTP
│   ├── PROJECT_SUMMARY.md          # Resumo do projeto
│   ├── DOCS_INDEX.md               # Índice de docs
│   ├── explanation.md              # Explicações técnicas
│   ├── GITHUB_SECRETS_SETUP.md     # Setup GitHub Secrets
│   ├── phase2_summary.md           # Resumo fase 2
│   ├── phase3_summary.md           # Resumo fase 3
│   ├── PHASE3_COMPLETE.md          # Fase 3 completa
│   ├── phase4_summary.md           # Resumo fase 4
│   ├── PHASE4_COMPLETE.md          # Fase 4 completa
│   ├── phase5_summary.md           # Resumo fase 5
│   ├── DOCKER_IMPLEMENTATION_SUMMARY.md
│   ├── DOCKER_INSTALL_GUIDE.md
│   └── images/                     # Imagens da documentação
│
├── 📊 reports/                     # Relatórios técnicos
│   ├── NOTEBOOK_EXECUTION_REPORT.md
│   ├── NOTEBOOK_VALIDATION_REPORT.md
│   ├── PROMPT_CHAT.md
│   ├── PYLANCE_FIXES_REPORT.md
│   └── figures/                    # Figuras dos relatórios
│
├── 📈 benchmarks/                  # Resultados de benchmarks
│   ├── hardware_benchmark_results.csv
│   └── scalability_results.csv
│
├── 💾 data/                        # Datasets
│   └── (datasets baixados)
│
├── 🎓 models/                      # Modelos treinados
│   └── fraud_snn_snntorch.pth
│
├── 📝 logs/                        # Logs de execução
│   └── (arquivos de log)
│
└── 🗄️ backups/                     # Backups (gerados)
    └── (backups automáticos)
```

---

## 🎯 Navegação Rápida

### 🚀 Para Começar

```bash
# Documentação inicial
README.md
docs/QUICKSTART_DOCKER.md
docs/QUICKSTART_VSCODE.md
```

### 🐍 Desenvolvimento Python

```bash
# Código principal
src/main.py                 # Pipeline completo
src/models_snn_snntorch.py  # Modelo SNN principal
src/api_server.py           # API REST

# Testes
tests/run_tests.py
make test
```

### 🐳 Docker & Deploy

```bash
# Configurações
config/docker-compose.yml           # Compose principal
docker/Dockerfile                   # Dockerfile principal

# Deploy
deployment/deploy.sh                # Deploy geral
deployment/deploy-remote.sh         # Deploy remoto

# Documentação
docs/DOCKER_DEPLOYMENT.md
docs/REMOTE_ACCESS.md
```

### 📓 Notebooks & Experimentação

```bash
# Notebooks Jupyter
notebooks/02-demo.ipynb             # Demo principal
notebooks/03-loihi_benchmark.ipynb  # Benchmark hardware

# Executar
make jupyter
# ou
docker compose -f config/docker-compose.yml up jupyter-lab
```

### 🌐 API & Web Interface

```bash
# API REST
src/api_server.py
api/main.py

# Interface Web
web/app.py

# Documentação
docs/API.md
docs/HTTP_ENDPOINTS.md

# Executar
make api
make web
```

### 💻 Hardware Neuromórfico

```bash
# Simuladores
hardware/loihi2_simulator.py        # Loihi 2
hardware/brainscales2_simulator.py  # BrainScaleS-2

# Benchmarks
hardware/energy_benchmark.py

# Documentação
hardware/README.md
```

---

## 🛠️ Comandos Makefile

```bash
make help           # Lista todos os comandos
make install-deps   # Instala dependências
make test           # Executa testes
make docker-build   # Build Docker images
make docker-up      # Inicia containers
make docker-down    # Para containers
make api            # Inicia API
make jupyter        # Inicia Jupyter
make web            # Inicia interface web
make clean          # Limpa artefatos
```

---

## 📦 Instalação de Dependências

```bash
# Dependências principais
pip install -r requirements/requirements.txt

# Dependências de produção
pip install -r requirements/requirements-production.txt

# Dependências CI/CD
pip install -r requirements/requirements-ci.txt

# Dependências edge computing
pip install -r requirements/requirements-edge.txt
```

---

## 🐳 Deploy Docker

### Desenvolvimento Local

```bash
# VS Code Dev Containers
deployment/deploy-remote.sh
# Escolha opção 1

# Ou manualmente
docker compose -f config/docker-compose.dev.yml up -d
```

### Produção

```bash
# Deploy completo (6 serviços)
deployment/deploy-production.sh

# Ou manualmente
docker compose -f config/docker-compose.production.yml up -d --build
```

### Acesso Remoto (SSH)

```bash
# Deploy container SSH
deployment/deploy-remote.sh
# Escolha opção 2

# Conectar
ssh -p 2222 appuser@localhost
# Senha: neuromorphic2025
```

---

## 📚 Documentação por Categoria

### Início Rápido
- `docs/QUICKSTART_DOCKER.md` - Começar com Docker
- `docs/QUICKSTART_VSCODE.md` - Desenvolver no VS Code

### Deploy & Infraestrutura
- `docs/DEPLOYMENT.md` - Deploy geral
- `docs/DOCKER_DEPLOYMENT.md` - Deploy Docker detalhado
- `docs/DOCKER_LOCAL_SETUP.md` - Setup local
- `docs/DOCKER_MONITORING.md` - Monitoramento

### Desenvolvimento
- `docs/architecture.md` - Arquitetura do sistema
- `docs/API.md` - API REST completa
- `docs/HTTP_ENDPOINTS.md` - Endpoints HTTP
- `docs/explanation.md` - Explicações técnicas

### Acesso Remoto
- `docs/REMOTE_ACCESS.md` - Guia completo de acesso remoto

### Histórico do Projeto
- `docs/PROJECT_SUMMARY.md` - Resumo completo
- `docs/phase*_summary.md` - Resumos das fases

---

## 🔍 Busca Rápida de Arquivos

### Por Funcionalidade

**Machine Learning / SNN:**
- `src/models_snn_snntorch.py` - Modelo principal
- `src/models_snn.py` - Modelo Brian2
- `src/encoders.py` - Codificadores básicos
- `src/advanced_encoders.py` - Codificadores avançados

**API REST:**
- `src/api_server.py` - API principal
- `api/main.py` - API alternativa
- `api/models.py` - Schemas Pydantic

**Data Processing:**
- `src/dataset_loader.py` - Carregamento de dados
- `src/dataset_kaggle.py` - Dataset Kaggle

**Testing:**
- `tests/test_encoders.py` - Testes de codificadores
- `tests/test_integration.py` - Testes de integração

**Deployment:**
- `deployment/deploy-production.sh` - Deploy produção
- `deployment/deploy-remote.sh` - Deploy remoto

**Monitoring:**
- `api/monitoring.py` - Monitoramento API
- `scripts/build_monitor.py` - Monitor de build

---

## 🎯 Workflows Comuns

### 1. Desenvolvimento Local (Python)

```bash
# 1. Instalar dependências
make install-deps

# 2. Executar testes
make test

# 3. Executar pipeline
python src/main.py

# 4. Iniciar API
make api
```

### 2. Desenvolvimento com Docker (VS Code)

```bash
# 1. Deploy dev container
deployment/deploy-remote.sh  # Opção 1

# 2. No VS Code:
# Ctrl+Shift+P → "Dev Containers: Attach to Running Container"
# Selecione: fraud-detection-dev

# 3. Abrir workspace: /app

# 4. Executar notebooks ou código
```

### 3. Deploy Produção

```bash
# 1. Deploy completo
deployment/deploy-production.sh

# 2. Verificar status
docker ps

# 3. Acessar serviços
# API: http://localhost:8000/api/docs
# Jupyter: http://localhost:8888
# Grafana: http://localhost:3000
# Web: http://localhost:8501
```

### 4. Experimentação com Notebooks

```bash
# 1. Iniciar Jupyter
make jupyter

# 2. Abrir navegador: http://localhost:8888

# 3. Abrir notebook:
# notebooks/02-demo.ipynb
```

---

## 🔧 Troubleshooting

### Docker Permission Issues
```bash
./scripts/fix-docker-permissions.sh
# Depois logout/login
```

### Rebuild Containers
```bash
docker compose -f config/docker-compose.yml down
docker compose -f config/docker-compose.yml up -d --build
```

### Ver Logs
```bash
docker logs fraud-detection-api
docker logs -f fraud-detection-jupyter  # Follow mode
```

---

## 📞 Contato

**Autor:** Mauro Risonho de Paula Assumpção  
**Email:** mauro.risonho@gmail.com  
**LinkedIn:** [linkedin.com/in/maurorisonho](https://linkedin.com/in/maurorisonho)  
**GitHub:** [github.com/maurorisonho](https://github.com/maurorisonho)

---

## 📄 Licença

MIT License - Veja `LICENSE` para detalhes.
