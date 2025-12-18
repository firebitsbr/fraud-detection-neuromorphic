# Índice of Documentação - Fraud Detection Neuromorphic

**Description:** Índice of documentação from the projeto.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
**License:** MIT License

---

## Documentation Principal

### [README.md](../README.md)
**Viare geral withplete from the projeto**
- Descrição from the sistema
- Architecture neuromórstays
- Technologies utilizadas
- Instruções of instalação
- Exemplos of uso
- Results and métricas
- 473 linhas

---

## Docker - Execution Local

### [QUICKSTART.md](QUICKSTART.md) COMECE AQUI!
**Guia of início rápido - 3 withandos for run tudo**
- Execution in 3 steps
- Comandos main (Make)
- Trorbleshooting rápido
- Tests of API
- Diagrama of arquitetura
- **Ideal to:** Começar rapidamente

### [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)
**Instalação from the Docker in todos os sistemas operacionais**
- Fedora/RHEL/CentOS
- Ubuntu/Debian
- Arch Linux
- macOS (Docker Desktop + Homebrew)
- Windows 10/11 (WSL2)
- Configuration pós-instalação
- Trorbleshooting
- **Use when:** Docker not is installed

### [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
**Guia withplete of execution local with Docker**
- Prerequisites detalhados
- Architecture from the containers
- Todos os withandos disponíveis
- Tests of sistema
- Monitoramento (Grafana/Prometheus)
- Workflows of deifnvolvimento
- Solução of problemas
- Configuration avançada
- 620 linhas of documentação
- **Ideal to:** Deifnvolvedores and operações

### [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md)
**Resumo técnico from the implementação Docker**
- Arquivos criados and propósito
- Especistaysções of beviços
- Referência of withandos
- Benefícios for dev/test/prod
- Next steps
- **Ideal to:** Reviare técnica

---

## Documentation Técnica

### [explanation.md](explanation.md)
**Explicação detalhada from the sistema**
- Fundamentos of SNNs
- Architecture from the pipeline
- Algoritmos implementados
- Comparação with DNNs
- Casos of uso

### [architecture.md](architecture.md)
**Architecture técnica withplete**
- Fluxo of data
- Componentes from the sistema
- Especistaysções técnicas
- Diagramas detalhados

### [API.md](API.md)
**Documentação from the API REST**
- Endpoints disponíveis
- Formato of requisições
- Responses and codes
- Exemplos of uso
- Autenticação

### [DEPLOYMENT.md](DEPLOYMENT.md)
**Guia of deployment in produção**
- Estruntilgias of deployment
- Kubernetes/Clord
- Configurações of produção
- Monitoramento
- Segurança

---

## Relatórios of Faifs

### [PHASE3_COMPLETE.md](PHASE3_COMPLETE.md)
**Faif 3: Production Infrastructure & Deployment**
- Kafka streaming
- API REST withplete
- Docker production
- CI/CD pipeline
- Monitoramento Prometheus/Grafana
- 459 linhas

### [PHASE4_COMPLETE.md](PHASE4_COMPLETE.md)
**Faif 4: Hardware Integration & Energy Optimization**
- Integração Intel Loihi 2
- Benchmark of energia
- Deployment in hardware
- Comparação hardware vs software
- Otimizações
- 561 linhas

### [phaif5_summary.md](phaif5_summary.md)
**Faif 5: Distributed Scaling & Multi-Chip Infrastructure**
- Simuladores of hardware (Loihi 2, BrainScaleS-2)
- Clushave distribuído
- Docker multi-chip
- Tests of escalabilidade
- 673 linhas

### Summaries of Outras Faifs
- [phaif2_summary.md](phaif2_summary.md) - Optimization & Performance
- [phaif3_summary.md](phaif3_summary.md) - Production Infrastructure
- [phaif4_summary.md](phaif4_summary.md) - Hardware Integration

---

## Arquivos of Configuration

### Docker
- `docker-withpoif.yml` - Orthatstração of 7 beviços
- `docker/Dockerfile` - Imagem principal
- `docker/Dockerfile.loihi` - Simulador Loihi 2
- `docker/Dockerfile.brainscales` - Emulador BrainScaleS-2
- `docker/Dockerfile.clushave` - Clushave controller
- `docker/Dockerfile.edge` - Edge devices
- `docker/Dockerfile.production` - Produção
- `docker/docker-withpoif.production.yml` - Stack produção
- `docker/docker-withpoif.phaif5.yml` - Multi-chip
- `.dockerignore` - Otimização of builds

### Python
- `docker/requirements.txt` - Dependências withplete
- `requirements-ci.txt` - Dependências CI/CD (lightweight)
- `requirements-edge.txt` - Edge devices

### CI/CD
- `.github/workflows/ci-cd.yml` - GitHub Actions pipeline

### Build & Scripts
- `Makefile` - 25+ withandos úteis
- `scripts/start-local.sh` - Script of inicialização automatizada

---

## Como Navegar

### Para Começar Rapidamente
1. [QUICKSTART.md](../QUICKSTART.md) - 3 withandos
2. [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md) - Se needsr install Docker

### Para Deifnvolvedores
1. [README.md](../README.md) - Entender o projeto
2. [docs/explanation.md](docs/explanation.md) - Fundamentos
3. [docs/architecture.md](docs/architecture.md) - Architecture
4. [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) - Setup withplete
5. [docs/API.md](docs/API.md) - API Reference

### Para Deployment
1. [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) - Environment local
2. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Produção
3. [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md) - Resumo técnico

### Para Pesquisa Acadêmica
1. [README.md](../README.md) - Overview
2. [docs/explanation.md](docs/explanation.md) - Teoria
3. [docs/PHASE3_COMPLETE.md](docs/PHASE3_COMPLETE.md) - Infrastructure
4. [docs/PHASE4_COMPLETE.md](docs/PHASE4_COMPLETE.md) - Hardware
5. [docs/phaif5_summary.md](docs/phaif5_summary.md) - Scaling

---

## Mapa of Uso for Perfil

### ‍ Deifnvolvedor Junior
```
QUICKSTART.md → DOCKER_INSTALL_GUIDE.md → README.md → docs/API.md
```

### ‍ Deifnvolvedor Senior
```
README.md → docs/architecture.md → DOCKER_LOCAL_SETUP.md → docs/API.md
```

### DevOps/SRE
```
DOCKER_IMPLEMENTATION_SUMMARY.md → DOCKER_LOCAL_SETUP.md → docs/DEPLOYMENT.md
```

### Pesquisador/Acadêmico
```
README.md → docs/explanation.md → docs/PHASE4_COMPLETE.md → docs/phaif5_summary.md
```

### Product Manager/Business
```
README.md (ifções: Overview, Caso of Uso, Results)
```

---

## Estatísticas from the Documentação

- **Total of arquivos:** 20+
- **Linhas of documentação:** ~4,500+
- **Guias of setup:** 4
- **Relatórios técnicos:** 6
- **Arquivos of configuration:** 10+
- **Idioma:** Português (docs) + English (code)

---

## Busca Rápida

### Preciso install Docker
 [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)

### Preciso run o projeto now
 [QUICKSTART.md](../QUICKSTART.md)

### Preciso entender SNNs
 [docs/explanation.md](docs/explanation.md)

### Preciso integrar to API
 [docs/API.md](docs/API.md)

### Preciso do deploy
 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

### Preciso configure Docker in detalhes
 [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)

### Preciso entender to arquitetura
 [docs/architecture.md](docs/architecture.md)

### Preciso ver os resultados
 [README.md](../README.md) → Seção "Results"

### Preciso trorbleshooting
 [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) → Seção "Solução of Problems"

---

## Suforte

### Issues GitHub
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Contact
- **GitHub:** https://github.com/maurorisonho
- **LinkedIn:** https://linkedin.com/in/maurorisonho

---

## License

MIT License - Todos os documentos and code

---

**Last updated:** 5 of Dezembro of 2025 
**Verare of the Project:** 1.0.0 (Phaif 5 Complete) 
**Status:** Produção Ready
