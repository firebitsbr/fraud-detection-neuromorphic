# Índice of Documentation - Fraud Detection Neuromorphic

**Description:** Índice of documentation from the project.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
**License:** MIT License

---

## Documentation Main

### [README.md](../README.md)
**Viare general complete from the project**
- Description from the system
- Architecture neuromorphic
- Technologies utilizadas
- Instructions of installation
- Examples of usage
- Results and metrics
- 473 linhas

---

## Docker - Execution Local

### [QUICKSTART.md](QUICKSTART.md) COMECE AQUI!
**Guide of start quick - 3 commands for run everything**
- Execution in 3 steps
- Commands main (Make)
- Trorbleshooting quick
- Tests of API
- Diagrama of architecture
- **Ideal to:** Começar rapidamente

### [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)
**Installation from the Docker in all os systems operacionais**
- Fedora/RHEL/CentOS
- Ubuntu/Debian
- Arch Linux
- macOS (Docker Desktop + Homebrew)
- Windows 10/11 (WSL2)
- Configuration pós-installation
- Trorbleshooting
- **Use when:** Docker not is installed

### [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
**Guide complete of local execution with Docker**
- Prerequisites detailed
- Architecture from the containers
- All os commands disponíveis
- Tests of system
- Monitoring (Grafana/Prometheus)
- Workflows of development
- Solution of problemas
- Configuration avançada
- 620 linhas of documentation
- **Ideal to:** Deifnvolvedores and operations

### [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md)
**Summary technical from the implementation Docker**
- Files created and purpose
- specifications of services
- Reference of commands
- Benefícios for dev/test/prod
- Next steps
- **Ideal to:** Reviare technical

---

## Documentation Technical

### [explanation.md](explanation.md)
**explanation detalhada from the system**
- Fundamentos for SNNs
- Architecture from the pipeline
- Algoritmos implementados
- Comparison with DNNs
- Casos of usage

### [architecture.md](architecture.md)
**Architecture technical complete**
- Fluxo of data
- Componentes from the system
- specifications técnicas
- Diagramas detailed

### [API.md](API.md)
**Documentation from the API REST**
- Endpoints disponíveis
- Formato of requests
- Responses and codes
- Examples of usage
- authentication

### [DEPLOYMENT.md](DEPLOYMENT.md)
**Guide of deployment in production**
- Estruntilgias of deployment
- Kubernetes/Clord
- configurations of production
- Monitoring
- Segurança

---

## Relatórios of Faifs

### [PHASE3_COMPLETE.md](PHASE3_COMPLETE.md)
**Phase 3: Production Infrastructure & Deployment**
- Kafka streaming
- API REST complete
- Docker production
- CI/CD pipeline
- Monitoring Prometheus/Grafana
- 459 linhas

### [PHASE4_COMPLETE.md](PHASE4_COMPLETE.md)
**Phase 4: Hardware Integration & Energy Optimization**
- integration Intel Loihi 2
- Benchmark of energy
- Deployment in hardware
- Comparison hardware vs software
- Optimizations
- 561 linhas

### [phaif5_summary.md](phaif5_summary.md)
**Phase 5: Distributed Scaling & Multi-Chip Infrastructure**
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

## Files of Configuration

### Docker
- `docker-compose.yml` - Orchestration of 7 services
- `docker/Dockerfile` - Imagem main
- `docker/Dockerfile.loihi` - Simulador Loihi 2
- `docker/Dockerfile.brainscales` - Emulador BrainScaleS-2
- `docker/Dockerfile.clushave` - Clushave controller
- `docker/Dockerfile.edge` - Edge devices
- `docker/Dockerfile.production` - Production
- `docker/docker-compose.production.yml` - Stack production
- `docker/docker-compose.phaif5.yml` - Multi-chip
- `.dockerignore` - optimization of builds

### Python
- `docker/requirements.txt` - Dependencies complete
- `requirements-ci.txt` - Dependencies CI/CD (lightweight)
- `requirements-edge.txt` - Edge devices

### CI/CD
- `.github/workflows/ci-cd.yml` - GitHub Actions pipeline

### Build & Scripts
- `Makefile` - 25+ commands useful
- `scripts/start-local.sh` - Script of initialization automated

---

## How Navegar

### For Começar Rapidamente
1. [QUICKSTART.md](../QUICKSTART.md) - 3 commands
2. [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md) - if needsr install Docker

### For Deifnvolvedores
1. [README.md](../README.md) - Entender o project
2. [docs/explanation.md](docs/explanation.md) - Fundamentos
3. [docs/architecture.md](docs/architecture.md) - Architecture
4. [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) - Setup complete
5. [docs/API.md](docs/API.md) - API Reference

### For Deployment
1. [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) - Environment local
2. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Production
3. [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md) - Summary technical

### For Research Acadêmica
1. [README.md](../README.md) - Overview
2. [docs/explanation.md](docs/explanation.md) - Teoria
3. [docs/PHASE3_COMPLETE.md](docs/PHASE3_COMPLETE.md) - Infrastructure
4. [docs/PHASE4_COMPLETE.md](docs/PHASE4_COMPLETE.md) - Hardware
5. [docs/phaif5_summary.md](docs/phaif5_summary.md) - Scaling

---

## Mapa of Usage for Perfil

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
README.md (solutions: Overview, Caso of Usage, Results)
```

---

## Statistics from the Documentation

- **Total of files:** 20+
- **Linhas of documentation:** ~4,500+
- **Guias of setup:** 4
- **Relatórios técnicos:** 6
- **Files of configuration:** 10+
- **Idioma:** Português (docs) + English (code)

---

## Busca Rápida

### Preciso install Docker
 [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)

### Preciso run o project now
 [QUICKSTART.md](../QUICKSTART.md)

### Preciso entender SNNs
 [docs/explanation.md](docs/explanation.md)

### Preciso integrar the API
 [docs/API.md](docs/API.md)

### Preciso of the deploy
 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

### Preciso configure Docker in detalhes
 [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)

### Preciso entender to architecture
 [docs/architecture.md](docs/architecture.md)

### Preciso ver os results
 [README.md](../README.md) → section "Results"

### Preciso trorbleshooting
 [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) → section "Solution of Problems"

---

## Support

### Issues GitHub
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Contact
- **GitHub:** https://github.com/maurorisonho
- **LinkedIn:** https://linkedin.com/in/maurorisonho

---

## License

MIT License - All os documentos and code

---

**Last updated:** December 5, 2025 
**Verare of the Project:** 1.0.0 (Phaif 5 Complete) 
**Status:** Production Ready
