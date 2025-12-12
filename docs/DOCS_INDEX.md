# Índice de Documentação - Fraud Detection Neuromorphic

**Descrição:** Índice de documentação do projeto.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
**Licença:** MIT License

---

## Documentação Principal

### [README.md](../README.md)
**Visão geral completa do projeto**
- Descrição do sistema
- Arquitetura neuromórfica
- Tecnologias utilizadas
- Instruções de instalação
- Exemplos de uso
- Resultados e métricas
- 473 linhas

---

## Docker - Execução Local

### [QUICKSTART.md](QUICKSTART.md) COMECE AQUI!
**Guia de início rápido - 3 comandos para rodar tudo**
- Execução em 3 passos
- Comandos principais (Make)
- Troubleshooting rápido
- Testes de API
- Diagrama de arquitetura
- **Ideal para:** Começar rapidamente

### [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)
**Instalação do Docker em todos os sistemas operacionais**
- Fedora/RHEL/CentOS
- Ubuntu/Debian
- Arch Linux
- macOS (Docker Desktop + Homebrew)
- Windows 10/11 (WSL2)
- Configuração pós-instalação
- Troubleshooting
- **Use quando:** Docker não está instalado

### [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
**Guia completo de execução local com Docker**
- Pré-requisitos detalhados
- Arquitetura dos containers
- Todos os comandos disponíveis
- Testes de sistema
- Monitoramento (Grafana/Prometheus)
- Workflows de desenvolvimento
- Solução de problemas
- Configuração avançada
- 620 linhas de documentação
- **Ideal para:** Desenvolvedores e operações

### [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md)
**Resumo técnico da implementação Docker**
- Arquivos criados e propósito
- Especificações de serviços
- Referência de comandos
- Benefícios para dev/test/prod
- Próximos passos
- **Ideal para:** Revisão técnica

---

## Documentação Técnica

### [explanation.md](explanation.md)
**Explicação detalhada do sistema**
- Fundamentos de SNNs
- Arquitetura do pipeline
- Algoritmos implementados
- Comparação com DNNs
- Casos de uso

### [architecture.md](architecture.md)
**Arquitetura técnica completa**
- Fluxo de dados
- Componentes do sistema
- Especificações técnicas
- Diagramas detalhados

### [API.md](API.md)
**Documentação da API REST**
- Endpoints disponíveis
- Formato de requisições
- Respostas e códigos
- Exemplos de uso
- Autenticação

### [DEPLOYMENT.md](DEPLOYMENT.md)
**Guia de deployment em produção**
- Estratégias de deployment
- Kubernetes/Cloud
- Configurações de produção
- Monitoramento
- Segurança

---

## Relatórios de Fases

### [PHASE3_COMPLETE.md](PHASE3_COMPLETE.md)
**Fase 3: Production Infrastructure & Deployment**
- Kafka streaming
- API REST completa
- Docker production
- CI/CD pipeline
- Monitoramento Prometheus/Grafana
- 459 linhas

### [PHASE4_COMPLETE.md](PHASE4_COMPLETE.md)
**Fase 4: Hardware Integration & Energy Optimization**
- Integração Intel Loihi 2
- Benchmark de energia
- Deployment em hardware
- Comparação hardware vs software
- Otimizações
- 561 linhas

### [phase5_summary.md](phase5_summary.md)
**Fase 5: Distributed Scaling & Multi-Chip Infrastructure**
- Simuladores de hardware (Loihi 2, BrainScaleS-2)
- Cluster distribuído
- Docker multi-chip
- Testes de escalabilidade
- 673 linhas

### Summaries de Outras Fases
- [phase2_summary.md](phase2_summary.md) - Optimization & Performance
- [phase3_summary.md](phase3_summary.md) - Production Infrastructure
- [phase4_summary.md](phase4_summary.md) - Hardware Integration

---

## Arquivos de Configuração

### Docker
- `docker-compose.yml` - Orquestração de 7 serviços
- `docker/Dockerfile` - Imagem principal
- `docker/Dockerfile.loihi` - Simulador Loihi 2
- `docker/Dockerfile.brainscales` - Emulador BrainScaleS-2
- `docker/Dockerfile.cluster` - Cluster controller
- `docker/Dockerfile.edge` - Edge devices
- `docker/Dockerfile.production` - Produção
- `docker/docker-compose.production.yml` - Stack produção
- `docker/docker-compose.phase5.yml` - Multi-chip
- `.dockerignore` - Otimização de builds

### Python
- `docker/requirements.txt` - Dependências completas
- `requirements-ci.txt` - Dependências CI/CD (lightweight)
- `requirements-edge.txt` - Edge devices

### CI/CD
- `.github/workflows/ci-cd.yml` - GitHub Actions pipeline

### Build & Scripts
- `Makefile` - 25+ comandos úteis
- `scripts/start-local.sh` - Script de inicialização automatizada

---

## Como Navegar

### Para Começar Rapidamente
1. [QUICKSTART.md](../QUICKSTART.md) - 3 comandos
2. [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md) - Se precisar instalar Docker

### Para Desenvolvedores
1. [README.md](../README.md) - Entender o projeto
2. [docs/explanation.md](docs/explanation.md) - Fundamentos
3. [docs/architecture.md](docs/architecture.md) - Arquitetura
4. [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) - Setup completo
5. [docs/API.md](docs/API.md) - API Reference

### Para Deployment
1. [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) - Ambiente local
2. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Produção
3. [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md) - Resumo técnico

### Para Pesquisa Acadêmica
1. [README.md](../README.md) - Overview
2. [docs/explanation.md](docs/explanation.md) - Teoria
3. [docs/PHASE3_COMPLETE.md](docs/PHASE3_COMPLETE.md) - Infrastructure
4. [docs/PHASE4_COMPLETE.md](docs/PHASE4_COMPLETE.md) - Hardware
5. [docs/phase5_summary.md](docs/phase5_summary.md) - Scaling

---

## Mapa de Uso por Perfil

### ‍ Desenvolvedor Junior
```
QUICKSTART.md → DOCKER_INSTALL_GUIDE.md → README.md → docs/API.md
```

### ‍ Desenvolvedor Senior
```
README.md → docs/architecture.md → DOCKER_LOCAL_SETUP.md → docs/API.md
```

### DevOps/SRE
```
DOCKER_IMPLEMENTATION_SUMMARY.md → DOCKER_LOCAL_SETUP.md → docs/DEPLOYMENT.md
```

### Pesquisador/Acadêmico
```
README.md → docs/explanation.md → docs/PHASE4_COMPLETE.md → docs/phase5_summary.md
```

### Product Manager/Business
```
README.md (seções: Visão Geral, Caso de Uso, Resultados)
```

---

## Estatísticas da Documentação

- **Total de arquivos:** 20+
- **Linhas de documentação:** ~4,500+
- **Guias de setup:** 4
- **Relatórios técnicos:** 6
- **Arquivos de configuração:** 10+
- **Idioma:** Português (docs) + English (code)

---

## Busca Rápida

### Preciso instalar Docker
 [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)

### Preciso rodar o projeto agora
 [QUICKSTART.md](../QUICKSTART.md)

### Preciso entender SNNs
 [docs/explanation.md](docs/explanation.md)

### Preciso integrar a API
 [docs/API.md](docs/API.md)

### Preciso fazer deploy
 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

### Preciso configurar Docker em detalhes
 [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)

### Preciso entender a arquitetura
 [docs/architecture.md](docs/architecture.md)

### Preciso ver os resultados
 [README.md](../README.md) → Seção "Resultados"

### Preciso troubleshooting
 [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md) → Seção "Solução de Problemas"

---

## Suporte

### Issues GitHub
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Contato
- **GitHub:** https://github.com/maurorisonho
- **LinkedIn:** https://linkedin.com/in/maurorisonho

---

## Licença

MIT License - Todos os documentos e código

---

**Última atualização:** 5 de Dezembro de 2025 
**Versão do Projeto:** 1.0.0 (Phase 5 Complete) 
**Status:** Produção Ready
