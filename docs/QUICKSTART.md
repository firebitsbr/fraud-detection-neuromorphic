# Quick Start - Docker Local

**Descrição:** Guia rápido de execução local com Docker.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025

**Execução local em 3 comandos:**

```bash
# 1. Clone o repositório
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic

# 2. Inicie o sistema
./scripts/start-local.sh
# ou
make start

# 3. Acesse os serviços
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

## Requisitos

### Docker Não Instalado?

**Fedora/RHEL:**
```bash
sudo ../scripts/install-docker-fedora.sh
newgrp docker
```

**Outros sistemas:** Ver [DOCKER_INSTALL_GUIDE.md](DOCKER_INSTALL_GUIDE.md)

### Requisitos Mínimos
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM, 10GB disco

## Comandos Principais

| Comando | Descrição |
|---------|-----------|
| `make start` | Inicia todos os serviços |
| `make stop` | Para todos os serviços |
| `make logs` | Visualiza logs em tempo real |
| `make status` | Status dos containers |
| `make health` | Verifica saúde dos serviços |
| `make urls` | Lista URLs de acesso |

## Comandos Avançados

```bash
# Reconstruir imagens
make build

# Reiniciar serviços
make restart

# Limpeza completa
make clean-all

# Executar testes
make test

# Shell no container
make shell-api

# Monitoramento
make monitor
```

## Troubleshooting Rápido

### Container não inicia
```bash
make logs-api
make build
make restart
```

### Porta ocupada
```bash
sudo lsof -i :8000
# Edite porta em docker-compose.yml
```

### Falta de memória
```bash
docker stats
# Reduza resources em docker-compose.yml
```

## Testar API

```bash
# Health check
curl http://localhost:8000/health

# Predição de fraude
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
 "amount": 1500.50,
 "merchant": "Electronics Store",
 "location": "New York"
 }'
```

## Monitoramento

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **API Metrics:** http://localhost:8000/metrics

## Documentação Completa

 [DOCKER_LOCAL_SETUP.md](docs/DOCKER_LOCAL_SETUP.md) - Guia completo

## Arquitetura

```

 fraud_api (8000) 
 Main REST API 

 
 
 
 
 
 Loihi 2 BrainScale Cluster 
 (8001) (8002) (8003) 
 
 
 
 
 
 Redis Prometheus
 (6379) (9090) 
 
 
 
 
 Grafana 
 (3000) 
 
```

**Autor:** Mauro Risonho de Paula Assumpção 
**Licença:** MIT 
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic
