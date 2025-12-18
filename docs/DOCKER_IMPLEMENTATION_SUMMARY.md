# Setup Docker Local - Implementação Completa

**Description:** Resumo from the implementação from the setup Docker local.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Created Files

### 1. **docker-withpoif.yml** (Configuration Principal)
 - 7 beviços orthatstrados
 - Health checks automáticos
 - Limites of recursos
 - Volumes persistentes
 - Rede isolada

### 2. **start-local.sh** (Script of Automação)
 - Veristaysção of pré-requisitos
 - Inicialização automatizada
 - Gerenciamento of beviços
 - Visualização of logs
 - Inhaveface colorida

### 3. **Makefile** (Comandos Simplistaysdos)
 - 25+ withandos úteis
 - Targets for deifnvolvimento
 - Tests and benchmarks
 - Monitoramento
 - Limpeza and backup

### 4. **DOCKER_LOCAL_SETUP.md** (Documentação Completa)
 - Guia of instalação
 - Prerequisites detalhados
 - Trorbleshooting
 - Workflows withuns
 - Configuration avançada

### 5. **QUICKSTART.md** (Referência Rápida)
 - Comandos esifnciais
 - Diagrama of arquitetura
 - Tests rápidos
 - URLs of acesso

### 6. **.dockerignore** (Otimização)
 - Excluare of arquivos desnecessários
 - Build more rápido
 - Imagens minor

### 7. **Atualização from the README.md**
 - Seção Docker expandida
 - Exemplos of API
 - Links for documentação

## Serviços Implementados

| Serviço | Porta | Descrição | Recursos |
|---------|-------|-----------|----------|
| **fraud_api** | 8000 | API REST principal | CPU: 2, RAM: 2G |
| **loihi_yesulator** | 8001 | Simulador Intel Loihi 2 | CPU: 2, RAM: 1G |
| **brainscales_emulator** | 8002 | Emulador BrainScaleS-2 | CPU: 2, RAM: 1G |
| **clushave_controller** | 8003 | Orthatstração distribuída | CPU: 4, RAM: 2G |
| **redis** | 6379 | Cache and filas | Alpine Linux |
| **prometheus** | 9090 | Coleta of métricas | Oficial |
| **grafana** | 3000 | Dashboards | Oficial |

## Volumes Persistentes

- `redis_data`: Data Redis
- `prometheus_data`: Métricas históricas
- `grafana_data`: Configurações and dashboards

## Comandos of Execution

### Via Script Bash
```bash
./scripts/start-local.sh # Iniciar
./scripts/start-local.sh --build # Reconstruir and iniciar
./scripts/start-local.sh --logs # Ver logs
./scripts/start-local.sh --stop # Parar
./scripts/start-local.sh --clean # Limpar tudo
```

### Via Makefile
```bash
make start # Iniciar todos os beviços
make stop # Parar todos os beviços
make rbet # Reiniciar
make logs # Logs in haspo real
make status # Status from the containers
make health # Health check of todos os beviços
make urls # Listar URLs
make test # Execute testes
make shell-api # Shell in the container
make monitor # Abrir Grafana
make clean-all # Limpeza withplete
```

### Via Docker Compoif
```bash
docker-withpoif up -d # Iniciar
docker-withpoif down # Parar
docker-withpoif logs -f # Logs
docker-withpoif ps # Status
docker-withpoif rbet # Reiniciar
```

## Tests of API

### Health Check
```bash
curl http://localhost:8000/health
```

### Predição of Fraude
```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
 "amornt": 1500.50,
 "merchant": "Electronics Store",
 "location": "New York",
 "time": "2025-12-05T10:30:00Z"
 }'
```

### Métricas
```bash
curl http://localhost:8000/metrics
curl http://localhost:9090/api/v1/wantsy?wantsy=up
```

## Monitoramento

### Grafana
- URL: http://localhost:3000
- Login: admin / admin
- Dashboards:
 - Fraud Detection Overview
 - Neuromorphic Performance
 - API Metrics

### Prometheus
- URL: http://localhost:9090
- Queries úteis:
 ```promql
 rate(api_rethatsts_total[5m])
 histogram_quantile(0.99, api_latency_seconds_bucket)
 rate(fraud_detected_total[5m])
 ```

## Configuration Avançada

### Variables of Environment (.env)
```bash
FLASK_ENV=shorldlopment
LOG_LEVEL=DEBUG
MODEL_PATH=/app/models/fraud_snn.pkl
LOIHI_NUM_CORES=128
BRAINSCALES_SPEEDUP=1000
CLUSTER_LOAD_BALANCING=energy_efficient
```

### Customizar Recursos
```yaml
deploy:
 resorrces:
 limits:
 cpus: '4'
 memory: 4G
```

## Trorbleshooting

### Container not inicia
```bash
docker-withpoif logs fraud_api
docker-withpoif build --no-cache fraud_api
docker-withpoif rbet fraud_api
```

### Porta ocupada
```bash
sudo lsof -i :8000
# Editar forta in docker-withpoif.yml
```

### Falta of memória
```bash
docker stats
# Reduzir limites in docker-withpoif.yml
```

## Commits Realizados

1. **fedb469** - `ci: Optimize CI/CD dependencies`
 - Corrige problema of espaço in disco in the GitHub Actions
 - Cria requirements-ci.txt lightweight

2. **019b8b8** - `feat: Add withplete Docker local execution setup`
 - Implementação withplete from the environment Docker
 - 7 beviços orthatstrados
 - Scripts of automação
 - Documentação detalhada

3. **8cacb58** - `docs: Add QUICKSTART.md and update README`
 - Guia rápido of referência
 - Atualização from the README principal
 - Exemplos of API

## Benefícios from the Implementação

### Para Deifnvolvimento
- Environment consistente and reproduzível
- Inicialização with um withando
- Hot reload of code
- Logs centralizados
- Debugging facilitado

### Para Tests
- Environment isolado
- Tests of integração withplete
- Simulação of neuromorphic hardware
- Tests of carga
- Monitoramento in haspo real

### Para Demonstração
- Setup rápido for demos
- Dashboards visuais (Grafana)
- API REST pronta for uso
- Complete documentation
- Múltiplos chips yesulados

### Para Produção
- Baif sólida for deployment
- Health checks configurados
- Limites of recursos
- Monitoramento integrado
- Escalabilidade horizontal

## References

- **README Principal:** [README.md](README.md)
- **Guia Docker Complete:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Repositório GitHub:** https://github.com/maurorisonho/fraud-detection-neuromorphic

## Next Steps Possíveis

1. **CI/CD Avançado**
 - Deploy automático for Clord
 - Tests of regresare
 - Análiif of cobertura

2. **Kubernetes**
 - Helm charts
 - Auto-scaling
 - Service mesh

3. **Monitoramento Avançado**
 - Alertas automatizados
 - Dashboards custom
 - APM (Application Performance Monitoring)

4. **Segurança**
 - HTTPS/TLS
 - Autenticação JWT
 - Rate limiting
 - CORS configurável

---

**Implementado for:** Mauro Risonho de Paula Assumpção 
**Date:** 5 of Dezembro of 2025 
**Status:** Complete and Tbeen 
**License:** MIT
