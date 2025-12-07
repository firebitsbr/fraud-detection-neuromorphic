# Neuromorphic Fraud Detection - Makefile
#
# Description: Comandos simplificados para gerenciar o projeto
#
# Author: Mauro Risonho de Paula Assumpção
# Date: December 5, 2025
# License: MIT License

.PHONY: help start stop restart logs build clean test status health

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

# Detect Docker Compose command (v2 uses 'docker compose', v1 uses '$(COMPOSE_CMD)')
COMPOSE_CMD := $(shell docker compose version > /dev/null 2>&1 && echo 'docker compose' || echo '$(COMPOSE_CMD)')

help: ## Mostra esta mensagem de ajuda
	@echo "$(BLUE)╔═══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Neuromorphic Fraud Detection - Comandos Make        ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

start: ## Inicia todos os serviços Docker
	@echo "$(GREEN)→ Iniciando serviços...$(NC)"
	@$(COMPOSE_CMD) up -d
	@echo "$(GREEN)✓ Serviços iniciados!$(NC)"
	@make status

stop: ## Para todos os serviços Docker
	@echo "$(YELLOW)→ Parando serviços...$(NC)"
	@$(COMPOSE_CMD) down
	@echo "$(GREEN)✓ Serviços parados!$(NC)"

restart: ## Reinicia todos os serviços
	@make stop
	@make start

logs: ## Mostra logs em tempo real
	@$(COMPOSE_CMD) logs -f

build: ## Reconstrói todas as imagens Docker
	@echo "$(YELLOW)→ Reconstruindo imagens...$(NC)"
	@$(COMPOSE_CMD) build --no-cache
	@echo "$(GREEN)✓ Imagens reconstruídas!$(NC)"

build-start: build start ## Reconstrói e inicia

clean: ## Para serviços e remove volumes
	@echo "$(YELLOW)→ Limpando sistema...$(NC)"
	@$(COMPOSE_CMD) down -v
	@echo "$(GREEN)✓ Sistema limpo!$(NC)"

clean-all: clean ## Limpeza completa (incluindo imagens)
	@echo "$(YELLOW)→ Removendo imagens...$(NC)"
	@docker rmi $$(docker images -q fraud-detection* 2>/dev/null) 2>/dev/null || true
	@echo "$(GREEN)✓ Limpeza completa!$(NC)"

status: ## Mostra status dos containers
	@echo ""
	@echo "$(BLUE)Status dos Serviços:$(NC)"
	@$(COMPOSE_CMD) ps
	@echo ""

health: ## Verifica saúde dos serviços
	@echo "$(BLUE)→ Verificando saúde dos serviços...$(NC)"
	@echo ""
	@echo "$(GREEN)API Principal:$(NC)"
	@curl -s http://localhost:8000/health || echo "$(YELLOW)⚠ Não disponível$(NC)"
	@echo ""
	@echo "$(GREEN)Loihi Simulator:$(NC)"
	@curl -s http://localhost:8001/health || echo "$(YELLOW)⚠ Não disponível$(NC)"
	@echo ""
	@echo "$(GREEN)BrainScaleS:$(NC)"
	@curl -s http://localhost:8002/health || echo "$(YELLOW)⚠ Não disponível$(NC)"
	@echo ""
	@echo "$(GREEN)Cluster Controller:$(NC)"
	@curl -s http://localhost:8003/health || echo "$(YELLOW)⚠ Não disponível$(NC)"
	@echo ""

test: ## Executa testes
	@echo "$(BLUE)→ Executando testes...$(NC)"
	@$(COMPOSE_CMD) exec fraud_api pytest tests/ -v
	@echo "$(GREEN)✓ Testes completos!$(NC)"

test-coverage: ## Executa testes com cobertura
	@echo "$(BLUE)→ Executando testes com cobertura...$(NC)"
	@$(COMPOSE_CMD) exec fraud_api pytest tests/ --cov=src --cov=api --cov-report=html
	@echo "$(GREEN)✓ Relatório gerado em htmlcov/index.html$(NC)"

shell-api: ## Abre shell no container da API
	@$(COMPOSE_CMD) exec fraud_api bash

shell-loihi: ## Abre shell no Loihi simulator
	@$(COMPOSE_CMD) exec loihi_simulator bash

shell-cluster: ## Abre shell no cluster controller
	@$(COMPOSE_CMD) exec cluster_controller bash

logs-api: ## Mostra logs da API
	@$(COMPOSE_CMD) logs -f fraud_api

logs-loihi: ## Mostra logs do Loihi
	@$(COMPOSE_CMD) logs -f loihi_simulator

logs-cluster: ## Mostra logs do cluster
	@$(COMPOSE_CMD) logs -f cluster_controller

urls: ## Mostra URLs dos serviços
	@echo ""
	@echo "$(BLUE)╔═══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  URLs dos Serviços                                    ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "  $(GREEN)API Principal:$(NC)        http://localhost:8000"
	@echo "  $(GREEN)JupyterLab:$(NC)           http://localhost:8888"
	@echo "  $(GREEN)Loihi Simulator:$(NC)      http://localhost:8001"
	@echo "  $(GREEN)BrainScaleS:$(NC)          http://localhost:8002"
	@echo "  $(GREEN)Cluster Controller:$(NC)   http://localhost:8003"
	@echo "  $(GREEN)Grafana:$(NC)              http://localhost:3000 (admin/admin)"
	@echo "  $(GREEN)Prometheus:$(NC)           http://localhost:9090"
	@echo "  $(GREEN)Redis:$(NC)                localhost:6379"
	@echo ""

dev: ## Modo desenvolvimento (com reload automático)
	@echo "$(BLUE)→ Iniciando em modo desenvolvimento...$(NC)"
	@$(COMPOSE_CMD) up

load-test: ## Executa teste de carga
	@echo "$(BLUE)→ Executando teste de carga...$(NC)"
	@python examples/load_test.py

benchmark: ## Executa benchmarks de performance
	@echo "$(BLUE)→ Executando benchmarks...$(NC)"
	@$(COMPOSE_CMD) exec fraud_api python tests/test_scaling.py

monitor: ## Abre Grafana para monitoramento
	@echo "$(GREEN)→ Abrindo Grafana...$(NC)"
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Acesse: http://localhost:3000"

install-deps: ## Instala dependências locais (sem Docker)
	@echo "$(BLUE)→ Instalando dependências...$(NC)"
	@pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependências instaladas!$(NC)"

format: ## Formata código com Black
	@echo "$(BLUE)→ Formatando código...$(NC)"
	@black src/ api/ tests/
	@echo "$(GREEN)✓ Código formatado!$(NC)"

lint: ## Verifica qualidade do código
	@echo "$(BLUE)→ Verificando código...$(NC)"
	@flake8 src/ api/ tests/
	@echo "$(GREEN)✓ Verificação completa!$(NC)"

backup: ## Backup dos volumes Docker
	@echo "$(YELLOW)→ Fazendo backup...$(NC)"
	@mkdir -p backups
	@docker run --rm -v fraud_redis_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/redis_backup_$$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
	@echo "$(GREEN)✓ Backup completo!$(NC)"

.DEFAULT_GOAL := help
