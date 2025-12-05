#!/bin/bash

# Neuromorphic Fraud Detection - Local Docker Startup Script
#
# Description: Script para inicializar o sistema completo localmente usando Docker.
#
# Author: Mauro Risonho de Paula Assumpção
# Date: December 5, 2025
# License: MIT License
#
# Usage:
#   ./start-local.sh          # Start all services
#   ./start-local.sh --build  # Rebuild and start
#   ./start-local.sh --stop   # Stop all services
#   ./start-local.sh --logs   # View logs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Neuromorphic Fraud Detection - Local Docker Environment  ║${NC}"
    echo -e "${BLUE}║  Author: Mauro Risonho de Paula Assumpção                  ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker não está instalado!"
        echo "Instale o Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker encontrado: $(docker --version)"
}

check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose não está instalado!"
        echo "Instale o Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose encontrado"
}

create_directories() {
    print_info "Criando diretórios necessários..."
    mkdir -p models data logs
    print_success "Diretórios criados"
}

start_services() {
    print_info "Iniciando serviços Docker..."
    
    if [ "$1" == "--build" ]; then
        print_info "Reconstruindo imagens..."
        docker-compose build --no-cache
    fi
    
    docker-compose up -d
    
    print_success "Serviços iniciados!"
    echo ""
    print_info "Aguardando serviços ficarem prontos (30s)..."
    sleep 30
    
    echo ""
    print_header
    print_success "Sistema iniciado com sucesso!"
    echo ""
    echo -e "${GREEN}Serviços disponíveis:${NC}"
    echo -e "  ${BLUE}API Principal:${NC}        http://localhost:8000"
    echo -e "  ${BLUE}JupyterLab:${NC}           http://localhost:8888"
    echo -e "  ${BLUE}Loihi Simulator:${NC}      http://localhost:8001"
    echo -e "  ${BLUE}BrainScaleS:${NC}          http://localhost:8002"
    echo -e "  ${BLUE}Cluster Controller:${NC}   http://localhost:8003"
    echo -e "  ${BLUE}Grafana:${NC}              http://localhost:3000 (admin/admin)"
    echo -e "  ${BLUE}Prometheus:${NC}           http://localhost:9090"
    echo -e "  ${BLUE}Redis:${NC}                localhost:6379"
    echo ""
    echo -e "${YELLOW}Comandos úteis:${NC}"
    echo -e "  ${BLUE}Ver logs:${NC}             docker-compose logs -f"
    echo -e "  ${BLUE}Ver status:${NC}           docker-compose ps"
    echo -e "  ${BLUE}Parar serviços:${NC}       docker-compose down"
    echo -e "  ${BLUE}Reiniciar:${NC}            docker-compose restart"
    echo ""
}

stop_services() {
    print_info "Parando serviços..."
    docker-compose down
    print_success "Serviços parados!"
}

stop_services_and_clean() {
    print_warning "Parando serviços e removendo volumes..."
    docker-compose down -v
    print_success "Serviços parados e volumes removidos!"
}

view_logs() {
    print_info "Visualizando logs (Ctrl+C para sair)..."
    docker-compose logs -f
}

show_status() {
    print_info "Status dos serviços:"
    docker-compose ps
}

show_usage() {
    echo "Uso: $0 [OPÇÃO]"
    echo ""
    echo "Opções:"
    echo "  (nenhuma)       Inicia todos os serviços"
    echo "  --build         Reconstrói as imagens e inicia"
    echo "  --stop          Para todos os serviços"
    echo "  --clean         Para serviços e remove volumes"
    echo "  --logs          Visualiza logs em tempo real"
    echo "  --status        Mostra status dos serviços"
    echo "  --help          Mostra esta mensagem"
    echo ""
    echo "Exemplos:"
    echo "  $0              # Inicia o sistema"
    echo "  $0 --build      # Reconstrói e inicia"
    echo "  $0 --logs       # Visualiza logs"
    echo "  $0 --stop       # Para o sistema"
    echo ""
}

# Main
print_header

case "${1:-start}" in
    start)
        check_docker
        check_docker_compose
        create_directories
        start_services
        ;;
    --build)
        check_docker
        check_docker_compose
        create_directories
        start_services "--build"
        ;;
    --stop)
        stop_services
        ;;
    --clean)
        stop_services_and_clean
        ;;
    --logs)
        view_logs
        ;;
    --status)
        show_status
        ;;
    --help)
        show_usage
        ;;
    *)
        print_error "Opção inválida: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
