#!/usr/bin/env bash
#
# **Descrição:** Monitoring Script - Real-time service health and metrics.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

set -euo pipefail

PROJECT_NAME="neuromorphic-fraud-detection"
COMPOSE_FILE="docker-compose.production.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear_screen() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Neuromorphic Fraud Detection - Real-time Monitoring         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

get_container_status() {
    local service=$1
    local container="${PROJECT_NAME}-${service}-1"
    
    if docker ps --filter "name=$container" --format "{{.Status}}" 2>/dev/null | grep -q "Up"; then
        echo -e "${GREEN}●${NC} Running"
    else
        echo -e "${RED}●${NC} Stopped"
    fi
}

get_container_health() {
    local service=$1
    local container="${PROJECT_NAME}-${service}-1"
    
    health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
    
    case $health in
        healthy)
            echo -e "${GREEN}✓ Healthy${NC}"
            ;;
        unhealthy)
            echo -e "${RED}✗ Unhealthy${NC}"
            ;;
        starting)
            echo -e "${YELLOW}⟳ Starting${NC}"
            ;;
        none)
            echo -e "${BLUE}− No check${NC}"
            ;;
        *)
            echo -e "${YELLOW}? Unknown${NC}"
            ;;
    esac
}

get_container_cpu() {
    local service=$1
    local container="${PROJECT_NAME}-${service}-1"
    
    docker stats --no-stream --format "{{.CPUPerc}}" "$container" 2>/dev/null || echo "N/A"
}

get_container_memory() {
    local service=$1
    local container="${PROJECT_NAME}-${service}-1"
    
    docker stats --no-stream --format "{{.MemUsage}}" "$container" 2>/dev/null || echo "N/A"
}

get_api_requests() {
    curl -s http://localhost:8000/health 2>/dev/null | grep -o '"requests_count":[0-9]*' | cut -d: -f2 || echo "0"
}

get_redis_keys() {
    docker exec -it "${PROJECT_NAME}-redis-1" redis-cli DBSIZE 2>/dev/null | grep -o '[0-9]*' || echo "0"
}

show_services() {
    echo -e "${BLUE}════════════════════════ Services Status ════════════════════════${NC}"
    echo ""
    
    services=("fraud-api" "jupyter-lab" "web-interface" "redis" "prometheus" "grafana")
    
    printf "%-20s %-15s %-15s %-12s %-15s\n" "SERVICE" "STATUS" "HEALTH" "CPU" "MEMORY"
    echo "────────────────────────────────────────────────────────────────────────"
    
    for service in "${services[@]}"; do
        status=$(get_container_status "$service")
        health=$(get_container_health "$service")
        cpu=$(get_container_cpu "$service")
        memory=$(get_container_memory "$service")
        
        printf "%-20s %-25s %-25s %-12s %-15s\n" "$service" "$status" "$health" "$cpu" "$memory"
    done
    
    echo ""
}

show_metrics() {
    echo -e "${BLUE}═══════════════════════ System Metrics ══════════════════════════${NC}"
    echo ""
    
    # API metrics
    api_requests=$(get_api_requests)
    echo -e "  ${CYAN}API Requests:${NC}      $api_requests"
    
    # Redis metrics
    redis_keys=$(get_redis_keys)
    echo -e "  ${CYAN}Redis Keys:${NC}        $redis_keys"
    
    # Disk usage
    disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    echo -e "  ${CYAN}Disk Usage:${NC}        $disk_usage"
    
    # Memory usage
    mem_usage=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
    echo -e "  ${CYAN}System Memory:${NC}     $mem_usage"
    
    echo ""
}

show_endpoints() {
    echo -e "${BLUE}═══════════════════════ Service Endpoints ═══════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}●${NC} API:              http://localhost:8000"
    echo -e "  ${GREEN}●${NC} API Docs:         http://localhost:8000/docs"
    echo -e "  ${GREEN}●${NC} Jupyter Lab:      http://localhost:8888"
    echo -e "  ${GREEN}●${NC} Streamlit:        http://localhost:8501"
    echo -e "  ${GREEN}●${NC} Grafana:          http://localhost:3000 (admin/neuromorphic2025)"
    echo -e "  ${GREEN}●${NC} Prometheus:       http://localhost:9090"
    echo ""
}

show_recent_logs() {
    echo -e "${BLUE}══════════════════════ Recent API Logs ══════════════════════════${NC}"
    echo ""
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs --tail=5 fraud-api 2>/dev/null | sed 's/^/  /'
    echo ""
}

monitor_loop() {
    while true; do
        clear_screen
        show_services
        show_metrics
        show_endpoints
        show_recent_logs
        
        echo -e "${YELLOW}Press Ctrl+C to exit | Auto-refresh every 5 seconds${NC}"
        sleep 5
    done
}

monitor_once() {
    clear_screen
    show_services
    show_metrics
    show_endpoints
    show_recent_logs
}

case "${1:-loop}" in
    loop)
        monitor_loop
        ;;
    once)
        monitor_once
        ;;
    services)
        clear_screen
        show_services
        ;;
    metrics)
        clear_screen
        show_metrics
        ;;
    endpoints)
        clear_screen
        show_endpoints
        ;;
    *)
        echo "Usage: $0 {loop|once|services|metrics|endpoints}"
        exit 1
        ;;
esac
