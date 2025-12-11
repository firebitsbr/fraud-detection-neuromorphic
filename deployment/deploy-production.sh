#!/usr/bin/env bash
#
# **Descrição:** Production Deployment Script - Ubuntu 24.04 LTS. Automated deployment of Neuromorphic Fraud Detection System.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#
# License: MIT

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="config/docker-compose.production.yml"
PROJECT_NAME="neuromorphic-fraud-detection"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deployment.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "$1"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    log "✓ Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker compose &> /dev/null; then
        error "Docker Compose is not installed."
    fi
    log "✓ Docker Compose found: $(docker compose version)"
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. Consider using a non-root user with docker group."
    fi
    
    # Check disk space
    available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if (( available_space < 20 )); then
        warn "Low disk space: ${available_space}GB available. At least 20GB recommended."
    else
        log "✓ Disk space: ${available_space}GB available"
    fi
    
    # Check memory
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if (( total_mem < 8 )); then
        warn "Low memory: ${total_mem}GB. At least 8GB recommended for optimal performance."
    else
        log "✓ Memory: ${total_mem}GB available"
    fi
}

create_directories() {
    print_header "Creating Directories"
    
    directories=(
        "logs"
        "data/kaggle"
        "models"
        "backups"
        "notebooks"
        "web"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "✓ Created directory: $dir"
        fi
    done
}

setup_environment() {
    print_header "Setting Up Environment"
    
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Neuromorphic Fraud Detection - Environment Variables

# Jupyter
JUPYTER_TOKEN=neuromorphic2025

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=neuromorphic2025

# API
API_WORKERS=4
API_MAX_REQUESTS=1000

# Logging
LOG_LEVEL=INFO

# Redis
REDIS_MAX_MEMORY=512mb
EOF
        log "✓ Created .env file"
    else
        log "✓ .env file already exists"
    fi
}

build_images() {
    print_header "Building Docker Images"
    
    log "Building base image..."
    docker build -t fraud-detection-base:ubuntu24.04 -f Dockerfile --target builder . || error "Failed to build base image"
    
    log "Building API image..."
    docker build -t fraud-detection-api:ubuntu24.04 -f Dockerfile . || error "Failed to build API image"
    
    log "✓ All images built successfully"
}

pull_images() {
    print_header "Pulling External Images"
    
    images=(
        "redis:7.4-alpine"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
    )
    
    for image in "${images[@]}"; do
        log "Pulling $image..."
        docker pull "$image" || warn "Failed to pull $image"
    done
}

start_services() {
    print_header "Starting Services"
    
    log "Starting containers..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d || error "Failed to start services"
    
    log "Waiting for services to be healthy..."
    sleep 10
    
    log "✓ Services started successfully"
}

check_health() {
    print_header "Checking Service Health"
    
    services=("fraud-api" "redis" "prometheus" "grafana")
    
    for service in "${services[@]}"; do
        container="${PROJECT_NAME}-${service}-1"
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            log "✓ $service is running"
        else
            warn "$service is not running"
        fi
    done
}

show_status() {
    print_header "Deployment Status"
    
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    
    echo ""
    info "Services are available at:"
    echo "  • API:        http://localhost:8000"
    echo "  • API Docs:   http://localhost:8000/docs"
    echo "  • Jupyter:    http://localhost:8888"
    echo "  • Streamlit:  http://localhost:8501"
    echo "  • Grafana:    http://localhost:3000"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Redis:      localhost:6379"
    echo ""
}

show_logs() {
    print_header "Recent Logs"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs --tail=20
}

cleanup() {
    print_header "Cleanup"
    
    log "Removing stopped containers..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" rm -f
    
    log "Pruning unused images..."
    docker image prune -f
    
    log "✓ Cleanup completed"
}

# Main execution
main() {
    # Create logs directory first
    mkdir -p "$(dirname "$LOG_FILE")"
    
    print_header "🚀 Neuromorphic Fraud Detection - Production Deployment"
    
    log "Starting deployment on $(hostname) at $(date)"
    
    check_requirements
    create_directories
    setup_environment
    
    build_images
    pull_images
    start_services
    check_health
    show_status
    
    print_header "✅ Deployment Completed Successfully"
    
    info "To view logs: docker compose -f $COMPOSE_FILE logs -f"
    info "To stop:      docker compose -f $COMPOSE_FILE down"
    info "To restart:   docker compose -f $COMPOSE_FILE restart"
    
    log "Deployment completed at $(date)"
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        print_header "Stopping Services"
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
        log "✓ Services stopped"
        ;;
    restart)
        print_header "Restarting Services"
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" restart
        log "✓ Services restarted"
        ;;
    status)
        show_status
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|status|logs|cleanup}"
        exit 1
        ;;
esac
