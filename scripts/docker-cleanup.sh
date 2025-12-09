#!/usr/bin/env bash
#
# **Descrição:** Docker Cleanup Script - Remove old containers, images and volumes.
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
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    echo ""
}

confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

cleanup_containers() {
    print_header "Stopping and Removing Containers"
    
    log "Stopping containers..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down || warn "Some containers may not exist"
    
    log "Removing stopped containers..."
    docker container prune -f
    
    log "✓ Containers cleaned"
}

cleanup_images() {
    print_header "Removing Unused Images"
    
    log "Current images:"
    docker images | grep -E "fraud-detection|REPOSITORY" || true
    
    if confirm "Remove dangling images?"; then
        docker image prune -f
        log "✓ Dangling images removed"
    fi
    
    if confirm "Remove ALL unused images (including tagged)?"; then
        docker image prune -a -f
        log "✓ All unused images removed"
    fi
}

cleanup_volumes() {
    print_header "Managing Volumes"
    
    log "Named volumes:"
    docker volume ls | grep "$PROJECT_NAME" || echo "No project volumes found"
    
    warn "Removing volumes will delete ALL data (models, notebooks, Grafana configs)"
    
    if confirm "Remove project volumes? THIS CANNOT BE UNDONE"; then
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v
        log "✓ Volumes removed"
    else
        log "Volumes preserved"
    fi
    
    if confirm "Remove dangling volumes?"; then
        docker volume prune -f
        log "✓ Dangling volumes removed"
    fi
}

cleanup_networks() {
    print_header "Cleaning Networks"
    
    log "Removing unused networks..."
    docker network prune -f
    
    log "✓ Networks cleaned"
}

cleanup_build_cache() {
    print_header "Cleaning Build Cache"
    
    log "Current cache usage:"
    docker system df
    
    if confirm "Remove build cache?"; then
        docker builder prune -f
        log "✓ Build cache cleared"
    fi
}

show_disk_usage() {
    print_header "Docker Disk Usage"
    docker system df
    echo ""
}

full_cleanup() {
    warn "FULL CLEANUP will remove containers, images, volumes, and cache"
    warn "This operation is IRREVERSIBLE"
    
    if confirm "Proceed with full cleanup?"; then
        cleanup_containers
        cleanup_images
        cleanup_volumes
        cleanup_networks
        cleanup_build_cache
        
        print_header "✅ Full Cleanup Complete"
        show_disk_usage
    else
        log "Full cleanup cancelled"
    fi
}

case "${1:-interactive}" in
    containers)
        cleanup_containers
        ;;
    images)
        cleanup_images
        ;;
    volumes)
        cleanup_volumes
        ;;
    networks)
        cleanup_networks
        ;;
    cache)
        cleanup_build_cache
        ;;
    full)
        full_cleanup
        ;;
    status)
        show_disk_usage
        ;;
    interactive|*)
        print_header "Docker Cleanup Menu"
        echo "1) Clean containers"
        echo "2) Clean images"
        echo "3) Clean volumes (DANGEROUS)"
        echo "4) Clean networks"
        echo "5) Clean build cache"
        echo "6) Full cleanup (ALL)"
        echo "7) Show disk usage"
        echo "0) Exit"
        echo ""
        
        read -p "Select option: " choice
        
        case $choice in
            1) cleanup_containers ;;
            2) cleanup_images ;;
            3) cleanup_volumes ;;
            4) cleanup_networks ;;
            5) cleanup_build_cache ;;
            6) full_cleanup ;;
            7) show_disk_usage ;;
            0) log "Exiting..." ;;
            *) error "Invalid option" ;;
        esac
        ;;
esac
