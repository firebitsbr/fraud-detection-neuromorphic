#!/bin/bash

# Production Deployment Script
# Author: Mauro Risonho de Paula Assumpção
# Date: December 5, 2025
# Description: Deploy Neuromorphic Fraud Detection stack to production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker/docker-compose.production.yml"
PROJECT_NAME="fraud-detection"

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_info "All requirements satisfied"
}

pull_images() {
    print_info "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" pull
}

build_images() {
    print_info "Building images..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build --no-cache
}

start_services() {
    print_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
    
    print_info "Waiting for services to be healthy..."
    sleep 10
}

check_health() {
    print_info "Checking service health..."
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_info "API is healthy"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    print_error "API health check failed"
    return 1
}

show_status() {
    print_info "Service status:"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
}

show_logs() {
    print_info "Recent logs:"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs --tail=20
}

main() {
    echo "========================================"
    echo "Fraud Detection Production Deployment"
    echo "========================================"
    echo ""
    
    check_requirements
    
    # Build or pull images
    if [ "$1" == "--build" ]; then
        build_images
    else
        pull_images
    fi
    
    # Start services
    start_services
    
    # Check health
    if check_health; then
        show_status
        
        echo ""
        print_info "Deployment successful!"
        echo ""
        echo "Services available at:"
        echo "  - API:        http://localhost:8000"
        echo "  - API Docs:   http://localhost:8000/api/docs"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana:    http://localhost:3000 (admin/admin)"
        echo ""
        print_info "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
        print_info "To stop:      docker-compose -f $COMPOSE_FILE down"
    else
        print_error "Deployment failed"
        show_logs
        exit 1
    fi
}

# Run main function
main "$@"
