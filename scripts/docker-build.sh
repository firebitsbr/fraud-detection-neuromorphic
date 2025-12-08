#!/usr/bin/env bash
#
# Docker Build Script with Progress Tracking
#
# Author: Mauro Risonho de Paula Assumpção
# Email: mauro.risonho@gmail.com
# LinkedIn: linkedin.com/in/maurorisonho
# GitHub: github.com/maurorisonho

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check Docker availability
if ! command -v docker &> /dev/null; then
    error "Docker not found. Please install Docker first."
fi

print_header "Building Neuromorphic Fraud Detection - Docker Images"

# Build base image
log "Building base image (this may take several minutes)..."
docker build \
    --target builder \
    --tag fraud-detection-base:ubuntu24.04 \
    --tag fraud-detection-base:latest \
    --file Dockerfile \
    --progress=plain \
    . || error "Failed to build base image"

log "✓ Base image built successfully"

# Build API image
log "Building API production image..."
docker build \
    --tag fraud-detection-api:ubuntu24.04 \
    --tag fraud-detection-api:latest \
    --file Dockerfile \
    --progress=plain \
    . || error "Failed to build API image"

log "✓ API image built successfully"

# Show built images
print_header "Built Images"
docker images | grep -E "fraud-detection|REPOSITORY"

# Image sizes
print_header "Image Sizes"
log "Base image:  $(docker images fraud-detection-base:latest --format '{{.Size}}')"
log "API image:   $(docker images fraud-detection-api:latest --format '{{.Size}}')"

# Optional: Tag for registry
read -p "Tag images for registry push? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter registry URL (e.g., docker.io/username): " registry
    
    if [ -n "$registry" ]; then
        log "Tagging for registry: $registry"
        docker tag fraud-detection-base:latest "$registry/fraud-detection-base:ubuntu24.04"
        docker tag fraud-detection-api:latest "$registry/fraud-detection-api:ubuntu24.04"
        
        log "✓ Images tagged for registry"
        echo ""
        echo "To push images, run:"
        echo "  docker push $registry/fraud-detection-base:ubuntu24.04"
        echo "  docker push $registry/fraud-detection-api:ubuntu24.04"
    fi
fi

print_header "✅ Build Complete"
log "Images are ready for deployment"
