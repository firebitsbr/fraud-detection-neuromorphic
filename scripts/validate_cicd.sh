#!/bin/bash
#
# **Descrição:** Script de validação do pipeline CI/CD.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

# CI/CD Validation Script
set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║  CI/CD Pipeline Validation                             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

total=0
passed=0

check_file() {
    ((total++))
    printf "${BLUE}Checking $2...${NC} "
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC}"
        ((passed++))
    else
        echo -e "${RED}✗${NC}"
    fi
}

check_dir() {
    ((total++))
    printf "${BLUE}Checking $2...${NC} "
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC}"
        ((passed++))
    else
        echo -e "${RED}✗${NC}"
    fi
}

echo "=== Files ==="
check_file ".github/workflows/ci-cd.yml" "CI/CD workflow"
check_file "requirements-ci.txt" "CI requirements"
check_file "docker/requirements-production.txt" "Production requirements"
check_file "docker/Dockerfile.production" "Production Dockerfile"

echo ""
echo "=== Directories ==="
check_dir "src" "Source code"
check_dir "api" "API"
check_dir "tests" "Tests"
check_dir "docker" "Docker"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Summary: $passed/$total checks passed                      ║"
echo "╚════════════════════════════════════════════════════════╝"

[ $passed -eq $total ] && exit 0 || exit 1
