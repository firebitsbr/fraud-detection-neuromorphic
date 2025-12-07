#!/bin/bash

# Docker Installation Script for Fedora
#
# Description: Script automatizado para instalar Docker no Fedora
#
# Author: Mauro Risonho de Paula Assumpção
# Date: December 5, 2025
# License: MIT License
#
# Usage: sudo ./install-docker-fedora.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Docker Installation - Fedora/RHEL                 ║${NC}"
echo -e "${BLUE}║  Author: Mauro Risonho de Paula Assumpção          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}✗ Este script precisa ser executado como root${NC}"
   echo "Execute: sudo $0"
   exit 1
fi

echo -e "${GREEN}→ Iniciando instalação do Docker...${NC}"
echo ""

# Step 1: Remove old versions
echo -e "${YELLOW}[1/7] Removendo versões antigas do Docker...${NC}"
dnf remove -y docker \
    docker-client \
    docker-client-latest \
    docker-common \
    docker-latest \
    docker-latest-logrotate \
    docker-logrotate \
    docker-selinux \
    docker-engine-selinux \
    docker-engine 2>/dev/null || true
echo -e "${GREEN}✓ Versões antigas removidas${NC}"
echo ""

# Step 2: Install dnf-plugins-core
echo -e "${YELLOW}[2/7] Instalando dnf-plugins-core...${NC}"
dnf -y install dnf-plugins-core
echo -e "${GREEN}✓ dnf-plugins-core instalado${NC}"
echo ""

# Step 3: Add Docker repository
echo -e "${YELLOW}[3/7] Adicionando repositório oficial do Docker...${NC}"
dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
echo -e "${GREEN}✓ Repositório adicionado${NC}"
echo ""

# Step 4: Install Docker Engine
echo -e "${YELLOW}[4/7] Instalando Docker Engine (pode demorar alguns minutos)...${NC}"
dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
echo -e "${GREEN}✓ Docker Engine instalado${NC}"
echo ""

# Step 5: Start and enable Docker
echo -e "${YELLOW}[5/7] Iniciando serviço Docker...${NC}"
systemctl start docker
systemctl enable docker
echo -e "${GREEN}✓ Docker iniciado e habilitado${NC}"
echo ""

# Step 6: Add user to docker group
echo -e "${YELLOW}[6/7] Configurando permissões de usuário...${NC}"
if [ -n "$SUDO_USER" ]; then
    usermod -aG docker $SUDO_USER
    echo -e "${GREEN}✓ Usuário '$SUDO_USER' adicionado ao grupo docker${NC}"
    echo -e "${YELLOW}⚠ IMPORTANTE: Faça logout e login novamente para aplicar as permissões${NC}"
    echo -e "${YELLOW}   ou execute: newgrp docker${NC}"
else
    echo -e "${YELLOW}⚠ Execute manualmente: sudo usermod -aG docker \$USER${NC}"
fi
echo ""

# Step 7: Verify installation
echo -e "${YELLOW}[7/7] Verificando instalação...${NC}"
docker --version
docker compose version
echo ""

# Test Docker
echo -e "${BLUE}Testando Docker com Hello World...${NC}"
if docker run hello-world > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker está funcionando perfeitamente!${NC}"
else
    echo -e "${YELLOW}⚠ Teste não realizado (execute como usuário normal)${NC}"
fi
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ Instalação concluída com sucesso!              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Próximos passos:${NC}"
echo -e "  1. ${YELLOW}Faça logout e login novamente${NC}"
echo -e "     ${BLUE}ou execute:${NC} newgrp docker"
echo ""
echo -e "  2. ${YELLOW}Teste Docker:${NC}"
echo -e "     ${BLUE}docker run hello-world${NC}"
echo ""
echo -e "  3. ${YELLOW}Inicie o projeto:${NC}"
echo -e "     ${BLUE}cd /path/to/fraud-detection-neuromorphic${NC}"
echo -e "     ${BLUE}./start-local.sh${NC}"
echo ""
echo -e "${GREEN}Docker instalado com sucesso! 🐳${NC}"
