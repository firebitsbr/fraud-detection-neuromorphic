#!/usr/bin/env bash
#
# **Descrição:** Deploy Remote Access - Acesso remoto ao container via VS Code.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Deploy Remote Access - Neuromorphic Fraud Detection   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Escolher modo de deployment
echo -e "${YELLOW}Escolha o modo de deployment:${NC}"
echo ""
echo "  1) Dev Container (local - attach ao container)"
echo "  2) Remote SSH (acesso via SSH - porta 2222)"
echo "  3) Ambos"
echo ""
read -p "Opção [1-3]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}🚀 Deploying Dev Container...${NC}"
        sudo docker compose -f config/docker-compose.dev.yml up -d
        
        echo -e "\n${GREEN}✅ Container deployed!${NC}"
        echo ""
        echo "📝 Próximos passos no VS Code:"
        echo "  1. Ctrl+Shift+P"
        echo "  2. 'Dev Containers: Attach to Running Container'"
        echo "  3. Selecione: fraud-detection-dev"
        ;;
        
    2)
        echo -e "\n${GREEN}🚀 Building Remote SSH Container...${NC}"
        
        # Build da imagem base primeiro
        echo "Building base API image..."
        sudo docker build -t fraud-detection-api:ubuntu24.04 -f Dockerfile .
        
        # Build container remoto
        sudo docker compose -f docker-compose.remote.yml up -d --build
        
        echo -e "\n${GREEN}✅ Remote container deployed!${NC}"
        echo ""
        echo "📊 Connection Info:"
        echo "  SSH: ssh -p 2222 appuser@localhost"
        echo "  Password: neuromorphic2025"
        echo ""
        echo "📝 Configure VS Code SSH (~/.ssh/config):"
        echo "Host fraud-docker"
        echo "    HostName localhost"
        echo "    User appuser"
        echo "    Port 2222"
        echo ""
        echo "Depois conecte:"
        echo "  Ctrl+Shift+P → 'Remote-SSH: Connect to Host' → fraud-docker"
        ;;
        
    3)
        echo -e "\n${GREEN}🚀 Deploying both containers...${NC}"
        
        # Dev container
        sudo docker compose -f docker-compose.dev.yml up -d
        
        # Remote SSH
        sudo docker build -t fraud-detection-api:ubuntu24.04 -f Dockerfile .
        sudo docker compose -f docker-compose.remote.yml up -d --build
        
        echo -e "\n${GREEN}✅ Both containers deployed!${NC}"
        echo ""
        echo "Dev Container:"
        echo "  Ctrl+Shift+P → 'Attach to Running Container' → fraud-detection-dev"
        echo ""
        echo "Remote SSH:"
        echo "  ssh -p 2222 appuser@localhost"
        ;;
        
    *)
        echo -e "${YELLOW}Opção inválida${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Ver status:"
echo "  docker ps"
echo ""
echo "Ver logs:"
echo "  docker logs fraud-detection-dev"
echo "  docker logs fraud-detection-remote"
echo ""
echo "Documentação completa:"
echo "  docs/REMOTE_ACCESS.md"
