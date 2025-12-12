#!/bin/bash
#
# **Descrição:** Build Docker com Progresso Visual em Tempo Real. Script que mostra EXATAMENTE o que está acontecendo dentro do Docker durante o build, com barras de progresso e status detalhado.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Função para mostrar progresso
show_progress() {
 local current=$1
 local total=$2
 local package=$3
 local percent=$((current * 100 / total))
 local filled=$((percent / 2))
 local empty=$((50 - filled))
 
 printf "\r${CYAN}[${GREEN}"
 printf "%${filled}s" | tr ' ' ''
 printf "${NC}%${empty}s${CYAN}]${NC} ${BOLD}%3d%%${NC} ${YELLOW}%s${NC}" " " "$percent" "$package"
}

# Limpar tela
clear

echo -e "${BOLD}${MAGENTA}"
echo ""
echo " DOCKER BUILD COM MONITORAMENTO EM TEMPO REAL "
echo " "
echo " Sistema de Detecção de Fraude Neuromorfico "
echo ""
echo -e "${NC}"

echo -e "${CYAN} Iniciando build da imagem base...${NC}\n"

# Criar pipe para capturar output
LOGFILE="/tmp/docker_build_$(date +%s).log"

# Build com progress=plain para ver TODOS os detalhes
echo -e "${YELLOW}⏳ Construindo imagem base (isso pode levar 5-10 minutos)...${NC}\n"

docker compose build --progress=plain base_image 2>&1 | tee "$LOGFILE" | while IFS= read -r line; do
 # Detectar estágios
 if echo "$line" | grep -q "FROM docker.io"; then
 echo -e "\n${BLUE} Baixando imagem base Python...${NC}"
 elif echo "$line" | grep -q "apt-get update"; then
 echo -e "\n${BLUE} Instalando dependências do sistema...${NC}"
 elif echo "$line" | grep -q "pip install --upgrade pip"; then
 echo -e "\n${GREEN} Atualizando pip...${NC}"
 elif echo "$line" | grep -q "Collecting"; then
 package=$(echo "$line" | sed -n 's/.*Collecting \([^ ]*\).*/\1/p')
 echo -e "${CYAN} ↓ Baixando: ${YELLOW}$package${NC}"
 elif echo "$line" | grep -q "Downloading"; then
 size=$(echo "$line" | grep -oP '\d+\.\d+ [KMG]B' | tail -1)
 echo -e "${CYAN} Tamanho: ${GREEN}$size${NC}"
 elif echo "$line" | grep -q "Installing collected packages"; then
 echo -e "\n${GREEN} Instalando pacotes...${NC}"
 elif echo "$line" | grep -q "Successfully installed"; then
 echo -e "\n${GREEN} Pacotes instalados com sucesso!${NC}"
 # Mostrar quais pacotes foram instalados
 packages=$(echo "$line" | sed 's/Successfully installed //')
 echo -e "${CYAN} Instalados: ${YELLOW}$packages${NC}"
 elif echo "$line" | grep -q "Building wheel"; then
 package=$(echo "$line" | sed -n 's/.*Building wheel for \([^ ]*\).*/\1/p')
 echo -e "${MAGENTA} Compilando: ${YELLOW}$package${NC}"
 elif echo "$line" | grep -q "ERROR\|Error\|error"; then
 echo -e "${RED} ERRO: $line${NC}"
 elif echo "$line" | grep -q "WARNING\|Warning"; then
 echo -e "${YELLOW} Aviso: $line${NC}"
 elif echo "$line" | grep -q "exporting to image"; then
 echo -e "\n${BLUE} Salvando imagem...${NC}"
 elif echo "$line" | grep -q "naming to"; then
 echo -e "${GREEN} Nomeando imagem...${NC}"
 fi
done

# Verificar resultado
if [ $? -eq 0 ]; then
 echo -e "\n${GREEN}${BOLD}"
 echo ""
 echo " BUILD CONCLUÍDO COM SUCESSO! "
 echo ""
 echo -e "${NC}"
 
 # Mostrar informações da imagem
 echo -e "${CYAN} Informações da imagem:${NC}"
 docker images fraud-detection-base:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
 
 echo -e "\n${CYAN} Log completo salvo em: ${YELLOW}$LOGFILE${NC}"
 echo -e "${GREEN} Pronto para construir os serviços!${NC}\n"
else
 echo -e "\n${RED}${BOLD}"
 echo ""
 echo " BUILD FALHOU! "
 echo ""
 echo -e "${NC}"
 echo -e "${RED}Verifique o log em: ${YELLOW}$LOGFILE${NC}\n"
 exit 1
fi
