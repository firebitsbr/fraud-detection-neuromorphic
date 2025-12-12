#!/usr/bin/env bash
#
# **Descrição:** Docker Permission Fix - Adiciona usuário ao grupo docker.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

set -euo pipefail

echo " Corrigindo permissões do Docker..."
echo ""

# Adicionar ao grupo docker
sudo groupadd docker 2>/dev/null || echo " Grupo docker já existe"
sudo usermod -aG docker $USER

echo " Usuário $USER adicionado ao grupo docker"
echo ""
echo " IMPORTANTE: Você precisa fazer UMA das opções abaixo:"
echo ""
echo "Opção 1 - Logout/Login (RECOMENDADO):"
echo " 1. Fechar VS Code"
echo " 2. Fazer logout do sistema"
echo " 3. Fazer login novamente"
echo " 4. Abrir VS Code"
echo ""
echo "Opção 2 - Reiniciar Docker service:"
echo " sudo systemctl restart docker"
echo " newgrp docker"
echo ""
echo "Opção 3 - Reiniciar sistema (mais garantido):"
echo " sudo reboot"
echo ""

# Verificar se já funciona
if docker ps &>/dev/null; then
 echo " Docker já está funcionando sem sudo!"
else
 echo " Docker ainda precisa de permissões"
 echo ""
 echo "Execute um dos comandos:"
 echo " gnome-session-quit --logout --no-prompt # Para GNOME"
 echo " sudo reboot # Reiniciar sistema"
fi
