#!/bin/bash
#
# **Descrição:** Entrypoint for remote access container.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

set -e

# Iniciar SSH daemon
echo "🚀 Starting SSH daemon..."
sudo /usr/sbin/sshd -D &

# Aguardar SSH iniciar
sleep 2

echo "✅ Container ready for remote access!"
echo ""
echo "📊 Connection Info:"
echo "  SSH: ssh -p 2222 appuser@localhost"
echo "  Password: neuromorphic2025"
echo ""
echo "📁 Workspace: /app"
echo "🐍 Python: /opt/venv/bin/python"
echo ""

# Manter container rodando
tail -f /dev/null
