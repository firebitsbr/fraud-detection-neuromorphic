#!/bin/bash
#
# Entrypoint for remote access container
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
