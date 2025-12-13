#!/bin/bash
# Verificar PyTorch 1.13.1 + CUDA 11.6 no ambiente Conda
# PyTorch agora é instalado via pip durante criação do environment

set -e

ENV_NAME="fraud-detection-neuromorphic"

echo "=========================================="
echo "VERIFICANDO PYTORCH + GPU"
echo "=========================================="
echo ""

# Verificar se ambiente está ativo
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$CURRENT_ENV" != "$ENV_NAME" ]; then
    echo "⚠️  Ativando ambiente ${ENV_NAME}..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

echo "Ambiente ativo: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

echo "Verificando instalação do PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); cap = torch.cuda.get_device_capability(); print(f'Compute capability: {cap[0]}.{cap[1]}')"
    echo ""
    echo "✓✓✓ GPU HABILITADA E FUNCIONANDO! ✓✓✓"
else
    echo ""
    echo "⚠️  GPU não detectada. Verifique drivers NVIDIA."
    echo "   Execute: nvidia-smi para verificar driver"
fi

echo ""
echo "=========================================="
