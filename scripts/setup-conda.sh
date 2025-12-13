#!/bin/bash
# Setup do Ambiente Conda para Fraud Detection Neuromorphic
# Autor: Mauro Risonho de Paula Assumpção
# Data: Dezembro 2025

set -e  # Exit on error

echo "=========================================="
echo "SETUP: Fraud Detection Neuromorphic"
echo "=========================================="
echo ""

# Verificar se conda está instalado
if ! command -v conda &> /dev/null; then
    echo "❌ ERRO: Conda não encontrado!"
    echo "Instale Miniconda ou Anaconda primeiro:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda detectado: $(conda --version)"
echo ""

# Verificar se o ambiente já existe
ENV_NAME="fraud-detection-neuromorphic"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Ambiente '${ENV_NAME}' já existe."
    read -p "Deseja recriá-lo? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "Removendo ambiente existente..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Usando ambiente existente."
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

echo "=========================================="
echo "CRIANDO AMBIENTE CONDA"
echo "=========================================="
echo "Configurações:"
echo "  - Nome: ${ENV_NAME}"
echo "  - Python: 3.10"
echo "  - PyTorch: 1.13.1 + CUDA 11.6"
echo "  - GPU: GTX 1060 (compute capability 6.1)"
echo ""
echo "⏳ Este processo pode levar 10-15 minutos (PyTorch será instalado via pip)..."
echo ""

# Criar ambiente (PyTorch será instalado automaticamente via pip)
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "✓✓✓ SETUP COMPLETO! ✓✓✓"
echo "=========================================="
echo ""
echo "Verificando instalação do PyTorch + GPU..."
echo ""

# Ativar ambiente e verificar GPU
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Executar verificação do PyTorch
bash "$(dirname "$0")/install-pytorch.sh"

echo ""
echo "=========================================="
echo "PRÓXIMOS PASSOS:"
echo "=========================================="
echo ""
echo "1. Ative o ambiente:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Verifique GPU:"
echo "   python -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo ""
echo "3. Execute notebooks:"
echo "   jupyter lab notebooks/"
echo ""
echo "=========================================="
