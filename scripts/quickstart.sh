#!/bin/bash
# Quick Start - Fraud Detection Neuromorphic
# Execute este script após o setup-conda.sh terminar

echo "🚀 QUICK START - Fraud Detection Neuromorphic"
echo ""

# Verificar se ambiente existe
if ! conda env list | grep -q "fraud-detection-neuromorphic"; then
    echo "❌ Ambiente não encontrado!"
    echo "Execute primeiro: bash scripts/setup-conda.sh"
    exit 1
fi

# Ativar ambiente
echo "1️⃣  Ativando ambiente..."
eval "$(conda shell.bash hook)"
conda activate fraud-detection-neuromorphic

# Verificar Python
echo ""
echo "2️⃣  Verificando Python..."
python --version

# Verificar PyTorch
echo ""
echo "3️⃣  Verificando PyTorch e GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponível: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    capability = torch.cuda.get_device_capability()
    print(f'Compute capability: {capability[0]}.{capability[1]}')
"

echo ""
echo "4️⃣  Verificando snnTorch..."
python -c "import snntorch; print(f'snnTorch: {snntorch.__version__}')"

echo ""
echo "✅ TUDO PRONTO!"
echo ""
echo "Para iniciar Jupyter Lab:"
echo "  jupyter lab"
echo ""
echo "Notebook recomendado:"
echo "  notebooks/04_brian2_vs_snntorch.ipynb"
echo ""
