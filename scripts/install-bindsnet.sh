#!/bin/bash
# Instalar BindsNET com PyTorch 1.13.1 (compatível com GTX 1060)
# BindsNET tem dependência hard-coded para torch==2.9.0, precisamos modificar

set -e

ENV_NAME="fraud-detection-neuromorphic"

echo "=========================================="
echo "INSTALANDO BINDSNET"
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

# Clonar BindsNET
TEMP_DIR=$(mktemp -d)
echo "Clonando BindsNET em: $TEMP_DIR"
git clone https://github.com/BindsNET/bindsnet.git "$TEMP_DIR/bindsnet" --depth 1

cd "$TEMP_DIR/bindsnet"

# Modificar setup.py para aceitar PyTorch 1.13.1
echo ""
echo "Modificando setup.py para aceitar PyTorch 1.13.1..."

# Buscar e substituir a linha do torch
if grep -q "torch==2.9.0" setup.py; then
    sed -i 's/torch==2.9.0/torch>=1.13.0,<2.0.0/' setup.py
    echo "✓ Modificado: torch==2.9.0 → torch>=1.13.0,<2.0.0"
elif grep -q '"torch==2.9.0"' setup.py; then
    sed -i 's/"torch==2.9.0"/"torch>=1.13.0,<2.0.0"/' setup.py
    echo "✓ Modificado: \"torch==2.9.0\" → \"torch>=1.13.0,<2.0.0\""
fi

# Também modificar pyproject.toml se existir
if [ -f "pyproject.toml" ]; then
    if grep -q "torch==2.9.0" pyproject.toml; then
        sed -i 's/torch==2.9.0/torch>=1.13.0,<2.0.0/' pyproject.toml
        echo "✓ Modificado pyproject.toml"
    fi
fi

# Instalar BindsNET (não usar modo editable para evitar problemas de PATH)
echo ""
echo "Instalando BindsNET..."
pip install . --no-deps

# Instalar apenas as dependências que faltam (sem torch)
echo ""
echo "Instalando dependências do BindsNET (exceto torch)..."
pip install ale-py foolbox 'gymnasium[atari]' numba opencv-python scikit-build scikit-image tensorboardX --no-deps || true

echo ""
echo "=========================================="
echo "✓ BINDSNET INSTALADO COM SUCESSO!"
echo "=========================================="
echo ""

# Limpar
cd -
rm -rf "$TEMP_DIR"

# Verificar instalação
echo "Verificando instalação do BindsNET..."
python -c "import bindsnet; print('BindsNET instalado com sucesso!'); import bindsnet.network; print('Módulos principais carregados OK')"

echo ""
echo "✓✓✓ INSTALAÇÃO COMPLETA! ✓✓✓"
echo ""
echo "Para usar o BindsNET, execute:"
echo "  conda activate fraud-detection-neuromorphic"
echo "  python -c 'import bindsnet; print(bindsnet.__file__)'"
