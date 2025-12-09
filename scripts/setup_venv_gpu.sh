#!/bin/bash
#
# **Descrição:** Setup Python Virtual Environment for GPU (NVIDIA GTX 1060). Creates a .venv using Python 3.12 and installs PyTorch with CUDA support.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=== Setting up Development Environment for NVIDIA GTX 1060 ===${NC}"

# 1. Select Python Version (Prefer 3.12 for stability with PyTorch)
PYTHON_CMD="/usr/bin/python3.12"

if [ ! -f "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python 3.12 not found at $PYTHON_CMD${NC}"
    echo "Checking for other python3 versions..."
    PYTHON_CMD=$(which python3)
    VERSION=$($PYTHON_CMD --version)
    echo -e "${YELLOW}Using $PYTHON_CMD ($VERSION)${NC}"
else
    echo -e "${GREEN}Found Python 3.12 at $PYTHON_CMD${NC}"
fi

# 2. Create .venv
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Removing existing .venv...${NC}"
    rm -rf .venv
fi

echo -e "${YELLOW}Creating virtual environment (.venv)...${NC}"
$PYTHON_CMD -m venv .venv

# 3. Activate and Upgrade pip
source .venv/bin/activate
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA 12.1 support
# Driver 535 supports CUDA 12.2, so PyTorch with CUDA 12.1 is compatible.
echo -e "${YELLOW}Installing PyTorch with CUDA 12.1 support...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install Project Requirements
echo -e "${YELLOW}Installing project requirements...${NC}"
if [ -f "requirements/requirements.txt" ]; then
    pip install -r requirements/requirements.txt
else
    echo -e "${RED}Warning: requirements/requirements.txt not found!${NC}"
fi

# 6. Verify Installation
echo -e "\n${YELLOW}=== Verifying Installation ===${NC}"
python3 -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available!')
"

echo -e "\n${GREEN}✅ Environment setup complete!${NC}"
echo -e "To activate, run: ${YELLOW}source .venv/bin/activate${NC}"
