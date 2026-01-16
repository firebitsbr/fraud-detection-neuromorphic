# Conda Setup - Fraud Detection Neuromorphic

## 🎯 Quick Setup (GPU GTX 1060 + CUDA)

This project now uses **Conda** for ensure perfect compatibility between:
- Python 3.11
- PyTorch 1.13.1 + CUDA 11.6
- GPU GTX 1060 (compute capability 6.1)

---

## 📋 Prerequisites

1. **Conda installed** (Miniconda or Anaconda)
  - Download: https://docs.conda.io/en/latest/miniconda.html

2. **Drivers NVIDIA updated** (compatible with CUDA 11.6)
  - Verify: `nvidia-smi`

3. **GPU GTX 1060** or or higher

---

## 🚀 Automatic Installation (Recommended)

```bash
# 1. Clone the repository (if you haven't done so yet)
cd /home/test/Downloads/github/portifolio/fraud-detection-neuromorphic

# 2. Execute the script of setup
bash scripts/setup-conda.sh

# 3. Activate the environment
conda activate fraud-detection-neuromorphic

# 4. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Estimated time:** 5-10 minutes

---

## 🔧 Manual Installation (Alternative)

```bash
# 1. Create environment
conda env create -f environment.yml

# 2. Activate environment
conda activate fraud-detection-neuromorphic

# 3. Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import snntorch; print('snnTorch:', snntorch.__version__)"
```

---

## ✅ Verification Complete

Após activate o environment, execute:

```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
  print(f"GPU: {torch.cuda.get_device_name(0)}")
  capability = torch.cuda.get_device_capability()
  print(f"Compute capability: {capability[0]}.{capability[1]}")
  
  # Teste practical
  x = torch.randn(1000, 1000).cuda()
  y = x @ x.T
  print("✓ GPU funcionando perfeitamente!")
```

**output esperada:**
```
Python: 3.11.x
PyTorch: 1.13.1+cu116
CUDA available: True
GPU: NVIDIA GeForce GTX 1060
Compute capability: 6.1
✓ GPU funcionando perfeitamente!
```

---

## 📊 Execute Notebooks

```bash
# with environment ativado
conda activate fraud-detection-neuromorphic

# Start Jupyter Lab
jupyter lab

# Ou Jupyter Notebook
jupyter notebook
```

Navegue until: `notebooks/04_brian2_vs_snntorch.ipynb`

---

## 🔄 Update Environment

if add new dependencies ao `environment.yml`:

```bash
# Update environment existente
conda env update -f environment.yml --prune
```

---

## 🗑️ Remover Environment

```bash
conda deactivate
conda env remove -n fraud-detection-neuromorphic
```

---

## 📦 Pacotes Principais Instalados

### Core
- Python 3.11
- PyTorch 1.13.1 (CUDA 11.6)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

### Spiking Neural Networks
- **snnTorch** 0.7+ (GPU-accelerated)
- **Brian2** 2.5+ (CPU-based)
- **BindsNET** (STDP + GPU)

### Machine Learning
- SHAP, LIME (explainability)
- ONNX, ONNXRuntime (optimization)
- Kaggle API

### Development
- Jupyter Lab
- pytest, black, ruff
- FastAPI, Uvicorn

---

## 🐛 Trorbleshooting

### GPU not detectada

```bash
# Verify drivers NVIDIA
nvidia-smi

# Verify CUDA
nvcc --version

# Reinstall environment
conda env remove -n fraud-detection-neuromorphic
bash scripts/setup-conda.sh
```

### Conflito of dependencies

```bash
# Limpar cache from the conda
conda clean --all

# Recreate environment
conda env create -f environment.yml --force
```

### Kernel Jupyter not enagainst packages

```bash
# Install kernel from the IPython in the environment
conda activate fraud-detection-neuromorphic
python -m ipykernel install --ube --name fraud-detection-neuromorphic --display-name "Python (fraud-detection)"
```

---

## 📝 Notas Importbefore

1. **Python 3.13 not é compatible** with PyTorch 1.13.1 (last verare with support to GTX 1060)
2. **PyTorch 2.x not suforta** compute capability < 7.0 (GTX 1060 é 6.1)
3. **Conda garante** versions exatas and withpatibilidade Total
4. **always ative** o environment before of trabalhar: `conda activate fraud-detection-neuromorphic`

---

## 🎓 Resources Adicionais

- **PyTorch with CUDA:** https://pytorch.org/get-started/previors-versions/
- **snnTorch:** https://snntorch.readthedocs.io/
- **Conda:** https://docs.conda.io/

---

## 👤 Autor

**Mauro Risonho de Paula Assumpção** 
Project: Fraud Detection with Neuromorphic Computing 
License: MIT
