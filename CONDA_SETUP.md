# Setup Conda - Fraud Detection Neuromorphic

## 🎯 Configuração Rápida (GPU GTX 1060 + CUDA)

Este projeto agora usa **Conda** para garantir compatibilidade perfeita entre:
- Python 3.11
- PyTorch 1.13.1 + CUDA 11.6
- GPU GTX 1060 (compute capability 6.1)

---

## 📋 Pré-requisitos

1. **Conda instalado** (Miniconda ou Anaconda)
   - Download: https://docs.conda.io/en/latest/miniconda.html

2. **Drivers NVIDIA atualizados** (compatível com CUDA 11.6)
   - Verificar: `nvidia-smi`

3. **GPU GTX 1060** ou superior

---

## 🚀 Instalação Automática (Recomendado)

```bash
# 1. Clone o repositório (se ainda não fez)
cd /home/test/Downloads/github/portifolio/fraud-detection-neuromorphic

# 2. Execute o script de setup
bash scripts/setup-conda.sh

# 3. Ative o ambiente
conda activate fraud-detection-neuromorphic

# 4. Verifique GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Tempo estimado:** 5-10 minutos

---

## 🔧 Instalação Manual (Alternativa)

```bash
# 1. Criar ambiente
conda env create -f environment.yml

# 2. Ativar ambiente
conda activate fraud-detection-neuromorphic

# 3. Verificar instalação
python -c "import torch; print(torch.__version__)"
python -c "import snntorch; print('snnTorch:', snntorch.__version__)"
```

---

## ✅ Verificação Completa

Após ativar o ambiente, execute:

```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability()
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    
    # Teste prático
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T
    print("✓ GPU funcionando perfeitamente!")
```

**Saída esperada:**
```
Python: 3.11.x
PyTorch: 1.13.1+cu116
CUDA disponível: True
GPU: NVIDIA GeForce GTX 1060
Compute capability: 6.1
✓ GPU funcionando perfeitamente!
```

---

## 📊 Executar Notebooks

```bash
# Com ambiente ativado
conda activate fraud-detection-neuromorphic

# Iniciar Jupyter Lab
jupyter lab

# Ou Jupyter Notebook
jupyter notebook
```

Navegue até: `notebooks/04_brian2_vs_snntorch.ipynb`

---

## 🔄 Atualizar Ambiente

Se adicionar novas dependências ao `environment.yml`:

```bash
# Atualizar ambiente existente
conda env update -f environment.yml --prune
```

---

## 🗑️ Remover Ambiente

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

## 🐛 Troubleshooting

### GPU não detectada

```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar CUDA
nvcc --version

# Reinstalar ambiente
conda env remove -n fraud-detection-neuromorphic
bash scripts/setup-conda.sh
```

### Conflito de dependências

```bash
# Limpar cache do conda
conda clean --all

# Recriar ambiente
conda env create -f environment.yml --force
```

### Kernel Jupyter não encontra pacotes

```bash
# Instalar kernel do IPython no ambiente
conda activate fraud-detection-neuromorphic
python -m ipykernel install --user --name fraud-detection-neuromorphic --display-name "Python (fraud-detection)"
```

---

## 📝 Notas Importantes

1. **Python 3.13 não é compatível** com PyTorch 1.13.1 (última versão com suporte a GTX 1060)
2. **PyTorch 2.x não suporta** compute capability < 7.0 (GTX 1060 é 6.1)
3. **Conda garante** versões exatas e compatibilidade total
4. **Sempre ative** o ambiente antes de trabalhar: `conda activate fraud-detection-neuromorphic`

---

## 🎓 Recursos Adicionais

- **PyTorch com CUDA:** https://pytorch.org/get-started/previous-versions/
- **snnTorch:** https://snntorch.readthedocs.io/
- **Conda:** https://docs.conda.io/

---

## 👤 Autor

**Mauro Risonho de Paula Assumpção**  
Projeto: Fraud Detection with Neuromorphic Computing  
Licença: MIT
