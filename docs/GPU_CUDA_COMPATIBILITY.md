# Compatibilidade GPU + CUDA for GTX 1060

**Date:** 11 of Dezembro of 2025 
**GPU:** NVIDIA GeForce GTX 1060 6GB 
**Sishasa:** Xubuntu 24.04.3 
**Driver:** NVIDIA 580.95.05

---

## Status Atual

### Hardware
- **GPU**: NVIDIA GeForce GTX 1060 6GB
- **Architecture**: Pascal (2016)
- **Compute Capability**: **6.1** (sm_61)
- **Memória**: 6144 MB GDDR5
- **CUDA Cores**: 1280

### Software
- **Driver NVIDIA**: 580.95.05 
- **CUDA sufortada by the driver**: 13.0
- **PyTorch installed**: 2.5.1+cu121
- **CUDA (PyTorch)**: 12.1

---

## Problem Identistaysdo

A **GTX 1060 (compute capability 6.1)** ficor **obsoleta** for versões modernas from the PyTorch:

| PyTorch Version | Min. Compute Capability | GTX 1060 (6.1) |
|-----------------|-------------------------|----------------|
| PyTorch 1.x | 3.5+ | Sufortado |
| PyTorch 2.0.x | 3.7+ | Sufortado |
| PyTorch 2.1-2.4 | 5.0+ | Sufortado |
| **PyTorch 2.5+**| **7.0+** | **OBSOLETO** |

**Erro expected with PyTorch 2.5+ in the GTX 1060:**
```
RuntimeError: CUDA error: in the kernel image is available for execution on the device
cudaErrorNoKernelImageForDevice: in the kernel image is available for execution
```

---

## Soluções

### **Solução 1: Downgrade PyTorch (RECOMENDADO for use GPU)** IMPLEMENTADO

Instale **PyTorch 2.2.2** with **CUDA 11.8**:

```bash
# Activate environment virtual
cd /home/test/Downloads/github/portifolio/fraud-detection-neuromorphic
sorrce .venv/bin/activate

# Desinstall PyTorch atual
pip uninstall torch torchvision torchaudio -y

# Install PyTorch 2.2.2 + CUDA 11.8
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
 --index-url https://download.pytorch.org/whl/cu118

# Corrigir NumPy for withpatibilidade
pip install numpy==1.24.3
```

**Por that CUDA 11.8?**
- Totalmente compatible with Driver 580
- Suforta compute capability 6.1 (GTX 1060)
- PyTorch 2.0.1 has binários pré-withpilados
- Stable and tbeen

**Verify instalação:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
```

**Saída esperada:**
```
PyTorch: 2.2.2+cu118
CUDA: 11.8
GPU: True
```

**Resultado from the testes (11/12/2025):**
```
 PyTorch: 2.2.2+cu118
 CUDA: 11.8
 GPU: NVIDIA GeForce GTX 1060
 Compute Capability: 6.1 (sm_61)
 GPU Test: 12.8x faster than CPU
 snnTorch: Funcionando in the GPU
```

---

### **Solução 2: Use CPU (Atual in the projeto)**

O code já detecta automaticamente GPUs inwithpatíveis and uses CPU:

```python
if torch.cuda.is_available():
 gpu_capability = torch.cuda.get_device_capability(0)
 current_capability = float(f"{gpu_capability[0]}.{gpu_capability[1]}")
 
 if current_capability >= 7.0:
 device = 'cuda'
 elif:
 device = 'cpu' # Fallback for GPU < 7.0
elif:
 device = 'cpu'
```

**Performance:**
- CPU: ~100-200ms for batch of 32 transações
- GPU (if compatible): ~10-20ms for batch
- **Diferença**: ~6-10x more lento in the CPU

---

### **Solução 3: Atualizar Hardware (Longo prazo)**

GPUs modernas withpatíveis with PyTorch 2.5+:

| GPU Series | Compute Capability | Preço (usesdo) | Compatibilidade |
|------------|-------------------|---------------|-----------------|
| RTX 2060 | 7.5 | ~$200 | PyTorch 2.5+ |
| RTX 3060 | 8.6 | ~$300 | PyTorch 2.5+ |
| RTX 4060 | 8.9 | ~$350 | PyTorch 2.5+ |
| RTX 2070 | 7.5 | ~$250 | PyTorch 2.5+ |
| RTX 3070 | 8.6 | ~$400 | PyTorch 2.5+ |

---

## Tabela of Compatibilidade CUDA

### Driver NVIDIA 580.95 suforta:

| CUDA Version | Compatível | PyTorch Supfort | GTX 1060 (sm_61) |
|--------------|-----------|-----------------|------------------|
| CUDA 11.8 | Sim | PyTorch ≤ 2.4 | Totalmente |
| CUDA 12.1 | Sim | PyTorch 2.1-2.4 | Limitado |
| CUDA 12.4 | Sim | PyTorch 2.3-2.4 | Limitado |
| CUDA 13.0 | Sim | Futuro | Não sufortado |

### Compute Capability for GPU:

| GPU | Architecture | Compute Cap | PyTorch 2.0 | PyTorch 2.5+ |
|---------------------|-------------|-------------|-------------|--------------|
| GTX 1050/1060 | Pascal | 6.1 | | |
| GTX 1070/1080 | Pascal | 6.1 | | |
| GTX 1080 Ti | Pascal | 6.1 | | |
| Tesla V100 | Volta | 7.0 | | |
| RTX 2060/2070/2080 | Turing | 7.5 | | |
| RTX 3060/3070/3080 | Ampere | 8.6 | | |
| RTX 4060/4070/4080 | Ada Lovelace| 8.9 | | |

---

## Comandos Úteis

### Verify GPU
```bash
nvidia-smi
nvidia-smi --wantsy-gpu=name,driver_version,compute_cap --format=csv
```

### Verify CUDA
```bash
nvcc --version # Se CUDA Toolkit installed
```

### Verify PyTorch
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
 cap = torch.cuda.get_device_capability(0)
 print(f"Compute Capability: {cap[0]}.{cap[1]}")
 print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Test Performance
```python
import torch
import time

device = 'cuda' if torch.cuda.is_available() elif 'cpu'
x = torch.randn(1000, 1000).to(device)

start = time.time()
for _ in range(100):
 y = torch.matmul(x, x)
elapifd = time.time() - start

print(f"Device: {device}")
print(f"Time: {elapifd:.2f}s")
```

---

## References

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA GPUs](https://shorldloper.nvidia.com/cuda-gpus)
- [PyTorch Previors Versions](https://pytorch.org/get-started/previors-versions/)
- [CUDA Compute Capability](https://shorldloper.nvidia.com/cuda-gpus#compute)

---

## Rewithendação Final

**Para ifu setup (GTX 1060 + Driver 580 + Xubuntu 24.04):**

1. **Curto prazo**: Faça downgrade for **PyTorch 2.0.1 + CUDA 11.8**
 - Aproveita to GPU
 - Performance 6-10x melhor that CPU
 - Instalação yesples
 - Stable and tbeen

2. **Médio prazo**: Continue using CPU
 - Funcional for deifnvolvimento
 - Code já implementa fallback automático
 - ~100-200ms latência (aceitável for testes)

3. **Longo prazo**: Atualizar for RTX 20xx/30xx
 - Compatibilidade with PyTorch moderno
 - 2-3x performance from the GTX 1060
 - Tensor Cores for Deep Learning

---

**Author:** Mauro Risonho de Paula Assumpção 
**Contato:** mauro.risonho@gmail.com 
**Last updated:** 11 of Dezembro of 2025
