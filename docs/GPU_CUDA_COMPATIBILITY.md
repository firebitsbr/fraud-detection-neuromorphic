# Compatibilidade GPU + CUDA para GTX 1060

**Data:** 11 de Dezembro de 2025 
**GPU:** NVIDIA GeForce GTX 1060 6GB 
**Sistema:** Xubuntu 24.04.3 
**Driver:** NVIDIA 580.95.05

---

## Status Atual

### Hardware
- **GPU**: NVIDIA GeForce GTX 1060 6GB
- **Arquitetura**: Pascal (2016)
- **Compute Capability**: **6.1** (sm_61)
- **Memória**: 6144 MB GDDR5
- **CUDA Cores**: 1280

### Software
- **Driver NVIDIA**: 580.95.05 
- **CUDA suportada pelo driver**: 13.0
- **PyTorch instalado**: 2.5.1+cu121
- **CUDA (PyTorch)**: 12.1

---

## Problema Identificado

A **GTX 1060 (compute capability 6.1)** ficou **obsoleta** para versões modernas do PyTorch:

| PyTorch Version | Min. Compute Capability | GTX 1060 (6.1) |
|-----------------|-------------------------|----------------|
| PyTorch 1.x | 3.5+ | Suportado |
| PyTorch 2.0.x | 3.7+ | Suportado |
| PyTorch 2.1-2.4 | 5.0+ | Suportado |
| **PyTorch 2.5+**| **7.0+** | **OBSOLETO** |

**Erro esperado com PyTorch 2.5+ na GTX 1060:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
cudaErrorNoKernelImageForDevice: no kernel image is available for execution
```

---

## Soluções

### **Solução 1: Downgrade PyTorch (RECOMENDADO para usar GPU)** IMPLEMENTADO

Instale **PyTorch 2.2.2** com **CUDA 11.8**:

```bash
# Ativar ambiente virtual
cd /home/test/Downloads/github/portifolio/fraud-detection-neuromorphic
source .venv/bin/activate

# Desinstalar PyTorch atual
pip uninstall torch torchvision torchaudio -y

# Instalar PyTorch 2.2.2 + CUDA 11.8
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
 --index-url https://download.pytorch.org/whl/cu118

# Corrigir NumPy para compatibilidade
pip install numpy==1.24.3
```

**Por que CUDA 11.8?**
- Totalmente compatível com Driver 580
- Suporta compute capability 6.1 (GTX 1060)
- PyTorch 2.0.1 tem binários pré-compilados
- Stable e testado

**Verificar instalação:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
```

**Saída esperada:**
```
PyTorch: 2.2.2+cu118
CUDA: 11.8
GPU: True
```

**Resultado dos testes (11/12/2025):**
```
 PyTorch: 2.2.2+cu118
 CUDA: 11.8
 GPU: NVIDIA GeForce GTX 1060
 Compute Capability: 6.1 (sm_61)
 GPU Test: 12.8x mais rápido que CPU
 snnTorch: Funcionando na GPU
```

---

### **Solução 2: Usar CPU (Atual no projeto)**

O código já detecta automaticamente GPUs incompatíveis e usa CPU:

```python
if torch.cuda.is_available():
 gpu_capability = torch.cuda.get_device_capability(0)
 current_capability = float(f"{gpu_capability[0]}.{gpu_capability[1]}")
 
 if current_capability >= 7.0:
 device = 'cuda'
 else:
 device = 'cpu' # Fallback para GPU < 7.0
else:
 device = 'cpu'
```

**Performance:**
- CPU: ~100-200ms por batch de 32 transações
- GPU (se compatível): ~10-20ms por batch
- **Diferença**: ~6-10x mais lento na CPU

---

### **Solução 3: Atualizar Hardware (Longo prazo)**

GPUs modernas compatíveis com PyTorch 2.5+:

| GPU Series | Compute Capability | Preço (usado) | Compatibilidade |
|------------|-------------------|---------------|-----------------|
| RTX 2060 | 7.5 | ~$200 | PyTorch 2.5+ |
| RTX 3060 | 8.6 | ~$300 | PyTorch 2.5+ |
| RTX 4060 | 8.9 | ~$350 | PyTorch 2.5+ |
| RTX 2070 | 7.5 | ~$250 | PyTorch 2.5+ |
| RTX 3070 | 8.6 | ~$400 | PyTorch 2.5+ |

---

## Tabela de Compatibilidade CUDA

### Driver NVIDIA 580.95 suporta:

| CUDA Version | Compatível | PyTorch Support | GTX 1060 (sm_61) |
|--------------|-----------|-----------------|------------------|
| CUDA 11.8 | Sim | PyTorch ≤ 2.4 | Totalmente |
| CUDA 12.1 | Sim | PyTorch 2.1-2.4 | Limitado |
| CUDA 12.4 | Sim | PyTorch 2.3-2.4 | Limitado |
| CUDA 13.0 | Sim | Futuro | Não suportado |

### Compute Capability por GPU:

| GPU | Arquitetura | Compute Cap | PyTorch 2.0 | PyTorch 2.5+ |
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

### Verificar GPU
```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

### Verificar CUDA
```bash
nvcc --version # Se CUDA Toolkit instalado
```

### Verificar PyTorch
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

### Testar Performance
```python
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(1000, 1000).to(device)

start = time.time()
for _ in range(100):
 y = torch.matmul(x, x)
elapsed = time.time() - start

print(f"Device: {device}")
print(f"Time: {elapsed:.2f}s")
```

---

## Referências

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus#compute)

---

## Recomendação Final

**Para seu setup (GTX 1060 + Driver 580 + Xubuntu 24.04):**

1. **Curto prazo**: Faça downgrade para **PyTorch 2.0.1 + CUDA 11.8**
 - Aproveita a GPU
 - Performance 6-10x melhor que CPU
 - Instalação simples
 - Stable e testado

2. **Médio prazo**: Continue usando CPU
 - Funcional para desenvolvimento
 - Código já implementa fallback automático
 - ~100-200ms latência (aceitável para testes)

3. **Longo prazo**: Atualizar para RTX 20xx/30xx
 - Compatibilidade com PyTorch moderno
 - 2-3x performance da GTX 1060
 - Tensor Cores para Deep Learning

---

**Autor:** Mauro Risonho de Paula Assumpção 
**Contato:** mauro.risonho@gmail.com 
**Última atualização:** 11 de Dezembro de 2025
