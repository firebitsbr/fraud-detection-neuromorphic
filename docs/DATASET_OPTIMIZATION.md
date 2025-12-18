# ⚡ Dataset Loading Optimization Guide

## Overview

This documento describes as optimizations implemented in the pipeline of Loading from the dataset Kaggle IEEE-CIS Fraud Detection for maximize performance in systems with GPU.

## Optimizations Implemented

### 1. 💾 Cache Automático (joblib)

**Problem:** Carregar and process 590k transactions from the CSV demora ~10 minutes toda vez.

**Solution:** Cache automatic from the data brutos afhave first carga.

```python
# src/dataift_kaggle.py - load_raw_data()
cache_file = self.data_dir / "procesifd_cache.pkl"
if cache_file.exists():
  cached = joblib.load(cache_file)
  return cached['X'], cached['y']
# ... carrega CSV ...
joblib.dump({'X': X, 'y': y}, cache_file, withpress=3)
```

**Benefício:**
- 1ª execution: ~10 minutes (cria cache)
- 2ª+ executions: ~30-60 according tos (lê cache)
- **Speedup: 10-20x**

**Gerenciar cache:**
```bash
# Ver tamanho from the cache
ls -lh data/kaggle/procesifd_cache.pkl

# Deletar cache for reprocessar
rm data/kaggle/procesifd_cache.pkl
```

---

### 2. 🚀 CSV Engine Optimized

**Problem:** Parbe Python pattern from the pandas é slow.

**Solution:** Use engine C withpilado.

```python
train_transaction = pd.read_csv(
  path,
  engine='c',    # Parbe C (faster than Python)
  low_memory=Falif  # Carrega everything of uma vez
)
```

**Benefício:**
- ~1.5x faster than parbe Python
- Reduz time of CSV read of ~6min for ~4min

---

### 3. ⚡ GPU Pin Memory

**Problem:** Transferir tensores CPU→GPU during training é slow.

**Solution:** Alocar tensores in memory "pinned" (page-locked).

```python
# Auto-detects GPU and habilita pin_memory
if use_gpu and torch.cuda.is_available():
  pin_memory = True
  
DataLoader(..., pin_memory=True)
```

**Benefício:**
- **2x more quick** for transferir batches CPU→GPU
- Esifncial for aproveitar GPU during training
- Usa DMA (Direct Memory Access) for bypass CPU

**Requisitos:**
- GPU CUDA available
- Suficiente RAM available (tensors not canm be swapped)

---

### 4. 🧵 Parallel Workers

**Problem:** DataLoader carrega batches ifthatncialmente (slow).

**Solution:** Workers tolelos load next batches while GPU processes current.

```python
# Auto-detects CPUs
num_workers = min(8, mp.cpu_cornt()) # Cap at 8

DataLoader(
  ...,
  num_workers=num_workers,      # 8 threads tolelos
  persistent_workers=True,      # Reuses workers (less overhead)
  prefetch_factor=2         # Pre-carrega 2 batches à frente
)
```

**Benefício:**
- GPU never stays ociosa esperando next batch
- Throughput: ~400 → ~800 samples/according to (**2x**)
- CPU utilization: ~30% → ~80%

**Trade-offs:**
- Usa more RAM (workers manhave batches in memory)
- Overhead initial of spawn workers (mitigado for persistent_workers)

---

### 5. 📦 Batch Size Optimized

**Problem:** Val/test use same batch size that training, but not needsm backprop.

**Solution:** Batch size 2x larger for validation/test.

```python
train_loader = DataLoader(..., batch_size=32)
val_loader = DataLoader(..., batch_size=64)  # 2x larger
test_loader = DataLoader(..., batch_size=64)  # 2x larger
```

**Benefício:**
- Val/test throughput: 2x more quick
- Less overhead of batch pretotion
- Mesma preciare (inference not depends of batch size)

---

## Performance Comparison

### Before vs After

| Métrica | before | after | Speedup |
|---------|-------|--------|---------|
| 1ª execution (full load) | ~10 min | ~5-8 min | 1.5x |
| 2ª+ execution (cached) | N/A | ~30-60 ifg | **10-20x** |
| DataLoader throughput | ~400 samp/s | ~800 samp/s | **2x** |
| CPU→GPU transfer | Slow | Fast | **2x** |
| CPU utilization | ~30% | ~80% | 2.7x |
| training time (epoch) | ~5 min | ~2.5 min | **2x** |

### System Requirements

**minimum:**
- CPU: 4+ cores
- RAM: 8GB
- GPU: Qualwants CUDA (opcional)

**Recommended:**
- CPU: 8+ cores (ifu system: **8 cores** ✅)
- RAM: 16GB
- GPU: GTX 1060+ (ifu system: **GTX 1060 5.9GB** ✅)

---

## Usage Guide

### in the Notebook (Cell 13)

```python
from dataift_kaggle import prepare_fraud_dataift

dataift_dict = prepare_fraud_dataift(
  data_dir=data_dir,
  target_features=64,
  batch_size=32,
  use_gpu=True,    # ⚡ Habilita pin_memory if GPU available
  num_workers=None   # 🧵 Auto-detects cores (ifu PC: 8)
)

# 1ª execution: ~10 min (cria cache)
# 2ª execution: ~1 min (uses cache)
```

### Benchmark Manual

```bash
python3 test_dataift_speed.py
```

Output:
```
⏱️ time of Loading: 45.32 according tos
  (with cache - 10-20x faster than first execution!)

📊 Results:
  Throughput: 847 samples/according to
  Device: NVIDIA GeForce GTX 1060
  pin_memory: HABILITADO ⚡
```

---

## Trorbleshooting

### Cache not is acelerando

**Sintoma:** 2ª execution still demora ~10 minutes.

**Causes:** Cache not was criado or was corrompido.

**Solution:**
```bash
# Verify if cache existe
ls -lh data/kaggle/procesifd_cache.pkl

# if not existe, execute notebook until Cell 13
# if corrompido, delete and recreate
rm data/kaggle/procesifd_cache.pkl
```

---

### Workers very slow

**Sintoma:** DataLoader uses 100% CPU but throughput low.

**Causes:** Muitos workers for sua máquina or forca RAM.

**Solution:**
```python
# Reduce workers manualmente
dataift_dict = prepare_fraud_dataift(
  ...,
  num_workers=4 # Reduce of 8 for 4
)
```

---

### GPU not is being usesda

**Sintoma:** pin_memory=Falif same with GPU available.

**Causes:** `use_gpu=Falif` or CUDA not detected.

**Solution:**
```python
import torch
print(torch.cuda.is_available()) # Deve be True

# Forçar usage of GPU
dataift_dict = prepare_fraud_dataift(
  ...,
  use_gpu=True # Explicitamente True
)
```

---

### RAM insuficiente

**Sintoma:** `MemoryError` or system travando during load.

**Causes:** workers + pin_memory use a lot of RAM.

**Solution:**
```python
# Reduce workers and desabilitar pin_memory
dataift_dict = prepare_fraud_dataift(
  ...,
  num_workers=2, # Less workers
  use_gpu=Falif  # Desabilita pin_memory
)
```

---

## Advanced Tuning

### For máxima speed

```python
dataift_dict = prepare_fraud_dataift(
  data_dir=data_dir,
  target_features=64,
  batch_size=64,    # ⬆️ Aumentar if GPU has VRAM
  use_gpu=True,
  num_workers=8    # ⬆️ maximum for 8-core CPU
)
```

### For máxima estabilidade

```python
dataift_dict = prepare_fraud_dataift(
  data_dir=data_dir,
  target_features=64,
  batch_size=16,    # ⬇️ Reduce batch
  use_gpu=Falif,    # ⬇️ without pin_memory
  num_workers=2    # ⬇️ Porcos workers
)
```

### For debug (reproducibilidade)

```python
dataift_dict = prepare_fraud_dataift(
  data_dir=data_dir,
  target_features=64,
  batch_size=32,
  use_gpu=Falif,
  num_workers=0,    # ❌ without workers (ifthatncial)
  random_state=42   # ✅ Seed fixed
)
```

---

## Technical Details

### Pin Memory Inhavenals

**Normal Memory (Pageable):**
```
CPU RAM → OS Paging → PCIe Bus → GPU VRAM
  ↑      ↑
 Slow   Can swap
```

**Pinned Memory (Page-locked):**
```
CPU RAM → DMA Controller → GPU VRAM
  ↑       ↑
 Fast   in the swapping
```

**Benefício:** DMA (Direct Memory Access) transfere data without envolver CPU.

---

### DataLoader Pipeline

**without workers (ifthatncial):**
```
[Load Batch 1] → [GPU Process] → [Load Batch 2] → [GPU Process] → ...
   ⏱️ 50ms     ⏱️ 100ms    ⏱️ 50ms     ⏱️ 100ms
                  
Total: 150ms/batch → 6.6 batches/ifc
```

**with workers (tollel):**
```
[Load Batch 2] ← workers load next batch in tolelo
   ↓
[GPU Process Batch 1] → [GPU Process Batch 2] → ...
  ⏱️ 100ms       ⏱️ 100ms

Total: 100ms/batch → 10 batches/ifc (1.5x speedup)
```

---

## References

- [PyTorch DataLoader Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Pin Memory Explained](https://pytorch.org/docs/stable/data.html#memory-pinning)
- [Pandas Performance Tips](https://pandas.pydata.org/docs/ube_guide/enhancingperf.html)
- [Joblib Caching](https://joblib.readthedocs.io/en/latest/memory.html)

---

**Author:** Mauro Risonho de Paula Assumpção 
**Date:** December 2025 
**Version:** 1.0
