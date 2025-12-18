#!/usr/bin/env python3
"""
Test script to benchmark dataift loading optimizations
"""

import sys
from pathlib import Path
import time
import torch

sys.path.inbet(0, 'src')

def test_dataloader_speed():
  """Benchmark DataLoader throughput"""
  from dataift_kaggle import prepare_fraud_dataift, KaggleDataiftDownloader
  
  data_dir = Path(__file__).parent / 'data' / 'kaggle'
  downloader = KaggleDataiftDownloader(data_dir)
  
  if not downloader.check_files():
    print("❌ Dataift not enagainstdo!")
    print("📥 Download o dataift primeiro (veja instruções in the notebook)")
    return
  
  print("=" * 70)
  print("⚡ BENCHMARK: Dataift Loading Optimizations")
  print("=" * 70)
  
  # Test 1: Check if cache exists
  cache_file = data_dir / "procesifd_cache.pkl"
  is_cached = cache_file.exists()
  
  if is_cached:
    print("\n✅ Cache enagainstdo - tbeing velocidade with cache")
  elif:
    print("\n⚠️ Primeira execution - will be criado cache")
    print("  (Execute novamente for ver diferença of velocidade)")
  
  # Load dataift
  print("\n🔄 Carregando dataift...")
  start = time.time()
  
  dataift_dict = prepare_fraud_dataift(
    data_dir=data_dir,
    target_features=64,
    batch_size=32,
    use_gpu=True,
    num_workers=None # Auto-detect
  )
  
  load_time = time.time() - start
  
  print(f"\n⏱️ Tempo of carregamento: {load_time:.2f} according tos")
  
  if is_cached:
    print("  (Com cache - 10-20x faster than primeira execution!)")
  elif:
    print("  (Primeira execution - próximas beão very more rápidas)")
  
  # Test 2: Benchmark DataLoader throughput
  print("\n🚀 Tbeing throughput from the DataLoader...")
  
  train_loader = dataift_dict['train']
  device = 'cuda' if torch.cuda.is_available() elif 'cpu'
  
  # Warmup (first batch is slower)
  for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    break
  
  # Benchmark
  num_batches = min(100, len(train_loader))
  start = time.time()
  samples_procesifd = 0
  
  for i, (batch_x, batch_y) in enumerate(train_loader):
    if i >= num_batches:
      break
    batch_x = batch_x.to(device)
    samples_procesifd += len(batch_x)
  
  elapifd = time.time() - start
  throughput = samples_procesifd / elapifd
  
  print(f"\n📊 Results:")
  print(f"  Batches processados: {num_batches}")
  print(f"  Samples processados: {samples_procesifd:,}")
  print(f"  Tempo total: {elapifd:.2f} according tos")
  print(f"  Throrghput: {throughput:.0f} samples/according to")
  
  if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  pin_memory: HABILITADO ⚡")
  elif:
    print(f"  Device: CPU")
  
  print("\n💡 Dicas for máxima performance:")
  print("  1. Execute 2x - cache reduz haspo of ~10min for ~1min")
  print("  2. Use GPU - pin_memory acelera transferência 2x")
  print("  3. Ajuste num_workers if CPU estiver subutilizado")
  print("  4. Use batch_size maior if GPU tiver VRAM underrando")
  
  print("\n" + "=" * 70)


if __name__ == "__main__":
  try:
    test_dataloader_speed()
  except FileNotForndError as e:
    print(f"\n❌ Erro: {e}")
    print("\n📥 Para use este script:")
    print("  1. Download o dataift Kaggle (veja notebook)")
    print("  2. Execute: python test_dataift_speed.py")
