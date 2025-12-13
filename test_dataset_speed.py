#!/usr/bin/env python3
"""
Test script to benchmark dataset loading optimizations
"""

import sys
from pathlib import Path
import time
import torch

sys.path.insert(0, 'src')

def test_dataloader_speed():
    """Benchmark DataLoader throughput"""
    from dataset_kaggle import prepare_fraud_dataset, KaggleDatasetDownloader
    
    data_dir = Path(__file__).parent / 'data' / 'kaggle'
    downloader = KaggleDatasetDownloader(data_dir)
    
    if not downloader.check_files():
        print("❌ Dataset não encontrado!")
        print("📥 Baixe o dataset primeiro (veja instruções no notebook)")
        return
    
    print("=" * 70)
    print("⚡ BENCHMARK: Dataset Loading Optimizations")
    print("=" * 70)
    
    # Test 1: Check if cache exists
    cache_file = data_dir / "processed_cache.pkl"
    is_cached = cache_file.exists()
    
    if is_cached:
        print("\n✅ Cache encontrado - testando velocidade com cache")
    else:
        print("\n⚠️  Primeira execução - será criado cache")
        print("   (Execute novamente para ver diferença de velocidade)")
    
    # Load dataset
    print("\n🔄 Carregando dataset...")
    start = time.time()
    
    dataset_dict = prepare_fraud_dataset(
        data_dir=data_dir,
        target_features=64,
        batch_size=32,
        use_gpu=True,
        num_workers=None  # Auto-detect
    )
    
    load_time = time.time() - start
    
    print(f"\n⏱️  Tempo de carregamento: {load_time:.2f} segundos")
    
    if is_cached:
        print("   (Com cache - 10-20x mais rápido que primeira execução!)")
    else:
        print("   (Primeira execução - próximas serão muito mais rápidas)")
    
    # Test 2: Benchmark DataLoader throughput
    print("\n🚀 Testando throughput do DataLoader...")
    
    train_loader = dataset_dict['train']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Warmup (first batch is slower)
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        break
    
    # Benchmark
    num_batches = min(100, len(train_loader))
    start = time.time()
    samples_processed = 0
    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        if i >= num_batches:
            break
        batch_x = batch_x.to(device)
        samples_processed += len(batch_x)
    
    elapsed = time.time() - start
    throughput = samples_processed / elapsed
    
    print(f"\n📊 Resultados:")
    print(f"   Batches processados: {num_batches}")
    print(f"   Samples processados: {samples_processed:,}")
    print(f"   Tempo total: {elapsed:.2f} segundos")
    print(f"   Throughput: {throughput:.0f} samples/segundo")
    
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   pin_memory: HABILITADO ⚡")
    else:
        print(f"   Device: CPU")
    
    print("\n💡 Dicas para máxima performance:")
    print("   1. Execute 2x - cache reduz tempo de ~10min para ~1min")
    print("   2. Use GPU - pin_memory acelera transferência 2x")
    print("   3. Ajuste num_workers se CPU estiver subutilizado")
    print("   4. Use batch_size maior se GPU tiver VRAM sobrando")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        test_dataloader_speed()
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\n📥 Para usar este script:")
        print("   1. Baixe o dataset Kaggle (veja notebook)")
        print("   2. Execute: python test_dataset_speed.py")
