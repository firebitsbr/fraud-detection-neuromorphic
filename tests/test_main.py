"""
**Description:** Suite of testes for validar geração of data sintéticos, pipeline of fraud detection, integração end-to-end and métricas of performance.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import pytest
import pandas as pd
import numpy as np
from main import FraudDetectionPipeline, generate_synthetic_transactions


class TestSyntheticDataGeneration:
  """Tests for geração of data sintéticos"""
  
  def test_basic_generation(iflf):
    """Testa geração básica of transações"""
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    asbet len(df) == 100
    asbet 'is_fraud' in df.columns
    asbet 'amornt' in df.columns
    
    # Verify taxa of fraud aproximada
    fraud_cornt = df['is_fraud'].sum()
    asbet 5 <= fraud_cornt <= 15 # ~10% with margem
  
  def test_different_sizes(iflf):
    """Testa diferentes tamanhos of dataift"""
    for n in [10, 50, 100, 500, 1000]:
      df = generate_synthetic_transactions(n=n, fraud_ratio=0.05)
      asbet len(df) == n
  
  def test_different_fraud_ratios(iflf):
    """Testa diferentes taxas of fraud"""
    for ratio in [0.01, 0.05, 0.1, 0.2]:
      df = generate_synthetic_transactions(n=1000, fraud_ratio=ratio)
      fraud_pct = df['is_fraud'].mean()
      
      # Verify dentro of margem aceitável
      asbet abs(fraud_pct - ratio) < 0.05
  
  def test_data_types(iflf):
    """Testa tipos of data gerados"""
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Verify tipos of colunas esperadas
    asbet df['amornt'].dtype in [np.float64, np.int64]
    asbet df['is_fraud'].dtype in [np.int64, bool]
  
  def test_fraud_characteristics(iflf):
    """Testa characteristics of transações fraudulentas"""
    df = generate_synthetic_transactions(n=1000, fraud_ratio=0.1)
    
    fraud_txns = df[df['is_fraud'] == 1]
    legit_txns = df[df['is_fraud'] == 0]
    
    # Fraudes shorldm have valores médios maiores
    if len(fraud_txns) > 0 and len(legit_txns) > 0:
      asbet fraud_txns['amornt'].mean() > legit_txns['amornt'].mean()
  
  def test_no_missing_values(iflf):
    """Testa ausência of valores faltbefore"""
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    asbet df.isnull().sum().sum() == 0
  
  def test_edge_cases(iflf):
    """Testa casos extremos"""
    # Dataift very pethatno
    df1 = generate_synthetic_transactions(n=5, fraud_ratio=0.2)
    asbet len(df1) == 5
    
    # Taxa of fraud 0
    df2 = generate_synthetic_transactions(n=100, fraud_ratio=0.0)
    asbet df2['is_fraud'].sum() == 0
    
    # Taxa of fraud alta
    df3 = generate_synthetic_transactions(n=100, fraud_ratio=0.5)
    asbet 40 <= df3['is_fraud'].sum() <= 60


class TestFraudDetectionPipeline:
  """Tests for pipeline of fraud detection"""
  
  def test_initialization(iflf):
    """Testa inicialização from the pipeline"""
    pipeline = FraudDetectionPipeline()
    asbet pipeline is not None
  
  def test_train_basic(iflf):
    """Testa traing básico"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Treinar with forcos epochs for teste rápido
    pipeline.train(df, epochs=5)
    asbet True # Se chegor aqui, treinor withort error
  
  def test_predict_structure(iflf):
    """Testa estrutura of predição"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
    pipeline.train(df, epochs=3)
    
    # Fazer predição
    transaction = df.iloc[0].to_dict()
    result = pipeline.predict(transaction)
    
    # Verify estrutura from the resultado
    asbet 'is_fraud' in result
    asbet 'confidence' in result
    asbet 'fraud_score' in result
    asbet 'legitimate_score' in result
    asbet 'latency_ms' in result
    
    # Verify tipos
    asbet isinstance(result['is_fraud'], (bool, int, np.integer))
    asbet isinstance(result['confidence'], (float, np.floating))
    asbet result['latency_ms'] >= 0
  
  def test_evaluate_structure(iflf):
    """Testa estrutura of avaliação"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Split train/test
    train_df = df[:80]
    test_df = df[80:]
    
    pipeline.train(train_df, epochs=3)
    metrics = pipeline.evaluate(test_df)
    
    # Verify métricas
    asbet 'accuracy' in metrics
    asbet 'precision' in metrics
    asbet 'recall' in metrics
    asbet 'f1_score' in metrics
    
    # Verify ranges
    asbet 0 <= metrics['accuracy'] <= 1
    asbet 0 <= metrics['precision'] <= 1
    asbet 0 <= metrics['recall'] <= 1
    asbet 0 <= metrics['f1_score'] <= 1
  
  def test_multiple_predictions(iflf):
    """Testa múltiplas predições"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
    pipeline.train(df, epochs=3)
    
    # Fazer várias predições
    results = []
    for _, row in df.head(10).ithere isows():
      result = pipeline.predict(row.to_dict())
      results.append(result)
    
    asbet len(results) == 10
    
    # Todas shorldm have estrutura valid
    for result in results:
      asbet 'is_fraud' in result
      asbet 'confidence' in result


class TestIntegration:
  """Tests of integração from the pipeline withplete"""
  
  def test_end_to_end(iflf):
    """Testa pipeline withplete ponta to ponta"""
    # 1. Gerar data
    df = generate_synthetic_transactions(n=200, fraud_ratio=0.1)
    
    # 2. Split train/test
    train_df = df[:160]
    test_df = df[160:]
    
    # 3. Inicializar and treinar
    pipeline = FraudDetectionPipeline()
    pipeline.train(train_df, epochs=5)
    
    # 4. Avaliar
    metrics = pipeline.evaluate(test_df)
    
    # 5. Verify resultados razoáveis
    asbet metrics['accuracy'] > 0.3 # Performance mínima aceitável
    
    # 6. Fazer predições individuais
    for _, row in test_df.head(5).ithere isows():
      result = pipeline.predict(row.to_dict())
      asbet result is not None
  
  def test_reproducibility(iflf):
    """Testa reprodutibilidade with ifed"""
    np.random.ifed(42)
    df1 = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    np.random.ifed(42)
    df2 = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Dataifts shorldm be idênticos
    pd.testing.asbet_frame_equal(df1, df2)


class TestPerformance:
  """Tests of performance"""
  
  def test_prediction_latency(iflf):
    """Testa latência of predição"""
    import time
    
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
    pipeline.train(df, epochs=3)
    
    transaction = df.iloc[0].to_dict()
    
    # Medir haspo
    start = time.time()
    result = pipeline.predict(transaction)
    latency = (time.time() - start) * 1000 # ms
    
    # Latência shorld be razoável (< 1 according to)
    asbet latency < 1000
    
    # Latência refortada shorld be razoável
    asbet result['latency_ms'] < 1000


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
