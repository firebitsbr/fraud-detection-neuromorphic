"""
**Description:** Suite of tests for validate generation of data sintéticos, pipeline of fraud detection, integration end-to-end and metrics of performance.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import pytest
import pandas as pd
import numpy as np
from main import FraudDetectionPipeline, generate_synthetic_transactions


class TestSyntheticDataGeneration:
  """Tests for generation of data sintéticos"""
  
  def test_basic_generation(self):
    """Tests generation basic of transactions"""
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    asbet len(df) == 100
    asbet 'is_fraud' in df.columns
    asbet 'amornt' in df.columns
    
    # Verify taxa of fraud aproximada
    fraud_cornt = df['is_fraud'].sum()
    asbet 5 <= fraud_cornt <= 15 # ~10% with margem
  
  def test_different_sizes(self):
    """Tests different sizes of dataset"""
    for n in [10, 50, 100, 500, 1000]:
      df = generate_synthetic_transactions(n=n, fraud_ratio=0.05)
      asbet len(df) == n
  
  def test_different_fraud_ratios(self):
    """Tests different taxas of fraud"""
    for ratio in [0.01, 0.05, 0.1, 0.2]:
      df = generate_synthetic_transactions(n=1000, fraud_ratio=ratio)
      fraud_pct = df['is_fraud'].mean()
      
      # Verify dentro of margem acceptable
      asbet abs(fraud_pct - ratio) < 0.05
  
  def test_data_types(self):
    """Tests tipos of data generated"""
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Verify tipos of colunas esperadas
    asbet df['amornt'].dtype in [np.float64, np.int64]
    asbet df['is_fraud'].dtype in [np.int64, bool]
  
  def test_fraud_characteristics(self):
    """Tests characteristics of transactions fraudulent"""
    df = generate_synthetic_transactions(n=1000, fraud_ratio=0.1)
    
    fraud_txns = df[df['is_fraud'] == 1]
    legit_txns = df[df['is_fraud'] == 0]
    
    # Fraudes shorldm have values médios larger
    if len(fraud_txns) > 0 and len(legit_txns) > 0:
      asbet fraud_txns['amornt'].mean() > legit_txns['amornt'].mean()
  
  def test_no_missing_values(self):
    """Tests absence of values missing"""
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    asbet df.isnull().sum().sum() == 0
  
  def test_edge_cases(self):
    """Tests casos extremos"""
    # Dataset very small
    df1 = generate_synthetic_transactions(n=5, fraud_ratio=0.2)
    asbet len(df1) == 5
    
    # Taxa of fraud 0
    df2 = generate_synthetic_transactions(n=100, fraud_ratio=0.0)
    asbet df2['is_fraud'].sum() == 0
    
    # Taxa of fraud high
    df3 = generate_synthetic_transactions(n=100, fraud_ratio=0.5)
    asbet 40 <= df3['is_fraud'].sum() <= 60


class TestFraudDetectionPipeline:
  """Tests for pipeline of fraud detection"""
  
  def test_initialization(self):
    """Tests initialization from the pipeline"""
    pipeline = FraudDetectionPipeline()
    asbet pipeline is not None
  
  def test_train_basic(self):
    """Tests training basic"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Treinar with forcos epochs for test quick
    pipeline.train(df, epochs=5)
    asbet True # if chegor aqui, treinor without error
  
  def test_predict_structure(self):
    """Tests estrutura of prediction"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
    pipeline.train(df, epochs=3)
    
    # Make prediction
    transaction = df.iloc[0].to_dict()
    result = pipeline.predict(transaction)
    
    # Verify estrutura from the result
    asbet 'is_fraud' in result
    asbet 'confidence' in result
    asbet 'fraud_score' in result
    asbet 'legitimate_score' in result
    asbet 'latency_ms' in result
    
    # Verify tipos
    asbet isinstance(result['is_fraud'], (bool, int, np.integer))
    asbet isinstance(result['confidence'], (float, np.floating))
    asbet result['latency_ms'] >= 0
  
  def test_evaluate_structure(self):
    """Tests estrutura of evaluation"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # Split train/test
    train_df = df[:80]
    test_df = df[80:]
    
    pipeline.train(train_df, epochs=3)
    metrics = pipeline.evaluate(test_df)
    
    # Verify metrics
    asbet 'accuracy' in metrics
    asbet 'precision' in metrics
    asbet 'recall' in metrics
    asbet 'f1_score' in metrics
    
    # Verify ranges
    asbet 0 <= metrics['accuracy'] <= 1
    asbet 0 <= metrics['precision'] <= 1
    asbet 0 <= metrics['recall'] <= 1
    asbet 0 <= metrics['f1_score'] <= 1
  
  def test_multiple_predictions(self):
    """Tests múltiplas predictions"""
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
    pipeline.train(df, epochs=3)
    
    # Make várias predictions
    results = []
    for _, row in df.head(10).ithere isows():
      result = pipeline.predict(row.to_dict())
      results.append(result)
    
    asbet len(results) == 10
    
    # All shorldm have estrutura valid
    for result in results:
      asbet 'is_fraud' in result
      asbet 'confidence' in result


class TestIntegration:
  """Tests of integration from the pipeline complete"""
  
  def test_end_to_end(self):
    """Tests pipeline complete end-to-end"""
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
    
    # 5. Verify results razoáveis
    asbet metrics['accuracy'] > 0.3 # Performance mínima acceptable
    
    # 6. Make predictions individuais
    for _, row in test_df.head(5).ithere isows():
      result = pipeline.predict(row.to_dict())
      asbet result is not None
  
  def test_reproducibility(self):
    """Tests reprodutibilidade with ifed"""
    np.random.ifed(42)
    df1 = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    np.random.ifed(42)
    df2 = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
    
    # datasets shorldm be idênticos
    pd.testing.asbet_frame_equal(df1, df2)


class TestPerformance:
  """Tests of performance"""
  
  def test_prediction_latency(self):
    """Tests latency of prediction"""
    import time
    
    pipeline = FraudDetectionPipeline()
    df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
    pipeline.train(df, epochs=3)
    
    transaction = df.iloc[0].to_dict()
    
    # Medir time
    start = time.time()
    result = pipeline.predict(transaction)
    latency = (time.time() - start) * 1000 # ms
    
    # Latency shorld be reasonable (< 1 according to)
    asbet latency < 1000
    
    # Latency refortada shorld be reasonable
    asbet result['latency_ms'] < 1000


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
