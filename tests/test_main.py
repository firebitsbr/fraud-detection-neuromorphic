"""
**Descrição:** Suite de testes para validar geração de dados sintéticos, pipeline de detecção de fraude, integração end-to-end e métricas de performance.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import pytest
import pandas as pd
import numpy as np
from main import FraudDetectionPipeline, generate_synthetic_transactions


class TestSyntheticDataGeneration:
    """Testes para geração de dados sintéticos"""
    
    def test_basic_generation(self):
        """Testa geração básica de transações"""
        df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        
        assert len(df) == 100
        assert 'is_fraud' in df.columns
        assert 'amount' in df.columns
        
        # Verificar taxa de fraude aproximada
        fraud_count = df['is_fraud'].sum()
        assert 5 <= fraud_count <= 15  # ~10% com margem
    
    def test_different_sizes(self):
        """Testa diferentes tamanhos de dataset"""
        for n in [10, 50, 100, 500, 1000]:
            df = generate_synthetic_transactions(n=n, fraud_ratio=0.05)
            assert len(df) == n
    
    def test_different_fraud_ratios(self):
        """Testa diferentes taxas de fraude"""
        for ratio in [0.01, 0.05, 0.1, 0.2]:
            df = generate_synthetic_transactions(n=1000, fraud_ratio=ratio)
            fraud_pct = df['is_fraud'].mean()
            
            # Verificar dentro de margem aceitável
            assert abs(fraud_pct - ratio) < 0.05
    
    def test_data_types(self):
        """Testa tipos de dados gerados"""
        df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        
        # Verificar tipos de colunas esperadas
        assert df['amount'].dtype in [np.float64, np.int64]
        assert df['is_fraud'].dtype in [np.int64, bool]
    
    def test_fraud_characteristics(self):
        """Testa características de transações fraudulentas"""
        df = generate_synthetic_transactions(n=1000, fraud_ratio=0.1)
        
        fraud_txns = df[df['is_fraud'] == 1]
        legit_txns = df[df['is_fraud'] == 0]
        
        # Fraudes devem ter valores médios maiores
        if len(fraud_txns) > 0 and len(legit_txns) > 0:
            assert fraud_txns['amount'].mean() > legit_txns['amount'].mean()
    
    def test_no_missing_values(self):
        """Testa ausência de valores faltantes"""
        df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        assert df.isnull().sum().sum() == 0
    
    def test_edge_cases(self):
        """Testa casos extremos"""
        # Dataset muito pequeno
        df1 = generate_synthetic_transactions(n=5, fraud_ratio=0.2)
        assert len(df1) == 5
        
        # Taxa de fraude 0
        df2 = generate_synthetic_transactions(n=100, fraud_ratio=0.0)
        assert df2['is_fraud'].sum() == 0
        
        # Taxa de fraude alta
        df3 = generate_synthetic_transactions(n=100, fraud_ratio=0.5)
        assert 40 <= df3['is_fraud'].sum() <= 60


class TestFraudDetectionPipeline:
    """Testes para pipeline de detecção de fraude"""
    
    def test_initialization(self):
        """Testa inicialização do pipeline"""
        pipeline = FraudDetectionPipeline()
        assert pipeline is not None
    
    def test_train_basic(self):
        """Testa treinamento básico"""
        pipeline = FraudDetectionPipeline()
        df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        
        # Treinar com poucos epochs para teste rápido
        pipeline.train(df, epochs=5)
        assert True  # Se chegou aqui, treinou sem erro
    
    def test_predict_structure(self):
        """Testa estrutura de predição"""
        pipeline = FraudDetectionPipeline()
        df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
        pipeline.train(df, epochs=3)
        
        # Fazer predição
        transaction = df.iloc[0].to_dict()
        result = pipeline.predict(transaction)
        
        # Verificar estrutura do resultado
        assert 'is_fraud' in result
        assert 'confidence' in result
        assert 'fraud_score' in result
        assert 'legitimate_score' in result
        assert 'latency_ms' in result
        
        # Verificar tipos
        assert isinstance(result['is_fraud'], (bool, int, np.integer))
        assert isinstance(result['confidence'], (float, np.floating))
        assert result['latency_ms'] >= 0
    
    def test_evaluate_structure(self):
        """Testa estrutura de avaliação"""
        pipeline = FraudDetectionPipeline()
        df = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        
        # Split train/test
        train_df = df[:80]
        test_df = df[80:]
        
        pipeline.train(train_df, epochs=3)
        metrics = pipeline.evaluate(test_df)
        
        # Verificar métricas
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Verificar ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_multiple_predictions(self):
        """Testa múltiplas predições"""
        pipeline = FraudDetectionPipeline()
        df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
        pipeline.train(df, epochs=3)
        
        # Fazer várias predições
        results = []
        for _, row in df.head(10).iterrows():
            result = pipeline.predict(row.to_dict())
            results.append(result)
        
        assert len(results) == 10
        
        # Todas devem ter estrutura válida
        for result in results:
            assert 'is_fraud' in result
            assert 'confidence' in result


class TestIntegration:
    """Testes de integração do pipeline completo"""
    
    def test_end_to_end(self):
        """Testa pipeline completo ponta a ponta"""
        # 1. Gerar dados
        df = generate_synthetic_transactions(n=200, fraud_ratio=0.1)
        
        # 2. Split train/test
        train_df = df[:160]
        test_df = df[160:]
        
        # 3. Inicializar e treinar
        pipeline = FraudDetectionPipeline()
        pipeline.train(train_df, epochs=5)
        
        # 4. Avaliar
        metrics = pipeline.evaluate(test_df)
        
        # 5. Verificar resultados razoáveis
        assert metrics['accuracy'] > 0.3  # Performance mínima aceitável
        
        # 6. Fazer predições individuais
        for _, row in test_df.head(5).iterrows():
            result = pipeline.predict(row.to_dict())
            assert result is not None
    
    def test_reproducibility(self):
        """Testa reprodutibilidade com seed"""
        np.random.seed(42)
        df1 = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        
        np.random.seed(42)
        df2 = generate_synthetic_transactions(n=100, fraud_ratio=0.1)
        
        # Datasets devem ser idênticos
        pd.testing.assert_frame_equal(df1, df2)


class TestPerformance:
    """Testes de performance"""
    
    def test_prediction_latency(self):
        """Testa latência de predição"""
        import time
        
        pipeline = FraudDetectionPipeline()
        df = generate_synthetic_transactions(n=50, fraud_ratio=0.1)
        pipeline.train(df, epochs=3)
        
        transaction = df.iloc[0].to_dict()
        
        # Medir tempo
        start = time.time()
        result = pipeline.predict(transaction)
        latency = (time.time() - start) * 1000  # ms
        
        # Latência deve ser razoável (< 1 segundo)
        assert latency < 1000
        
        # Latência reportada deve ser razoável
        assert result['latency_ms'] < 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
