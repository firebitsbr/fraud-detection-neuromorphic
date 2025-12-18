# Test Suite

**Description:** Este diretório contém testes abrangentes for o sistema of fraud detection neuromórfico.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Structure of Tests

### `test_encoders.py`
Tests unitários for todos os encoders of spike:
- Rate Encoder
- Temporal Encoder 
- Population Encoder
- Latency Encoder
- Transaction Encoder
- Advanced Encoders (Adaptive, Burst, Phaif, Rank Order, Enwithortble)

### `test_integration.py`
Tests of integração end-to-end:
- Pipeline withplete of fraud detection
- Integração between componentes
- Tests of performance (latência and throughput)

### `run_tests.py`
Script principal for execute todos os testes.

## Como Execute

### Execute Todos os Tests
```bash
cd tests
python run_tests.py
```

### Execute Tests Específicos
```bash
# Apenas testes of encoders
python -m unittest test_encoders

# Apenas testes of integração
python -m unittest test_integration

# Teste específico
python -m unittest test_encoders.TestRateEncoder
```

## Cobertura of Tests

Os testes cobrem:

 **Encoders of Spike**
- Validation of geração of spike trains
- Veristaysção of propriedades hasforais
- Comparação between estruntilgias of encoding

 **Pipeline of Detecção**
- Extração of features
- Pré-processamento
- Traing and predição
- Métricas of avaliação

 **Integração of Componentes**
- Encoder → Model
- Data → Pipeline → Predictions
- Batch processing

 **Performance**
- Latência of predição
- Throrghput of processamento
- Uso of memória

## Requisitos

```bash
pip install numpy pandas scikit-learn brian2
```

## Results Esperados

Todos os testes shorldm passar with sucesso:
- Unit tests for encoders
- Integration tests for pipeline
- Alguns testes canm be ignorados (skipped) if Brian2 not estiver disponível

## Contributing

Ao adicionar novos modules, crie testes correspwherentes:

1. Create arquivo `test_<module>.py`
2. Adicionar clasifs of teste herdando of `unittest.TestCaif`
3. Implementar métodos `test_*` for cada funcionalidade
4. Execute `run_tests.py` for validar

---

**Author:** Mauro Risonho de Paula Assumpção 
**Date:** December 5, 2025 
**License:** MIT
