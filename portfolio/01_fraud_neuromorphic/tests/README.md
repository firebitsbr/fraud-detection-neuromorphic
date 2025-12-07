# Test Suite

Este diretório contém testes abrangentes para o sistema de detecção de fraude neuromórfico.

## Estrutura de Testes

### `test_encoders.py`
Testes unitários para todos os encoders de spike:
- Rate Encoder
- Temporal Encoder  
- Population Encoder
- Latency Encoder
- Transaction Encoder
- Advanced Encoders (Adaptive, Burst, Phase, Rank Order, Ensemble)

### `test_integration.py`
Testes de integração end-to-end:
- Pipeline completo de detecção de fraude
- Integração entre componentes
- Testes de performance (latência e throughput)

### `run_tests.py`
Script principal para executar todos os testes.

## Como Executar

### Executar Todos os Testes
```bash
cd tests
python run_tests.py
```

### Executar Testes Específicos
```bash
# Apenas testes de encoders
python -m unittest test_encoders

# Apenas testes de integração
python -m unittest test_integration

# Teste específico
python -m unittest test_encoders.TestRateEncoder
```

## Cobertura de Testes

Os testes cobrem:

✅ **Encoders de Spike**
- Validação de geração de spike trains
- Verificação de propriedades temporais
- Comparação entre estratégias de encoding

✅ **Pipeline de Detecção**
- Extração de features
- Pré-processamento
- Treinamento e predição
- Métricas de avaliação

✅ **Integração de Componentes**
- Encoder → Model
- Data → Pipeline → Predictions
- Batch processing

✅ **Performance**
- Latência de predição
- Throughput de processamento
- Uso de memória

## Requisitos

```bash
pip install numpy pandas scikit-learn brian2
```

## Resultados Esperados

Todos os testes devem passar com sucesso:
- ✅ Unit tests para encoders
- ✅ Integration tests para pipeline
- ⚠️ Alguns testes podem ser ignorados (skipped) se Brian2 não estiver disponível

## Contribuindo

Ao adicionar novos módulos, crie testes correspondentes:

1. Criar arquivo `test_<module>.py`
2. Adicionar classes de teste herdando de `unittest.TestCase`
3. Implementar métodos `test_*` para cada funcionalidade
4. Executar `run_tests.py` para validar

---

**Autor:** Mauro Risonho de Paula Assumpção  
**Data:** December 5, 2025  
**License:** MIT
