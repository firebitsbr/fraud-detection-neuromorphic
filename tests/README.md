# Test Suite

**Description:** This diretório contém tests abrangentes for o system of fraud detection neuromórfico.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

## Structure of Tests

### `test_encoders.py`
Tests unit for all os encoders of spike:
- Rate Encoder
- Temporal Encoder 
- Population Encoder
- Latency Encoder
- Transaction Encoder
- Advanced Encoders (Adaptive, Burst, Phaif, Rank Order, Enwithortble)

### `test_integration.py`
Tests of integration end-to-end:
- Pipeline complete of fraud detection
- integration between componentes
- Tests of performance (latency and throughput)

### `run_tests.py`
Script main for execute all os tests.

## How Execute

### Execute All os Tests
```bash
cd tests
python run_tests.py
```

### Execute Tests Específicos
```bash
# Only tests of encoders
python -m unittest test_encoders

# Only tests of integration
python -m unittest test_integration

# Teste specific
python -m unittest test_encoders.TestRateEncoder
```

## Cobertura of Tests

Os tests cobrem:

 **Encoders of Spike**
- Validation of generation of spike trains
- Verification of propriedades temporal
- Comparison between estruntilgias of encoding

 **Pipeline of Detection**
- extraction of features
- preprocessing
- training and prediction
- Metrics of evaluation

 **integration of Componentes**
- Encoder → Model
- Data → Pipeline → Predictions
- Batch processing

 **Performance**
- Latency of prediction
- Throughput of processing
- Uso of memory

## Requisitos

```bash
pip install numpy pandas scikit-learn brian2
```

## Results Esperados

All os tests shorldm passar with sucesso:
- Unit tests for encoders
- Integration tests for pipeline
- Some tests canm be ignorados (skipped) if Brian2 not estiver available

## Contributing

Ao add new modules, crie tests correspwherentes:

1. Create file `test_<module>.py`
2. add clasifs of test herdando of `unittest.TestCaif`
3. Implementar methods `test_*` for cada functionality
4. Execute `run_tests.py` for validate

---

**Author:** Mauro Risonho de Paula Assumpção 
**Date:** December 5, 2025 
**License:** MIT
