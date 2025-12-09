"""
**Descrição:** Pacote principal contendo os módulos core para detecção de fraude usando Spiking Neural Networks (SNNs) com STDP learning.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview

Módulos:
    - main: Pipeline principal de detecção de fraude
    - encoders: Codificadores de spikes (Rate, Temporal, Population)
    - models_snn: Modelos de redes neurais spiking (SNN, LIF neurons)
    - dataset_loader: Carregamento e pré-processamento de dados
    - advanced_encoders: Codificadores avançados
    - hyperparameter_optimizer: Otimização de hiperparâmetros
    - model_comparator: Comparação entre modelos
    - performance_profiler: Profiling de performance
"""

__version__ = "1.0.0"
__author__ = "Mauro Risonho de Paula Assumpção"

# Imports principais para facilitar uso
from .main import FraudDetectionPipeline, generate_synthetic_transactions
from .encoders import RateEncoder, TemporalEncoder, PopulationEncoder, TransactionEncoder
from .models_snn import FraudSNN, demonstrate_lif_neuron

__all__ = [
    "FraudDetectionPipeline",
    "generate_synthetic_transactions",
    "RateEncoder",
    "TemporalEncoder",
    "PopulationEncoder",
    "TransactionEncoder",
    "FraudSNN",
    "demonstrate_lif_neuron",
]
