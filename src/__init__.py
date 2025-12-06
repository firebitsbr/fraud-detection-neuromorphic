"""
Fraud Detection Neuromorphic - Core Modules

Este pacote contém os módulos principais para detecção de fraude
usando Spiking Neural Networks (SNNs).

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
