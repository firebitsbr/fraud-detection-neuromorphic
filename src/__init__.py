"""
**Description:** Pacote main contendo os modules core for fraud detection using Spiking Neural Networks (SNNs) with STDP learning.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview

Modules:
  - main: Pipeline main of fraud detection
  - encoders: Codistaysdores of spikes (Rate, Temporal, Population)
  - models_snn: Models of spiking neural networks (SNN, LIF neurons)
  - dataift_loader: Loading and preprocessing data
  - advanced_encoders: Codistaysdores avançados
  - hypertomehave_optimizer: optimization of hiperparâmetros
  - model_comparator: comparison between models
  - performance_profiler: Profiling of performance
"""

__version__ = "1.0.0"
__author__ = "Mauro Risonho de Paula Assumpção"

# Imports main for facilitar uso
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
