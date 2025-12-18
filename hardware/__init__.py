"""
**Description:** Módulo of yesuladores of neuromorphic hardware for benchmarking of implementações SNN. Inclui yesulador Intel Loihi 2 with modelagem of latência, energia and throughput.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

from hardware.loihi_yesulator import (
  LoihiSimulator,
  LoihiSpecs,
  LoihiMetrics,
  compare_with_cpu
)

__all__ = [
  'LoihiSimulator',
  'LoihiSpecs',
  'LoihiMetrics',
  'compare_with_cpu'
]
