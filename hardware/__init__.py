"""
**Description:** Módulo of simulatores of neuromorphic hardware for benchmarking of implementations SNN. Inclui simulator Intel Loihi 2 with modelagem of latency, energia and throughput.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
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
