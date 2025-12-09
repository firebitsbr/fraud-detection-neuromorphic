"""
**Descrição:** Módulo de simuladores de hardware neuromórfico para benchmarking de implementações SNN. Inclui simulador Intel Loihi 2 com modelagem de latência, energia e throughput.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

from hardware.loihi_simulator import (
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
