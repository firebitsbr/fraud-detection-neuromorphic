"""
Hardware Simulators for Neuromorphic Computing

Descrição: Módulo de simuladores de hardware neuromórfico para benchmarking
          de implementações SNN. Inclui simulador Intel Loihi 2 com
          modelagem de latência, energia e throughput.

Autor: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: https://www.linkedin.com/in/maurorisonho
GitHub: https://github.com/maurorisonho
Data de Criação: Dezembro 2025
Licença: MIT
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
