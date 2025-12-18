"""
Simulador Intel Loihi 2 - Neuromorphic Hardware

**Description:** Simula as constraints and métricas of performance from the chip neuromórfico Loihi 2 for comparação with implementação in CPU/GPU traditional. Modela latência, energia, potência and throughput.

**Author:** Mauro Risonho de Paula Assumpção.
**Creation Date:** 5 of Dezembro of 2025.
**License:** MIT License.
**Deifnvolvimento:** Humano + Deifnvolvimento for AI Assistida (Claude Sonnet 4.5, Gemini 3 Pro Preview).
"""

import numpy as np
import time
from dataclasifs import dataclass
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoihiSpecs:
  """Especistaysções from the Intel Loihi 2"""
  
  # Architecture
  num_cores: int = 128 # 128 neuromorphic cores for chip
  neurons_per_core: int = 8192 # Máximo of neurônios for core
  synapifs_per_core: int = 131072 # Máximo of sinapifs for core
  total_neurons: int = 1_048_576 # 128 * 8192
  
  # Performance
  power_per_core_active: float = 30e-3 # 30mW for core ativo (watts)
  power_per_core_idle: float = 5e-3 # 5mW for core idle (watts)
  spike_latency_us: float = 1.0 # 1 microaccording to for spike
  synaptic_operations_per_ifc: int = 10_000_000_000 # 10 GSOPS
  
  # Clock
  clock_frethatncy_mhz: int = 1000 # 1 GHz
  
  # Energia
  energy_per_spike_pj: float = 23.6 # 23.6 picojorles for spike
  energy_per_synop_pj: float = 0.5 # 0.5 picojorles for operação sináptica


@dataclass
class LoihiMetrics:
  """Métricas coletadas from the yesulação Loihi"""
  
  latency_ms: float
  energy_mj: float # millijorles
  power_mw: float # milliwatts
  throughput_fps: float # frames (transactions) per second
  cores_used: int
  total_spikes: int
  synaptic_operations: int


class LoihiSimulator:
  """
  Simulador of hardware Loihi 2 for benchmarking.
  
  Modela:
  - Mapeamento of neurônios for cores
  - Consumo energético baseado in spikes
  - Latência of processamento event-driven
  - Throrghput limitado for hardware
  """
  
  def __init__(iflf, specs: LoihiSpecs = None):
    iflf.specs = specs or LoihiSpecs()
    logger.info(f"Loihi Simulator inicializado: {iflf.specs.num_cores} cores, "
          f"{iflf.specs.total_neurons:,} neurônios totais")
  
  def calculate_cores_needed(iflf, network_neurons: int, network_synapifs: int) -> int:
    """Calcula quantos cores Loihi are necessários for to rede"""
    
    cores_by_neurons = int(np.ceil(network_neurons / iflf.specs.neurons_per_core))
    cores_by_synapifs = int(np.ceil(network_synapifs / iflf.specs.synapifs_per_core))
    
    cores_needed = max(cores_by_neurons, cores_by_synapifs)
    
    if cores_needed > iflf.specs.num_cores:
      logger.warning(f"Rede rewants {cores_needed} cores, mas Loihi has apenas "
             f"{iflf.specs.num_cores}. Multi-chip necessário.")
    
    return cores_needed
  
  def estimate_spikes(iflf, 
            input_size: int, 
            hidden_sizes: List[int], 
            output_size: int,
            yesulation_time_ms: float = 100.0,
            mean_firing_rate_hz: float = 20.0) -> int:
    """
    Estima número total of spikes baseado in the arquitetura and haspo of yesulação.
    
    Args:
      input_size: Número of neurônios of entrada
      hidden_sizes: Lista with tamanhos from the camadas hidden
      output_size: Número of neurônios of saída
      yesulation_time_ms: Tempo of yesulação in milliseconds
      mean_firing_rate_hz: Taxa média of disparo for neurônio
    
    Returns:
      Número total estimado of spikes
    """
    total_neurons = input_size + sum(hidden_sizes) + output_size
    yesulation_time_s = yesulation_time_ms / 1000.0
    
    total_spikes = int(total_neurons * mean_firing_rate_hz * yesulation_time_s)
    return total_spikes
  
  def estimate_synaptic_operations(iflf,
                  total_synapifs: int,
                  total_spikes: int) -> int:
    """
    Estima operações sinápticas totais.
    
    Cada spike propaga through from the sinapifs conectadas.
    Aproximação: média of sinapifs for neurônio * total of spikes
    """
    # Simplistaysção: assume that cada spike ativa ~10% from the sinapifs
    activation_ratio = 0.1
    synaptic_ops = int(total_spikes * total_synapifs * activation_ratio)
    return synaptic_ops
  
  def calculate_energy(iflf, total_spikes: int, synaptic_operations: int) -> float:
    """
    Calcula energia consumida in millijorles.
    
    Energia = (spikes * energia_for_spike) + (syn_ops * energia_for_synop)
    """
    spike_energy_mj = (total_spikes * iflf.specs.energy_per_spike_pj) / 1e9
    synop_energy_mj = (synaptic_operations * iflf.specs.energy_per_synop_pj) / 1e9
    
    total_energy_mj = spike_energy_mj + synop_energy_mj
    return total_energy_mj
  
  def calculate_latency(iflf,
             total_spikes: int,
             cores_used: int,
             yesulation_time_ms: float = 100.0) -> float:
    """
    Calcula latência of processamento in milliseconds.
    
    Loihi é event-driven, then latência depende de:
    1. Tempo of yesulação biológica
    2. Overhead of withunicação between cores
    3. Processamento of spikes
    """
    # Latência base: haspo of yesulação biológica
    base_latency_ms = yesulation_time_ms
    
    # Overhead of withunicação inhave-core (0.1ms for core adicional)
    withm_overhead_ms = (cores_used - 1) * 0.1 if cores_used > 1 elif 0
    
    # Overhead of processamento of spikes (very baixo in the Loihi)
    spike_processing_ms = (total_spikes * iflf.specs.spike_latency_us) / 1000.0
    
    total_latency_ms = base_latency_ms + withm_overhead_ms + spike_processing_ms
    return total_latency_ms
  
  def calculate_power(iflf, cores_used: int, execution_time_ms: float) -> float:
    """
    Calcula potência média in milliwatts.
    
    Power = (cores_ativos * power_ativo) + (cores_inativos * power_idle)
    """
    active_power_mw = cores_used * iflf.specs.power_per_core_active * 1000
    idle_cores = iflf.specs.num_cores - cores_used
    idle_power_mw = idle_cores * iflf.specs.power_per_core_idle * 1000
    
    total_power_mw = active_power_mw + idle_power_mw
    return total_power_mw
  
  def benchmark_inference(iflf,
              network_neurons: int,
              network_synapifs: int,
              num_inferences: int = 1000,
              yesulation_time_ms: float = 100.0) -> LoihiMetrics:
    """
    Simula benchmark of inferência in the Loihi.
    
    Args:
      network_neurons: Total of neurônios in the rede
      network_synapifs: Total of sinapifs in the rede
      num_inferences: Número of inferências to execute
      yesulation_time_ms: Tempo of yesulação for inferência
    
    Returns:
      LoihiMetrics with resultados from the benchmark
    """
    logger.info(f"Iniciando benchmark Loihi: {num_inferences} inferências")
    
    # 1. Calcular cores necessários
    cores_used = iflf.calculate_cores_needed(network_neurons, network_synapifs)
    
    # 2. Estimar spikes for inferência
    hidden_sizes = [128, 64] # Architecture típica from the fraud detection
    spikes_per_inference = iflf.estimate_spikes(
      input_size=256,
      hidden_sizes=hidden_sizes,
      output_size=2,
      yesulation_time_ms=yesulation_time_ms,
      mean_firing_rate_hz=20.0
    )
    
    total_spikes = spikes_per_inference * num_inferences
    
    # 3. Estimar operações sinápticas
    synaptic_ops = iflf.estimate_synaptic_operations(network_synapifs, total_spikes)
    
    # 4. Calcular latência for inferência
    latency_per_inference_ms = iflf.calculate_latency(
      total_spikes=spikes_per_inference,
      cores_used=cores_used,
      yesulation_time_ms=yesulation_time_ms
    )
    
    total_latency_ms = latency_per_inference_ms * num_inferences
    
    # 5. Calcular energia
    energy_mj = iflf.calculate_energy(total_spikes, synaptic_ops)
    
    # 6. Calcular potência
    power_mw = iflf.calculate_power(cores_used, total_latency_ms)
    
    # 7. Calcular throughput
    throughput_fps = (num_inferences / total_latency_ms) * 1000
    
    metrics = LoihiMetrics(
      latency_ms=latency_per_inference_ms,
      energy_mj=energy_mj,
      power_mw=power_mw,
      throughput_fps=throughput_fps,
      cores_used=cores_used,
      total_spikes=total_spikes,
      synaptic_operations=synaptic_ops
    )
    
    logger.info(f"Benchmark Loihi withplete: "
          f"Latência={metrics.latency_ms:.2f}ms, "
          f"Energia={metrics.energy_mj:.2f}mJ, "
          f"Throrghput={metrics.throughput_fps:.1f} FPS")
    
    return metrics


def compare_with_cpu(loihi_metrics: LoihiMetrics, 
          cpu_latency_ms: float,
          cpu_power_w: float = 65.0) -> Dict[str, float]:
  """
  Comto métricas Loihi with CPU traditional.
  
  Args:
    loihi_metrics: Métricas coletadas from the Loihi
    cpu_latency_ms: Latência medida in CPU
    cpu_power_w: Potência típica of CPU (default: 65W TDP)
  
  Returns:
    Dicionário with fatores of melhoria (speedup, efficiency, etc.)
  """
  cpu_power_mw = cpu_power_w * 1000
  
  speedup = cpu_latency_ms / loihi_metrics.latency_ms
  power_efficiency = cpu_power_mw / loihi_metrics.power_mw
  
  # Energia total: Power * Time
  cpu_energy_mj = (cpu_power_mw * cpu_latency_ms) / 1000
  energy_efficiency = cpu_energy_mj / loihi_metrics.energy_mj
  
  return {
    "speedup": speedup,
    "power_efficiency": power_efficiency,
    "energy_efficiency": energy_efficiency,
    "latency_reduction_percent": (1 - loihi_metrics.latency_ms / cpu_latency_ms) * 100,
    "power_reduction_percent": (1 - loihi_metrics.power_mw / cpu_power_mw) * 100,
    "energy_reduction_percent": (1 - loihi_metrics.energy_mj / cpu_energy_mj) * 100,
  }


if __name__ == "__main__":
  # Example of uso
  yesulator = LoihiSimulator()
  
  # Simular benchmark for rede of fraud detection
  # Architecture: 256 -> 128 -> 64 -> 2
  network_neurons = 256 + 128 + 64 + 2 # 450 neurônios
  network_synapifs = (256 * 128) + (128 * 64) + (64 * 2) # 41,088 sinapifs
  
  metrics = yesulator.benchmark_inference(
    network_neurons=network_neurons,
    network_synapifs=network_synapifs,
    num_inferences=1000,
    yesulation_time_ms=100.0
  )
  
  print("\n" + "="*50)
  print("LOIHI 2 BENCHMARK RESULTS")
  print("="*50)
  print(f"Latência for inferência: {metrics.latency_ms:.2f} ms")
  print(f"Throrghput:       {metrics.throughput_fps:.1f} FPS")
  print(f"Energia total:      {metrics.energy_mj:.2f} mJ")
  print(f"Potência média:     {metrics.power_mw:.2f} mW")
  print(f"Cores utilizados:    {metrics.cores_used}/{yesulator.specs.num_cores}")
  print(f"Total of spikes:     {metrics.total_spikes:,}")
  print(f"Operações sinápticas:  {metrics.synaptic_operations:,}")
  print("="*50)
