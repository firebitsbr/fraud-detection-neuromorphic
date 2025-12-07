"""
Simula as constraints e métricas de performance do chip neuromórfico
Loihi 2 para comparação com implementação em CPU/GPU tradicional.
Modela latência, energia, potência e throughput.

Autor: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
Linkedin: https://www.linkedin.com/in/maurorisonho
github: https://github.com/maurorisonho
Data de criação: Dezembro 2025
LICENSE MIT

Referências:
- Intel Loihi 2: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html
- Davies et al. (2021): "Advancing Neuromorphic Computing with Loihi"
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoihiSpecs:
    """Especificações do Intel Loihi 2"""
    
    # Arquitetura
    num_cores: int = 128  # 128 neuromorphic cores por chip
    neurons_per_core: int = 8192  # Máximo de neurônios por core
    synapses_per_core: int = 131072  # Máximo de sinapses por core
    total_neurons: int = 1_048_576  # 128 * 8192
    
    # Performance
    power_per_core_active: float = 30e-3  # 30mW por core ativo (watts)
    power_per_core_idle: float = 5e-3  # 5mW por core idle (watts)
    spike_latency_us: float = 1.0  # 1 microsegundo por spike
    synaptic_operations_per_sec: int = 10_000_000_000  # 10 GSOPS
    
    # Clock
    clock_frequency_mhz: int = 1000  # 1 GHz
    
    # Energia
    energy_per_spike_pj: float = 23.6  # 23.6 picojoules por spike
    energy_per_synop_pj: float = 0.5  # 0.5 picojoules por operação sináptica


@dataclass
class LoihiMetrics:
    """Métricas coletadas da simulação Loihi"""
    
    latency_ms: float
    energy_mj: float  # millijoules
    power_mw: float  # milliwatts
    throughput_fps: float  # frames (transactions) per second
    cores_used: int
    total_spikes: int
    synaptic_operations: int


class LoihiSimulator:
    """
    Simulador de hardware Loihi 2 para benchmarking.
    
    Modela:
    - Mapeamento de neurônios para cores
    - Consumo energético baseado em spikes
    - Latência de processamento event-driven
    - Throughput limitado por hardware
    """
    
    def __init__(self, specs: LoihiSpecs = None):
        self.specs = specs or LoihiSpecs()
        logger.info(f"Loihi Simulator inicializado: {self.specs.num_cores} cores, "
                   f"{self.specs.total_neurons:,} neurônios totais")
    
    def calculate_cores_needed(self, network_neurons: int, network_synapses: int) -> int:
        """Calcula quantos cores Loihi são necessários para a rede"""
        
        cores_by_neurons = int(np.ceil(network_neurons / self.specs.neurons_per_core))
        cores_by_synapses = int(np.ceil(network_synapses / self.specs.synapses_per_core))
        
        cores_needed = max(cores_by_neurons, cores_by_synapses)
        
        if cores_needed > self.specs.num_cores:
            logger.warning(f"Rede requer {cores_needed} cores, mas Loihi tem apenas "
                          f"{self.specs.num_cores}. Multi-chip necessário.")
        
        return cores_needed
    
    def estimate_spikes(self, 
                       input_size: int, 
                       hidden_sizes: List[int], 
                       output_size: int,
                       simulation_time_ms: float = 100.0,
                       mean_firing_rate_hz: float = 20.0) -> int:
        """
        Estima número total de spikes baseado na arquitetura e tempo de simulação.
        
        Args:
            input_size: Número de neurônios de entrada
            hidden_sizes: Lista com tamanhos das camadas hidden
            output_size: Número de neurônios de saída
            simulation_time_ms: Tempo de simulação em milissegundos
            mean_firing_rate_hz: Taxa média de disparo por neurônio
        
        Returns:
            Número total estimado de spikes
        """
        total_neurons = input_size + sum(hidden_sizes) + output_size
        simulation_time_s = simulation_time_ms / 1000.0
        
        total_spikes = int(total_neurons * mean_firing_rate_hz * simulation_time_s)
        return total_spikes
    
    def estimate_synaptic_operations(self,
                                    total_synapses: int,
                                    total_spikes: int) -> int:
        """
        Estima operações sinápticas totais.
        
        Cada spike propaga através das sinapses conectadas.
        Aproximação: média de sinapses por neurônio * total de spikes
        """
        # Simplificação: assume que cada spike ativa ~10% das sinapses
        activation_ratio = 0.1
        synaptic_ops = int(total_spikes * total_synapses * activation_ratio)
        return synaptic_ops
    
    def calculate_energy(self, total_spikes: int, synaptic_operations: int) -> float:
        """
        Calcula energia consumida em millijoules.
        
        Energia = (spikes * energia_por_spike) + (syn_ops * energia_por_synop)
        """
        spike_energy_mj = (total_spikes * self.specs.energy_per_spike_pj) / 1e9
        synop_energy_mj = (synaptic_operations * self.specs.energy_per_synop_pj) / 1e9
        
        total_energy_mj = spike_energy_mj + synop_energy_mj
        return total_energy_mj
    
    def calculate_latency(self,
                         total_spikes: int,
                         cores_used: int,
                         simulation_time_ms: float = 100.0) -> float:
        """
        Calcula latência de processamento em milissegundos.
        
        Loihi é event-driven, então latência depende de:
        1. Tempo de simulação biológica
        2. Overhead de comunicação entre cores
        3. Processamento de spikes
        """
        # Latência base: tempo de simulação biológica
        base_latency_ms = simulation_time_ms
        
        # Overhead de comunicação inter-core (0.1ms por core adicional)
        comm_overhead_ms = (cores_used - 1) * 0.1 if cores_used > 1 else 0
        
        # Overhead de processamento de spikes (muito baixo no Loihi)
        spike_processing_ms = (total_spikes * self.specs.spike_latency_us) / 1000.0
        
        total_latency_ms = base_latency_ms + comm_overhead_ms + spike_processing_ms
        return total_latency_ms
    
    def calculate_power(self, cores_used: int, execution_time_ms: float) -> float:
        """
        Calcula potência média em milliwatts.
        
        Power = (cores_ativos * power_ativo) + (cores_inativos * power_idle)
        """
        active_power_mw = cores_used * self.specs.power_per_core_active * 1000
        idle_cores = self.specs.num_cores - cores_used
        idle_power_mw = idle_cores * self.specs.power_per_core_idle * 1000
        
        total_power_mw = active_power_mw + idle_power_mw
        return total_power_mw
    
    def benchmark_inference(self,
                           network_neurons: int,
                           network_synapses: int,
                           num_inferences: int = 1000,
                           simulation_time_ms: float = 100.0) -> LoihiMetrics:
        """
        Simula benchmark de inferência no Loihi.
        
        Args:
            network_neurons: Total de neurônios na rede
            network_synapses: Total de sinapses na rede
            num_inferences: Número de inferências a executar
            simulation_time_ms: Tempo de simulação por inferência
        
        Returns:
            LoihiMetrics com resultados do benchmark
        """
        logger.info(f"Iniciando benchmark Loihi: {num_inferences} inferências")
        
        # 1. Calcular cores necessários
        cores_used = self.calculate_cores_needed(network_neurons, network_synapses)
        
        # 2. Estimar spikes por inferência
        hidden_sizes = [128, 64]  # Arquitetura típica do fraud detection
        spikes_per_inference = self.estimate_spikes(
            input_size=256,
            hidden_sizes=hidden_sizes,
            output_size=2,
            simulation_time_ms=simulation_time_ms,
            mean_firing_rate_hz=20.0
        )
        
        total_spikes = spikes_per_inference * num_inferences
        
        # 3. Estimar operações sinápticas
        synaptic_ops = self.estimate_synaptic_operations(network_synapses, total_spikes)
        
        # 4. Calcular latência por inferência
        latency_per_inference_ms = self.calculate_latency(
            total_spikes=spikes_per_inference,
            cores_used=cores_used,
            simulation_time_ms=simulation_time_ms
        )
        
        total_latency_ms = latency_per_inference_ms * num_inferences
        
        # 5. Calcular energia
        energy_mj = self.calculate_energy(total_spikes, synaptic_ops)
        
        # 6. Calcular potência
        power_mw = self.calculate_power(cores_used, total_latency_ms)
        
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
        
        logger.info(f"Benchmark Loihi completo: "
                   f"Latência={metrics.latency_ms:.2f}ms, "
                   f"Energia={metrics.energy_mj:.2f}mJ, "
                   f"Throughput={metrics.throughput_fps:.1f} FPS")
        
        return metrics


def compare_with_cpu(loihi_metrics: LoihiMetrics, 
                    cpu_latency_ms: float,
                    cpu_power_w: float = 65.0) -> Dict[str, float]:
    """
    Compara métricas Loihi com CPU tradicional.
    
    Args:
        loihi_metrics: Métricas coletadas do Loihi
        cpu_latency_ms: Latência medida em CPU
        cpu_power_w: Potência típica de CPU (default: 65W TDP)
    
    Returns:
        Dicionário com fatores de melhoria (speedup, efficiency, etc.)
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
    # Exemplo de uso
    simulator = LoihiSimulator()
    
    # Simular benchmark para rede de detecção de fraude
    # Arquitetura: 256 -> 128 -> 64 -> 2
    network_neurons = 256 + 128 + 64 + 2  # 450 neurônios
    network_synapses = (256 * 128) + (128 * 64) + (64 * 2)  # 41,088 sinapses
    
    metrics = simulator.benchmark_inference(
        network_neurons=network_neurons,
        network_synapses=network_synapses,
        num_inferences=1000,
        simulation_time_ms=100.0
    )
    
    print("\n" + "="*50)
    print("LOIHI 2 BENCHMARK RESULTS")
    print("="*50)
    print(f"Latência por inferência: {metrics.latency_ms:.2f} ms")
    print(f"Throughput:              {metrics.throughput_fps:.1f} FPS")
    print(f"Energia total:           {metrics.energy_mj:.2f} mJ")
    print(f"Potência média:          {metrics.power_mw:.2f} mW")
    print(f"Cores utilizados:        {metrics.cores_used}/{simulator.specs.num_cores}")
    print(f"Total de spikes:         {metrics.total_spikes:,}")
    print(f"Operações sinápticas:    {metrics.synaptic_operations:,}")
    print("="*50)
