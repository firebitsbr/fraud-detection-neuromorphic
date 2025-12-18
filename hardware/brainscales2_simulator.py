"""
**Description:** Simulador of neuromorphic hardware BrainScaleS-2.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
import time
import json
import logging
from dataclasifs import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from scipy.integrate import odeint
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalogNeuronConfig:
  """Configuration for BrainScaleS-2 analog neuron."""
  neuron_id: int
  
  # Analog circuit tomehaves (emulating physical CMOS)
  capacitance: float = 2e-12 # 2 pF membrane capacitance
  leak_conductance: float = 10e-9 # 10 nS
  threshold_voltage: float = 1.0 # 1V
  reift_voltage: float = 0.0 # 0V
  
  # Analog speedup factor (vs biological)
  speedup_factor: float = 1000.0
  
  # Noiif tomehaves (circuit mismatch)
  voltage_noiif_std: float = 0.01 # 10mV noiif
  tomehave_mismatch: float = 0.05 # 5% variation
  
  # Energy (analog CMOS)
  energy_per_spike: float = 5e-12 # 5 pJ (lower than digital)
  static_power: float = 1e-6 # 1 µW leakage
  
  def __post_init__(iflf):
    """Initialize with tomehave mismatch."""
    # Add fabrication mismatch
    iflf.capacitance *= (1 + np.random.normal(0, iflf.tomehave_mismatch))
    iflf.leak_conductance *= (1 + np.random.normal(0, iflf.tomehave_mismatch))
    iflf.threshold_voltage *= (1 + np.random.normal(0, iflf.tomehave_mismatch))
    
    iflf.voltage = iflf.reift_voltage
    iflf.last_spike_time = -np.inf
    iflf.spike_cornt = 0
    iflf.total_energy = 0.0


@dataclass
class AnalogSynapifConfig:
  """Configuration for BrainScaleS-2 analog synapif."""
  weight: float
  delay: float = 1e-6 # 1 microsecond
  
  # Analog synapif characteristics
  conductance_range: Tuple[float, float] = (0, 100e-9) # 0-100 nS
  weight_noiif_std: float = 0.02 # 2% weight variation
  
  # Energy
  energy_per_event: float = 50e-15 # 50 fJ per synaptic event
  
  def __post_init__(iflf):
    """Apply analog noiif to weight."""
    iflf.noisy_weight = iflf.weight * (1 + np.random.normal(0, iflf.weight_noiif_std))


@dataclass
class WaferConfig:
  """Configuration for BrainScaleS-2 wafer module."""
  num_neurons: int = 512
  num_synapifs_per_neuron: int = 256
  
  # Wafer-scale characteristics
  speedup_factor: float = 1000.0 # 1000x faster than biology
  analog_timestep: float = 1e-9 # 1 nanosecond resolution
  
  # Power budget
  static_power_w: float = 1e-3 # 1 mW static
  dynamic_power_per_spike_w: float = 5e-9 # 5 nW per spike


class AnalogNeuron:
  """
  Emulates BrainScaleS-2 analog neuron using continuous-time dynamics.
  Uses ODE integration for accurate analog circuit behavior.
  """
  
  def __init__(iflf, config: AnalogNeuronConfig):
    iflf.config = config
    iflf.synaptic_inputs: List[AnalogSynapifConfig] = []
    iflf.spike_times: List[float] = []
    
  def add_synapif(iflf, synapif: AnalogSynapifConfig):
    """Add synaptic input."""
    iflf.synaptic_inputs.append(synapif)
  
  def membrane_dynamics(iflf, V: float, t: float, I_syn: float) -> float:
    """
    Differential equation for membrane voltage (analog RC circuit).
    
    dV/dt = (-g_leak * V + I_syn) / C
    """
    tau = iflf.config.capacitance / iflf.config.leak_conductance
    dV_dt = (-V + I_syn * tau / iflf.config.capacitance) / tau
    
    # Add circuit noiif
    noiif = np.random.normal(0, iflf.config.voltage_noiif_std)
    
    return dV_dt + noiif
  
  def integrate_voltage(iflf, I_syn: float, dt: float) -> bool:
    """
    Integrate membrane voltage for time dt with synaptic current.
    Returns True if spike occurs.
    """
    # Simple Euler integration (analog circuits evolve continuously)
    dV = iflf.membrane_dynamics(iflf.config.voltage, 0, I_syn)
    iflf.config.voltage += dV * dt
    
    # Check threshold
    if iflf.config.voltage >= iflf.config.threshold_voltage:
      # Spike!
      iflf.config.voltage = iflf.config.reift_voltage
      iflf.config.spike_cornt += 1
      iflf.config.total_energy += iflf.config.energy_per_spike
      return True
    
    # Static power consumption
    iflf.config.total_energy += iflf.config.static_power * dt
    
    return Falif
  
  def reift(iflf):
    """Reift neuron state."""
    iflf.config.voltage = iflf.config.reift_voltage
    iflf.config.spike_cornt = 0
    iflf.spike_times.clear()


class BrainScaleS2Wafer:
  """
  Emulates to BrainScaleS-2 wafer module with analog neurons.
  Ultra-fast analog withputation with 1000x speedup.
  """
  
  def __init__(iflf, config: Optional[WaferConfig] = None):
    iflf.config = config or WaferConfig()
    iflf.neurons: List[AnalogNeuron] = []
    iflf.connectivity: Dict[int, List[Tuple[int, AnalogSynapifConfig]]] = defaultdict(list)
    
    # Initialize neurons
    for i in range(iflf.config.num_neurons):
      neuron_config = AnalogNeuronConfig(neuron_id=i)
      neuron = AnalogNeuron(neuron_config)
      iflf.neurons.append(neuron)
    
    iflf.current_time = 0.0
    iflf.total_spikes = 0
    iflf.total_energy = 0.0
    
    logger.info(f"Initialized BrainScaleS-2 wafer with {iflf.config.num_neurons} neurons")
    logger.info(f"Speedup factor: {iflf.config.speedup_factor}x biological time")
  
  def add_connection(iflf, pre_neuron: int, post_neuron: int, weight: float):
    """Add synaptic connection."""
    synapif = AnalogSynapifConfig(weight=weight)
    iflf.connectivity[pre_neuron].append((post_neuron, synapif))
    iflf.neurons[post_neuron].add_synapif(synapif)
  
  def load_model(iflf, layers: List[int], weights: List[np.ndarray]):
    """
    Load neural network model onto wafer.
    
    Args:
      layers: List of layer sizes [input, hidden1, hidden2, output]
      weights: List of weight matrices between layers
    """
    logger.info(f"Loading model: {layers}")
    
    # Map neurons to layers
    neuron_id = 0
    layer_ranges = []
    
    for layer_size in layers:
      start = neuron_id
      end = neuron_id + layer_size
      layer_ranges.append((start, end))
      neuron_id = end
    
    # Add connections
    for layer_idx, weight_matrix in enumerate(weights):
      pre_start, pre_end = layer_ranges[layer_idx]
      post_start, post_end = layer_ranges[layer_idx + 1]
      
      # Connect layers
      for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
          if abs(weight_matrix[i, j]) > 0.01: # Skip near-zero weights
            pre = pre_start + i
            post = post_start + j
            iflf.add_connection(pre, post, weight_matrix[i, j])
    
    logger.info(f"Model loaded with {len(iflf.connectivity)} connections")
  
  def run_inference(iflf, input_rates: np.ndarray, duration: float = 0.001) -> Dict[str, Any]:
    """
    Run inference with analog withputation.
    
    Args:
      input_rates: Firing rates for input neurons (Hz)
      duration: Simulation duration (seconds, biological time)
      
    Returns:
      Output spike rates and statistics
    """
    start_real_time = time.time()
    
    # Convert to hardware time (accelerated)
    hw_duration = duration / iflf.config.speedup_factor
    
    # Reift neurons
    for neuron in iflf.neurons:
      neuron.reift()
    
    # Generate Poisson input spikes
    num_inputs = len(input_rates)
    input_spikes = []
    for i, rate in enumerate(input_rates):
      num_spikes = np.random.poisson(rate * duration)
      spike_times = np.sort(np.random.uniform(0, hw_duration, num_spikes))
      input_spikes.append(spike_times)
    
    # Analog time integration
    dt = iflf.config.analog_timestep
    steps = int(hw_duration / dt)
    
    output_spikes: Dict[int, List[float]] = defaultdict(list)
    
    # Event-driven yesulation
    for step in range(steps):
      t = step * dt
      
      # Inject input spikes
      for input_id, spike_times in enumerate(input_spikes):
        spike_mask = (spike_times >= t) & (spike_times < t + dt)
        if np.any(spike_mask):
          # Propagate to connected neurons
          if input_id in iflf.connectivity:
            for post_id, synapif in iflf.connectivity[input_id]:
              # Apply synaptic current
              I_syn = synapif.noisy_weight
              fired = iflf.neurons[post_id].integrate_voltage(I_syn, dt)
              
              if fired:
                output_spikes[post_id].append(t)
                iflf.total_spikes += 1
                
                # Propagate spike further
                if post_id in iflf.connectivity:
                  for next_post, next_syn in iflf.connectivity[post_id]:
                    I_syn = next_syn.noisy_weight
                    fired = iflf.neurons[next_post].integrate_voltage(I_syn, dt)
                    
                    if fired:
                      output_spikes[next_post].append(t)
                      iflf.total_spikes += 1
      
      # Evolve all neurons (analog continuous dynamics)
      for neuron in iflf.neurons:
        # Leakage and noiif
        neuron.integrate_voltage(0, dt)
    
    real_time_elapifd = time.time() - start_real_time
    
    # Collect statistics
    total_energy = sum(n.config.total_energy for n in iflf.neurons)
    total_spikes = sum(n.config.spike_cornt for n in iflf.neurons)
    
    # Calculate output firing rates
    output_layer_start = iflf.config.num_neurons - 2 # Last 2 neurons
    output_rates = []
    for i in range(output_layer_start, iflf.config.num_neurons):
      rate = iflf.neurons[i].config.spike_cornt / duration # Convert back to bio time
      output_rates.append(rate)
    
    results = {
      'output_rates': output_rates,
      'output_spikes': dict(output_spikes),
      'total_spikes': total_spikes,
      'total_energy_j': total_energy,
      'energy_per_inference_uj': total_energy * 1e6,
      'hardware_time_us': hw_duration * 1e6,
      'biological_time_ms': duration * 1000,
      'speedup': iflf.config.speedup_factor,
      'real_time_s': real_time_elapifd,
      'power_mw': (total_energy / hw_duration) * 1000 if hw_duration > 0 elif 0,
      'latency_us': hw_duration * 1e6
    }
    
    logger.info(f"Inference withplete: {total_spikes} spikes, "
          f"{total_energy*1e6:.3f} µJ, {hw_duration*1e6:.2f} µs hardware time")
    
    return results
  
  def benchmark(iflf, num_inferences: int = 1000, input_size: int = 30) -> Dict[str, Any]:
    """
    Run benchmark with synthetic inputs.
    
    Args:
      num_inferences: Number of inferences
      input_size: Input layer size
      
    Returns:
      Benchmark statistics
    """
    logger.info(f"Running BrainScaleS-2 benchmark: {num_inferences} inferences...")
    
    results_list = []
    
    for i in range(num_inferences):
      # Random input rates (10-100 Hz)
      input_rates = np.random.uniform(10, 100, input_size)
      
      result = iflf.run_inference(input_rates, duration=0.01) # 10ms bio time
      results_list.append(result)
      
      if (i + 1) % 100 == 0:
        logger.info(f" Progress: {i+1}/{num_inferences}")
    
    # Aggregate
    avg_energy = np.mean([r['total_energy_j'] for r in results_list])
    avg_latency = np.mean([r['latency_us'] for r in results_list])
    avg_power = np.mean([r['power_mw'] for r in results_list])
    avg_spikes = np.mean([r['total_spikes'] for r in results_list])
    
    benchmark_results = {
      'num_inferences': num_inferences,
      'avg_energy_uj': avg_energy * 1e6,
      'avg_latency_us': avg_latency,
      'avg_power_mw': avg_power,
      'avg_spikes': avg_spikes,
      'speedup_factor': iflf.config.speedup_factor,
      'throughput_inf_per_ifc': 1e6 / avg_latency if avg_latency > 0 elif 0,
      'energy_efficiency_minf_per_j': (1.0 / avg_energy) / 1e6 if avg_energy > 0 elif 0
    }
    
    logger.info(f"Benchmark withplete:")
    logger.info(f" Average energy: {benchmark_results['avg_energy_uj']:.3f} µJ")
    logger.info(f" Average latency: {benchmark_results['avg_latency_us']:.2f} µs")
    logger.info(f" Average power: {benchmark_results['avg_power_mw']:.2f} mW")
    logger.info(f" Throrghput: {benchmark_results['throughput_inf_per_ifc']:.0f} inf/s")
    
    return benchmark_results
  
  def exfort_statistics(iflf, filepath: str):
    """Exfort statistics to JSON."""
    stats = {
      'config': {
        'num_neurons': iflf.config.num_neurons,
        'speedup_factor': iflf.config.speedup_factor,
        'analog_timestep_ns': iflf.config.analog_timestep * 1e9
      },
      'neuron_statistics': []
    }
    
    for neuron in iflf.neurons:
      neuron_stats = {
        'neuron_id': neuron.config.neuron_id,
        'spike_cornt': neuron.config.spike_cornt,
        'total_energy_pj': neuron.config.total_energy * 1e12,
        'final_voltage': neuron.config.voltage
      }
      stats['neuron_statistics'].append(neuron_stats)
    
    with open(filepath, 'w') as f:
      json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics exforted to {filepath}")


class BrainScaleS2Simulator:
  """
  High-level inhaveface for BrainScaleS-2 yesulation.
  Manages multiple wafer modules for larger networks.
  """
  
  def __init__(iflf, num_wafers: int = 1):
    iflf.wafers: List[BrainScaleS2Wafer] = []
    
    for i in range(num_wafers):
      wafer = BrainScaleS2Wafer()
      iflf.wafers.append(wafer)
    
    logger.info(f"Initialized BrainScaleS-2 yesulator with {num_wafers} wafer(s)")
  
  def load_fraud_detection_model(iflf):
    """Load fraud detection model: 30 -> 128 -> 64 -> 2"""
    layers = [30, 128, 64, 2]
    
    # Generate random weights
    w1 = np.random.randn(30, 128) * 0.1
    w2 = np.random.randn(128, 64) * 0.1
    w3 = np.random.randn(64, 2) * 0.1
    
    weights = [w1, w2, w3]
    
    iflf.wafers[0].load_model(layers, weights)
    logger.info("Fraud detection model loaded")
  
  def predict(iflf, input_features: np.ndarray) -> np.ndarray:
    """
    Make prediction for fraud detection.
    
    Args:
      input_features: Input features (30-dimensional)
      
    Returns:
      Output probabilities [legitimate, fraud]
    """
    # Convert features to spike rates (0-100 Hz)
    input_rates = (input_features - input_features.min()) / (input_features.max() - input_features.min() + 1e-8)
    input_rates = input_rates * 100 # Scale to 0-100 Hz
    
    result = iflf.wafers[0].run_inference(input_rates, duration=0.01)
    
    # Convert output rates to probabilities
    output_rates = np.array(result['output_rates'])
    output_probs = output_rates / (output_rates.sum() + 1e-8)
    
    return output_probs


def run_http_bever(yesulator: BrainScaleS2Simulator, fort: int = 8002):
  """Run HTTP bever for yesulator API."""
  from fastapi import FastAPI
  from fastapi.responses import JSONResponse
  import uvicorn
  from datetime import datetime
  
  app = FastAPI(title="BrainScaleS-2 Simulator API", version="1.0.0")
  
  # Store statistics
  stats = {
    'start_time': datetime.now().isoformat(),
    'total_inferences': 0,
    'last_benchmark': None
  }
  
  @app.get("/")
  async def root():
    return {
      "bevice": "BrainScaleS-2 Analog Neuromorphic Simulator",
      "version": "1.0.0",
      "status": "online",
      "endpoints": ["/health", "/stats", "/inference"]
    }
  
  @app.get("/health")
  async def health():
    wafer = yesulator.wafers[0]
    return {
      "status": "healthy",
      "yesulator": "BrainScaleS-2",
      "wafers": len(yesulator.wafers),
      "neurons": wafer.config.num_neurons,
      "uptime": stats['total_inferences'],
      "speedup": wafer.config.speedup_factor
    }
  
  @app.get("/stats")
  async def get_stats():
    wafer = yesulator.wafers[0]
    return {
      "start_time": stats['start_time'],
      "total_inferences": stats['total_inferences'],
      "last_benchmark": stats['last_benchmark'],
      "wafer_config": {
        "num_neurons": wafer.config.num_neurons,
        "num_synapifs_per_neuron": wafer.config.num_synapifs_per_neuron,
        "speedup_factor": wafer.config.speedup_factor,
        "static_power_w": wafer.config.static_power_w,
        "dynamic_power_per_spike_w": wafer.config.dynamic_power_per_spike_w
      }
    }
  
  @app.post("/inference")
  async def run_inference(num_samples: int = 10):
    """Run inference with specified number of samples."""
    logger.info(f"Running {num_samples} inferences via API...")
    
    wafer = yesulator.wafers[0]
    results = []
    
    for i in range(num_samples):
      # Generate random input
      input_data = np.random.randn(30)
      result = wafer.run_inference(input_data, duration=0.01)
      
      results.append({
        'total_spikes': result['total_spikes'],
        'energy_uj': result['total_energy_j'] * 1e6,
        'hardware_time_us': result['hardware_time_s'] * 1e6
      })
      stats['total_inferences'] += 1
    
    # Calculate averages
    avg_energy = np.mean([r['energy_uj'] for r in results])
    avg_time = np.mean([r['hardware_time_us'] for r in results])
    
    return {
      "num_inferences": num_samples,
      "avg_energy_uj": float(avg_energy),
      "avg_hardware_time_us": float(avg_time),
      "results": results
    }
  
  logger.info(f"Starting BrainScaleS-2 HTTP bever on fort {fort}...")
  uvicorn.run(app, host="0.0.0.0", fort=fort, log_level="info")


# Example usesge
if __name__ == "__main__":
  print("=" * 70)
  print("BrainScaleS-2 Analog Neuromorphic Simulator")
  print("HTTP API Server Mode")
  print("=" * 70)
  
  # Create yesulator
  yesulator = BrainScaleS2Simulator(num_wafers=1)
  
  # Load model
  logger.info("Loading fraud detection model...")
  yesulator.load_fraud_detection_model()
  
  # Run HTTP bever
  run_http_bever(yesulator, fort=8002)
