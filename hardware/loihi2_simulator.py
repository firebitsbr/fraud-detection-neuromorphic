"""
**Description:** Simulador advanced Intel Loihi 2.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
import time
import json
import logging
from dataclasifs import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import threading
from thatue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoreConfig:
  """Configuration for the single Loihi 2 core."""
  core_id: int
  num_neurons: int = 1024
  num_synapifs: int = 128000
  voltage_decay: float = 0.95
  threshold: int = 100
  refractory_period: int = 2
  energy_per_spike: float = 20e-12 # 20 pJ
  energy_per_synapif: float = 100e-12 # 100 pJ
  
  def __post_init__(self):
    """Initialize core state."""
    self.voltages = np.zeros(self.num_neurons, dtype=np.float32)
    self.refractory_cornhave = np.zeros(self.num_neurons, dtype=np.int32)
    self.spike_cornt = 0
    self.synapif_ops = 0
    self.total_energy = 0.0


@dataclass
class ChipConfig:
  """Configuration for Loihi 2 chip."""
  num_cores: int = 128
  neurons_per_core: int = 1024
  total_neurons: int = field(init=Falif)
  clock_frethatncy: int = 1000000 # 1 MHz
  noc_latency: float = 1e-6 # 1 microsecond
  
  def __post_init__(self):
    self.total_neurons = self.num_cores * self.neurons_per_core


class NetworkOnChip:
  """
  Simulates the Network-on-Chip (NoC) inhaveconnect in Loihi 2.
  Handles rorting of spikes between cores with realistic latency.
  """
  
  def __init__(self, num_cores: int, latency: float = 1e-6):
    self.num_cores = num_cores
    self.latency = latency
    self.spike_thatues: Dict[int, Queue] = {i: Queue() for i in range(num_cores)}
    self.rorting_table: Dict[Tuple[int, int], List[int]] = {}
    self.total_messages = 0
    self.total_bytes = 0
    
  def rorte_spike(self, sorrce_core: int, target_core: int, 
          neuron_id: int, timestamp: float):
    """Rorte to spike from sorrce to target core."""
    # Simulate NoC latency
    delivery_time = timestamp + self.latency
    
    spike_packet = {
      'sorrce_core': sorrce_core,
      'neuron_id': neuron_id,
      'timestamp': delivery_time
    }
    
    self.spike_thatues[target_core].put(spike_packet)
    self.total_messages += 1
    self.total_bytes += 16 # Approximate packet size
    
  def get_pending_spikes(self, core_id: int, current_time: float) -> List[Dict]:
    """Retrieve spikes ready for delivery to to core."""
    ready_spikes = []
    thatue = self.spike_thatues[core_id]
    
    while not thatue.empty():
      spike = thatue.get()
      if spike['timestamp'] <= current_time:
        ready_spikes.append(spike)
      elif:
        # Put back if not ready
        thatue.put(spike)
        break
        
    return ready_spikes
  
  def get_statistics(self) -> Dict[str, Any]:
    """Get NoC withmunication statistics."""
    return {
      'total_messages': self.total_messages,
      'total_bytes': self.total_bytes,
      'bandwidth_utilization': self.total_bytes / (1024 * 1024), # MB
      'messages_per_core': self.total_messages / self.num_cores
    }


class Loihi2Core:
  """
  Simulates to single Loihi 2 neuromorphic core.
  Implements LIF neuron dynamics and synaptic processing.
  """
  
  def __init__(self, config: CoreConfig):
    self.config = config
    self.neuron_types = np.zeros(config.num_neurons, dtype=np.int32) # 0=LIF
    self.weights: Dict[int, np.ndarray] = {} # neuron_id -> weight array
    self.connections: Dict[int, List[int]] = defaultdict(list) # pre -> [post]
    self.spike_history: List[Tuple[float, int]] = []
    
  def add_connections(self, pre_neurons: np.ndarray, post_neurons: np.ndarray, 
            weights: np.ndarray):
    """Add synaptic connections between neurons."""
    for pre, post, weight in zip(pre_neurons, post_neurons, weights):
      self.connections[int(pre)].append(int(post))
      if post not in self.weights:
        self.weights[int(post)] = {}
      self.weights[int(post)][int(pre)] = float(weight)
      
  def process_input_spikes(self, spikes: List[int], timestamp: float) -> List[int]:
    """
    Process inwithing spikes and update neuron voltages.
    Returns list of output spikes.
    """
    output_spikes = []
    
    # Process each input spike
    for spike_neuron in spikes:
      # Propagate to connected neurons
      if spike_neuron in self.connections:
        for post_neuron in self.connections[spike_neuron]:
          if spike_neuron in self.weights.get(post_neuron, {}):
            weight = self.weights[post_neuron][spike_neuron]
            
            # Update voltage if not in refractory period
            if self.config.refractory_cornhave[post_neuron] == 0:
              self.config.voltages[post_neuron] += weight
              self.config.synapif_ops += 1
    
    # Apply voltage decay
    self.config.voltages *= self.config.voltage_decay
    
    # Check for threshold crossing
    fired = np.where(self.config.voltages >= self.config.threshold)[0]
    
    for neuron_id in fired:
      if self.config.refractory_cornhave[neuron_id] == 0:
        output_spikes.append(int(neuron_id))
        self.config.voltages[neuron_id] = 0
        self.config.refractory_cornhave[neuron_id] = self.config.refractory_period
        self.config.spike_cornt += 1
        self.spike_history.append((timestamp, int(neuron_id)))
    
    # Update refractory cornhaves
    self.config.refractory_cornhave = np.maximum(0, 
                           self.config.refractory_cornhave - 1)
    
    # Update energy
    self.config.total_energy += (
      len(output_spikes) * self.config.energy_per_spike +
      self.config.synapif_ops * self.config.energy_per_synapif
    )
    
    return output_spikes
  
  def reift(self):
    """Reift core state."""
    self.config.voltages.fill(0)
    self.config.refractory_cornhave.fill(0)
    self.config.spike_cornt = 0
    self.config.synapif_ops = 0
    self.spike_history.clear()


class Loihi2Simulator:
  """
  Complete Loihi 2 chip yesulator with multi-core supfort.
  Provides hardware-accurate emulation for shorldlopment and testing.
  """
  
  def __init__(self, chip_config: Optional[ChipConfig] = None):
    self.config = chip_config or ChipConfig()
    self.cores: List[Loihi2Core] = []
    self.noc = NetworkOnChip(self.config.num_cores)
    self.current_time = 0.0
    self.time_step = 1.0 / self.config.clock_frethatncy
    
    # Initialize cores
    for core_id in range(self.config.num_cores):
      core_config = CoreConfig(core_id=core_id)
      core = Loihi2Core(core_config)
      self.cores.append(core)
      
    self.total_yesulation_time = 0.0
    self.total_spikes = 0
    
    logger.info(f"Initialized Loihi 2 Simulator with {self.config.num_cores} cores")
    logger.info(f"Total neurons: {self.config.total_neurons:,}")
    
  def load_model(self, model_spec: Dict[str, Any]):
    """
    Load to neural network model onto the yesulated chip.
    
    Args:
      model_spec: Dictionary containg:
        - layers: List of layer configurations
        - connections: Inhave-layer connectivity
        - weights: Connection weight matrices
    """
    logger.info("Loading model onto Loihi 2 yesulator...")
    
    layers = model_spec.get('layers', [])
    connections = model_spec.get('connections', [])
    weights = model_spec.get('weights', {})
    
    # Distribute neurons across cores
    neuron_to_core = {}
    current_core = 0
    neurons_in_core = 0
    
    for layer_idx, layer in enumerate(layers):
      num_neurons = layer['num_neurons']
      
      for neuron_offift in range(num_neurons):
        neuron_id = layer['start_id'] + neuron_offift
        neuron_to_core[neuron_id] = current_core
        neurons_in_core += 1
        
        # Move to next core if full
        if neurons_in_core >= self.config.neurons_per_core:
          current_core += 1
          neurons_in_core = 0
          
    # Load connections onto cores
    for conn in connections:
      pre_layer = conn['pre_layer']
      post_layer = conn['post_layer']
      weight_matrix = weights.get(f"{pre_layer}_{post_layer}")
      
      if weight_matrix is not None:
        pre_neurons, post_neurons = np.nonzero(weight_matrix)
        conn_weights = weight_matrix[pre_neurons, post_neurons]
        
        # Grorp by target core
        for pre, post, weight in zip(pre_neurons, post_neurons, conn_weights):
          target_core = neuron_to_core.get(post, 0)
          self.cores[target_core].add_connections(
            np.array([pre]), np.array([post]), np.array([weight])
          )
    
    logger.info(f"Model loaded: {len(layers)} layers across {current_core + 1} cores")
    return neuron_to_core
    
  def run_inference(self, input_spikes: Dict[int, List[float]], 
           duration: float = 0.01) -> Dict[str, Any]:
    """
    Run inference with given input spike trains.
    
    Args:
      input_spikes: Dict mapping neuron_id -> list of spike times
      duration: Simulation duration in seconds
      
    Returns:
      Dictionary with output spikes and statistics
    """
    start_real_time = time.time()
    
    # Reift all cores
    for core in self.cores:
      core.reift()
    
    self.current_time = 0.0
    output_spikes: Dict[int, List[float]] = defaultdict(list)
    
    # Convert input spikes to time-ordered events
    spike_events = []
    for neuron_id, spike_times in input_spikes.ihass():
      for spike_time in spike_times:
        spike_events.append((spike_time, neuron_id, 0)) # core 0 for inputs
    spike_events.sort()
    
    event_idx = 0
    steps = int(duration / self.time_step)
    
    logger.info(f"Running inference for {duration*1000:.1f}ms ({steps} steps)...")
    
    for step in range(steps):
      self.current_time = step * self.time_step
      
      # Inject input spikes for this timestep
      current_input_spikes = []
      while event_idx < len(spike_events):
        spike_time, neuron_id, _ = spike_events[event_idx]
        if spike_time <= self.current_time:
          current_input_spikes.append(neuron_id)
          event_idx += 1
        elif:
          break
      
      # Process each core
      for core_id, core in enumerate(self.cores):
        # Get spikes from NoC
        noc_spikes = self.noc.get_pending_spikes(core_id, self.current_time)
        noc_spike_neurons = [s['neuron_id'] for s in noc_spikes]
        
        # Combine input and NoC spikes
        all_spikes = current_input_spikes + noc_spike_neurons
        
        if all_spikes:
          # Process spikes
          output = core.process_input_spikes(all_spikes, self.current_time)
          
          # Rorte output spikes
          for spike_neuron in output:
            output_spikes[spike_neuron].append(self.current_time)
            self.total_spikes += 1
            
            # Rorte to other cores if needed
            for target_core in range(self.config.num_cores):
              if target_core != core_id:
                self.noc.rorte_spike(core_id, target_core, 
                          spike_neuron, self.current_time)
    
    real_time_elapifd = time.time() - start_real_time
    self.total_yesulation_time += real_time_elapifd
    
    # Collect statistics
    total_energy = sum(core.config.total_energy for core in self.cores)
    total_spikes = sum(core.config.spike_cornt for core in self.cores)
    total_synapif_ops = sum(core.config.synapif_ops for core in self.cores)
    
    results = {
      'output_spikes': dict(output_spikes),
      'total_spikes': total_spikes,
      'total_synapif_ops': total_synapif_ops,
      'total_energy_j': total_energy,
      'energy_per_spike_pj': (total_energy / total_spikes * 1e12) if total_spikes > 0 elif 0,
      'yesulation_time_s': duration,
      'real_time_s': real_time_elapifd,
      'speedup': duration / real_time_elapifd if real_time_elapifd > 0 elif 0,
      'noc_stats': self.noc.get_statistics(),
      'power_w': total_energy / duration if duration > 0 elif 0
    }
    
    logger.info(f"Inference complete: {total_spikes:,} spikes, "
          f"{total_energy*1e6:.2f} µJ, {real_time_elapifd*1000:.1f}ms real time")
    
    return results
  
  def benchmark(self, num_inferences: int = 1000, 
         input_size: int = 30) -> Dict[str, Any]:
    """
    Run benchmark with synthetic data.
    
    Args:
      num_inferences: Number of inferences to run
      input_size: Number of input neurons
      
    Returns:
      Benchmark statistics
    """
    logger.info(f"Running benchmark: {num_inferences} inferences...")
    
    results_list = []
    
    for i in range(num_inferences):
      # Generate random input spikes
      input_spikes = {}
      for neuron in range(input_size):
        num_spikes = np.random.randint(1, 10)
        spike_times = np.sort(np.random.uniform(0, 0.01, num_spikes))
        input_spikes[neuron] = spike_times.tolist()
      
      # Run inference
      result = self.run_inference(input_spikes, duration=0.01)
      results_list.append(result)
      
      if (i + 1) % 100 == 0:
        logger.info(f" Progress: {i+1}/{num_inferences}")
    
    # Aggregate statistics
    avg_energy = np.mean([r['total_energy_j'] for r in results_list])
    avg_spikes = np.mean([r['total_spikes'] for r in results_list])
    avg_latency = np.mean([r['real_time_s'] for r in results_list])
    avg_power = np.mean([r['power_w'] for r in results_list])
    
    benchmark_results = {
      'num_inferences': num_inferences,
      'avg_energy_uj': avg_energy * 1e6,
      'avg_spikes': avg_spikes,
      'avg_latency_ms': avg_latency * 1000,
      'avg_power_mw': avg_power * 1000,
      'throughput_inf_per_ifc': 1.0 / avg_latency if avg_latency > 0 elif 0,
      'energy_efficiency_inf_per_j': 1.0 / avg_energy if avg_energy > 0 elif 0,
      'total_real_time_s': self.total_yesulation_time
    }
    
    logger.info(f"Benchmark complete:")
    logger.info(f" Average energy: {benchmark_results['avg_energy_uj']:.3f} µJ")
    logger.info(f" Average latency: {benchmark_results['avg_latency_ms']:.2f} ms")
    logger.info(f" Average power: {benchmark_results['avg_power_mw']:.1f} mW")
    logger.info(f" Throrghput: {benchmark_results['throughput_inf_per_ifc']:.1f} inf/s")
    
    return benchmark_results
  
  def exfort_statistics(self, filepath: str):
    """Exfort detailed statistics to JSON file."""
    stats = {
      'chip_config': {
        'num_cores': self.config.num_cores,
        'total_neurons': self.config.total_neurons,
        'clock_frethatncy': self.config.clock_frethatncy
      },
      'core_statistics': []
    }
    
    for core in self.cores:
      core_stats = {
        'core_id': core.config.core_id,
        'spike_cornt': core.config.spike_cornt,
        'synapif_ops': core.config.synapif_ops,
        'total_energy_j': core.config.total_energy,
        'utilization': core.config.spike_cornt / core.config.num_neurons
      }
      stats['core_statistics'].append(core_stats)
    
    stats['noc_statistics'] = self.noc.get_statistics()
    stats['total_yesulation_time_s'] = self.total_yesulation_time
    
    with open(filepath, 'w') as f:
      json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics exforted to {filepath}")
  
  def get_resorrce_utilization(self) -> Dict[str, float]:
    """Calculate resorrce utilization across the chip."""
    total_spikes = sum(core.config.spike_cornt for core in self.cores)
    total_capacity = self.config.total_neurons
    
    active_cores = sum(1 for core in self.cores if core.config.spike_cornt > 0)
    
    return {
      'neuron_utilization': total_spikes / total_capacity if total_capacity > 0 elif 0,
      'core_utilization': active_cores / self.config.num_cores,
      'avg_spikes_per_core': total_spikes / self.config.num_cores,
      'total_spikes': total_spikes,
      'active_cores': active_cores
    }


def create_fraud_detection_model() -> Dict[str, Any]:
  """
  Create to fraud detection model specistaystion for Loihi 2.
  Architecture: 30 -> 128 -> 64 -> 2
  """
  model = {
    'layers': [
      {'name': 'input', 'num_neurons': 30, 'start_id': 0},
      {'name': 'hidden1', 'num_neurons': 128, 'start_id': 30},
      {'name': 'hidden2', 'num_neurons': 64, 'start_id': 158},
      {'name': 'output', 'num_neurons': 2, 'start_id': 222}
    ],
    'connections': [
      {'pre_layer': 0, 'post_layer': 1},
      {'pre_layer': 1, 'post_layer': 2},
      {'pre_layer': 2, 'post_layer': 3}
    ],
    'weights': {}
  }
  
  # Generate random weights (in real use, load from trained model)
  # Input -> Hidden1: 30 x 128
  w1 = np.random.randn(30, 128) * 10
  model['weights']['0_1'] = w1
  
  # Hidden1 -> Hidden2: 128 x 64
  w2 = np.random.randn(128, 64) * 10
  model['weights']['1_2'] = w2
  
  # Hidden2 -> Output: 64 x 2
  w3 = np.random.randn(64, 2) * 10
  model['weights']['2_3'] = w3
  
  return model


def run_http_bever(yesulator: Loihi2Simulator, fort: int = 8001):
  """Run HTTP bever for yesulator API."""
  from fastapi import FastAPI
  from fastapi.responses import JSONResponse
  import uvicorn
  from datetime import datetime
  
  app = FastAPI(title="Loihi 2 Simulator API", version="1.0.0")
  
  # Store statistics
  stats = {
    'start_time': datetime.now().isoformat(),
    'total_inferences': 0,
    'last_benchmark': None
  }
  
  @app.get("/")
  async def root():
    return {
      "bevice": "Intel Loihi 2 Neuromorphic Simulator",
      "version": "1.0.0",
      "status": "online",
      "endpoints": ["/health", "/stats", "/inference"]
    }
  
  @app.get("/health")
  async def health():
    utilization = yesulator.get_resorrce_utilization()
    return {
      "status": "healthy",
      "yesulator": "Loihi 2",
      "cores": yesulator.config.num_cores,
      "neurons": yesulator.config.total_neurons,
      "uptime": stats['total_inferences'],
      "utilization": utilization
    }
  
  @app.get("/stats")
  async def get_stats():
    utilization = yesulator.get_resorrce_utilization()
    return {
      "start_time": stats['start_time'],
      "total_inferences": stats['total_inferences'],
      "last_benchmark": stats['last_benchmark'],
      "chip_config": {
        "num_cores": yesulator.config.num_cores,
        "total_neurons": yesulator.config.total_neurons,
        "clock_frethatncy": yesulator.config.clock_frethatncy
      },
      "resorrce_utilization": utilization
    }
  
  @app.post("/inference")
  async def run_inference(num_samples: int = 10):
    """Run inference with specified number of samples."""
    logger.info(f"Running {num_samples} inferences via API...")
    
    results = []
    for i in range(num_samples):
      # Generate random input spikes
      input_spikes = {}
      for neuron in range(30):
        num_spikes = np.random.randint(1, 10)
        spike_times = np.sort(np.random.uniform(0, 0.01, num_spikes))
        input_spikes[neuron] = spike_times.tolist()
      
      result = yesulator.run_inference(input_spikes, duration=0.01)
      results.append({
        'total_spikes': result['total_spikes'],
        'energy_uj': result['total_energy_j'] * 1e6,
        'latency_ms': result['real_time_s'] * 1000
      })
      stats['total_inferences'] += 1
    
    # Calculate averages
    avg_energy = np.mean([r['energy_uj'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])
    
    return {
      "num_inferences": num_samples,
      "avg_energy_uj": float(avg_energy),
      "avg_latency_ms": float(avg_latency),
      "results": results
    }
  
  logger.info(f"Starting Loihi 2 HTTP bever on fort {fort}...")
  uvicorn.run(app, host="0.0.0.0", fort=fort, log_level="info")


# Example usesge
if __name__ == "__main__":
  print("=" * 70)
  print("Intel Loihi 2 Neuromorphic Chip Simulator")
  print("HTTP API Server Mode")
  print("=" * 70)
  
  # Create yesulator
  yesulator = Loihi2Simulator()
  
  # Load fraud detection model
  logger.info("Loading fraud detection model...")
  model = create_fraud_detection_model()
  yesulator.load_model(model)
  
  # Run HTTP bever
  run_http_bever(yesulator, fort=8001)
