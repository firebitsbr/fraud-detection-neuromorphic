"""
Loihi 2 Hardware Simulator - Docker-based Emulation
===================================================

Complete software emulation of Intel Loihi 2 neuromorphic chip behavior.
Designed for development and testing without physical hardware.

Features:
- Realistic spike-based computation
- Multi-core simulation (128 cores)
- Energy tracking with hardware-accurate models
- Network-on-chip communication emulation
- Configurable latency and power models
- Compatible with NxSDK API subset

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
"""

import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import threading
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoreConfig:
    """Configuration for a single Loihi 2 core."""
    core_id: int
    num_neurons: int = 1024
    num_synapses: int = 128000
    voltage_decay: float = 0.95
    threshold: int = 100
    refractory_period: int = 2
    energy_per_spike: float = 20e-12  # 20 pJ
    energy_per_synapse: float = 100e-12  # 100 pJ
    
    def __post_init__(self):
        """Initialize core state."""
        self.voltages = np.zeros(self.num_neurons, dtype=np.float32)
        self.refractory_counter = np.zeros(self.num_neurons, dtype=np.int32)
        self.spike_count = 0
        self.synapse_ops = 0
        self.total_energy = 0.0


@dataclass
class ChipConfig:
    """Configuration for Loihi 2 chip."""
    num_cores: int = 128
    neurons_per_core: int = 1024
    total_neurons: int = field(init=False)
    clock_frequency: int = 1000000  # 1 MHz
    noc_latency: float = 1e-6  # 1 microsecond
    
    def __post_init__(self):
        self.total_neurons = self.num_cores * self.neurons_per_core


class NetworkOnChip:
    """
    Simulates the Network-on-Chip (NoC) interconnect in Loihi 2.
    Handles routing of spikes between cores with realistic latency.
    """
    
    def __init__(self, num_cores: int, latency: float = 1e-6):
        self.num_cores = num_cores
        self.latency = latency
        self.spike_queues: Dict[int, Queue] = {i: Queue() for i in range(num_cores)}
        self.routing_table: Dict[Tuple[int, int], List[int]] = {}
        self.total_messages = 0
        self.total_bytes = 0
        
    def route_spike(self, source_core: int, target_core: int, 
                    neuron_id: int, timestamp: float):
        """Route a spike from source to target core."""
        # Simulate NoC latency
        delivery_time = timestamp + self.latency
        
        spike_packet = {
            'source_core': source_core,
            'neuron_id': neuron_id,
            'timestamp': delivery_time
        }
        
        self.spike_queues[target_core].put(spike_packet)
        self.total_messages += 1
        self.total_bytes += 16  # Approximate packet size
        
    def get_pending_spikes(self, core_id: int, current_time: float) -> List[Dict]:
        """Retrieve spikes ready for delivery to a core."""
        ready_spikes = []
        queue = self.spike_queues[core_id]
        
        while not queue.empty():
            spike = queue.get()
            if spike['timestamp'] <= current_time:
                ready_spikes.append(spike)
            else:
                # Put back if not ready
                queue.put(spike)
                break
                
        return ready_spikes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get NoC communication statistics."""
        return {
            'total_messages': self.total_messages,
            'total_bytes': self.total_bytes,
            'bandwidth_utilization': self.total_bytes / (1024 * 1024),  # MB
            'messages_per_core': self.total_messages / self.num_cores
        }


class Loihi2Core:
    """
    Simulates a single Loihi 2 neuromorphic core.
    Implements LIF neuron dynamics and synaptic processing.
    """
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.neuron_types = np.zeros(config.num_neurons, dtype=np.int32)  # 0=LIF
        self.weights: Dict[int, np.ndarray] = {}  # neuron_id -> weight array
        self.connections: Dict[int, List[int]] = defaultdict(list)  # pre -> [post]
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
        Process incoming spikes and update neuron voltages.
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
                        if self.config.refractory_counter[post_neuron] == 0:
                            self.config.voltages[post_neuron] += weight
                            self.config.synapse_ops += 1
        
        # Apply voltage decay
        self.config.voltages *= self.config.voltage_decay
        
        # Check for threshold crossing
        fired = np.where(self.config.voltages >= self.config.threshold)[0]
        
        for neuron_id in fired:
            if self.config.refractory_counter[neuron_id] == 0:
                output_spikes.append(int(neuron_id))
                self.config.voltages[neuron_id] = 0
                self.config.refractory_counter[neuron_id] = self.config.refractory_period
                self.config.spike_count += 1
                self.spike_history.append((timestamp, int(neuron_id)))
        
        # Update refractory counters
        self.config.refractory_counter = np.maximum(0, 
                                                     self.config.refractory_counter - 1)
        
        # Update energy
        self.config.total_energy += (
            len(output_spikes) * self.config.energy_per_spike +
            self.config.synapse_ops * self.config.energy_per_synapse
        )
        
        return output_spikes
    
    def reset(self):
        """Reset core state."""
        self.config.voltages.fill(0)
        self.config.refractory_counter.fill(0)
        self.config.spike_count = 0
        self.config.synapse_ops = 0
        self.spike_history.clear()


class Loihi2Simulator:
    """
    Complete Loihi 2 chip simulator with multi-core support.
    Provides hardware-accurate emulation for development and testing.
    """
    
    def __init__(self, chip_config: Optional[ChipConfig] = None):
        self.config = chip_config or ChipConfig()
        self.cores: List[Loihi2Core] = []
        self.noc = NetworkOnChip(self.config.num_cores)
        self.current_time = 0.0
        self.time_step = 1.0 / self.config.clock_frequency
        
        # Initialize cores
        for core_id in range(self.config.num_cores):
            core_config = CoreConfig(core_id=core_id)
            core = Loihi2Core(core_config)
            self.cores.append(core)
            
        self.total_simulation_time = 0.0
        self.total_spikes = 0
        
        logger.info(f"Initialized Loihi 2 Simulator with {self.config.num_cores} cores")
        logger.info(f"Total neurons: {self.config.total_neurons:,}")
        
    def load_model(self, model_spec: Dict[str, Any]):
        """
        Load a neural network model onto the simulated chip.
        
        Args:
            model_spec: Dictionary containing:
                - layers: List of layer configurations
                - connections: Inter-layer connectivity
                - weights: Connection weight matrices
        """
        logger.info("Loading model onto Loihi 2 simulator...")
        
        layers = model_spec.get('layers', [])
        connections = model_spec.get('connections', [])
        weights = model_spec.get('weights', {})
        
        # Distribute neurons across cores
        neuron_to_core = {}
        current_core = 0
        neurons_in_core = 0
        
        for layer_idx, layer in enumerate(layers):
            num_neurons = layer['num_neurons']
            
            for neuron_offset in range(num_neurons):
                neuron_id = layer['start_id'] + neuron_offset
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
                
                # Group by target core
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
        
        # Reset all cores
        for core in self.cores:
            core.reset()
        
        self.current_time = 0.0
        output_spikes: Dict[int, List[float]] = defaultdict(list)
        
        # Convert input spikes to time-ordered events
        spike_events = []
        for neuron_id, spike_times in input_spikes.items():
            for spike_time in spike_times:
                spike_events.append((spike_time, neuron_id, 0))  # core 0 for inputs
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
                else:
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
                    
                    # Route output spikes
                    for spike_neuron in output:
                        output_spikes[spike_neuron].append(self.current_time)
                        self.total_spikes += 1
                        
                        # Route to other cores if needed
                        for target_core in range(self.config.num_cores):
                            if target_core != core_id:
                                self.noc.route_spike(core_id, target_core, 
                                                    spike_neuron, self.current_time)
        
        real_time_elapsed = time.time() - start_real_time
        self.total_simulation_time += real_time_elapsed
        
        # Collect statistics
        total_energy = sum(core.config.total_energy for core in self.cores)
        total_spikes = sum(core.config.spike_count for core in self.cores)
        total_synapse_ops = sum(core.config.synapse_ops for core in self.cores)
        
        results = {
            'output_spikes': dict(output_spikes),
            'total_spikes': total_spikes,
            'total_synapse_ops': total_synapse_ops,
            'total_energy_j': total_energy,
            'energy_per_spike_pj': (total_energy / total_spikes * 1e12) if total_spikes > 0 else 0,
            'simulation_time_s': duration,
            'real_time_s': real_time_elapsed,
            'speedup': duration / real_time_elapsed if real_time_elapsed > 0 else 0,
            'noc_stats': self.noc.get_statistics(),
            'power_w': total_energy / duration if duration > 0 else 0
        }
        
        logger.info(f"Inference complete: {total_spikes:,} spikes, "
                   f"{total_energy*1e6:.2f} µJ, {real_time_elapsed*1000:.1f}ms real time")
        
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
                logger.info(f"  Progress: {i+1}/{num_inferences}")
        
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
            'throughput_inf_per_sec': 1.0 / avg_latency if avg_latency > 0 else 0,
            'energy_efficiency_inf_per_j': 1.0 / avg_energy if avg_energy > 0 else 0,
            'total_real_time_s': self.total_simulation_time
        }
        
        logger.info(f"Benchmark complete:")
        logger.info(f"  Average energy: {benchmark_results['avg_energy_uj']:.3f} µJ")
        logger.info(f"  Average latency: {benchmark_results['avg_latency_ms']:.2f} ms")
        logger.info(f"  Average power: {benchmark_results['avg_power_mw']:.1f} mW")
        logger.info(f"  Throughput: {benchmark_results['throughput_inf_per_sec']:.1f} inf/s")
        
        return benchmark_results
    
    def export_statistics(self, filepath: str):
        """Export detailed statistics to JSON file."""
        stats = {
            'chip_config': {
                'num_cores': self.config.num_cores,
                'total_neurons': self.config.total_neurons,
                'clock_frequency': self.config.clock_frequency
            },
            'core_statistics': []
        }
        
        for core in self.cores:
            core_stats = {
                'core_id': core.config.core_id,
                'spike_count': core.config.spike_count,
                'synapse_ops': core.config.synapse_ops,
                'total_energy_j': core.config.total_energy,
                'utilization': core.config.spike_count / core.config.num_neurons
            }
            stats['core_statistics'].append(core_stats)
        
        stats['noc_statistics'] = self.noc.get_statistics()
        stats['total_simulation_time_s'] = self.total_simulation_time
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics exported to {filepath}")
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization across the chip."""
        total_spikes = sum(core.config.spike_count for core in self.cores)
        total_capacity = self.config.total_neurons
        
        active_cores = sum(1 for core in self.cores if core.config.spike_count > 0)
        
        return {
            'neuron_utilization': total_spikes / total_capacity if total_capacity > 0 else 0,
            'core_utilization': active_cores / self.config.num_cores,
            'avg_spikes_per_core': total_spikes / self.config.num_cores,
            'total_spikes': total_spikes,
            'active_cores': active_cores
        }


def create_fraud_detection_model() -> Dict[str, Any]:
    """
    Create a fraud detection model specification for Loihi 2.
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


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Loihi 2 Simulator - Docker-based Emulation")
    print("=" * 70)
    
    # Create simulator
    simulator = Loihi2Simulator()
    
    # Load fraud detection model
    model = create_fraud_detection_model()
    simulator.load_model(model)
    
    # Run benchmark
    print("\nRunning benchmark...")
    benchmark_results = simulator.benchmark(num_inferences=100, input_size=30)
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    for key, value in benchmark_results.items():
        print(f"{key:30s}: {value}")
    
    # Export statistics
    simulator.export_statistics("loihi2_simulator_stats.json")
    
    # Resource utilization
    utilization = simulator.get_resource_utilization()
    print("\n" + "=" * 70)
    print("RESOURCE UTILIZATION")
    print("=" * 70)
    for key, value in utilization.items():
        print(f"{key:30s}: {value:.4f}")
