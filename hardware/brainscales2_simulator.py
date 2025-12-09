"""
**Descrição:** Simulador de hardware neuromórfico BrainScaleS-2.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from scipy.integrate import odeint
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalogNeuronConfig:
    """Configuration for BrainScaleS-2 analog neuron."""
    neuron_id: int
    
    # Analog circuit parameters (emulating physical CMOS)
    capacitance: float = 2e-12  # 2 pF membrane capacitance
    leak_conductance: float = 10e-9  # 10 nS
    threshold_voltage: float = 1.0  # 1V
    reset_voltage: float = 0.0  # 0V
    
    # Analog speedup factor (vs biological)
    speedup_factor: float = 1000.0
    
    # Noise parameters (circuit mismatch)
    voltage_noise_std: float = 0.01  # 10mV noise
    parameter_mismatch: float = 0.05  # 5% variation
    
    # Energy (analog CMOS)
    energy_per_spike: float = 5e-12  # 5 pJ (lower than digital)
    static_power: float = 1e-6  # 1 µW leakage
    
    def __post_init__(self):
        """Initialize with parameter mismatch."""
        # Add fabrication mismatch
        self.capacitance *= (1 + np.random.normal(0, self.parameter_mismatch))
        self.leak_conductance *= (1 + np.random.normal(0, self.parameter_mismatch))
        self.threshold_voltage *= (1 + np.random.normal(0, self.parameter_mismatch))
        
        self.voltage = self.reset_voltage
        self.last_spike_time = -np.inf
        self.spike_count = 0
        self.total_energy = 0.0


@dataclass
class AnalogSynapseConfig:
    """Configuration for BrainScaleS-2 analog synapse."""
    weight: float
    delay: float = 1e-6  # 1 microsecond
    
    # Analog synapse characteristics
    conductance_range: Tuple[float, float] = (0, 100e-9)  # 0-100 nS
    weight_noise_std: float = 0.02  # 2% weight variation
    
    # Energy
    energy_per_event: float = 50e-15  # 50 fJ per synaptic event
    
    def __post_init__(self):
        """Apply analog noise to weight."""
        self.noisy_weight = self.weight * (1 + np.random.normal(0, self.weight_noise_std))


@dataclass
class WaferConfig:
    """Configuration for BrainScaleS-2 wafer module."""
    num_neurons: int = 512
    num_synapses_per_neuron: int = 256
    
    # Wafer-scale characteristics
    speedup_factor: float = 1000.0  # 1000x faster than biology
    analog_timestep: float = 1e-9  # 1 nanosecond resolution
    
    # Power budget
    static_power_w: float = 1e-3  # 1 mW static
    dynamic_power_per_spike_w: float = 5e-9  # 5 nW per spike


class AnalogNeuron:
    """
    Emulates BrainScaleS-2 analog neuron using continuous-time dynamics.
    Uses ODE integration for accurate analog circuit behavior.
    """
    
    def __init__(self, config: AnalogNeuronConfig):
        self.config = config
        self.synaptic_inputs: List[AnalogSynapseConfig] = []
        self.spike_times: List[float] = []
        
    def add_synapse(self, synapse: AnalogSynapseConfig):
        """Add synaptic input."""
        self.synaptic_inputs.append(synapse)
    
    def membrane_dynamics(self, V: float, t: float, I_syn: float) -> float:
        """
        Differential equation for membrane voltage (analog RC circuit).
        
        dV/dt = (-g_leak * V + I_syn) / C
        """
        tau = self.config.capacitance / self.config.leak_conductance
        dV_dt = (-V + I_syn * tau / self.config.capacitance) / tau
        
        # Add circuit noise
        noise = np.random.normal(0, self.config.voltage_noise_std)
        
        return dV_dt + noise
    
    def integrate_voltage(self, I_syn: float, dt: float) -> bool:
        """
        Integrate membrane voltage for time dt with synaptic current.
        Returns True if spike occurs.
        """
        # Simple Euler integration (analog circuits evolve continuously)
        dV = self.membrane_dynamics(self.config.voltage, 0, I_syn)
        self.config.voltage += dV * dt
        
        # Check threshold
        if self.config.voltage >= self.config.threshold_voltage:
            # Spike!
            self.config.voltage = self.config.reset_voltage
            self.config.spike_count += 1
            self.config.total_energy += self.config.energy_per_spike
            return True
        
        # Static power consumption
        self.config.total_energy += self.config.static_power * dt
        
        return False
    
    def reset(self):
        """Reset neuron state."""
        self.config.voltage = self.config.reset_voltage
        self.config.spike_count = 0
        self.spike_times.clear()


class BrainScaleS2Wafer:
    """
    Emulates a BrainScaleS-2 wafer module with analog neurons.
    Ultra-fast analog computation with 1000x speedup.
    """
    
    def __init__(self, config: Optional[WaferConfig] = None):
        self.config = config or WaferConfig()
        self.neurons: List[AnalogNeuron] = []
        self.connectivity: Dict[int, List[Tuple[int, AnalogSynapseConfig]]] = defaultdict(list)
        
        # Initialize neurons
        for i in range(self.config.num_neurons):
            neuron_config = AnalogNeuronConfig(neuron_id=i)
            neuron = AnalogNeuron(neuron_config)
            self.neurons.append(neuron)
        
        self.current_time = 0.0
        self.total_spikes = 0
        self.total_energy = 0.0
        
        logger.info(f"Initialized BrainScaleS-2 wafer with {self.config.num_neurons} neurons")
        logger.info(f"Speedup factor: {self.config.speedup_factor}x biological time")
    
    def add_connection(self, pre_neuron: int, post_neuron: int, weight: float):
        """Add synaptic connection."""
        synapse = AnalogSynapseConfig(weight=weight)
        self.connectivity[pre_neuron].append((post_neuron, synapse))
        self.neurons[post_neuron].add_synapse(synapse)
    
    def load_model(self, layers: List[int], weights: List[np.ndarray]):
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
                    if abs(weight_matrix[i, j]) > 0.01:  # Skip near-zero weights
                        pre = pre_start + i
                        post = post_start + j
                        self.add_connection(pre, post, weight_matrix[i, j])
        
        logger.info(f"Model loaded with {len(self.connectivity)} connections")
    
    def run_inference(self, input_rates: np.ndarray, duration: float = 0.001) -> Dict[str, Any]:
        """
        Run inference with analog computation.
        
        Args:
            input_rates: Firing rates for input neurons (Hz)
            duration: Simulation duration (seconds, biological time)
            
        Returns:
            Output spike rates and statistics
        """
        start_real_time = time.time()
        
        # Convert to hardware time (accelerated)
        hw_duration = duration / self.config.speedup_factor
        
        # Reset neurons
        for neuron in self.neurons:
            neuron.reset()
        
        # Generate Poisson input spikes
        num_inputs = len(input_rates)
        input_spikes = []
        for i, rate in enumerate(input_rates):
            num_spikes = np.random.poisson(rate * duration)
            spike_times = np.sort(np.random.uniform(0, hw_duration, num_spikes))
            input_spikes.append(spike_times)
        
        # Analog time integration
        dt = self.config.analog_timestep
        steps = int(hw_duration / dt)
        
        output_spikes: Dict[int, List[float]] = defaultdict(list)
        
        # Event-driven simulation
        for step in range(steps):
            t = step * dt
            
            # Inject input spikes
            for input_id, spike_times in enumerate(input_spikes):
                spike_mask = (spike_times >= t) & (spike_times < t + dt)
                if np.any(spike_mask):
                    # Propagate to connected neurons
                    if input_id in self.connectivity:
                        for post_id, synapse in self.connectivity[input_id]:
                            # Apply synaptic current
                            I_syn = synapse.noisy_weight
                            fired = self.neurons[post_id].integrate_voltage(I_syn, dt)
                            
                            if fired:
                                output_spikes[post_id].append(t)
                                self.total_spikes += 1
                                
                                # Propagate spike further
                                if post_id in self.connectivity:
                                    for next_post, next_syn in self.connectivity[post_id]:
                                        I_syn = next_syn.noisy_weight
                                        fired = self.neurons[next_post].integrate_voltage(I_syn, dt)
                                        
                                        if fired:
                                            output_spikes[next_post].append(t)
                                            self.total_spikes += 1
            
            # Evolve all neurons (analog continuous dynamics)
            for neuron in self.neurons:
                # Leakage and noise
                neuron.integrate_voltage(0, dt)
        
        real_time_elapsed = time.time() - start_real_time
        
        # Collect statistics
        total_energy = sum(n.config.total_energy for n in self.neurons)
        total_spikes = sum(n.config.spike_count for n in self.neurons)
        
        # Calculate output firing rates
        output_layer_start = self.config.num_neurons - 2  # Last 2 neurons
        output_rates = []
        for i in range(output_layer_start, self.config.num_neurons):
            rate = self.neurons[i].config.spike_count / duration  # Convert back to bio time
            output_rates.append(rate)
        
        results = {
            'output_rates': output_rates,
            'output_spikes': dict(output_spikes),
            'total_spikes': total_spikes,
            'total_energy_j': total_energy,
            'energy_per_inference_uj': total_energy * 1e6,
            'hardware_time_us': hw_duration * 1e6,
            'biological_time_ms': duration * 1000,
            'speedup': self.config.speedup_factor,
            'real_time_s': real_time_elapsed,
            'power_mw': (total_energy / hw_duration) * 1000 if hw_duration > 0 else 0,
            'latency_us': hw_duration * 1e6
        }
        
        logger.info(f"Inference complete: {total_spikes} spikes, "
                   f"{total_energy*1e6:.3f} µJ, {hw_duration*1e6:.2f} µs hardware time")
        
        return results
    
    def benchmark(self, num_inferences: int = 1000, input_size: int = 30) -> Dict[str, Any]:
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
            
            result = self.run_inference(input_rates, duration=0.01)  # 10ms bio time
            results_list.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{num_inferences}")
        
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
            'speedup_factor': self.config.speedup_factor,
            'throughput_inf_per_sec': 1e6 / avg_latency if avg_latency > 0 else 0,
            'energy_efficiency_minf_per_j': (1.0 / avg_energy) / 1e6 if avg_energy > 0 else 0
        }
        
        logger.info(f"Benchmark complete:")
        logger.info(f"  Average energy: {benchmark_results['avg_energy_uj']:.3f} µJ")
        logger.info(f"  Average latency: {benchmark_results['avg_latency_us']:.2f} µs")
        logger.info(f"  Average power: {benchmark_results['avg_power_mw']:.2f} mW")
        logger.info(f"  Throughput: {benchmark_results['throughput_inf_per_sec']:.0f} inf/s")
        
        return benchmark_results
    
    def export_statistics(self, filepath: str):
        """Export statistics to JSON."""
        stats = {
            'config': {
                'num_neurons': self.config.num_neurons,
                'speedup_factor': self.config.speedup_factor,
                'analog_timestep_ns': self.config.analog_timestep * 1e9
            },
            'neuron_statistics': []
        }
        
        for neuron in self.neurons:
            neuron_stats = {
                'neuron_id': neuron.config.neuron_id,
                'spike_count': neuron.config.spike_count,
                'total_energy_pj': neuron.config.total_energy * 1e12,
                'final_voltage': neuron.config.voltage
            }
            stats['neuron_statistics'].append(neuron_stats)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics exported to {filepath}")


class BrainScaleS2Simulator:
    """
    High-level interface for BrainScaleS-2 simulation.
    Manages multiple wafer modules for larger networks.
    """
    
    def __init__(self, num_wafers: int = 1):
        self.wafers: List[BrainScaleS2Wafer] = []
        
        for i in range(num_wafers):
            wafer = BrainScaleS2Wafer()
            self.wafers.append(wafer)
        
        logger.info(f"Initialized BrainScaleS-2 simulator with {num_wafers} wafer(s)")
    
    def load_fraud_detection_model(self):
        """Load fraud detection model: 30 -> 128 -> 64 -> 2"""
        layers = [30, 128, 64, 2]
        
        # Generate random weights
        w1 = np.random.randn(30, 128) * 0.1
        w2 = np.random.randn(128, 64) * 0.1
        w3 = np.random.randn(64, 2) * 0.1
        
        weights = [w1, w2, w3]
        
        self.wafers[0].load_model(layers, weights)
        logger.info("Fraud detection model loaded")
    
    def predict(self, input_features: np.ndarray) -> np.ndarray:
        """
        Make prediction for fraud detection.
        
        Args:
            input_features: Input features (30-dimensional)
            
        Returns:
            Output probabilities [legitimate, fraud]
        """
        # Convert features to spike rates (0-100 Hz)
        input_rates = (input_features - input_features.min()) / (input_features.max() - input_features.min() + 1e-8)
        input_rates = input_rates * 100  # Scale to 0-100 Hz
        
        result = self.wafers[0].run_inference(input_rates, duration=0.01)
        
        # Convert output rates to probabilities
        output_rates = np.array(result['output_rates'])
        output_probs = output_rates / (output_rates.sum() + 1e-8)
        
        return output_probs


def run_http_server(simulator: BrainScaleS2Simulator, port: int = 8002):
    """Run HTTP server for simulator API."""
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
            "service": "BrainScaleS-2 Analog Neuromorphic Simulator",
            "version": "1.0.0",
            "status": "online",
            "endpoints": ["/health", "/stats", "/inference"]
        }
    
    @app.get("/health")
    async def health():
        wafer = simulator.wafers[0]
        return {
            "status": "healthy",
            "simulator": "BrainScaleS-2",
            "wafers": len(simulator.wafers),
            "neurons": wafer.config.num_neurons,
            "uptime": stats['total_inferences'],
            "speedup": wafer.config.speedup_factor
        }
    
    @app.get("/stats")
    async def get_stats():
        wafer = simulator.wafers[0]
        return {
            "start_time": stats['start_time'],
            "total_inferences": stats['total_inferences'],
            "last_benchmark": stats['last_benchmark'],
            "wafer_config": {
                "num_neurons": wafer.config.num_neurons,
                "num_synapses_per_neuron": wafer.config.num_synapses_per_neuron,
                "speedup_factor": wafer.config.speedup_factor,
                "static_power_w": wafer.config.static_power_w,
                "dynamic_power_per_spike_w": wafer.config.dynamic_power_per_spike_w
            }
        }
    
    @app.post("/inference")
    async def run_inference(num_samples: int = 10):
        """Run inference with specified number of samples."""
        logger.info(f"Running {num_samples} inferences via API...")
        
        wafer = simulator.wafers[0]
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
    
    logger.info(f"Starting BrainScaleS-2 HTTP server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("BrainScaleS-2 Analog Neuromorphic Simulator")
    print("HTTP API Server Mode")
    print("=" * 70)
    
    # Create simulator
    simulator = BrainScaleS2Simulator(num_wafers=1)
    
    # Load model
    logger.info("Loading fraud detection model...")
    simulator.load_fraud_detection_model()
    
    # Run HTTP server
    run_http_server(simulator, port=8002)
