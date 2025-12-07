"""
Intel Loihi 2 Adapter for Neuromorphic Fraud Detection

Converts Brian2 SNN models to Intel Loihi 2 neuromorphic hardware format.
Provides seamless deployment from simulation to physical hardware.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
License: MIT License
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass


# Mock Loihi imports (for development without hardware)
try:
    import nxsdk
    from nxsdk.graph.graph import Graph
    from nxsdk.graph.processes.phase_enums import Phase
    LOIHI_AVAILABLE = True
except ImportError:
    LOIHI_AVAILABLE = False
    logging.warning("NxSDK not available. Using simulation mode.")


@dataclass
class LoihiNeuronConfig:
    """Configuration for Loihi neuron compartment."""
    vth: int = 100  # Voltage threshold
    v_decay: int = 128  # Voltage decay (u8)
    c_decay: int = 4096  # Current decay (u12)
    refractory_period: int = 2  # Refractory period in timesteps
    bias: int = 0  # Bias current
    
    
@dataclass
class LoihiSynapseConfig:
    """Configuration for Loihi synaptic connections."""
    weight: int = 1  # Synaptic weight (int)
    delay: int = 0  # Axonal delay in timesteps
    weight_exponent: int = 0  # Weight scaling exponent


class LoihiAdapter:
    """
    Adapter to convert and deploy SNN models to Intel Loihi 2 hardware.
    
    Features:
    - Model conversion from Brian2 to Loihi
    - Weight quantization and scaling
    - Hardware resource allocation
    - Real-time inference
    - Energy measurement
    """
    
    def __init__(
        self,
        n_cores: int = 128,
        timestep_ms: float = 1.0,
        use_hardware: bool = True
    ):
        """
        Initialize Loihi adapter.
        
        Args:
            n_cores: Number of Loihi cores to use
            timestep_ms: Timestep duration in milliseconds
            use_hardware: Use physical hardware if available
        """
        self.n_cores = n_cores
        self.timestep_ms = timestep_ms
        self.use_hardware = use_hardware and LOIHI_AVAILABLE
        
        self.graph = None
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        
        self.neuron_groups = {}
        self.synapse_groups = {}
        
        self.energy_stats = {
            'total_energy_uj': 0.0,
            'spike_count': 0,
            'inference_count': 0
        }
        
        logging.info(
            f"LoihiAdapter initialized (hardware={'enabled' if self.use_hardware else 'simulated'})"
        )
    
    def convert_model(
        self,
        layer_sizes: List[int],
        weights: List[np.ndarray],
        neuron_params: Optional[Dict] = None
    ) -> bool:
        """
        Convert a trained SNN model to Loihi format.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            weights: List of weight matrices between layers
            neuron_params: Optional neuron configuration parameters
            
        Returns:
            True if conversion successful
        """
        if not self.use_hardware:
            logging.warning("Running in simulation mode")
        
        # Create Loihi graph
        if self.use_hardware:
            self.graph = Graph()
        
        # Set neuron parameters
        if neuron_params is None:
            neuron_params = {
                'vth': 100,
                'v_decay': 128,
                'c_decay': 4096,
                'refractory_period': 2
            }
        
        # Create input layer
        self.input_layer = self._create_neuron_group(
            'input',
            layer_sizes[0],
            neuron_params
        )
        
        # Create hidden layers
        self.hidden_layers = []
        for i, size in enumerate(layer_sizes[1:-1]):
            layer = self._create_neuron_group(
                f'hidden_{i}',
                size,
                neuron_params
            )
            self.hidden_layers.append(layer)
        
        # Create output layer
        self.output_layer = self._create_neuron_group(
            'output',
            layer_sizes[-1],
            neuron_params
        )
        
        # Create synaptic connections
        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        
        for i, (pre_layer, post_layer, weight_matrix) in enumerate(
            zip(all_layers[:-1], all_layers[1:], weights)
        ):
            self._create_synapse_group(
                f'synapse_{i}',
                pre_layer,
                post_layer,
                weight_matrix
            )
        
        logging.info(f"Model converted successfully with {len(layer_sizes)} layers")
        return True
    
    def _create_neuron_group(
        self,
        name: str,
        size: int,
        params: Dict
    ) -> Dict:
        """Create a group of Loihi neuron compartments."""
        neuron_group = {
            'name': name,
            'size': size,
            'params': params,
            'spike_counts': np.zeros(size, dtype=np.int32)
        }
        
        if self.use_hardware:
            # Create actual Loihi compartments
            compartments = self.graph.createCompartmentGroup(
                size=size,
                vThMant=params['vth'],
                refractoryDelay=params['refractory_period']
            )
            neuron_group['compartments'] = compartments
        
        self.neuron_groups[name] = neuron_group
        return neuron_group
    
    def _create_synapse_group(
        self,
        name: str,
        pre_layer: Dict,
        post_layer: Dict,
        weights: np.ndarray
    ) -> Dict:
        """Create synaptic connections with quantized weights."""
        # Quantize weights for Loihi (8-bit signed)
        weight_scale = 127.0 / np.abs(weights).max()
        quantized_weights = np.clip(
            np.round(weights * weight_scale),
            -128,
            127
        ).astype(np.int8)
        
        synapse_group = {
            'name': name,
            'pre_layer': pre_layer['name'],
            'post_layer': post_layer['name'],
            'weights': quantized_weights,
            'weight_scale': weight_scale
        }
        
        if self.use_hardware:
            # Create actual Loihi connections
            pre_compartments = pre_layer['compartments']
            post_compartments = post_layer['compartments']
            
            # Connect with quantized weights
            pre_compartments.connect(
                post_compartments,
                prototype=self._create_connection_prototype(quantized_weights)
            )
        
        self.synapse_groups[name] = synapse_group
        return synapse_group
    
    def _create_connection_prototype(self, weights: np.ndarray):
        """Create Loihi connection prototype for synapses."""
        if not self.use_hardware:
            return None
        
        # This would create actual Loihi connection prototypes
        # Simplified for this implementation
        return {
            'weights': weights,
            'delay': 0,
            'weightExponent': 0
        }
    
    def encode_input(
        self,
        features: np.ndarray,
        encoding_type: str = 'rate',
        duration_ms: int = 10
    ) -> np.ndarray:
        """
        Encode input features as spike trains for Loihi.
        
        Args:
            features: Input feature vector (normalized 0-1)
            encoding_type: 'rate', 'temporal', or 'population'
            duration_ms: Encoding duration in milliseconds
            
        Returns:
            Spike train array [neurons x timesteps]
        """
        n_steps = int(duration_ms / self.timestep_ms)
        n_neurons = len(features)
        spike_train = np.zeros((n_neurons, n_steps), dtype=np.int8)
        
        if encoding_type == 'rate':
            # Rate coding: spike frequency proportional to feature value
            for i, value in enumerate(features):
                spike_rate = value  # Normalized 0-1
                n_spikes = int(spike_rate * n_steps)
                spike_times = np.random.choice(n_steps, n_spikes, replace=False)
                spike_train[i, spike_times] = 1
        
        elif encoding_type == 'temporal':
            # Temporal coding: spike timing encodes value
            for i, value in enumerate(features):
                if value > 0:
                    spike_time = int((1 - value) * (n_steps - 1))
                    spike_train[i, spike_time] = 1
        
        elif encoding_type == 'population':
            # Population coding: multiple neurons per feature
            # Simplified implementation
            spike_train = self._population_encode(features, n_steps)
        
        return spike_train
    
    def _population_encode(
        self,
        features: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """Population coding with Gaussian receptive fields."""
        n_features = len(features)
        n_neurons_per_feature = 4
        n_total = n_features * n_neurons_per_feature
        
        spike_train = np.zeros((n_total, n_steps), dtype=np.int8)
        
        # Create Gaussian receptive fields
        centers = np.linspace(0, 1, n_neurons_per_feature)
        sigma = 0.3
        
        for i, value in enumerate(features):
            for j, center in enumerate(centers):
                neuron_idx = i * n_neurons_per_feature + j
                activation = np.exp(-((value - center) ** 2) / (2 * sigma ** 2))
                n_spikes = int(activation * n_steps)
                
                if n_spikes > 0:
                    spike_times = np.random.choice(n_steps, n_spikes, replace=False)
                    spike_train[neuron_idx, spike_times] = 1
        
        return spike_train
    
    def predict(
        self,
        features: np.ndarray,
        duration_ms: int = 10
    ) -> Dict:
        """
        Run inference on Loihi hardware.
        
        Args:
            features: Input feature vector
            duration_ms: Inference duration in milliseconds
            
        Returns:
            Dictionary with prediction results and energy stats
        """
        # Encode input
        spike_train = self.encode_input(features, duration_ms=duration_ms)
        
        # Run simulation/hardware
        if self.use_hardware:
            output_spikes = self._run_hardware(spike_train, duration_ms)
        else:
            output_spikes = self._run_simulation(spike_train, duration_ms)
        
        # Decode output
        spike_counts = np.sum(output_spikes, axis=1)
        prediction = np.argmax(spike_counts)
        confidence = spike_counts[prediction] / np.sum(spike_counts) if np.sum(spike_counts) > 0 else 0
        
        # Update energy stats
        self._update_energy_stats(spike_train, output_spikes)
        
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'spike_counts': spike_counts.tolist(),
            'energy_uj': self.energy_stats['total_energy_uj'],
            'latency_ms': duration_ms
        }
    
    def _run_hardware(
        self,
        spike_train: np.ndarray,
        duration_ms: int
    ) -> np.ndarray:
        """Run inference on physical Loihi hardware."""
        n_steps = int(duration_ms / self.timestep_ms)
        
        # Inject input spikes
        for t in range(n_steps):
            spike_indices = np.where(spike_train[:, t] > 0)[0]
            if len(spike_indices) > 0:
                self.input_layer['compartments'].injectSpikes(spike_indices)
        
        # Run network
        self.graph.run(n_steps)
        
        # Read output spikes
        output_spikes = self.output_layer['compartments'].getSpikeCounters()
        
        return output_spikes
    
    def _run_simulation(
        self,
        spike_train: np.ndarray,
        duration_ms: int
    ) -> np.ndarray:
        """Simulate Loihi behavior in software."""
        n_steps = int(duration_ms / self.timestep_ms)
        output_size = self.output_layer['size']
        output_spikes = np.zeros((output_size, n_steps), dtype=np.int8)
        
        # Simplified LIF simulation
        voltage = np.zeros(output_size)
        threshold = 100
        
        for t in range(n_steps):
            # Propagate spikes through network (simplified)
            input_spikes = spike_train[:, t]
            
            # Accumulate weighted input
            for synapse_name, synapse in self.synapse_groups.items():
                if synapse['pre_layer'] == 'input':
                    weights = synapse['weights']
                    voltage += np.dot(weights.T, input_spikes)
            
            # Check threshold
            fired = voltage >= threshold
            output_spikes[fired, t] = 1
            voltage[fired] = 0  # Reset
            
            # Decay
            voltage *= 0.9
        
        return output_spikes
    
    def _update_energy_stats(
        self,
        input_spikes: np.ndarray,
        output_spikes: np.ndarray
    ):
        """Update energy consumption statistics."""
        # Loihi 2 approximate energy costs:
        # - Spike: ~20 pJ per spike
        # - Synaptic operation: ~100 pJ per operation
        
        spike_energy_pj = 20
        synapse_energy_pj = 100
        
        n_spikes = np.sum(input_spikes) + np.sum(output_spikes)
        n_synaptic_ops = n_spikes * np.mean([
            np.prod(s['weights'].shape) 
            for s in self.synapse_groups.values()
        ])
        
        energy_pj = (n_spikes * spike_energy_pj) + (n_synaptic_ops * synapse_energy_pj)
        self.energy_stats['total_energy_uj'] += energy_pj / 1e6
        self.energy_stats['spike_count'] += n_spikes
        self.energy_stats['inference_count'] += 1
    
    def get_energy_stats(self) -> Dict:
        """Get energy consumption statistics."""
        avg_energy = (
            self.energy_stats['total_energy_uj'] / 
            max(self.energy_stats['inference_count'], 1)
        )
        
        return {
            'total_energy_uj': self.energy_stats['total_energy_uj'],
            'average_energy_per_inference_uj': avg_energy,
            'total_spikes': self.energy_stats['spike_count'],
            'inference_count': self.energy_stats['inference_count'],
            'power_efficiency_inferences_per_joule': 1e6 / max(avg_energy, 1e-9)
        }
    
    def benchmark_energy(
        self,
        test_features: List[np.ndarray],
        duration_ms: int = 10
    ) -> Dict:
        """
        Benchmark energy consumption on test dataset.
        
        Args:
            test_features: List of feature vectors
            duration_ms: Duration per inference
            
        Returns:
            Detailed energy statistics
        """
        results = []
        
        for features in test_features:
            result = self.predict(features, duration_ms)
            results.append(result)
        
        # Aggregate statistics
        stats = self.get_energy_stats()
        stats['per_sample_results'] = results
        
        return stats
    
    def reset(self):
        """Reset the adapter and clear energy statistics."""
        self.energy_stats = {
            'total_energy_uj': 0.0,
            'spike_count': 0,
            'inference_count': 0
        }
        
        if self.use_hardware and self.graph:
            self.graph.reset()


def main():
    """Example usage of Loihi adapter."""
    
    # Initialize adapter
    adapter = LoihiAdapter(n_cores=128, use_hardware=False)
    
    # Define model architecture
    layer_sizes = [30, 128, 64, 2]  # input, hidden1, hidden2, output
    
    # Create dummy weights (in practice, load from trained model)
    weights = [
        np.random.randn(30, 128) * 0.1,
        np.random.randn(128, 64) * 0.1,
        np.random.randn(64, 2) * 0.1
    ]
    
    # Convert model
    adapter.convert_model(layer_sizes, weights)
    
    # Test prediction
    test_features = np.random.rand(30)
    result = adapter.predict(test_features, duration_ms=10)
    
    print("Prediction Result:")
    print(f"  Class: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Energy: {result['energy_uj']:.6f} µJ")
    print(f"  Latency: {result['latency_ms']} ms")
    
    # Benchmark energy
    test_dataset = [np.random.rand(30) for _ in range(100)]
    energy_stats = adapter.benchmark_energy(test_dataset)
    
    print("\nEnergy Statistics:")
    print(f"  Total Energy: {energy_stats['total_energy_uj']:.2f} µJ")
    print(f"  Avg per Inference: {energy_stats['average_energy_per_inference_uj']:.6f} µJ")
    print(f"  Power Efficiency: {energy_stats['power_efficiency_inferences_per_joule']:.0f} inf/J")


if __name__ == "__main__":
    main()
