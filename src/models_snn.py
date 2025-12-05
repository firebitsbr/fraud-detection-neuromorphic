"""
Spiking Neural Network Models for Fraud Detection

Description: Implementa modelos de Spiking Neural Networks (SNNs) usando neurônios
             Leaky Integrate-and-Fire (LIF) com aprendizado STDP para detecção
             de fraude neuromórfica em transações bancárias.

Author: Mauro Risonho de Paula Assumpção
Created: December 5, 2025
License: MIT License

Implements Leaky Integrate-and-Fire (LIF) neurons with STDP learning
for neuromorphic fraud detection in banking transactions.
"""

import numpy as np
from brian2 import *
from typing import List, Tuple, Dict, Any, Optional
import pickle
from pathlib import Path


class FraudSNN:
    """
    Spiking Neural Network for fraud detection using Brian2.
    
    Architecture:
        - Input layer: Spike-encoded transaction features
        - Hidden layers: LIF neurons with STDP
        - Output layer: 2 neurons (legitimate / fraudulent)
    
    Learning: Spike-Timing-Dependent Plasticity (STDP)
    """
    
    def __init__(self, input_size: int = 256, hidden_sizes: List[int] = [128, 64],
                 output_size: int = 2, dt: float = 0.1*ms):
        """
        Initialize SNN architecture.
        
        Args:
            input_size: Number of input neurons (spike encoding dimension)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons (2 for binary classification)
            dt: Simulation timestep
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dt = dt
        
        # Brian2 setup
        start_scope()
        defaultclock.dt = dt
        
        # Neuron parameters (LIF model)
        self.neuron_params = {
            'tau_m': 10*ms,      # Membrane time constant
            'tau_s': 5*ms,       # Synapse time constant
            'v_rest': -70*mV,    # Resting potential
            'v_reset': -70*mV,   # Reset potential
            'v_thresh': -50*mV,  # Spike threshold
            'tau_refrac': 2*ms   # Refractory period
        }
        
        # STDP parameters
        self.stdp_params = {
            'tau_pre': 20*ms,    # Pre-synaptic trace time constant
            'tau_post': 20*ms,   # Post-synaptic trace time constant
            'A_pre': 0.01,       # Pre-synaptic learning rate
            'A_post': -0.012,    # Post-synaptic learning rate (depression)
            'w_max': 1.0,        # Maximum synaptic weight
            'w_min': 0.0         # Minimum synaptic weight
        }
        
        self.network = None
        self.layers = {}
        self.synapses = {}
        self.monitors = {}
        
        self._build_network()
    
    def _build_network(self):
        """Build the SNN architecture using Brian2."""
        
        # 1. INPUT LAYER (Spike Generator Group)
        self.layers['input'] = SpikeGeneratorGroup(
            self.input_size, 
            indices=[], 
            times=[]
        )
        
        # 2. HIDDEN LAYERS (LIF neurons)
        layer_sizes = [self.input_size] + self.hidden_sizes
        
        for i, size in enumerate(self.hidden_sizes):
            layer_name = f'hidden_{i}'
            
            # LIF neuron model
            eqs = '''
            dv/dt = (v_rest - v + I_syn) / tau_m : volt (unless refractory)
            dI_syn/dt = -I_syn / tau_s : volt
            '''
            
            self.layers[layer_name] = NeuronGroup(
                size,
                eqs,
                threshold='v > v_thresh',
                reset='v = v_reset',
                refractory=self.neuron_params['tau_refrac'],
                method='euler',
                namespace=self.neuron_params
            )
            
            # Initialize membrane potential
            self.layers[layer_name].v = self.neuron_params['v_rest']
        
        # 3. OUTPUT LAYER
        eqs_output = '''
        dv/dt = (v_rest - v + I_syn) / tau_m : volt (unless refractory)
        dI_syn/dt = -I_syn / tau_s : volt
        '''
        
        self.layers['output'] = NeuronGroup(
            self.output_size,
            eqs_output,
            threshold='v > v_thresh',
            reset='v = v_reset',
            refractory=self.neuron_params['tau_refrac'],
            method='euler',
            namespace=self.neuron_params
        )
        self.layers['output'].v = self.neuron_params['v_rest']
        
        # 4. SYNAPTIC CONNECTIONS WITH STDP
        
        # Input → Hidden[0]
        self.synapses['input_hidden0'] = self._create_synapse_with_stdp(
            self.layers['input'],
            self.layers['hidden_0'],
            connectivity='i != j' if self.input_size == self.hidden_sizes[0] else True
        )
        
        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            syn_name = f'hidden{i}_hidden{i+1}'
            self.synapses[syn_name] = self._create_synapse_with_stdp(
                self.layers[f'hidden_{i}'],
                self.layers[f'hidden_{i+1}']
            )
        
        # Last hidden → Output
        last_hidden_idx = len(self.hidden_sizes) - 1
        self.synapses['hidden_output'] = self._create_synapse_with_stdp(
            self.layers[f'hidden_{last_hidden_idx}'],
            self.layers['output']
        )
        
        # 5. MONITORS
        self.monitors['input_spikes'] = SpikeMonitor(self.layers['input'])
        self.monitors['output_spikes'] = SpikeMonitor(self.layers['output'])
        self.monitors['output_rate'] = PopulationRateMonitor(self.layers['output'])
        
        # Monitor first hidden layer
        self.monitors['hidden0_spikes'] = SpikeMonitor(self.layers['hidden_0'])
        
        # 6. BUILD NETWORK
        network_objects = (
            list(self.layers.values()) + 
            list(self.synapses.values()) + 
            list(self.monitors.values())
        )
        self.network = Network(network_objects)
    
    def _create_synapse_with_stdp(self, source, target, connectivity=True):
        """
        Create synaptic connection with STDP learning rule.
        
        Args:
            source: Source neuron group
            target: Target neuron group
            connectivity: Connection pattern
        
        Returns:
            Synapses object with STDP
        """
        # Synaptic equations with STDP
        synapse_eqs = '''
        w : 1
        dApre/dt = -Apre / tau_pre : 1 (event-driven)
        dApost/dt = -Apost / tau_post : 1 (event-driven)
        '''
        
        # Pre-synaptic spike: potentiation
        on_pre = '''
        I_syn_post += w * mV
        Apre += A_pre
        w = clip(w + Apost, w_min, w_max)
        '''
        
        # Post-synaptic spike: depression
        on_post = '''
        Apost += A_post
        w = clip(w + Apre, w_min, w_max)
        '''
        
        synapses = Synapses(
            source, target,
            model=synapse_eqs,
            on_pre=on_pre,
            on_post=on_post,
            namespace=self.stdp_params,
            method='euler'
        )
        
        synapses.connect(connectivity)
        
        # Initialize weights randomly
        synapses.w = 'rand() * 0.5'
        
        return synapses
    
    def forward(self, spike_times: np.ndarray, spike_indices: np.ndarray,
                duration: float = 0.1) -> Dict[str, Any]:
        """
        Run forward pass through the SNN.
        
        Args:
            spike_times: Array of spike times (seconds)
            spike_indices: Array of neuron indices corresponding to spikes
            duration: Simulation duration (seconds)
        
        Returns:
            Dictionary with output spike counts and rates
        """
        # Reset network state
        self.network.restore()
        
        # Set input spikes
        self.layers['input'].set_spikes(
            indices=spike_indices.astype(int),
            times=spike_times * second
        )
        
        # Run simulation
        self.network.run(duration * second)
        
        # Collect output
        output_spikes = self.monitors['output_spikes']
        
        # Count spikes per output neuron
        spike_counts = np.zeros(self.output_size)
        for neuron_idx in output_spikes.i:
            spike_counts[neuron_idx] += 1
        
        # Calculate rates
        rates = spike_counts / duration
        
        return {
            'spike_counts': spike_counts,
            'rates': rates,
            'output_spikes': output_spikes,
            'prediction': np.argmax(spike_counts),
            'confidence': spike_counts[np.argmax(spike_counts)] / (spike_counts.sum() + 1e-6)
        }
    
    def train_stdp(self, spike_data: List[Tuple[np.ndarray, np.ndarray, int]], 
                   epochs: int = 100, duration: float = 0.1):
        """
        Train the SNN using STDP on labeled spike data.
        
        Args:
            spike_data: List of (spike_times, spike_indices, label) tuples
            epochs: Number of training epochs
            duration: Duration of each presentation (seconds)
        """
        print(f"Training SNN with STDP for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_correct = 0
            
            # Shuffle data
            np.random.shuffle(spike_data)
            
            for spike_times, spike_indices, label in spike_data:
                # Forward pass
                result = self.forward(spike_times, spike_indices, duration)
                
                # Check if prediction is correct
                if result['prediction'] == label:
                    epoch_correct += 1
                
                # STDP happens automatically in Brian2 synapses
                # Optional: Apply reward modulation here
            
            accuracy = epoch_correct / len(spike_data)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Accuracy: {accuracy:.2%}")
        
        print("Training complete!")
    
    def predict(self, spike_times: np.ndarray, spike_indices: np.ndarray,
                duration: float = 0.1) -> Dict[str, Any]:
        """
        Predict fraud/legitimate for input spikes.
        
        Args:
            spike_times: Array of spike times
            spike_indices: Array of neuron indices
            duration: Simulation duration
        
        Returns:
            Prediction dictionary
        """
        result = self.forward(spike_times, spike_indices, duration)
        
        return {
            'is_fraud': bool(result['prediction'] == 1),
            'confidence': float(result['confidence']),
            'output_rates': result['rates'].tolist(),
            'spike_counts': result['spike_counts'].tolist()
        }
    
    def save(self, filepath: str):
        """Save model weights and configuration."""
        save_data = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'neuron_params': self.neuron_params,
            'stdp_params': self.stdp_params,
            'weights': {}
        }
        
        # Extract synaptic weights
        for syn_name, syn in self.synapses.items():
            save_data['weights'][syn_name] = np.array(syn.w)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights and configuration."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Rebuild network with loaded config
        self.__init__(
            input_size=save_data['input_size'],
            hidden_sizes=save_data['hidden_sizes'],
            output_size=save_data['output_size']
        )
        
        # Restore weights
        for syn_name, weights in save_data['weights'].items():
            if syn_name in self.synapses:
                self.synapses[syn_name].w = weights
        
        print(f"Model loaded from {filepath}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the network."""
        stats = {
            'total_neurons': sum([
                self.input_size,
                sum(self.hidden_sizes),
                self.output_size
            ]),
            'total_synapses': sum([len(syn.w) for syn in self.synapses.values()]),
            'layers': {
                'input': self.input_size,
                'hidden': self.hidden_sizes,
                'output': self.output_size
            }
        }
        
        # Weight statistics
        all_weights = np.concatenate([np.array(syn.w) for syn in self.synapses.values()])
        stats['weights'] = {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights))
        }
        
        return stats


class SimpleLIFNeuron:
    """
    Simple Leaky Integrate-and-Fire neuron for educational purposes.
    Useful for understanding neuron dynamics without Brian2 complexity.
    """
    
    def __init__(self, tau_m: float = 10.0, v_rest: float = -70.0,
                 v_reset: float = -70.0, v_thresh: float = -50.0,
                 tau_refrac: float = 2.0):
        """
        Args:
            tau_m: Membrane time constant (ms)
            v_rest: Resting potential (mV)
            v_reset: Reset potential after spike (mV)
            v_thresh: Spike threshold (mV)
            tau_refrac: Refractory period (ms)
        """
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_refrac = tau_refrac
        
        self.v = v_rest
        self.refrac_counter = 0.0
    
    def step(self, I_input: float, dt: float = 0.1) -> bool:
        """
        Simulate one timestep.
        
        Args:
            I_input: Input current (arbitrary units)
            dt: Timestep (ms)
        
        Returns:
            True if neuron spiked, False otherwise
        """
        spiked = False
        
        # Refractory period
        if self.refrac_counter > 0:
            self.refrac_counter -= dt
            return False
        
        # Update membrane potential (Euler integration)
        dv = ((self.v_rest - self.v) + I_input) / self.tau_m
        self.v += dv * dt
        
        # Check for spike
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refrac_counter = self.tau_refrac
            spiked = True
        
        return spiked
    
    def reset(self):
        """Reset neuron to resting state."""
        self.v = self.v_rest
        self.refrac_counter = 0.0


def demonstrate_lif_neuron():
    """
    Demonstrate LIF neuron behavior with step current.
    Useful for notebooks and teaching.
    """
    neuron = SimpleLIFNeuron()
    
    # Simulation parameters
    dt = 0.1  # ms
    duration = 100  # ms
    n_steps = int(duration / dt)
    
    # Input current (step at t=20ms)
    I_input = np.zeros(n_steps)
    I_input[200:] = 25.0  # Step current
    
    # Record
    v_trace = []
    spike_times = []
    
    for step in range(n_steps):
        spiked = neuron.step(I_input[step], dt)
        v_trace.append(neuron.v)
        
        if spiked:
            spike_times.append(step * dt)
    
    return {
        'time': np.arange(n_steps) * dt,
        'voltage': np.array(v_trace),
        'spikes': spike_times,
        'input': I_input
    }
