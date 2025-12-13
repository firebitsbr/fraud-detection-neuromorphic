"""
Modelos SNN com STDP para Detecção de Fraude

**Descrição:** Tutorial interativo sobre o mecanismo de aprendizado biológico STDP (Spike-Timing-Dependent Plasticity) utilizado em redes neurais neuromórficas. Demonstra como neurônios aprendem correlações temporais automaticamente.

**Autor:** Mauro Risonho de Paula Assumpção.
**Data de Criação:** 5 de Dezembro de 2025.
**Licença:** MIT License.
**Desenvolvimento:** Humano + Desenvolvimento por AI Assistida (Claude Sonnet 4.5, Gemini 3 Pro Preview).

---
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from brian2 import Network, NeuronGroup, PopulationRateMonitor, SpikeGeneratorGroup, SpikeMonitor, Synapses, defaultclock, mV, ms, second, start_scope
from tqdm import tqdm


class FraudSNN:
    """
    Spiking Neural Network for fraud detection using Brian2.

    Architecture:
    - Input layer: Spike-encoded transaction features
    - Hidden layers: LIF neurons with STDP
    - Output layer: 2 neurons (legitimate / fraudulent)

    Learning: Spike-Timing-Dependent Plasticity (STDP)
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_sizes: List[int] | None = None,
        output_size: int = 2,
        dt: float = 0.1 * ms,
    ) -> None:
        """
        Initialize SNN architecture.

        Args:
            input_size: Number of input neurons (spike encoding dimension)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons (2 for binary classification)
            dt: Simulation timestep
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.output_size = output_size
        self.dt = dt

        start_scope()
        defaultclock.dt = dt

        self.neuron_params = {
            "tau_m": 10 * ms,
            "tau_s": 5 * ms,
            "v_rest": -70 * mV,
            "v_reset": -70 * mV,
            "v_thresh": -50 * mV,
            "tau_refrac": 2 * ms,
        }

        self.stdp_params = {
            "tau_pre": 20 * ms,
            "tau_post": 20 * ms,
            "A_pre": 0.01,
            "A_post": -0.012,
            "w_max": 1.0,
            "w_min": 0.0,
        }

        self.network: Network | None = None
        self.layers: Dict[str, Any] = {}
        self.synapses: Dict[str, Synapses] = {}
        self.monitors: Dict[str, Any] = {}

        self._build_network()

    def _build_network(self) -> None:
        """Build the SNN architecture using Brian2."""

        defaultclock.dt = self.dt

        # Reset collections when rebuilding
        self.layers = {}
        self.synapses = {}
        self.monitors = {}

        self.network = Network()

        self.layers["input"] = SpikeGeneratorGroup(
            self.input_size,
            indices=np.array([], dtype=int),
            times=np.array([]) * ms,
        )

        self.network.add(self.layers["input"])

        for i, size in enumerate(self.hidden_sizes):
            layer_name = f"hidden_{i}"

            eqs = """
            dv/dt = (v_rest - v + I_syn) / tau_m : volt (unless refractory)
            dI_syn/dt = -I_syn / tau_s : volt
            """

            self.layers[layer_name] = NeuronGroup(
                size,
                eqs,
                threshold="v > v_thresh",
                reset="v = v_reset",
                refractory=self.neuron_params["tau_refrac"],
                method="euler",
                namespace=self.neuron_params,
            )

            self.layers[layer_name].v = self.neuron_params["v_rest"]
            self.network.add(self.layers[layer_name])

        eqs_output = """
        dv/dt = (v_rest - v + I_syn) / tau_m : volt (unless refractory)
        dI_syn/dt = -I_syn / tau_s : volt
        """

        self.layers["output"] = NeuronGroup(
            self.output_size,
            eqs_output,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=self.neuron_params["tau_refrac"],
            method="euler",
            namespace=self.neuron_params,
        )
        self.layers["output"].v = self.neuron_params["v_rest"]
        self.network.add(self.layers["output"])

        self.synapses["input_hidden0"] = self._create_synapse_with_stdp(
            self.layers["input"],
            self.layers["hidden_0"],
            connectivity="i != j" if self.input_size == self.hidden_sizes[0] else True,
        )
        self.network.add(self.synapses["input_hidden0"])

        for i in range(len(self.hidden_sizes) - 1):
            syn_name = f"hidden{i}_hidden{i+1}"
            self.synapses[syn_name] = self._create_synapse_with_stdp(
                self.layers[f"hidden_{i}"],
                self.layers[f"hidden_{i+1}"],
            )
            self.network.add(self.synapses[syn_name])

        last_hidden_idx = len(self.hidden_sizes) - 1
        self.synapses["hidden_output"] = self._create_synapse_with_stdp(
            self.layers[f"hidden_{last_hidden_idx}"],
            self.layers["output"],
        )
        self.network.add(self.synapses["hidden_output"])

        self.monitors["input_spikes"] = SpikeMonitor(self.layers["input"])
        self.monitors["output_spikes"] = SpikeMonitor(self.layers["output"])
        self.monitors["output_rate"] = PopulationRateMonitor(self.layers["output"])
        self.monitors["hidden0_spikes"] = SpikeMonitor(self.layers["hidden_0"])

        self.network.add(*self.monitors.values())
        self.network.store()

    def _create_synapse_with_stdp(
        self,
        source: NeuronGroup,
        target: NeuronGroup,
        connectivity: Any = True,
    ) -> Synapses:
        """
        Create synaptic connection with STDP learning rule.

        Args:
            source: Source neuron group
            target: Target neuron group
            connectivity: Connection pattern

        Returns:
            Synapses object with STDP
        """

        synapse_eqs = """
        w : 1
        dApre/dt = -Apre / tau_pre : 1 (event-driven)
        dApost/dt = -Apost / tau_post : 1 (event-driven)
        """

        on_pre = """
        I_syn_post += w * mV
        Apre += A_pre
        w = clip(w + Apost, w_min, w_max)
        """

        on_post = """
        Apost += A_post
        w = clip(w + Apre, w_min, w_max)
        """

        synapses = Synapses(
            source,
            target,
            model=synapse_eqs,
            on_pre=on_pre,
            on_post=on_post,
            namespace=self.stdp_params,
            method="euler",
        )

        synapses.connect(connectivity)
        synapses.w = "rand() * 0.5"
        return synapses

    def forward(
        self,
        spike_times: np.ndarray,
        spike_indices: np.ndarray,
        duration: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run forward pass through the SNN.

        Args:
            spike_times: Array of spike times (seconds)
            spike_indices: Array of neuron indices corresponding to spikes
            duration: Simulation duration (seconds)

        Returns:
            Dictionary with output spike counts and rates
        """

        if self.network is None:
            raise RuntimeError("Network not initialized. Call _build_network first.")

        current_time = self.network.t
        self.layers["input"].set_spikes(
            spike_indices,
            spike_times * second + current_time,
        )

        temp_monitor = SpikeMonitor(self.layers["output"])
        self.network.add(temp_monitor)

        for layer_name, layer in self.layers.items():
            if layer_name != "input" and hasattr(layer, "v"):
                layer.v = layer.namespace["v_rest"]

        self.network.run(duration * second)
        self.network.remove(temp_monitor)

        spike_counts = np.zeros(self.output_size, dtype=float)
        for neuron_idx in temp_monitor.i:
            spike_counts[neuron_idx] += 1

        rates = spike_counts / duration

        if spike_counts.sum() == 0:
            confidence = 0.0
        else:
            winner = int(np.argmax(spike_counts))
            confidence = spike_counts[winner] / spike_counts.sum()

        return {
            "spike_counts": spike_counts,
            "rates": rates,
            "output_spikes": temp_monitor,
            "prediction": int(np.argmax(spike_counts)),
            "confidence": float(confidence),
        }

    def train_stdp(
        self,
        spike_data: List[Tuple[np.ndarray, np.ndarray, int]],
        epochs: int = 100,
        duration: float = 0.1,
    ) -> None:
        """
        Train the SNN using STDP on labeled spike data.

        Args:
            spike_data: List of (spike_times, spike_indices, label) tuples
            epochs: Number of training epochs
            duration: Duration of each presentation (seconds)
        """

        if not spike_data:
            raise ValueError("Spike data is empty; cannot train SNN.")

        print(f"Training SNN with STDP for {epochs} epochs...")

        with tqdm(total=epochs, desc=" Treinando Brian2", unit="epoch") as pbar:
            for _ in range(epochs):
                epoch_correct = 0

                np.random.shuffle(spike_data)

                for spike_times, spike_indices, label in spike_data:
                    result = self.forward(spike_times, spike_indices, duration)
                    if result["prediction"] == label:
                        epoch_correct += 1

                accuracy = epoch_correct / len(spike_data)
                pbar.set_postfix({"accuracy": f"{accuracy:.2%}"})
                pbar.update(1)

        print(" Training complete!")

    def predict(
        self,
        spike_times: np.ndarray,
        spike_indices: np.ndarray,
        duration: float = 0.1,
    ) -> Dict[str, Any]:
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
            "is_fraud": bool(result["prediction"] == 1),
            "confidence": float(result["confidence"]),
            "output_rates": result["rates"].tolist(),
            "spike_counts": result["spike_counts"].tolist(),
        }

    def save(self, filepath: str) -> None:
        """Save model weights and configuration."""

        save_data: Dict[str, Any] = {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "neuron_params": self.neuron_params,
            "stdp_params": self.stdp_params,
            "weights": {},
        }

        for syn_name, syn in self.synapses.items():
            save_data["weights"][syn_name] = np.array(syn.w)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as model_file:
            pickle.dump(save_data, model_file)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model weights and configuration."""

        with open(filepath, "rb") as model_file:
            save_data = pickle.load(model_file)

        self.__init__(
            input_size=save_data["input_size"],
            hidden_sizes=save_data["hidden_sizes"],
            output_size=save_data["output_size"],
        )

        for syn_name, weights in save_data["weights"].items():
            if syn_name in self.synapses:
                self.synapses[syn_name].w = weights

        print(f"Model loaded from {filepath}")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the network."""

        stats: Dict[str, Any] = {
            "total_neurons": self.input_size + sum(self.hidden_sizes) + self.output_size,
            "total_synapses": int(sum(len(syn.w) for syn in self.synapses.values())),
            "layers": {
                "input": self.input_size,
                "hidden": self.hidden_sizes,
                "output": self.output_size,
            },
        }

        if self.synapses:
            all_weights = np.concatenate([np.array(syn.w) for syn in self.synapses.values()])
            stats["weights"] = {
                "mean": float(np.mean(all_weights)),
                "std": float(np.std(all_weights)),
                "min": float(np.min(all_weights)),
                "max": float(np.max(all_weights)),
            }
        else:
            stats["weights"] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return stats


class SimpleLIFNeuron:
    """
    Simple Leaky Integrate-and-Fire neuron for educational purposes.
    Useful for understanding neuron dynamics without Brian2 complexity.
    """

    def __init__(
        self,
        tau_m: float = 10.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -50.0,
        tau_refrac: float = 2.0,
    ) -> None:
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
        """Simulate one timestep."""

        if self.refrac_counter > 0:
            self.refrac_counter -= dt
            return False

        dv = ((self.v_rest - self.v) + I_input) / self.tau_m
        self.v += dv * dt

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refrac_counter = self.tau_refrac
            return True

        return False

    def reset(self) -> None:
        """Reset neuron to resting state."""

        self.v = self.v_rest
        self.refrac_counter = 0.0


def demonstrate_lif_neuron() -> Dict[str, Any]:
    """
    Demonstrate LIF neuron behavior with step current.
    Useful for notebooks and teaching.
    """

    neuron = SimpleLIFNeuron()

    dt = 0.1
    duration = 100.0
    n_steps = int(duration / dt)

    I_input = np.zeros(n_steps)
    I_input[200:] = 25.0

    v_trace: List[float] = []
    spike_times: List[float] = []

    for step in range(n_steps):
        if neuron.step(float(I_input[step]), dt):
            spike_times.append(step * dt)
        v_trace.append(neuron.v)

    return {
        "time": np.arange(n_steps) * dt,
        "voltage": np.array(v_trace),
        "spikes": spike_times,
        "input": I_input,
    }
