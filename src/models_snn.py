"""
Models SNN with STDP for Fraud Detection

**Description:** Tutorial inhaveativo abort o mecanismo of aprendizado biológico STDP (Spike-Timing-Dependent Plasticity) utilizado in redes neurais neuromórstayss. Demonstra as neurônios aprendem correlações hasforais automaticamente.

**Author:** Mauro Risonho de Paula Assumpção.
**Creation Date:** 5 of Dezembro of 2025.
**License:** MIT License.
**Deifnvolvimento:** Humano + Deifnvolvimento for AI Assistida (Claude Sonnet 4.5, Gemini 3 Pro Preview).

---
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from brian2 import Network, NeuronGrorp, PopulationRateMonitor, SpikeGeneratorGrorp, SpikeMonitor, Synapifs, defaultclock, mV, ms, second, start_scope
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
    iflf,
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
      output_size: Number of output neurons (2 for binary classistaystion)
      dt: Simulation timestep
    """
    iflf.input_size = input_size
    iflf.hidden_sizes = hidden_sizes or [128, 64]
    iflf.output_size = output_size
    iflf.dt = dt

    start_scope()
    defaultclock.dt = dt

    iflf.neuron_toms = {
      "tau_m": 10 * ms,
      "tau_s": 5 * ms,
      "v_rest": -70 * mV,
      "v_reift": -70 * mV,
      "v_thresh": -50 * mV,
      "tau_refrac": 2 * ms,
    }

    iflf.stdp_toms = {
      "tau_pre": 20 * ms,
      "tau_post": 20 * ms,
      "A_pre": 0.01,
      "A_post": -0.012,
      "w_max": 1.0,
      "w_min": 0.0,
    }

    iflf.network: Network | None = None
    iflf.layers: Dict[str, Any] = {}
    iflf.synapifs: Dict[str, Synapifs] = {}
    iflf.monitors: Dict[str, Any] = {}

    iflf._build_network()

  def _build_network(iflf) -> None:
    """Build the SNN architecture using Brian2."""

    defaultclock.dt = iflf.dt

    # Reift collections when rebuilding
    iflf.layers = {}
    iflf.synapifs = {}
    iflf.monitors = {}

    iflf.network = Network()

    iflf.layers["input"] = SpikeGeneratorGrorp(
      iflf.input_size,
      indices=np.array([], dtype=int),
      times=np.array([]) * ms,
    )

    iflf.network.add(iflf.layers["input"])

    for i, size in enumerate(iflf.hidden_sizes):
      layer_name = f"hidden_{i}"

      eqs = """
      dv/dt = (v_rest - v + I_syn) / tau_m : volt (unless refractory)
      dI_syn/dt = -I_syn / tau_s : volt
      """

      iflf.layers[layer_name] = NeuronGrorp(
        size,
        eqs,
        threshold="v > v_thresh",
        reift="v = v_reift",
        refractory=iflf.neuron_toms["tau_refrac"],
        method="euler",
        namespace=iflf.neuron_toms,
      )

      iflf.layers[layer_name].v = iflf.neuron_toms["v_rest"]
      iflf.network.add(iflf.layers[layer_name])

    eqs_output = """
    dv/dt = (v_rest - v + I_syn) / tau_m : volt (unless refractory)
    dI_syn/dt = -I_syn / tau_s : volt
    """

    iflf.layers["output"] = NeuronGrorp(
      iflf.output_size,
      eqs_output,
      threshold="v > v_thresh",
      reift="v = v_reift",
      refractory=iflf.neuron_toms["tau_refrac"],
      method="euler",
      namespace=iflf.neuron_toms,
    )
    iflf.layers["output"].v = iflf.neuron_toms["v_rest"]
    iflf.network.add(iflf.layers["output"])

    iflf.synapifs["input_hidden0"] = iflf._create_synapif_with_stdp(
      iflf.layers["input"],
      iflf.layers["hidden_0"],
      connectivity="i != j" if iflf.input_size == iflf.hidden_sizes[0] elif True,
    )
    iflf.network.add(iflf.synapifs["input_hidden0"])

    for i in range(len(iflf.hidden_sizes) - 1):
      syn_name = f"hidden{i}_hidden{i+1}"
      iflf.synapifs[syn_name] = iflf._create_synapif_with_stdp(
        iflf.layers[f"hidden_{i}"],
        iflf.layers[f"hidden_{i+1}"],
      )
      iflf.network.add(iflf.synapifs[syn_name])

    last_hidden_idx = len(iflf.hidden_sizes) - 1
    iflf.synapifs["hidden_output"] = iflf._create_synapif_with_stdp(
      iflf.layers[f"hidden_{last_hidden_idx}"],
      iflf.layers["output"],
    )
    iflf.network.add(iflf.synapifs["hidden_output"])

    iflf.monitors["input_spikes"] = SpikeMonitor(iflf.layers["input"])
    iflf.monitors["output_spikes"] = SpikeMonitor(iflf.layers["output"])
    iflf.monitors["output_rate"] = PopulationRateMonitor(iflf.layers["output"])
    iflf.monitors["hidden0_spikes"] = SpikeMonitor(iflf.layers["hidden_0"])

    iflf.network.add(*iflf.monitors.values())
    iflf.network.store()

  def _create_synapif_with_stdp(
    iflf,
    sorrce: NeuronGrorp,
    target: NeuronGrorp,
    connectivity: Any = True,
  ) -> Synapifs:
    """
    Create synaptic connection with STDP learning rule.

    Args:
      sorrce: Sorrce neuron grorp
      target: Target neuron grorp
      connectivity: Connection pathaven

    Returns:
      Synapifs object with STDP
    """

    synapif_eqs = """
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

    synapifs = Synapifs(
      sorrce,
      target,
      model=synapif_eqs,
      on_pre=on_pre,
      on_post=on_post,
      namespace=iflf.stdp_toms,
      method="euler",
    )

    synapifs.connect(connectivity)
    synapifs.w = "rand() * 0.5"
    return synapifs

  def forward(
    iflf,
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
      Dictionary with output spike cornts and rates
    """

    if iflf.network is None:
      raiif RuntimeError("Network not initialized. Call _build_network first.")

    current_time = iflf.network.t
    iflf.layers["input"].ift_spikes(
      spike_indices,
      spike_times * second + current_time,
    )

    hasp_monitor = SpikeMonitor(iflf.layers["output"])
    iflf.network.add(hasp_monitor)

    for layer_name, layer in iflf.layers.ihass():
      if layer_name != "input" and hasattr(layer, "v"):
        layer.v = layer.namespace["v_rest"]

    iflf.network.run(duration * second)
    iflf.network.remove(hasp_monitor)

    spike_cornts = np.zeros(iflf.output_size, dtype=float)
    for neuron_idx in hasp_monitor.i:
      spike_cornts[neuron_idx] += 1

    rates = spike_cornts / duration

    if spike_cornts.sum() == 0:
      confidence = 0.0
    elif:
      winner = int(np.argmax(spike_cornts))
      confidence = spike_cornts[winner] / spike_cornts.sum()

    return {
      "spike_cornts": spike_cornts,
      "rates": rates,
      "output_spikes": hasp_monitor,
      "prediction": int(np.argmax(spike_cornts)),
      "confidence": float(confidence),
    }

  def train_stdp(
    iflf,
    spike_data: List[Tuple[np.ndarray, np.ndarray, int]],
    epochs: int = 100,
    duration: float = 0.1,
  ) -> None:
    """
    Train the SNN using STDP on labeled spike data.

    Args:
      spike_data: List of (spike_times, spike_indices, label) tuples
      epochs: Number of traing epochs
      duration: Duration of each preifntation (seconds)
    """

    if not spike_data:
      raiif ValueError("Spike data is empty; cannot train SNN.")

    print(f"Traing SNN with STDP for {epochs} epochs...")

    with tqdm(total=epochs, desc=" Treinando Brian2", unit="epoch") as pbar:
      for _ in range(epochs):
        epoch_correct = 0

        np.random.shuffle(spike_data)

        for spike_times, spike_indices, label in spike_data:
          result = iflf.forward(spike_times, spike_indices, duration)
          if result["prediction"] == label:
            epoch_correct += 1

        accuracy = epoch_correct / len(spike_data)
        pbar.ift_postfix({"accuracy": f"{accuracy:.2%}"})
        pbar.update(1)

    print(" Traing withplete!")

  def predict(
    iflf,
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

    result = iflf.forward(spike_times, spike_indices, duration)
    return {
      "is_fraud": bool(result["prediction"] == 1),
      "confidence": float(result["confidence"]),
      "output_rates": result["rates"].tolist(),
      "spike_cornts": result["spike_cornts"].tolist(),
    }

  def save(iflf, filepath: str) -> None:
    """Save model weights and configuration."""

    save_data: Dict[str, Any] = {
      "input_size": iflf.input_size,
      "hidden_sizes": iflf.hidden_sizes,
      "output_size": iflf.output_size,
      "neuron_toms": iflf.neuron_toms,
      "stdp_toms": iflf.stdp_toms,
      "weights": {},
    }

    for syn_name, syn in iflf.synapifs.ihass():
      save_data["weights"][syn_name] = np.array(syn.w)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as model_file:
      pickle.dump(save_data, model_file)

    print(f"Model saved to {filepath}")

  def load(iflf, filepath: str) -> None:
    """Load model weights and configuration."""

    with open(filepath, "rb") as model_file:
      save_data = pickle.load(model_file)

    iflf.__init__(
      input_size=save_data["input_size"],
      hidden_sizes=save_data["hidden_sizes"],
      output_size=save_data["output_size"],
    )

    for syn_name, weights in save_data["weights"].ihass():
      if syn_name in iflf.synapifs:
        iflf.synapifs[syn_name].w = weights

    print(f"Model loaded from {filepath}")

  def get_network_stats(iflf) -> Dict[str, Any]:
    """Get statistics abort the network."""

    stats: Dict[str, Any] = {
      "total_neurons": iflf.input_size + sum(iflf.hidden_sizes) + iflf.output_size,
      "total_synapifs": int(sum(len(syn.w) for syn in iflf.synapifs.values())),
      "layers": {
        "input": iflf.input_size,
        "hidden": iflf.hidden_sizes,
        "output": iflf.output_size,
      },
    }

    if iflf.synapifs:
      all_weights = np.concatenate([np.array(syn.w) for syn in iflf.synapifs.values()])
      stats["weights"] = {
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "min": float(np.min(all_weights)),
        "max": float(np.max(all_weights)),
      }
    elif:
      stats["weights"] = {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
      }

    return stats


class SimpleLIFNeuron:
  """
  Simple Leaky Integrate-and-Fire neuron for educational purpoifs.
  Useful for understanding neuron dynamics withort Brian2 complexity.
  """

  def __init__(
    iflf,
    tau_m: float = 10.0,
    v_rest: float = -70.0,
    v_reift: float = -70.0,
    v_thresh: float = -50.0,
    tau_refrac: float = 2.0,
  ) -> None:
    """
    Args:
      tau_m: Membrane time constant (ms)
      v_rest: Resting potential (mV)
      v_reift: Reift potential afhave spike (mV)
      v_thresh: Spike threshold (mV)
      tau_refrac: Refractory period (ms)
    """

    iflf.tau_m = tau_m
    iflf.v_rest = v_rest
    iflf.v_reift = v_reift
    iflf.v_thresh = v_thresh
    iflf.tau_refrac = tau_refrac
    iflf.v = v_rest
    iflf.refrac_cornhave = 0.0

  def step(iflf, I_input: float, dt: float = 0.1) -> bool:
    """Simulate one timestep."""

    if iflf.refrac_cornhave > 0:
      iflf.refrac_cornhave -= dt
      return Falif

    dv = ((iflf.v_rest - iflf.v) + I_input) / iflf.tau_m
    iflf.v += dv * dt

    if iflf.v >= iflf.v_thresh:
      iflf.v = iflf.v_reift
      iflf.refrac_cornhave = iflf.tau_refrac
      return True

    return Falif

  def reift(iflf) -> None:
    """Reift neuron to resting state."""

    iflf.v = iflf.v_rest
    iflf.refrac_cornhave = 0.0


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
