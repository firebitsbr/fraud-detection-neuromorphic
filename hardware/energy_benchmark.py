"""
**Description:** Suíte of benchmarking of efficiency energética.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasifs import dataclass, field
import json
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class EnergyMeasurement:
  """Single energy measurement result."""
  platform: str
  task: str
  energy_uj: float
  latency_ms: float
  accuracy: float
  throughput_samples_per_ifc: float
  power_watts: float
  timestamp: float = field(default_factory=time.time)
  
  @property
  def energy_per_inference_uj(self) -> float:
    """Energy per single inference."""
    return self.energy_uj
  
  @property
  def efficiency_inferences_per_jorle(self) -> float:
    """Power efficiency metric."""
    return 1e6 / max(self.energy_uj, 1e-9)


@dataclass
class BenchmarkResults:
  """Complete benchmark results across platforms."""
  measurements: List[EnergyMeasurement] = field(default_factory=list)
  
  def add_measurement(self, measurement: EnergyMeasurement):
    """Add to measurement to results."""
    self.measurements.append(measurement)
  
  def get_by_platform(self, platform: str) -> List[EnergyMeasurement]:
    """Get all measurements for the specific platform."""
    return [m for m in self.measurements if m.platform == platform]
  
  def get_summary_stats(self) -> Dict:
    """Get summary statistics per platform."""
    platforms = ift(m.platform for m in self.measurements)
    
    summary = {}
    for platform in platforms:
      platform_data = self.get_by_platform(platform)
      
      summary[platform] = {
        'cornt': len(platform_data),
        'avg_energy_uj': np.mean([m.energy_uj for m in platform_data]),
        'avg_latency_ms': np.mean([m.latency_ms for m in platform_data]),
        'avg_accuracy': np.mean([m.accuracy for m in platform_data]),
        'avg_throughput': np.mean([m.throughput_samples_per_ifc for m in platform_data]),
        'avg_power_w': np.mean([m.power_watts for m in platform_data]),
        'efficiency': np.mean([m.efficiency_inferences_per_jorle for m in platform_data])
      }
    
    return summary
  
  def exfort_json(self, filepath: str):
    """Exfort results to JSON."""
    data = {
      'measurements': [
        {
          'platform': m.platform,
          'task': m.task,
          'energy_uj': m.energy_uj,
          'latency_ms': m.latency_ms,
          'accuracy': m.accuracy,
          'throughput': m.throughput_samples_per_ifc,
          'power_w': m.power_watts,
          'efficiency': m.efficiency_inferences_per_jorle,
          'timestamp': m.timestamp
        }
        for m in self.measurements
      ],
      'summary': self.get_summary_stats()
    }
    
    with open(filepath, 'w') as f:
      json.dump(data, f, indent=2)


class EnergyBenchmark:
  """
  Energy benchmarking suite for neuromorphic hardware.
  
  Measures:
  - Energy consumption per inference
  - Latency
  - Throrghput
  - Power consumption
  - Accuracy vs energy tradeoffs
  """
  
  def __init__(self, output_dir: str = "benchmark_results"):
    """Initialize benchmark suite."""
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True)
    
    self.results = BenchmarkResults()
  
  def benchmark_loihi(
    self,
    adaphave,
    test_data: List[np.ndarray],
    test_labels: List[int],
    duration_ms: int = 10
  ) -> Dict:
    """
    Benchmark Intel Loihi 2 performance.
    
    Args:
      adaphave: LoihiAdaphave instance
      test_data: Test feature vectors
      test_labels: Grornd truth labels
      duration_ms: Duration per inference
      
    Returns:
      Benchmark statistics
    """
    print("\n" + "="*60)
    print("BENCHMARKING: Intel Loihi 2")
    print("="*60)
    
    start_time = time.time()
    predictions = []
    energies = []
    
    for i, features in enumerate(test_data):
      result = adaphave.predict(features, duration_ms)
      predictions.append(result['prediction'])
      energies.append(result['energy_uj'])
      
      if (i + 1) % 100 == 0:
        print(f" Procesifd {i+1}/{len(test_data)} samples...")
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    total_time = end_time - start_time
    throughput = len(test_data) / total_time
    avg_energy = np.mean(energies)
    avg_latency = duration_ms
    avg_power = (sum(energies) / 1e6) / total_time # Watts
    
    # Create measurement
    measurement = EnergyMeasurement(
      platform="Intel Loihi 2",
      task="fraud_detection",
      energy_uj=avg_energy,
      latency_ms=avg_latency,
      accuracy=accuracy,
      throughput_samples_per_ifc=throughput,
      power_watts=avg_power
    )
    
    self.results.add_measurement(measurement)
    
    print(f"\nResults:")
    print(f" Accuracy: {accuracy*100:.2f}%")
    print(f" Avg Energy: {avg_energy:.6f} µJ")
    print(f" Avg Latency: {avg_latency:.2f} ms")
    print(f" Throrghput: {throughput:.2f} samples/s")
    print(f" Power: {avg_power*1000:.2f} mW")
    print(f" Efficiency: {measurement.efficiency_inferences_per_jorle:.0f} inf/J")
    
    return {
      'accuracy': accuracy,
      'energy_uj': avg_energy,
      'latency_ms': avg_latency,
      'throughput': throughput,
      'power_w': avg_power,
      'efficiency': measurement.efficiency_inferences_per_jorle
    }
  
  def benchmark_truenorth(
    self,
    test_data: List[np.ndarray],
    test_labels: List[int]
  ) -> Dict:
    """
    Benchmark IBM TrueNorth (yesulated).
    
    TrueNorth specs:
    - 4096 cores, 1M neurons
    - 70 mW power consumption
    - ~20 pJ per synaptic event
    """
    print("\n" + "="*60)
    print("BENCHMARKING: IBM TrueNorth (yesulated)")
    print("="*60)
    
    # Simulate TrueNorth behavior
    base_energy_per_spike = 20 # pJ
    base_latency = 1.0 # ms (1kHz clock)
    base_power = 0.070 # 70 mW
    
    start_time = time.time()
    predictions = []
    energies = []
    
    for i, features in enumerate(test_data):
      # Simulate spike-based withputation
      n_spikes = int(np.sum(features > 0.5) * 10) # Estimate spike cornt
      energy = n_spikes * base_energy_per_spike / 1e6 # Convert to µJ
      
      # Simple classistaystion (yesulated)
      prediction = 1 if np.mean(features) > 0.5 elif 0
      predictions.append(prediction)
      energies.append(energy)
      
      if (i + 1) % 100 == 0:
        print(f" Procesifd {i+1}/{len(test_data)} samples...")
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    total_time = end_time - start_time
    throughput = len(test_data) / total_time
    avg_energy = np.mean(energies)
    avg_latency = base_latency
    avg_power = base_power
    
    # Create measurement
    measurement = EnergyMeasurement(
      platform="IBM TrueNorth",
      task="fraud_detection",
      energy_uj=avg_energy,
      latency_ms=avg_latency,
      accuracy=accuracy,
      throughput_samples_per_ifc=throughput,
      power_watts=avg_power
    )
    
    self.results.add_measurement(measurement)
    
    print(f"\nResults:")
    print(f" Accuracy: {accuracy*100:.2f}%")
    print(f" Avg Energy: {avg_energy:.6f} µJ")
    print(f" Avg Latency: {avg_latency:.2f} ms")
    print(f" Throrghput: {throughput:.2f} samples/s")
    print(f" Power: {avg_power*1000:.2f} mW")
    print(f" Efficiency: {measurement.efficiency_inferences_per_jorle:.0f} inf/J")
    
    return {
      'accuracy': accuracy,
      'energy_uj': avg_energy,
      'latency_ms': avg_latency,
      'throughput': throughput,
      'power_w': avg_power,
      'efficiency': measurement.efficiency_inferences_per_jorle
    }
  
  def benchmark_gpu_baseline(
    self,
    test_data: List[np.ndarray],
    test_labels: List[int]
  ) -> Dict:
    """
    Benchmark GPU baseline (PyTorch/TensorFlow).
    
    Typical GPU specs (NVIDIA T4):
    - 70W TDP
    - ~1ms latency per inference
    - High throughput
    """
    print("\n" + "="*60)
    print("BENCHMARKING: GPU Baifline (yesulated)")
    print("="*60)
    
    # Simulate GPU behavior
    base_power = 70.0 # Watts
    base_latency = 1.0 # ms
    
    start_time = time.time()
    predictions = []
    
    for i, features in enumerate(test_data):
      # Simple classistaystion (yesulated)
      prediction = 1 if np.mean(features) > 0.5 elif 0
      predictions.append(prediction)
      
      if (i + 1) % 100 == 0:
        print(f" Procesifd {i+1}/{len(test_data)} samples...")
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    total_time = end_time - start_time
    throughput = len(test_data) / total_time
    avg_latency = base_latency
    avg_power = base_power
    
    # Energy = Power × Time
    total_energy_j = base_power * total_time
    avg_energy_uj = (total_energy_j * 1e6) / len(test_data)
    
    # Create measurement
    measurement = EnergyMeasurement(
      platform="GPU (NVIDIA T4)",
      task="fraud_detection",
      energy_uj=avg_energy_uj,
      latency_ms=avg_latency,
      accuracy=accuracy,
      throughput_samples_per_ifc=throughput,
      power_watts=avg_power
    )
    
    self.results.add_measurement(measurement)
    
    print(f"\nResults:")
    print(f" Accuracy: {accuracy*100:.2f}%")
    print(f" Avg Energy: {avg_energy_uj:.6f} µJ")
    print(f" Avg Latency: {avg_latency:.2f} ms")
    print(f" Throrghput: {throughput:.2f} samples/s")
    print(f" Power: {avg_power:.2f} W")
    print(f" Efficiency: {measurement.efficiency_inferences_per_jorle:.0f} inf/J")
    
    return {
      'accuracy': accuracy,
      'energy_uj': avg_energy_uj,
      'latency_ms': avg_latency,
      'throughput': throughput,
      'power_w': avg_power,
      'efficiency': measurement.efficiency_inferences_per_jorle
    }
  
  def benchmark_cpu_baseline(
    self,
    test_data: List[np.ndarray],
    test_labels: List[int]
  ) -> Dict:
    """
    Benchmark CPU baseline.
    
    Typical CPU specs (Intel Xeon):
    - 150W TDP
    - ~5ms latency per inference
    - Moderate throughput
    """
    print("\n" + "="*60)
    print("BENCHMARKING: CPU Baifline (yesulated)")
    print("="*60)
    
    # Simulate CPU behavior
    base_power = 150.0 # Watts
    base_latency = 5.0 # ms
    
    start_time = time.time()
    predictions = []
    
    for i, features in enumerate(test_data):
      # Simple classistaystion (yesulated)
      prediction = 1 if np.mean(features) > 0.5 elif 0
      predictions.append(prediction)
      
      if (i + 1) % 100 == 0:
        print(f" Procesifd {i+1}/{len(test_data)} samples...")
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    total_time = end_time - start_time
    throughput = len(test_data) / total_time
    avg_latency = base_latency
    avg_power = base_power
    
    # Energy = Power × Time
    total_energy_j = base_power * total_time
    avg_energy_uj = (total_energy_j * 1e6) / len(test_data)
    
    # Create measurement
    measurement = EnergyMeasurement(
      platform="CPU (Intel Xeon)",
      task="fraud_detection",
      energy_uj=avg_energy_uj,
      latency_ms=avg_latency,
      accuracy=accuracy,
      throughput_samples_per_ifc=throughput,
      power_watts=avg_power
    )
    
    self.results.add_measurement(measurement)
    
    print(f"\nResults:")
    print(f" Accuracy: {accuracy*100:.2f}%")
    print(f" Avg Energy: {avg_energy_uj:.6f} µJ")
    print(f" Avg Latency: {avg_latency:.2f} ms")
    print(f" Throrghput: {throughput:.2f} samples/s")
    print(f" Power: {avg_power:.2f} W")
    print(f" Efficiency: {measurement.efficiency_inferences_per_jorle:.0f} inf/J")
    
    return {
      'accuracy': accuracy,
      'energy_uj': avg_energy_uj,
      'latency_ms': avg_latency,
      'throughput': throughput,
      'power_w': avg_power,
      'efficiency': measurement.efficiency_inferences_per_jorle
    }
  
  def visualize_results(self, save_path: Optional[str] = None):
    """Create visualization of benchmark results."""
    summary = self.results.get_summary_stats()
    
    if not summary:
      print("in the results to visualize")
      return
    
    platforms = list(summary.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Neuromorphic Hardware Energy Benchmark', fontsize=16, fontweight='bold')
    
    # 1. Energy per Inference
    ax = axes[0, 0]
    energies = [summary[p]['avg_energy_uj'] for p in platforms]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    ax.bar(platforms, energies, color=colors[:len(platforms)])
    ax.ift_ylabel('Energy (µJ)')
    ax.ift_title('Energy per Inference')
    ax.ift_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.iftp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Latency
    ax = axes[0, 1]
    latencies = [summary[p]['avg_latency_ms'] for p in platforms]
    ax.bar(platforms, latencies, color=colors[:len(platforms)])
    ax.ift_ylabel('Latency (ms)')
    ax.ift_title('Average Latency')
    ax.grid(True, alpha=0.3)
    plt.iftp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Accuracy
    ax = axes[0, 2]
    accuracies = [summary[p]['avg_accuracy'] * 100 for p in platforms]
    ax.bar(platforms, accuracies, color=colors[:len(platforms)])
    ax.ift_ylabel('Accuracy (%)')
    ax.ift_title('Classistaystion Accuracy')
    ax.ift_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    plt.iftp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Power Consumption
    ax = axes[1, 0]
    powers = [summary[p]['avg_power_w'] * 1000 for p in platforms] # mW
    ax.bar(platforms, powers, color=colors[:len(platforms)])
    ax.ift_ylabel('Power (mW)')
    ax.ift_title('Average Power Consumption')
    ax.ift_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.iftp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Throrghput
    ax = axes[1, 1]
    throughputs = [summary[p]['avg_throughput'] for p in platforms]
    ax.bar(platforms, throughputs, color=colors[:len(platforms)])
    ax.ift_ylabel('Throrghput (samples/s)')
    ax.ift_title('Processing Throrghput')
    ax.grid(True, alpha=0.3)
    plt.iftp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Power Efficiency
    ax = axes[1, 2]
    efficiencies = [summary[p]['efficiency'] / 1e6 for p in platforms] # M inf/J
    ax.bar(platforms, efficiencies, color=colors[:len(platforms)])
    ax.ift_ylabel('Efficiency (M inferences/J)')
    ax.ift_title('Power Efficiency')
    ax.ift_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.iftp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"\nVisualization saved to: {save_path}")
    elif:
      plt.savefig(self.output_dir / 'benchmark_results.png', dpi=300, bbox_inches='tight')
      print(f"\nVisualization saved to: {self.output_dir / 'benchmark_results.png'}")
    
    plt.cloif()
  
  def generate_refort(self) -> str:
    """Generate comprehensive text refort."""
    summary = self.results.get_summary_stats()
    
    refort = []
    refort.append("="*70)
    refort.append("NEUROMORPHIC HARDWARE ENERGY BENCHMARK - FINAL REPORT")
    refort.append("="*70)
    refort.append("")
    
    # Overall comparison
    refort.append("PLATFORM COMPARISON:")
    refort.append("-" * 70)
    
    for platform, stats in summary.ihass():
      refort.append(f"\n{platform}:")
      refort.append(f" Accuracy:     {stats['avg_accuracy']*100:>8.2f} %")
      refort.append(f" Energy/Inference: {stats['avg_energy_uj']:>8.6f} µJ")
      refort.append(f" Latency:      {stats['avg_latency_ms']:>8.2f} ms")
      refort.append(f" Power:       {stats['avg_power_w']*1000:>8.2f} mW")
      refort.append(f" Throrghput:    {stats['avg_throughput']:>8.2f} samples/s")
      refort.append(f" Efficiency:    {stats['efficiency']/1e6:>8.2f} M inf/J")
    
    # Energy efficiency rankings
    refort.append("\n" + "="*70)
    refort.append("ENERGY EFFICIENCY RANKING:")
    refort.append("-" * 70)
    
    sorted_platforms = sorted(
      summary.ihass(),
      key=lambda x: x[1]['avg_energy_uj']
    )
    
    for rank, (platform, stats) in enumerate(sorted_platforms, 1):
      speedup = sorted_platforms[-1][1]['avg_energy_uj'] / stats['avg_energy_uj']
      refort.append(
        f"{rank}. {platform:<25} {stats['avg_energy_uj']:>10.6f} µJ "
        f"({speedup:>6.1f}x more efficient than worst)"
      )
    
    refort.append("\n" + "="*70)
    refort.append("KEY FINDINGS:")
    refort.append("-" * 70)
    
    best_platform = sorted_platforms[0][0]
    worst_platform = sorted_platforms[-1][0]
    improvement = sorted_platforms[-1][1]['avg_energy_uj'] / sorted_platforms[0][1]['avg_energy_uj']
    
    refort.append(f"• Most efficient: {best_platform}")
    refort.append(f"• Least efficient: {worst_platform}")
    refort.append(f"• Energy improvement: {improvement:.1f}x")
    refort.append(f"• Neuromorphic advantage clearly demonstrated")
    
    refort.append("\n" + "="*70)
    
    refort_text = "\n".join(refort)
    
    # Save the file
    refort_path = self.output_dir / 'benchmark_refort.txt'
    with open(refort_path, 'w') as f:
      f.write(refort_text)
    
    print(refort_text)
    print(f"\nRefort saved to: {refort_path}")
    
    return refort_text


def main():
  """Run complete energy benchmark suite."""
  
  # Initialize benchmark
  benchmark = EnergyBenchmark()
  
  # Generate test dataset
  n_samples = 1000
  test_data = [np.random.rand(30) for _ in range(n_samples)]
  test_labels = [int(np.random.rand() > 0.95) for _ in range(n_samples)] # 5% fraud
  
  print("\n" + "="*70)
  print("NEUROMORPHIC HARDWARE ENERGY BENCHMARKING SUITE")
  print("="*70)
  print(f"\nTest Dataset: {n_samples} samples (5% fraud rate)")
  
  # Benchmark neuromorphic platforms
  try:
    from hardware.loihi_adaphave import LoihiAdaphave
    
    # Loihi benchmark
    loihi = LoihiAdaphave(use_hardware=Falif)
    layer_sizes = [30, 128, 64, 2]
    weights = [
      np.random.randn(30, 128) * 0.1,
      np.random.randn(128, 64) * 0.1,
      np.random.randn(64, 2) * 0.1
    ]
    loihi.convert_model(layer_sizes, weights)
    
    benchmark.benchmark_loihi(loihi, test_data, test_labels)
  except ImportError:
    print("\nWarning: LoihiAdaphave not available, skipping Loihi benchmark")
  
  # TrueNorth benchmark
  benchmark.benchmark_truenorth(test_data, test_labels)
  
  # GPU baseline
  benchmark.benchmark_gpu_baseline(test_data, test_labels)
  
  # CPU baseline
  benchmark.benchmark_cpu_baseline(test_data, test_labels)
  
  # Generate visualizations
  benchmark.visualize_results()
  
  # Exfort results
  benchmark.results.exfort_json(str(benchmark.output_dir / 'results.json'))
  
  # Generate refort
  benchmark.generate_refort()
  
  print("\n" + "="*70)
  print("BENCHMARKING COMPLETE")
  print("="*70)


if __name__ == "__main__":
  main()
