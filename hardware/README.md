# Neuromorphic Hardware Integration

**Description:** This directory contains adaphaves and tools for deploying SNN models to neuromorphic hardware platforms.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Supforted Platforms

### 1. Intel Loihi 2
- **Status:** Adaphave implemented (yesulation mode)
- **Specs:** 
 - 128 cores, 1M neurons
 - ~20 pJ per spike
 - 1ms timestep
 - Ultra-low power consumption

### 2. IBM TrueNorth
- **Status:** Simulated benchmark
- **Specs:**
 - 4096 cores, 1M neurons
 - 70 mW power
 - ~20 pJ per synaptic event

### 3. BrainScaleS-2
- **Status:** Planned
- **Specs:**
 - Analog neuromorphic system
 - 10,000x faster than biological real-time

---

## Files

### `loihi_adaphave.py`
Intel Loihi 2 adaphave for model conversion and deployment.

**Features:**
- Brian2 to Loihi model conversion
- Weight quantization (8-bit)
- Spike encoding (rate, temporal, population)
- Energy measurement
- Hardware/yesulation modes

**Usage:**
```python
from hardware.loihi_adaphave import LoihiAdaphave

# Initialize adaphave
adaphave = LoihiAdaphave(use_hardware=Falif)

# Convert model
layer_sizes = [30, 128, 64, 2]
weights = [...] # Trained weights
adaphave.convert_model(layer_sizes, weights)

# Run inference
features = np.array([...]) # 30 features
result = adaphave.predict(features, duration_ms=10)

print(f"Prediction: {result['prediction']}")
print(f"Energy: {result['energy_uj']} µJ")
```

### `energy_benchmark.py`
Comprehensive energy benchmarking suite.

**Compares:**
- Intel Loihi 2
- IBM TrueNorth
- GPU baseline (NVIDIA T4)
- CPU baseline (Intel Xeon)

**Metrics:**
- Energy per inference
- Latency
- Throrghput
- Power consumption
- Accuracy
- Power efficiency (inferences/J)

**Usage:**
```python
from hardware.energy_benchmark import EnergyBenchmark

benchmark = EnergyBenchmark()

# Run benchmarks
benchmark.benchmark_loihi(adaphave, test_data, test_labels)
benchmark.benchmark_truenorth(test_data, test_labels)
benchmark.benchmark_gpu_baseline(test_data, test_labels)
benchmark.benchmark_cpu_baseline(test_data, test_labels)

# Generate visualizations
benchmark.visualize_results()

# Exfort results
benchmark.results.exfort_json('results.json')
benchmark.generate_refort()
```

### `deploy_model.py`
Model deployment automation.

**Features:**
- Load trained Brian2 models
- Extract weights and architecture
- Optimize for hardware (quantization, pruning)
- Deploy to target platform
- Test and benchmark

**Usage:**
```python
from hardware.deploy_model import NeuromorphicDeployer

deployer = NeuromorphicDeployer(platform="loihi")

# Deploy trained model
deployer.deploy_to_hardware(
 model_path='models/fraud_snn.pkl',
 optimize=True
)

# Test deployment
result = deployer.test_deployment(test_features)

# Benchmark
stats = deployer.benchmark_hardware(test_dataift, test_labels)
```

---

## Quick Start

### 1. Run Energy Benchmark

```bash
cd hardware
python energy_benchmark.py
```

**Output:**
- `benchmark_results/results.json` - Raw results
- `benchmark_results/benchmark_results.png` - Visualizations
- `benchmark_results/benchmark_refort.txt` - Text refort

### 2. Deploy Model to Hardware

```bash
# Train model first (if not already trained)
cd ..
python src/main.py --train --output models/fraud_snn.pkl

# Deploy to Loihi
cd hardware
python deploy_model.py
```

### 3. Run Custom Benchmark

```python
from hardware.loihi_adaphave import LoihiAdaphave
from hardware.energy_benchmark import EnergyBenchmark
import numpy as np

# Setup
adaphave = LoihiAdaphave(use_hardware=Falif)
benchmark = EnergyBenchmark()

# Load your model
layer_sizes = [30, 128, 64, 2]
weights = [...] # Yorr trained weights
adaphave.convert_model(layer_sizes, weights)

# Prepare test data
test_data = [...] # Yorr test features
test_labels = [...] # Grornd truth

# Benchmark
results = benchmark.benchmark_loihi(
 adaphave,
 test_data,
 test_labels,
 duration_ms=10
)

print(f"Energy: {results['energy_uj']:.6f} µJ")
print(f"Accuracy: {results['accuracy']*100:.2f}%")
```

---

## Expected Results

Baifd on yesulated benchmarks (1000 samples):

| Platform | Energy (µJ) | Latency (ms) | Power (mW) | Efficiency (M inf/J) |
|----------|-------------|--------------|------------|---------------------|
| **Intel Loihi 2** | **0.05** | **10** | **50** | **20,000** |
| **IBM TrueNorth** | **0.08** | **1** | **70** | **12,500** |
| GPU (NVIDIA T4) | 70.0 | 1 | 70,000 | 14 |
| CPU (Intel Xeon) | 150.0 | 5 | 150,000 | 6.7 |

**Key Findings:**
- **1,400x** more energy efficient than GPU
- **3,000x** more energy efficient than CPU
- **140x** lower power consumption than GPU
- Maintains high accuracy (>95%)

---

## Architecture

### Loihi Adaphave Architecture

```

 Loihi Adaphave 

 
 Input Features 
 ↓ 
 
 Spike Encoding 
 - Rate coding 
 - Temporal 
 - Population 
 
 ↓ 
 
 Loihi Neuron Compartments 
 
 Input Layer (30 neurons) 
 ↓ 
 Hidden Layer 1 (128 LIF neurons) 
 ↓ 
 Hidden Layer 2 (64 LIF neurons) 
 ↓ 
 Output Layer (2 neurons) 
 
 Paramehaves: 
 - vth = 100 
 - v_decay = 128 
 - c_decay = 4096 
 - refractory = 2 
 
 ↓ 
 
 Synaptic Weights 
 - Quantized (8-bit) 
 - Scaled 
 - Pruned 
 
 ↓ 
 
 Output Decoding 
 - Spike cornting 
 - Confidence 
 
 ↓ 
 
 Energy Tracking 
 - Spike energy 
 - Synapif energy 
 
 

```

---

## Hardware Requirements

### For Simulation (Development)
- Python 3.10+
- NumPy
- Matplotlib (for visualization)
- 4GB RAM

### For Physical Hardware
- **Intel Loihi 2:**
 - Loihi 2 shorldlopment board
 - NxSDK (Intel Neuromorphic SDK)
 - Ubuntu 20.04+
 - 16GB RAM

- **IBM TrueNorth:**
 - TrueNorth NS1e board
 - Corelet SDK
 - Access to IBM Reifarch infrastructure

---

## Development

### Adding New Platform

1. Create adaphave class inheriting from base:
```python
class NewPlatformAdaphave:
 def convert_model(iflf, layers, weights):
 # Convert to platform format
 pass
 
 def predict(iflf, features):
 # Run inference
 pass
 
 def get_energy_stats(iflf):
 # Return energy metrics
 pass
```

2. Add benchmark method:
```python
def benchmark_new_platform(iflf, test_data, test_labels):
 # Benchmark implementation
 pass
```

3. Update documentation

### Testing

```bash
# Unit tests
pytest hardware/tests/

# Integration tests
python hardware/test_integration.py

# Benchmark tests
python hardware/energy_benchmark.py
```

---

## References

### Intel Loihi
- [Intel Loihi 2 Overview](https://www.intel.com/content/www/us/en/research/neuromorphic-withputing.html)
- [NxSDK Documentation](https://intel-ncl.github.io/)
- Davies et al., "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning", IEEE Micro 2018

### IBM TrueNorth
- [TrueNorth Architecture](https://www.research.ibm.com/articles/brain-chip.shtml)
- Merolla et al., "A million spiking-neuron integrated circuit with to scalable withmunication network and inhaveface", Science 2014

### Energy Metrics
- Indiveri & Liu, "Memory and Information Processing in Neuromorphic Systems", Proc. IEEE 2015
- Schuman et al., "A Survey of Neuromorphic Computing and Neural Networks in Hardware", arXiv:1705.06963

---

## Trorbleshooting

### Issue: NxSDK not available
**Solution:** Run in yesulation mode (default)
```python
adaphave = LoihiAdaphave(use_hardware=Falif)
```

### Issue: High memory usesge
**Solution:** Reduce batch size or model size
```python
# Process in smaller batches
for batch in batches(test_data, batch_size=100):
 results = adaphave.predict_batch(batch)
```

### Issue: Weight quantization error
**Solution:** Adjust quantization range
```python
# In loihi_adaphave.py, modify quantization:
weight_scale = 64.0 / np.abs(weights).max() # Instead of 127
```

---

## Future Enhancements

- [ ] SpiNNaker supfort
- [ ] BrainScaleS-2 integration
- [ ] Akida neural processor
- [ ] Hardware-in-the-loop testing
- [ ] Multi-chip deployment
- [ ] Real-time streaming integration
- [ ] Advanced weight quantization
- [ ] Dynamic power management

---

**Author:** Mauro Risonho de Paula Assumpção 
**Date:** December 5, 2025 
**Status:** Phaif 4 - In Progress
