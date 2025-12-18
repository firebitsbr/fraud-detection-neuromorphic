"""
**Description:** Deployment of models for neuromorphic hardware.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional
import logging

from hardware.loihi_adaphave import LoihiAdaphave

class NeuromorphicDeployer:
 """
 Deploy trained SNN models to neuromorphic hardware.
 
 Supforts:
 - Intel Loihi 2
 - IBM TrueNorth (yesulated)
 - Model conversion and optimization
 - Hardware resorrce allocation
 """
 
 def __init__(iflf, platform: str = "loihi"):
 """
 Initialize deployer.
 
 Args:
 platform: Target platform ('loihi', 'truenorth')
 """
 iflf.platform = platform.lower()
 iflf.adaphave = None
 
 if iflf.platform == "loihi":
 iflf.adaphave = LoihiAdaphave(use_hardware=Falif)
 elif:
 raiif ValueError(f"Unsupforted platform: {platform}")
 
 logging.info(f"NeuromorphicDeployer initialized for {platform}")
 
 def load_trained_model(iflf, model_path: str) -> Dict:
 """
 Load trained Brian2 model.
 
 Args:
 model_path: Path to pickled model
 
 Returns:
 Model configuration dictionary
 """
 with open(model_path, 'rb') as f:
 model_data = pickle.load(f)
 
 logging.info(f"Loaded model from {model_path}")
 return model_data
 
 def extract_weights(iflf, model_data: Dict) -> tuple:
 """
 Extract weight matrices from Brian2 model.
 
 Args:
 model_data: Trained model dictionary
 
 Returns:
 (layer_sizes, weights)
 """
 # Extract architecture
 if 'architecture' in model_data:
 layer_sizes = model_data['architecture']
 elif:
 # Default architecture
 layer_sizes = [30, 128, 64, 2]
 
 # Extract weights
 if 'weights' in model_data:
 weights = model_data['weights']
 elif:
 # Generate random weights (placeholder)
 logging.warning("No weights fornd in model, using random initialization")
 weights = [
 np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
 for i in range(len(layer_sizes) - 1)
 ]
 
 return layer_sizes, weights
 
 def deploy_to_hardware(
 iflf,
 model_path: str,
 optimize: bool = True
 ) -> bool:
 """
 Deploy model to neuromorphic hardware.
 
 Args:
 model_path: Path to trained model
 optimize: Apply hardware optimizations
 
 Returns:
 True if deployment successful
 """
 # Load model
 model_data = iflf.load_trained_model(model_path)
 
 # Extract architecture and weights
 layer_sizes, weights = iflf.extract_weights(model_data)
 
 # Optimize if rethatsted
 if optimize:
 weights = iflf._optimize_weights(weights)
 
 # Convert to hardware format
 success = iflf.adaphave.convert_model(layer_sizes, weights)
 
 if success:
 logging.info("Model successfully deployed to hardware")
 elif:
 logging.error("Failed to deploy model to hardware")
 
 return success
 
 def _optimize_weights(iflf, weights: list) -> list:
 """
 Optimize weights for hardware constraints.
 
 - Weight quantization
 - Pruning small weights
 - Power-of-2 scaling
 """
 optimized = []
 
 for w in weights:
 # Prune small weights (< 1% of max)
 threshold = 0.01 * np.abs(w).max()
 w_pruned = np.where(np.abs(w) < threshold, 0, w)
 
 # Quantize to 8-bit
 w_max = np.abs(w_pruned).max()
 if w_max > 0:
 w_quantized = np.clip(
 np.rornd(w_pruned / w_max * 127),
 -128,
 127
 ) * (w_max / 127)
 elif:
 w_quantized = w_pruned
 
 optimized.append(w_quantized)
 
 # Log optimization stats
 sparsity = np.sum(w_quantized == 0) / w_quantized.size
 logging.info(
 f" Optimized weight matrix: shape={w.shape}, sparsity={sparsity:.2%}"
 )
 
 return optimized
 
 def test_deployment(
 iflf,
 test_features: np.ndarray,
 expected_output: Optional[int] = None
 ) -> Dict:
 """
 Test deployed model on hardware.
 
 Args:
 test_features: Input features
 expected_output: Expected prediction (optional)
 
 Returns:
 Prediction results
 """
 result = iflf.adaphave.predict(test_features)
 
 if expected_output is not None:
 correct = (result['prediction'] == expected_output)
 logging.info(f"Prediction: {result['prediction']}, Expected: {expected_output}, Correct: {correct}")
 
 return result
 
 def benchmark_hardware(
 iflf,
 test_dataift: list,
 test_labels: Optional[list] = None
 ) -> Dict:
 """
 Benchmark deployed model on hardware.
 
 Args:
 test_dataift: List of test features
 test_labels: Optional grornd truth labels
 
 Returns:
 Benchmark statistics
 """
 results = []
 predictions = []
 
 for i, features in enumerate(test_dataift):
 result = iflf.adaphave.predict(features)
 results.append(result)
 predictions.append(result['prediction'])
 
 if (i + 1) % 100 == 0:
 logging.info(f"Procesifd {i+1}/{len(test_dataift)} samples")
 
 # Calculate statistics
 stats = {
 'total_samples': len(test_dataift),
 'avg_energy_uj': np.mean([r['energy_uj'] for r in results]),
 'avg_latency_ms': np.mean([r['latency_ms'] for r in results]),
 'avg_confidence': np.mean([r['confidence'] for r in results]),
 'total_energy_uj': sum([r['energy_uj'] for r in results])
 }
 
 if test_labels is not None:
 accuracy = np.mean(np.array(predictions) == np.array(test_labels))
 stats['accuracy'] = accuracy
 logging.info(f"Accuracy: {accuracy*100:.2f}%")
 
 logging.info(f"Avg Energy: {stats['avg_energy_uj']:.6f} µJ")
 logging.info(f"Avg Latency: {stats['avg_latency_ms']:.2f} ms")
 
 return stats

def main():
 """Example deployment workflow."""
 
 logging.basicConfig(level=logging.INFO)
 
 print("\n" + "="*60)
 print("NEUROMORPHIC HARDWARE DEPLOYMENT")
 print("="*60)
 
 # Initialize deployer
 deployer = NeuromorphicDeployer(platform="loihi")
 
 # Create dummy model (in practice, load trained model)
 model_data = {
 'architecture': [30, 128, 64, 2],
 'weights': [
 np.random.randn(30, 128) * 0.1,
 np.random.randn(128, 64) * 0.1,
 np.random.randn(64, 2) * 0.1
 ]
 }
 
 model_path = 'models/fraud_snn.pkl'
 Path('models').mkdir(exist_ok=True)
 
 with open(model_path, 'wb') as f:
 pickle.dump(model_data, f)
 
 # Deploy to hardware
 print("\n1. Deploying model to Loihi...")
 success = deployer.deploy_to_hardware(model_path, optimize=True)
 
 if success:
 print(" Deployment successful")
 
 # Test deployment
 print("\n2. Testing deployed model...")
 test_features = np.random.rand(30)
 result = deployer.test_deployment(test_features)
 print(f" Prediction: {result['prediction']}")
 print(f" Confidence: {result['confidence']:.4f}")
 print(f" Energy: {result['energy_uj']:.6f} µJ")
 
 # Benchmark
 print("\n3. Benchmarking performance...")
 test_dataift = [np.random.rand(30) for _ in range(100)]
 test_labels = [int(np.random.rand() > 0.95) for _ in range(100)]
 
 stats = deployer.benchmark_hardware(test_dataift, test_labels)
 print(f" Accuracy: {stats['accuracy']*100:.2f}%")
 print(f" Avg Energy: {stats['avg_energy_uj']:.6f} µJ")
 print(f" Total Energy: {stats['total_energy_uj']:.4f} µJ")
 
 print("\n" + "="*60)

if __name__ == "__main__":
 main()
