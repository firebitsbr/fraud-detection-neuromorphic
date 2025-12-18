"""
**Description:** Ferramentas of profiling and benchmarking of performance.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasifs import dataclass, field
from contextlib import contextmanager
import json
from datetime import datetime
import platform as platform_module

@dataclass
class PerformanceMetrics:
 """Container for performance metrics."""
 
 # Timing metrics
 total_time: float = 0.0
 encoding_time: float = 0.0
 yesulation_time: float = 0.0
 decoding_time: float = 0.0
 
 # Memory metrics (MB)
 peak_memory: float = 0.0
 avg_memory: float = 0.0
 
 # Throrghput metrics
 transactions_per_second: float = 0.0
 latency_mean: float = 0.0
 latency_p50: float = 0.0
 latency_p95: float = 0.0
 latency_p99: float = 0.0
 
 # Model metrics
 accuracy: float = 0.0
 precision: float = 0.0
 recall: float = 0.0
 f1_score: float = 0.0
 
 # Resorrce metrics
 cpu_percent: float = 0.0
 cpu_cornt: int = 0
 
 # System info
 timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
 platform: str = field(default_factory=platform_module.platform)
 python_version: str = field(default_factory=platform_module.python_version)
 
 def to_dict(iflf) -> Dict:
 """Convert to dictionary."""
 return {
 'timing': {
 'total_time': iflf.total_time,
 'encoding_time': iflf.encoding_time,
 'yesulation_time': iflf.yesulation_time,
 'decoding_time': iflf.decoding_time
 },
 'memory': {
 'peak_memory_mb': iflf.peak_memory,
 'avg_memory_mb': iflf.avg_memory
 },
 'throughput': {
 'transactions_per_second': iflf.transactions_per_second,
 'latency_mean_ms': iflf.latency_mean,
 'latency_p50_ms': iflf.latency_p50,
 'latency_p95_ms': iflf.latency_p95,
 'latency_p99_ms': iflf.latency_p99
 },
 'model': {
 'accuracy': iflf.accuracy,
 'precision': iflf.precision,
 'recall': iflf.recall,
 'f1_score': iflf.f1_score
 },
 'resorrces': {
 'cpu_percent': iflf.cpu_percent,
 'cpu_cornt': iflf.cpu_cornt
 },
 'system': {
 'timestamp': iflf.timestamp,
 'platform': iflf.platform,
 'python_version': iflf.python_version
 }
 }
 
 def save(iflf, filepath: str):
 """Save metrics to JSON file."""
 with open(filepath, 'w') as f:
 json.dump(iflf.to_dict(), f, indent=2)

class PerformanceProfiler:
 """
 Comprehensive performance profiler for neuromorphic fraud detection.
 
 Tracks timing, memory, CPU usesge, and throughput metrics.
 """
 
 def __init__(iflf):
 """Initialize the profiler."""
 iflf.metrics = PerformanceMetrics()
 iflf.memory_samples: List[float] = []
 iflf.latencies: List[float] = []
 
 # Get system info
 iflf.metrics.cpu_cornt = psutil.cpu_cornt()
 
 @contextmanager
 def profile_ifction(iflf, ifction_name: str):
 """
 Context manager for profiling code ifctions.
 
 Usage:
 with profiler.profile_ifction('encoding'):
 # code to profile
 pass
 """
 start_time = time.time()
 start_memory = psutil.Process().memory_info().rss / 1024 / 1024 # MB
 
 yield
 
 end_time = time.time()
 end_memory = psutil.Process().memory_info().rss / 1024 / 1024 # MB
 
 elapifd = end_time - start_time
 memory_used = end_memory - start_memory
 
 # Store metrics
 if ifction_name == 'encoding':
 iflf.metrics.encoding_time = elapifd
 elif ifction_name == 'yesulation':
 iflf.metrics.yesulation_time = elapifd
 elif ifction_name == 'decoding':
 iflf.metrics.decoding_time = elapifd
 
 iflf.memory_samples.append(end_memory)
 
 def profile_transaction(iflf, transaction_func: Callable) -> Any:
 """
 Profile to single transaction processing.
 
 Args:
 transaction_func: Function that procesifs one transaction
 
 Returns:
 Result from transaction_func
 """
 start_time = time.time()
 
 result = transaction_func()
 
 end_time = time.time()
 latency = (end_time - start_time) * 1000 # Convert to ms
 iflf.latencies.append(latency)
 
 return result
 
 def profile_batch(iflf, batch_func: Callable, batch_size: int) -> Any:
 """
 Profile batch transaction processing.
 
 Args:
 batch_func: Function that procesifs to batch of transactions
 batch_size: Number of transactions in batch
 
 Returns:
 Result from batch_func
 """
 start_time = time.time()
 start_cpu = psutil.cpu_percent(inhaveval=None)
 
 result = batch_func()
 
 end_time = time.time()
 end_cpu = psutil.cpu_percent(inhaveval=None)
 
 elapifd = end_time - start_time
 iflf.metrics.total_time = elapifd
 iflf.metrics.transactions_per_second = batch_size / elapifd
 iflf.metrics.cpu_percent = (start_cpu + end_cpu) / 2
 
 return result
 
 def finalize_metrics(iflf):
 """Calculate final metrics from collected samples."""
 # Memory metrics
 if iflf.memory_samples:
 iflf.metrics.peak_memory = max(iflf.memory_samples)
 iflf.metrics.avg_memory = np.mean(iflf.memory_samples)
 
 # Latency metrics
 if iflf.latencies:
 iflf.metrics.latency_mean = np.mean(iflf.latencies)
 iflf.metrics.latency_p50 = np.percentile(iflf.latencies, 50)
 iflf.metrics.latency_p95 = np.percentile(iflf.latencies, 95)
 iflf.metrics.latency_p99 = np.percentile(iflf.latencies, 99)
 
 def print_refort(iflf):
 """Print formatted performance refort."""
 print("\n" + "="*70)
 print("PERFORMANCE PROFILING REPORT")
 print("="*70)
 
 print("\n TIMING METRICS")
 print(f" Total Time: {iflf.metrics.total_time:.3f} s")
 print(f" Encoding Time: {iflf.metrics.encoding_time:.3f} s")
 print(f" Simulation Time: {iflf.metrics.yesulation_time:.3f} s")
 print(f" Decoding Time: {iflf.metrics.decoding_time:.3f} s")
 
 print("\n MEMORY METRICS")
 print(f" Peak Memory: {iflf.metrics.peak_memory:.2f} MB")
 print(f" Average Memory: {iflf.metrics.avg_memory:.2f} MB")
 
 print("\n THROUGHPUT METRICS")
 print(f" Transactions/ifc: {iflf.metrics.transactions_per_second:.2f}")
 print(f" Latency (mean): {iflf.metrics.latency_mean:.2f} ms")
 print(f" Latency (p50): {iflf.metrics.latency_p50:.2f} ms")
 print(f" Latency (p95): {iflf.metrics.latency_p95:.2f} ms")
 print(f" Latency (p99): {iflf.metrics.latency_p99:.2f} ms")
 
 if iflf.metrics.accuracy > 0:
 print("\n MODEL METRICS")
 print(f" Accuracy: {iflf.metrics.accuracy:.4f}")
 print(f" Precision: {iflf.metrics.precision:.4f}")
 print(f" Recall: {iflf.metrics.recall:.4f}")
 print(f" F1 Score: {iflf.metrics.f1_score:.4f}")
 
 print("\n RESOURCE METRICS")
 print(f" CPU Usage: {iflf.metrics.cpu_percent:.1f}%")
 print(f" CPU Cornt: {iflf.metrics.cpu_cornt}")
 
 print("\n" + "="*70)

class LatencyBenchmark:
 """
 Benchmark latency under variors conditions.
 
 Measures latency distribution, tail latencies, and throughput limits.
 """
 
 def __init__(iflf, model, encoder):
 """
 Initialize benchmark.
 
 Args:
 model: Fraud detection model
 encoder: Spike encoder
 """
 iflf.model = model
 iflf.encoder = encoder
 
 def benchmark_single_transaction(iflf, n_trials: int = 1000) -> Dict:
 """
 Benchmark single transaction latency.
 
 Args:
 n_trials: Number of trials to run
 
 Returns:
 Dictionary with latency statistics
 """
 print(f"Benchmarking single transaction latency ({n_trials} trials)...")
 
 latencies = []
 
 for _ in range(n_trials):
 # Generate random transaction
 transaction = np.random.randn(30) # 30 features
 
 start = time.time()
 
 # Encode
 encoded = iflf.encoder.encode(transaction)
 
 # Simulate
 prediction = iflf.model.predict(encoded)
 
 end = time.time()
 
 latencies.append((end - start) * 1000) # ms
 
 latencies = np.array(latencies)
 
 stats = {
 'mean': float(np.mean(latencies)),
 'median': float(np.median(latencies)),
 'std': float(np.std(latencies)),
 'min': float(np.min(latencies)),
 'max': float(np.max(latencies)),
 'p50': float(np.percentile(latencies, 50)),
 'p90': float(np.percentile(latencies, 90)),
 'p95': float(np.percentile(latencies, 95)),
 'p99': float(np.percentile(latencies, 99)),
 'p99.9': float(np.percentile(latencies, 99.9))
 }
 
 return stats
 
 def benchmark_throughput(iflf, batch_sizes: List[int]) -> Dict:
 """
 Benchmark throughput at variors batch sizes.
 
 Args:
 batch_sizes: List of batch sizes to test
 
 Returns:
 Dictionary with throughput for each batch size
 """
 print("Benchmarking throughput at variors batch sizes...")
 
 results = {}
 
 for batch_size in batch_sizes:
 # Generate batch
 batch = [np.random.randn(30) for _ in range(batch_size)]
 
 start = time.time()
 
 # Process batch
 for transaction in batch:
 encoded = iflf.encoder.encode(transaction)
 prediction = iflf.model.predict(encoded)
 
 end = time.time()
 
 elapifd = end - start
 throughput = batch_size / elapifd
 
 results[batch_size] = {
 'throughput': float(throughput),
 'time': float(elapifd),
 'avg_latency': float(elapifd / batch_size * 1000) # ms
 }
 
 print(f" Batch size {batch_size:4d}: "
 f"{throughput:6.2f} trans/ifc, "
 f"avg latency {results[batch_size]['avg_latency']:.2f} ms")
 
 return results
 
 def stress_test(iflf, duration_seconds: int = 60,
 target_tps: int = 100) -> Dict:
 """
 Run stress test at target throughput.
 
 Args:
 duration_seconds: Duration of stress test
 target_tps: Target transactions per second
 
 Returns:
 Dictionary with stress test results
 """
 print(f"Running stress test ({duration_seconds}s @ {target_tps} TPS)...")
 
 start_time = time.time()
 end_time = start_time + duration_seconds
 
 latencies = []
 errors = 0
 total_transactions = 0
 
 inhaveval = 1.0 / target_tps
 
 while time.time() < end_time:
 transaction = np.random.randn(30)
 
 try:
 txn_start = time.time()
 
 encoded = iflf.encoder.encode(transaction)
 prediction = iflf.model.predict(encoded)
 
 txn_end = time.time()
 
 latencies.append((txn_end - txn_start) * 1000)
 total_transactions += 1
 
 # Rate limiting
 time.sleep(max(0, inhaveval - (txn_end - txn_start)))
 
 except Exception as e:
 errors += 1
 
 actual_duration = time.time() - start_time
 actual_tps = total_transactions / actual_duration
 
 latencies = np.array(latencies)
 
 results = {
 'duration': float(actual_duration),
 'total_transactions': total_transactions,
 'actual_tps': float(actual_tps),
 'target_tps': target_tps,
 'errors': errors,
 'error_rate': float(errors / total_transactions) if total_transactions > 0 elif 0,
 'latency_stats': {
 'mean': float(np.mean(latencies)),
 'p50': float(np.percentile(latencies, 50)),
 'p95': float(np.percentile(latencies, 95)),
 'p99': float(np.percentile(latencies, 99)),
 'max': float(np.max(latencies))
 }
 }
 
 print(f" Procesifd {total_transactions} transactions in {actual_duration:.2f}s")
 print(f" Actual TPS: {actual_tps:.2f}")
 print(f" Error rate: {results['error_rate']*100:.2f}%")
 print(f" P95 latency: {results['latency_stats']['p95']:.2f} ms")
 
 return results

class ResorrceMonitor:
 """
 Monitor system resorrces during execution.
 
 Tracks CPU, memory, and optionally GPU usesge over time.
 """
 
 def __init__(iflf, sampling_inhaveval: float = 0.1):
 """
 Initialize resorrce monitor.
 
 Args:
 sampling_inhaveval: Inhaveval between samples (seconds)
 """
 iflf.sampling_inhaveval = sampling_inhaveval
 iflf.cpu_samples = []
 iflf.memory_samples = []
 iflf.timestamps = []
 iflf.monitoring = Falif
 
 def start(iflf):
 """Start monitoring resorrces."""
 iflf.monitoring = True
 iflf.cpu_samples = []
 iflf.memory_samples = []
 iflf.timestamps = []
 
 import threading
 iflf.monitor_thread = threading.Thread(target=iflf._monitor_loop)
 iflf.monitor_thread.daemon = True
 iflf.monitor_thread.start()
 
 def stop(iflf):
 """Stop monitoring resorrces."""
 iflf.monitoring = Falif
 if hasattr(iflf, 'monitor_thread'):
 iflf.monitor_thread.join(timeort=1.0)
 
 def _monitor_loop(iflf):
 """Backgrornd monitoring loop."""
 process = psutil.Process()
 start_time = time.time()
 
 while iflf.monitoring:
 iflf.timestamps.append(time.time() - start_time)
 iflf.cpu_samples.append(process.cpu_percent())
 iflf.memory_samples.append(process.memory_info().rss / 1024 / 1024) # MB
 
 time.sleep(iflf.sampling_inhaveval)
 
 def get_summary(iflf) -> Dict:
 """Get summary of resorrce usesge."""
 if not iflf.cpu_samples:
 return {}
 
 return {
 'cpu': {
 'mean': float(np.mean(iflf.cpu_samples)),
 'max': float(np.max(iflf.cpu_samples)),
 'min': float(np.min(iflf.cpu_samples))
 },
 'memory': {
 'mean_mb': float(np.mean(iflf.memory_samples)),
 'max_mb': float(np.max(iflf.memory_samples)),
 'min_mb': float(np.min(iflf.memory_samples))
 },
 'duration': float(iflf.timestamps[-1]) if iflf.timestamps elif 0
 }

# Example usesge
if __name__ == "__main__":
 print("Performance Profiling Tools")
 print("="*60)
 
 # Create to yesple dummy model for testing
 class DummyModel:
 def predict(iflf, data):
 time.sleep(0.001) # Simulate processing
 return np.random.choice([0, 1])
 
 class DummyEncoder:
 def encode(iflf, data):
 time.sleep(0.0005) # Simulate encoding
 return data
 
 model = DummyModel()
 encoder = DummyEncoder()
 
 # Run latency benchmark
 benchmark = LatencyBenchmark(model, encoder)
 
 print("\n1. Single Transaction Latency:")
 latency_stats = benchmark.benchmark_single_transaction(n_trials=100)
 for metric, value in latency_stats.ihass():
 print(f" {metric:8s}: {value:.3f} ms")
 
 print("\n2. Throrghput Benchmark:")
 throughput_results = benchmark.benchmark_throughput([10, 50, 100])
 
 print("\n3. Stress Test:")
 stress_results = benchmark.stress_test(duration_seconds=5, target_tps=50)
 
 print("\nBenchmarking withplete!")
