"""
Performance Profiling and Benchmarking Tools

This module provides comprehensive profiling and benchmarking utilities for
analyzing and optimizing neuromorphic fraud detection performance.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
License: MIT License
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from datetime import datetime
import platform


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Timing metrics
    total_time: float = 0.0
    encoding_time: float = 0.0
    simulation_time: float = 0.0
    decoding_time: float = 0.0
    
    # Memory metrics (MB)
    peak_memory: float = 0.0
    avg_memory: float = 0.0
    
    # Throughput metrics
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
    
    # Resource metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # System info
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    platform: str = field(default_factory=platform.platform)
    python_version: str = field(default_factory=platform.python_version)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timing': {
                'total_time': self.total_time,
                'encoding_time': self.encoding_time,
                'simulation_time': self.simulation_time,
                'decoding_time': self.decoding_time
            },
            'memory': {
                'peak_memory_mb': self.peak_memory,
                'avg_memory_mb': self.avg_memory
            },
            'throughput': {
                'transactions_per_second': self.transactions_per_second,
                'latency_mean_ms': self.latency_mean,
                'latency_p50_ms': self.latency_p50,
                'latency_p95_ms': self.latency_p95,
                'latency_p99_ms': self.latency_p99
            },
            'model': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score
            },
            'resources': {
                'cpu_percent': self.cpu_percent,
                'cpu_count': self.cpu_count
            },
            'system': {
                'timestamp': self.timestamp,
                'platform': self.platform,
                'python_version': self.python_version
            }
        }
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for neuromorphic fraud detection.
    
    Tracks timing, memory, CPU usage, and throughput metrics.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.metrics = PerformanceMetrics()
        self.memory_samples: List[float] = []
        self.latencies: List[float] = []
        
        # Get system info
        self.metrics.cpu_count = psutil.cpu_count()
        
    @contextmanager
    def profile_section(self, section_name: str):
        """
        Context manager for profiling code sections.
        
        Usage:
            with profiler.profile_section('encoding'):
                # code to profile
                pass
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        elapsed = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Store metrics
        if section_name == 'encoding':
            self.metrics.encoding_time = elapsed
        elif section_name == 'simulation':
            self.metrics.simulation_time = elapsed
        elif section_name == 'decoding':
            self.metrics.decoding_time = elapsed
            
        self.memory_samples.append(end_memory)
        
    def profile_transaction(self, transaction_func: Callable) -> Any:
        """
        Profile a single transaction processing.
        
        Args:
            transaction_func: Function that processes one transaction
            
        Returns:
            Result from transaction_func
        """
        start_time = time.time()
        
        result = transaction_func()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        self.latencies.append(latency)
        
        return result
        
    def profile_batch(self, batch_func: Callable, batch_size: int) -> Any:
        """
        Profile batch transaction processing.
        
        Args:
            batch_func: Function that processes a batch of transactions
            batch_size: Number of transactions in batch
            
        Returns:
            Result from batch_func
        """
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        
        result = batch_func()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        
        elapsed = end_time - start_time
        self.metrics.total_time = elapsed
        self.metrics.transactions_per_second = batch_size / elapsed
        self.metrics.cpu_percent = (start_cpu + end_cpu) / 2
        
        return result
        
    def finalize_metrics(self):
        """Calculate final metrics from collected samples."""
        # Memory metrics
        if self.memory_samples:
            self.metrics.peak_memory = max(self.memory_samples)
            self.metrics.avg_memory = np.mean(self.memory_samples)
            
        # Latency metrics
        if self.latencies:
            self.metrics.latency_mean = np.mean(self.latencies)
            self.metrics.latency_p50 = np.percentile(self.latencies, 50)
            self.metrics.latency_p95 = np.percentile(self.latencies, 95)
            self.metrics.latency_p99 = np.percentile(self.latencies, 99)
            
    def print_report(self):
        """Print formatted performance report."""
        print("\n" + "="*70)
        print("PERFORMANCE PROFILING REPORT")
        print("="*70)
        
        print("\n📊 TIMING METRICS")
        print(f"  Total Time:       {self.metrics.total_time:.3f} s")
        print(f"  Encoding Time:    {self.metrics.encoding_time:.3f} s")
        print(f"  Simulation Time:  {self.metrics.simulation_time:.3f} s")
        print(f"  Decoding Time:    {self.metrics.decoding_time:.3f} s")
        
        print("\n💾 MEMORY METRICS")
        print(f"  Peak Memory:      {self.metrics.peak_memory:.2f} MB")
        print(f"  Average Memory:   {self.metrics.avg_memory:.2f} MB")
        
        print("\n⚡ THROUGHPUT METRICS")
        print(f"  Transactions/sec: {self.metrics.transactions_per_second:.2f}")
        print(f"  Latency (mean):   {self.metrics.latency_mean:.2f} ms")
        print(f"  Latency (p50):    {self.metrics.latency_p50:.2f} ms")
        print(f"  Latency (p95):    {self.metrics.latency_p95:.2f} ms")
        print(f"  Latency (p99):    {self.metrics.latency_p99:.2f} ms")
        
        if self.metrics.accuracy > 0:
            print("\n🎯 MODEL METRICS")
            print(f"  Accuracy:         {self.metrics.accuracy:.4f}")
            print(f"  Precision:        {self.metrics.precision:.4f}")
            print(f"  Recall:           {self.metrics.recall:.4f}")
            print(f"  F1 Score:         {self.metrics.f1_score:.4f}")
        
        print("\n🖥️  RESOURCE METRICS")
        print(f"  CPU Usage:        {self.metrics.cpu_percent:.1f}%")
        print(f"  CPU Count:        {self.metrics.cpu_count}")
        
        print("\n" + "="*70)


class LatencyBenchmark:
    """
    Benchmark latency under various conditions.
    
    Measures latency distribution, tail latencies, and throughput limits.
    """
    
    def __init__(self, model, encoder):
        """
        Initialize benchmark.
        
        Args:
            model: Fraud detection model
            encoder: Spike encoder
        """
        self.model = model
        self.encoder = encoder
        
    def benchmark_single_transaction(self, n_trials: int = 1000) -> Dict:
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
            transaction = np.random.randn(30)  # 30 features
            
            start = time.time()
            
            # Encode
            encoded = self.encoder.encode(transaction)
            
            # Simulate
            prediction = self.model.predict(encoded)
            
            end = time.time()
            
            latencies.append((end - start) * 1000)  # ms
            
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
        
    def benchmark_throughput(self, batch_sizes: List[int]) -> Dict:
        """
        Benchmark throughput at various batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with throughput for each batch size
        """
        print("Benchmarking throughput at various batch sizes...")
        
        results = {}
        
        for batch_size in batch_sizes:
            # Generate batch
            batch = [np.random.randn(30) for _ in range(batch_size)]
            
            start = time.time()
            
            # Process batch
            for transaction in batch:
                encoded = self.encoder.encode(transaction)
                prediction = self.model.predict(encoded)
                
            end = time.time()
            
            elapsed = end - start
            throughput = batch_size / elapsed
            
            results[batch_size] = {
                'throughput': float(throughput),
                'time': float(elapsed),
                'avg_latency': float(elapsed / batch_size * 1000)  # ms
            }
            
            print(f"  Batch size {batch_size:4d}: "
                  f"{throughput:6.2f} trans/sec, "
                  f"avg latency {results[batch_size]['avg_latency']:.2f} ms")
            
        return results
        
    def stress_test(self, duration_seconds: int = 60,
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
        
        interval = 1.0 / target_tps
        
        while time.time() < end_time:
            transaction = np.random.randn(30)
            
            try:
                txn_start = time.time()
                
                encoded = self.encoder.encode(transaction)
                prediction = self.model.predict(encoded)
                
                txn_end = time.time()
                
                latencies.append((txn_end - txn_start) * 1000)
                total_transactions += 1
                
                # Rate limiting
                time.sleep(max(0, interval - (txn_end - txn_start)))
                
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
            'error_rate': float(errors / total_transactions) if total_transactions > 0 else 0,
            'latency_stats': {
                'mean': float(np.mean(latencies)),
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'max': float(np.max(latencies))
            }
        }
        
        print(f"  Processed {total_transactions} transactions in {actual_duration:.2f}s")
        print(f"  Actual TPS: {actual_tps:.2f}")
        print(f"  Error rate: {results['error_rate']*100:.2f}%")
        print(f"  P95 latency: {results['latency_stats']['p95']:.2f} ms")
        
        return results


class ResourceMonitor:
    """
    Monitor system resources during execution.
    
    Tracks CPU, memory, and optionally GPU usage over time.
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: Interval between samples (seconds)
        """
        self.sampling_interval = sampling_interval
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.monitoring = False
        
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process()
        start_time = time.time()
        
        while self.monitoring:
            self.timestamps.append(time.time() - start_time)
            self.cpu_samples.append(process.cpu_percent())
            self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
            
            time.sleep(self.sampling_interval)
            
    def get_summary(self) -> Dict:
        """Get summary of resource usage."""
        if not self.cpu_samples:
            return {}
            
        return {
            'cpu': {
                'mean': float(np.mean(self.cpu_samples)),
                'max': float(np.max(self.cpu_samples)),
                'min': float(np.min(self.cpu_samples))
            },
            'memory': {
                'mean_mb': float(np.mean(self.memory_samples)),
                'max_mb': float(np.max(self.memory_samples)),
                'min_mb': float(np.min(self.memory_samples))
            },
            'duration': float(self.timestamps[-1]) if self.timestamps else 0
        }


# Example usage
if __name__ == "__main__":
    print("Performance Profiling Tools")
    print("="*60)
    
    # Create a simple dummy model for testing
    class DummyModel:
        def predict(self, data):
            time.sleep(0.001)  # Simulate processing
            return np.random.choice([0, 1])
    
    class DummyEncoder:
        def encode(self, data):
            time.sleep(0.0005)  # Simulate encoding
            return data
    
    model = DummyModel()
    encoder = DummyEncoder()
    
    # Run latency benchmark
    benchmark = LatencyBenchmark(model, encoder)
    
    print("\n1. Single Transaction Latency:")
    latency_stats = benchmark.benchmark_single_transaction(n_trials=100)
    for metric, value in latency_stats.items():
        print(f"  {metric:8s}: {value:.3f} ms")
    
    print("\n2. Throughput Benchmark:")
    throughput_results = benchmark.benchmark_throughput([10, 50, 100])
    
    print("\n3. Stress Test:")
    stress_results = benchmark.stress_test(duration_seconds=5, target_tps=50)
    
    print("\nBenchmarking complete!")
