"""
Monitoring and Metrics Collection

This module provides comprehensive monitoring and metrics collection
for the fraud detection API.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
License: MIT License
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from collections import deque
import numpy as np
from datetime import datetime, timedelta


class MetricsCollector:
    """
    Collect and aggregate metrics for API monitoring.
    
    Tracks prediction latencies, throughput, error rates, and system resources.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        
        # Prediction metrics
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Counters
        self.total_predictions = 0
        self.total_errors = 0
        self.fraud_count = 0
        
        # Start time
        self.start_time = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def record_prediction(self, latency_ms: float, is_fraud: bool):
        """
        Record a prediction.
        
        Args:
            latency_ms: Prediction latency in milliseconds
            is_fraud: Whether fraud was detected
        """
        with self.lock:
            self.latencies.append(latency_ms)
            self.predictions.append(int(is_fraud))
            self.timestamps.append(time.time())
            
            self.total_predictions += 1
            if is_fraud:
                self.fraud_count += 1
    
    def record_error(self):
        """Record an error occurrence."""
        with self.lock:
            self.total_errors += 1
    
    def get_current_metrics(self) -> Dict:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary with current metrics
        """
        with self.lock:
            if not self.latencies:
                return {
                    'total_predictions': 0,
                    'total_errors': 0,
                    'avg_latency_ms': 0.0,
                    'p95_latency_ms': 0.0,
                    'p99_latency_ms': 0.0,
                    'throughput_per_second': 0.0,
                    'fraud_rate': 0.0,
                    'cpu_percent': 0.0,
                    'memory_mb': 0.0
                }
            
            latencies_array = np.array(list(self.latencies))
            
            # Calculate throughput (last minute)
            now = time.time()
            recent_count = sum(1 for ts in self.timestamps if now - ts < 60)
            throughput = recent_count / 60.0
            
            # Get system metrics
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                'total_predictions': self.total_predictions,
                'total_errors': self.total_errors,
                'avg_latency_ms': float(np.mean(latencies_array)),
                'p95_latency_ms': float(np.percentile(latencies_array, 95)),
                'p99_latency_ms': float(np.percentile(latencies_array, 99)),
                'throughput_per_second': float(throughput),
                'fraud_rate': float(self.fraud_count / self.total_predictions) 
                             if self.total_predictions > 0 else 0.0,
                'cpu_percent': float(cpu_percent),
                'memory_mb': float(memory_mb)
            }
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with detailed statistics
        """
        metrics = self.get_current_metrics()
        
        uptime = time.time() - self.start_time
        
        return {
            **metrics,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'error_rate': float(self.total_errors / self.total_predictions)
                         if self.total_predictions > 0 else 0.0,
            'window_size': len(self.latencies)
        }
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.latencies.clear()
            self.predictions.clear()
            self.timestamps.clear()
            self.total_predictions = 0
            self.total_errors = 0
            self.fraud_count = 0
            self.start_time = time.time()


class MonitoringService:
    """
    Background monitoring service.
    
    Continuously monitors system health and alerts on anomalies.
    """
    
    def __init__(self):
        """Initialize monitoring service."""
        self.start_time = time.time()
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Alert thresholds
        self.latency_threshold_ms = 50.0
        self.error_rate_threshold = 0.05
        self.cpu_threshold = 90.0
        self.memory_threshold_mb = 2048.0
        
        # Alert history
        self.alerts = deque(maxlen=100)
    
    def start(self):
        """Start the monitoring service."""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop(self):
        """Stop the monitoring service."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                self._check_health()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def _check_health(self):
        """Check system health and raise alerts if needed."""
        metrics = metrics_collector.get_current_metrics()
        
        # Check latency
        if metrics['p95_latency_ms'] > self.latency_threshold_ms:
            self._raise_alert(
                'high_latency',
                f"P95 latency {metrics['p95_latency_ms']:.2f}ms exceeds threshold"
            )
        
        # Check error rate
        error_rate = metrics['total_errors'] / max(metrics['total_predictions'], 1)
        if error_rate > self.error_rate_threshold:
            self._raise_alert(
                'high_error_rate',
                f"Error rate {error_rate*100:.2f}% exceeds threshold"
            )
        
        # Check CPU
        if metrics['cpu_percent'] > self.cpu_threshold:
            self._raise_alert(
                'high_cpu',
                f"CPU usage {metrics['cpu_percent']:.1f}% exceeds threshold"
            )
        
        # Check memory
        if metrics['memory_mb'] > self.memory_threshold_mb:
            self._raise_alert(
                'high_memory',
                f"Memory usage {metrics['memory_mb']:.1f}MB exceeds threshold"
            )
    
    def _raise_alert(self, alert_type: str, message: str):
        """
        Raise an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
        """
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.alerts.append(alert)
        print(f"ALERT [{alert_type}]: {message}")
    
    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return list(self.alerts)[-limit:]


# Global instances
metrics_collector = MetricsCollector()
monitoring_service = MonitoringService()


# Prometheus-compatible metrics export
def export_prometheus_metrics() -> str:
    """
    Export metrics in Prometheus format.
    
    Returns:
        Metrics in Prometheus text format
    """
    metrics = metrics_collector.get_current_metrics()
    
    lines = [
        f"# HELP fraud_detection_predictions_total Total number of predictions",
        f"# TYPE fraud_detection_predictions_total counter",
        f"fraud_detection_predictions_total {metrics['total_predictions']}",
        "",
        f"# HELP fraud_detection_errors_total Total number of errors",
        f"# TYPE fraud_detection_errors_total counter",
        f"fraud_detection_errors_total {metrics['total_errors']}",
        "",
        f"# HELP fraud_detection_latency_ms Prediction latency in milliseconds",
        f"# TYPE fraud_detection_latency_ms summary",
        f"fraud_detection_latency_ms{{quantile=\"0.5\"}} {metrics['avg_latency_ms']}",
        f"fraud_detection_latency_ms{{quantile=\"0.95\"}} {metrics['p95_latency_ms']}",
        f"fraud_detection_latency_ms{{quantile=\"0.99\"}} {metrics['p99_latency_ms']}",
        "",
        f"# HELP fraud_detection_throughput_per_second Current throughput",
        f"# TYPE fraud_detection_throughput_per_second gauge",
        f"fraud_detection_throughput_per_second {metrics['throughput_per_second']}",
        "",
        f"# HELP fraud_detection_fraud_rate Detected fraud rate",
        f"# TYPE fraud_detection_fraud_rate gauge",
        f"fraud_detection_fraud_rate {metrics['fraud_rate']}",
        "",
        f"# HELP fraud_detection_cpu_percent CPU usage percentage",
        f"# TYPE fraud_detection_cpu_percent gauge",
        f"fraud_detection_cpu_percent {metrics['cpu_percent']}",
        "",
        f"# HELP fraud_detection_memory_mb Memory usage in MB",
        f"# TYPE fraud_detection_memory_mb gauge",
        f"fraud_detection_memory_mb {metrics['memory_mb']}",
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics collection
    print("Testing metrics collector...")
    
    # Simulate predictions
    import random
    for _ in range(100):
        latency = random.uniform(5, 20)
        is_fraud = random.random() < 0.05
        metrics_collector.record_prediction(latency, is_fraud)
    
    # Get metrics
    metrics = metrics_collector.get_current_metrics()
    print("\nCurrent Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Export Prometheus format
    print("\nPrometheus Format:")
    print(export_prometheus_metrics())
