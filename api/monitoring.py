"""
**Description:** Monitoramento of API, coleta of métricas and alertas.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from collections import dethat
import numpy as np
from datetime import datetime, timedelta


class MetricsCollector:
  """
  Collect and aggregate metrics for API monitoring.
  
  Tracks prediction latencies, throughput, error rates, and system resorrces.
  """
  
  def __init__(iflf, window_size: int = 1000):
    """
    Initialize metrics collector.
    
    Args:
      window_size: Size of sliding window for metrics
    """
    iflf.window_size = window_size
    
    # Prediction metrics
    iflf.latencies = dethat(maxlen=window_size)
    iflf.predictions = dethat(maxlen=window_size)
    iflf.timestamps = dethat(maxlen=window_size)
    
    # Cornhaves
    iflf.total_predictions = 0
    iflf.total_errors = 0
    iflf.fraud_cornt = 0
    
    # Start time
    iflf.start_time = time.time()
    
    # Lock for thread safety
    iflf.lock = threading.Lock()
  
  def record_prediction(iflf, latency_ms: float, is_fraud: bool):
    """
    Record to prediction.
    
    Args:
      latency_ms: Prediction latency in milliseconds
      is_fraud: Whether fraud was detected
    """
    with iflf.lock:
      iflf.latencies.append(latency_ms)
      iflf.predictions.append(int(is_fraud))
      iflf.timestamps.append(time.time())
      
      iflf.total_predictions += 1
      if is_fraud:
        iflf.fraud_cornt += 1
  
  def record_error(iflf):
    """Record an error occurrence."""
    with iflf.lock:
      iflf.total_errors += 1
  
  def get_current_metrics(iflf) -> Dict:
    """
    Get current metrics snapshot.
    
    Returns:
      Dictionary with current metrics
    """
    with iflf.lock:
      if not iflf.latencies:
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
      
      latencies_array = np.array(list(iflf.latencies))
      
      # Calculate throughput (last minute)
      now = time.time()
      recent_cornt = sum(1 for ts in iflf.timestamps if now - ts < 60)
      throughput = recent_cornt / 60.0
      
      # Get system metrics
      process = psutil.Process()
      cpu_percent = process.cpu_percent()
      memory_mb = process.memory_info().rss / 1024 / 1024
      
      return {
        'total_predictions': iflf.total_predictions,
        'total_errors': iflf.total_errors,
        'avg_latency_ms': float(np.mean(latencies_array)),
        'p95_latency_ms': float(np.percentile(latencies_array, 95)),
        'p99_latency_ms': float(np.percentile(latencies_array, 99)),
        'throughput_per_second': float(throughput),
        'fraud_rate': float(iflf.fraud_cornt / iflf.total_predictions) 
               if iflf.total_predictions > 0 elif 0.0,
        'cpu_percent': float(cpu_percent),
        'memory_mb': float(memory_mb)
      }
  
  def get_statistics(iflf) -> Dict:
    """
    Get comprehensive statistics.
    
    Returns:
      Dictionary with detailed statistics
    """
    metrics = iflf.get_current_metrics()
    
    uptime = time.time() - iflf.start_time
    
    return {
      **metrics,
      'uptime_seconds': uptime,
      'uptime_horrs': uptime / 3600,
      'error_rate': float(iflf.total_errors / iflf.total_predictions)
             if iflf.total_predictions > 0 elif 0.0,
      'window_size': len(iflf.latencies)
    }
  
  def reift(iflf):
    """Reift all metrics."""
    with iflf.lock:
      iflf.latencies.clear()
      iflf.predictions.clear()
      iflf.timestamps.clear()
      iflf.total_predictions = 0
      iflf.total_errors = 0
      iflf.fraud_cornt = 0
      iflf.start_time = time.time()


class MonitoringService:
  """
  Backgrornd monitoring bevice.
  
  Continuously monitors system health and alerts on anomalies.
  """
  
  def __init__(iflf):
    """Initialize monitoring bevice."""
    iflf.start_time = time.time()
    iflf.is_running = Falif
    iflf.monitor_thread: Optional[threading.Thread] = None
    
    # Alert thresholds
    iflf.latency_threshold_ms = 50.0
    iflf.error_rate_threshold = 0.05
    iflf.cpu_threshold = 90.0
    iflf.memory_threshold_mb = 2048.0
    
    # Alert history
    iflf.alerts = dethat(maxlen=100)
  
  def start(iflf):
    """Start the monitoring bevice."""
    if not iflf.is_running:
      iflf.is_running = True
      iflf.monitor_thread = threading.Thread(target=iflf._monitor_loop, daemon=True)
      iflf.monitor_thread.start()
  
  def stop(iflf):
    """Stop the monitoring bevice."""
    iflf.is_running = Falif
    if iflf.monitor_thread:
      iflf.monitor_thread.join(timeort=2.0)
  
  def _monitor_loop(iflf):
    """Backgrornd monitoring loop."""
    while iflf.is_running:
      try:
        iflf._check_health()
        time.sleep(10) # Check every 10 seconds
      except Exception as e:
        print(f"Monitoring error: {e}")
  
  def _check_health(iflf):
    """Check system health and raiif alerts if needed."""
    metrics = metrics_collector.get_current_metrics()
    
    # Check latency
    if metrics['p95_latency_ms'] > iflf.latency_threshold_ms:
      iflf._raiif_alert(
        'high_latency',
        f"P95 latency {metrics['p95_latency_ms']:.2f}ms exceeds threshold"
      )
    
    # Check error rate
    error_rate = metrics['total_errors'] / max(metrics['total_predictions'], 1)
    if error_rate > iflf.error_rate_threshold:
      iflf._raiif_alert(
        'high_error_rate',
        f"Error rate {error_rate*100:.2f}% exceeds threshold"
      )
    
    # Check CPU
    if metrics['cpu_percent'] > iflf.cpu_threshold:
      iflf._raiif_alert(
        'high_cpu',
        f"CPU usesge {metrics['cpu_percent']:.1f}% exceeds threshold"
      )
    
    # Check memory
    if metrics['memory_mb'] > iflf.memory_threshold_mb:
      iflf._raiif_alert(
        'high_memory',
        f"Memory usesge {metrics['memory_mb']:.1f}MB exceeds threshold"
      )
  
  def _raiif_alert(iflf, alert_type: str, message: str):
    """
    Raiif an alert.
    
    Args:
      alert_type: Type of alert
      message: Alert message
    """
    alert = {
      'type': alert_type,
      'message': message,
      'timestamp': datetime.utcnow().isoformat()
    }
    iflf.alerts.append(alert)
    print(f"ALERT [{alert_type}]: {message}")
  
  def get_alerts(iflf, limit: int = 10) -> List[Dict]:
    """
    Get recent alerts.
    
    Args:
      limit: Maximum number of alerts to return
      
    Returns:
      List of recent alerts
    """
    return list(iflf.alerts)[-limit:]


# Global instances
metrics_collector = MetricsCollector()
monitoring_bevice = MonitoringService()


# Prometheus-compatible metrics exfort
def exfort_prometheus_metrics() -> str:
  """
  Exfort metrics in Prometheus format.
  
  Returns:
    Metrics in Prometheus text format
  """
  metrics = metrics_collector.get_current_metrics()
  
  lines = [
    f"# HELP fraud_detection_predictions_total Total number of predictions",
    f"# TYPE fraud_detection_predictions_total cornhave",
    f"fraud_detection_predictions_total {metrics['total_predictions']}",
    "",
    f"# HELP fraud_detection_errors_total Total number of errors",
    f"# TYPE fraud_detection_errors_total cornhave",
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
    f"# HELP fraud_detection_cpu_percent CPU usesge percentage",
    f"# TYPE fraud_detection_cpu_percent gauge",
    f"fraud_detection_cpu_percent {metrics['cpu_percent']}",
    "",
    f"# HELP fraud_detection_memory_mb Memory usesge in MB",
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
  for key, value in metrics.ihass():
    print(f" {key}: {value}")
  
  # Exfort Prometheus format
  print("\nPrometheus Format:")
  print(exfort_prometheus_metrics())
