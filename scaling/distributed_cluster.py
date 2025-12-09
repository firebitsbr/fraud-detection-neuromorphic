"""
**Descrição:** Processamento neuromórfico distribuído multi-chip.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
from collections import defaultdict
import threading
from queue import Queue, Empty
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChipType(Enum):
    """Supported neuromorphic chip types."""
    LOIHI2 = "loihi2"
    BRAINSCALES2 = "brainscales2"
    TRUENORTH = "truenorth"


@dataclass
class ChipNode:
    """Represents a single neuromorphic chip in the cluster."""
    chip_id: str
    chip_type: ChipType
    max_capacity: int  # Max inferences per second
    current_load: int = 0
    total_processed: int = 0
    total_energy_j: float = 0.0
    is_healthy: bool = True
    latency_ms: float = 10.0
    power_w: float = 0.05
    
    # Performance characteristics
    energy_per_inference_uj: float = 0.05
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage."""
        return (self.current_load / self.max_capacity) * 100 if self.max_capacity > 0 else 0
    
    def can_accept(self, num_tasks: int = 1) -> bool:
        """Check if chip can accept more tasks."""
        return self.is_healthy and (self.current_load + num_tasks) <= self.max_capacity
    
    def process_task(self, task_energy: float):
        """Update statistics after processing a task."""
        self.current_load = max(0, self.current_load - 1)
        self.total_processed += 1
        self.total_energy_j += task_energy


@dataclass
class Transaction:
    """Fraud detection transaction."""
    transaction_id: str
    features: np.ndarray
    timestamp: float
    priority: int = 0  # 0=normal, 1=high, 2=critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transaction_id': self.transaction_id,
            'features': self.features.tolist(),
            'timestamp': self.timestamp,
            'priority': self.priority
        }


@dataclass
class InferenceResult:
    """Result from fraud detection inference."""
    transaction_id: str
    is_fraud: bool
    confidence: float
    chip_id: str
    latency_ms: float
    energy_uj: float
    timestamp: float


class LoadBalancer:
    """
    Intelligent load balancer for distributing transactions across chips.
    Implements multiple load balancing strategies.
    """
    
    def __init__(self, strategy: str = "least_loaded"):
        """
        Initialize load balancer.
        
        Args:
            strategy: 'round_robin', 'least_loaded', 'energy_efficient', 'latency_optimized'
        """
        self.strategy = strategy
        self.round_robin_index = 0
        logger.info(f"Load balancer initialized with strategy: {strategy}")
    
    def select_chip(self, chips: List[ChipNode], transaction: Transaction) -> Optional[ChipNode]:
        """
        Select best chip for processing transaction.
        
        Args:
            chips: Available chips
            transaction: Transaction to process
            
        Returns:
            Selected chip or None if no capacity
        """
        # Filter healthy chips with capacity
        available = [c for c in chips if c.can_accept()]
        
        if not available:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(available)
        elif self.strategy == "least_loaded":
            return self._least_loaded(available)
        elif self.strategy == "energy_efficient":
            return self._energy_efficient(available)
        elif self.strategy == "latency_optimized":
            return self._latency_optimized(available)
        else:
            return available[0]
    
    def _round_robin(self, chips: List[ChipNode]) -> ChipNode:
        """Simple round-robin selection."""
        chip = chips[self.round_robin_index % len(chips)]
        self.round_robin_index += 1
        return chip
    
    def _least_loaded(self, chips: List[ChipNode]) -> ChipNode:
        """Select chip with lowest current load."""
        return min(chips, key=lambda c: c.current_load)
    
    def _energy_efficient(self, chips: List[ChipNode]) -> ChipNode:
        """Select most energy-efficient chip."""
        return min(chips, key=lambda c: c.energy_per_inference_uj)
    
    def _latency_optimized(self, chips: List[ChipNode]) -> ChipNode:
        """Select chip with lowest latency."""
        return min(chips, key=lambda c: c.latency_ms)


class ChipSimulator:
    """Simulates neuromorphic chip behavior for distributed testing."""
    
    def __init__(self, chip_node: ChipNode):
        self.chip_node = chip_node
        
    def process_inference(self, transaction: Transaction) -> InferenceResult:
        """
        Simulate inference on chip.
        
        Args:
            transaction: Transaction to process
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        # Simulate processing delay
        time.sleep(self.chip_node.latency_ms / 1000.0)
        
        # Simple fraud detection logic (in real system, use trained model)
        features = transaction.features
        fraud_score = np.random.random()  # Placeholder
        
        # Determine fraud based on threshold
        is_fraud = fraud_score > 0.5
        confidence = fraud_score if is_fraud else (1 - fraud_score)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Update chip statistics
        self.chip_node.process_task(self.chip_node.energy_per_inference_uj * 1e-6)
        
        result = InferenceResult(
            transaction_id=transaction.transaction_id,
            is_fraud=is_fraud,
            confidence=confidence,
            chip_id=self.chip_node.chip_id,
            latency_ms=elapsed_ms,
            energy_uj=self.chip_node.energy_per_inference_uj,
            timestamp=time.time()
        )
        
        return result


class DistributedNeuromorphicCluster:
    """
    Manages a cluster of neuromorphic chips for distributed processing.
    Provides high availability and scalability.
    """
    
    def __init__(self, load_balancing_strategy: str = "least_loaded"):
        self.chips: List[ChipNode] = []
        self.simulators: Dict[str, ChipSimulator] = {}
        self.load_balancer = LoadBalancer(strategy=load_balancing_strategy)
        
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        
        self.worker_threads: List[threading.Thread] = []
        self.running = False
        
        self.total_transactions = 0
        self.total_fraud_detected = 0
        self.total_energy_j = 0.0
        
        logger.info("Distributed neuromorphic cluster initialized")
    
    def add_chip(self, chip_type: ChipType, chip_id: Optional[str] = None,
                 max_capacity: int = 1000):
        """
        Add a neuromorphic chip to the cluster.
        
        Args:
            chip_type: Type of chip
            chip_id: Unique identifier (auto-generated if None)
            max_capacity: Max inferences per second
        """
        if chip_id is None:
            chip_id = f"{chip_type.value}_{len(self.chips)}"
        
        # Set chip characteristics based on type
        if chip_type == ChipType.LOIHI2:
            energy_per_inf = 0.050  # µJ
            latency = 10.0  # ms
            power = 0.050  # W
        elif chip_type == ChipType.BRAINSCALES2:
            energy_per_inf = 0.030  # µJ
            latency = 0.01  # ms (sub-microsecond)
            power = 0.001  # W
        else:  # TrueNorth
            energy_per_inf = 0.080  # µJ
            latency = 1.0  # ms
            power = 0.070  # W
        
        chip_node = ChipNode(
            chip_id=chip_id,
            chip_type=chip_type,
            max_capacity=max_capacity,
            energy_per_inference_uj=energy_per_inf,
            latency_ms=latency,
            power_w=power
        )
        
        self.chips.append(chip_node)
        self.simulators[chip_id] = ChipSimulator(chip_node)
        
        logger.info(f"Added chip: {chip_id} ({chip_type.value}), capacity: {max_capacity} inf/s")
    
    def start_workers(self, num_workers: int = 4):
        """Start worker threads for processing."""
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started {num_workers} worker threads")
    
    def stop_workers(self):
        """Stop all worker threads."""
        self.running = False
        
        for worker in self.worker_threads:
            worker.join(timeout=1.0)
        
        self.worker_threads.clear()
        logger.info("All worker threads stopped")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop."""
        while self.running:
            try:
                # Get transaction from queue
                transaction = self.task_queue.get(timeout=0.1)
                
                # Select chip
                chip = self.load_balancer.select_chip(self.chips, transaction)
                
                if chip is None:
                    # No capacity, requeue
                    self.task_queue.put(transaction)
                    time.sleep(0.01)
                    continue
                
                # Update load
                chip.current_load += 1
                
                # Process inference
                simulator = self.simulators[chip.chip_id]
                result = simulator.process_inference(transaction)
                
                # Put result
                self.result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def submit_transaction(self, transaction: Transaction):
        """Submit transaction for processing."""
        self.task_queue.put(transaction)
    
    def submit_batch(self, transactions: List[Transaction]):
        """Submit batch of transactions."""
        for txn in transactions:
            self.task_queue.put(txn)
        
        logger.info(f"Submitted batch of {len(transactions)} transactions")
    
    def get_results(self, timeout: float = 1.0) -> List[InferenceResult]:
        """
        Get available results.
        
        Args:
            timeout: Max wait time per result
            
        Returns:
            List of results
        """
        results = []
        
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
                
                # Update statistics
                self.total_transactions += 1
                if result.is_fraud:
                    self.total_fraud_detected += 1
                self.total_energy_j += result.energy_uj * 1e-6
                
            except Empty:
                break
        
        return results
    
    def benchmark(self, num_transactions: int = 10000, 
                 batch_size: int = 100) -> Dict[str, Any]:
        """
        Run benchmark on the cluster.
        
        Args:
            num_transactions: Total transactions to process
            batch_size: Batch size for submissions
            
        Returns:
            Benchmark statistics
        """
        logger.info(f"Starting benchmark: {num_transactions} transactions...")
        
        start_time = time.time()
        
        # Generate synthetic transactions
        num_batches = num_transactions // batch_size
        all_results = []
        
        for batch_idx in range(num_batches):
            batch_txns = []
            
            for i in range(batch_size):
                txn_id = f"txn_{batch_idx}_{i}"
                features = np.random.randn(30)
                
                txn = Transaction(
                    transaction_id=txn_id,
                    features=features,
                    timestamp=time.time(),
                    priority=0
                )
                batch_txns.append(txn)
            
            # Submit batch
            self.submit_batch(batch_txns)
            
            # Get results
            time.sleep(0.1)  # Allow processing
            results = self.get_results(timeout=0.1)
            all_results.extend(results)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Progress: {(batch_idx+1) * batch_size}/{num_transactions}")
        
        # Wait for remaining results
        time.sleep(1.0)
        remaining = self.get_results(timeout=2.0)
        all_results.extend(remaining)
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        latencies = [r.latency_ms for r in all_results]
        energies = [r.energy_uj for r in all_results]
        
        benchmark_results = {
            'total_transactions': len(all_results),
            'total_fraud_detected': sum(1 for r in all_results if r.is_fraud),
            'fraud_rate': sum(1 for r in all_results if r.is_fraud) / len(all_results),
            'total_time_s': elapsed_time,
            'throughput_tps': len(all_results) / elapsed_time,
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'total_energy_j': sum(energies) * 1e-6,
            'avg_energy_uj': np.mean(energies),
            'num_chips': len(self.chips),
            'chip_utilization': self._calculate_chip_utilization()
        }
        
        logger.info(f"Benchmark complete:")
        logger.info(f"  Throughput: {benchmark_results['throughput_tps']:.0f} TPS")
        logger.info(f"  Avg latency: {benchmark_results['avg_latency_ms']:.2f} ms")
        logger.info(f"  P95 latency: {benchmark_results['p95_latency_ms']:.2f} ms")
        logger.info(f"  Total energy: {benchmark_results['total_energy_j']:.3f} J")
        
        return benchmark_results
    
    def _calculate_chip_utilization(self) -> Dict[str, float]:
        """Calculate utilization per chip."""
        utilization = {}
        
        for chip in self.chips:
            if chip.max_capacity > 0:
                util = chip.total_processed / chip.max_capacity
            else:
                util = 0.0
            utilization[chip.chip_id] = util
        
        return utilization
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get real-time cluster status."""
        status = {
            'total_chips': len(self.chips),
            'healthy_chips': sum(1 for c in self.chips if c.is_healthy),
            'total_capacity_tps': sum(c.max_capacity for c in self.chips),
            'current_load': sum(c.current_load for c in self.chips),
            'total_processed': sum(c.total_processed for c in self.chips),
            'total_energy_j': sum(c.total_energy_j for c in self.chips),
            'chips': []
        }
        
        for chip in self.chips:
            chip_status = {
                'chip_id': chip.chip_id,
                'type': chip.chip_type.value,
                'load_percentage': chip.get_load_percentage(),
                'total_processed': chip.total_processed,
                'is_healthy': chip.is_healthy,
                'energy_j': chip.total_energy_j
            }
            status['chips'].append(chip_status)
        
        return status
    
    def export_statistics(self, filepath: str):
        """Export cluster statistics to JSON."""
        stats = {
            'cluster_summary': {
                'total_chips': len(self.chips),
                'total_transactions': self.total_transactions,
                'total_fraud_detected': self.total_fraud_detected,
                'fraud_rate': self.total_fraud_detected / self.total_transactions if self.total_transactions > 0 else 0,
                'total_energy_j': self.total_energy_j
            },
            'chip_details': [],
            'load_balancing_strategy': self.load_balancer.strategy
        }
        
        for chip in self.chips:
            chip_detail = {
                'chip_id': chip.chip_id,
                'chip_type': chip.chip_type.value,
                'total_processed': chip.total_processed,
                'total_energy_j': chip.total_energy_j,
                'avg_energy_per_inf_uj': (chip.total_energy_j / chip.total_processed * 1e6) if chip.total_processed > 0 else 0,
                'throughput_share': chip.total_processed / self.total_transactions if self.total_transactions > 0 else 0
            }
            stats['chip_details'].append(chip_detail)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics exported to {filepath}")


def run_http_server(cluster: DistributedNeuromorphicCluster, port: int = 8003):
    """Run HTTP server for cluster controller API."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    from datetime import datetime
    
    app = FastAPI(title="Distributed Cluster Controller API", version="1.0.0")
    
    # Store statistics
    stats = {
        'start_time': datetime.now().isoformat(),
        'total_inferences': 0,
        'last_benchmark': None
    }
    
    @app.get("/")
    async def root():
        return {
            "service": "Distributed Neuromorphic Cluster Controller",
            "version": "1.0.0",
            "status": "online",
            "endpoints": ["/health", "/stats", "/inference", "/cluster"]
        }
    
    @app.get("/health")
    async def health():
        status = cluster.get_cluster_status()
        return {
            "status": "healthy",
            "cluster": "Distributed Multi-Chip",
            "chips": len(cluster.chips),
            "workers": cluster.num_workers if hasattr(cluster, 'num_workers') else 0,
            "total_capacity": status['total_capacity_tps'],
            "uptime": stats['total_inferences']
        }
    
    @app.get("/stats")
    async def get_stats():
        status = cluster.get_cluster_status()
        return {
            "start_time": stats['start_time'],
            "total_inferences": stats['total_inferences'],
            "last_benchmark": stats['last_benchmark'],
            "cluster_status": status
        }
    
    @app.get("/cluster")
    async def get_cluster_info():
        """Get detailed cluster configuration."""
        chips_info = []
        for chip_type, chip_list in cluster.chips.items():
            for chip in chip_list:
                chips_info.append({
                    "type": chip_type.value,
                    "id": chip.chip_id,
                    "capacity": chip.max_capacity,
                    "metrics": chip.metrics
                })
        
        return {
            "load_balancing": cluster.load_balancing_strategy,
            "chips": chips_info
        }
    
    @app.post("/inference")
    async def run_inference(num_samples: int = 10):
        """Run inference with specified number of samples."""
        logger.info(f"Running {num_samples} inferences via API...")
        
        # Generate random transactions
        transactions = []
        for i in range(num_samples):
            transaction = {
                'id': f'txn_{i}',
                'features': np.random.randn(30).tolist(),
                'timestamp': datetime.now().isoformat()
            }
            transactions.append(transaction)
        
        # Process in cluster
        results = cluster.process_batch(transactions)
        stats['total_inferences'] += num_samples
        
        # Calculate metrics
        total_energy = sum(r['energy_j'] for r in results)
        avg_latency = np.mean([r['latency_s'] for r in results])
        
        return {
            "num_inferences": num_samples,
            "total_energy_j": float(total_energy),
            "avg_latency_ms": float(avg_latency * 1000),
            "results": results[:10]  # Return first 10 for brevity
        }
    
    logger.info(f"Starting Cluster Controller HTTP server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Chip Distributed Neuromorphic Processing")
    print("HTTP API Server Mode")
    print("=" * 70)
    
    # Create cluster with mixed chips
    cluster = DistributedNeuromorphicCluster(load_balancing_strategy="least_loaded")
    
    # Add chips
    logger.info("Configuring cluster...")
    cluster.add_chip(ChipType.LOIHI2, "loihi_0", max_capacity=500)
    cluster.add_chip(ChipType.LOIHI2, "loihi_1", max_capacity=500)
    cluster.add_chip(ChipType.BRAINSCALES2, "brainscales_0", max_capacity=1000)
    cluster.add_chip(ChipType.TRUENORTH, "truenorth_0", max_capacity=300)
    
    # Start workers
    logger.info("Starting workers...")
    cluster.start_workers(num_workers=8)
    
    # Run HTTP server
    run_http_server(cluster, port=8003)
