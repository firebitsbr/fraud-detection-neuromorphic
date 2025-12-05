"""
Comprehensive Scaling Test Suite for Phase 5
============================================

Tests multi-chip distributed processing, load balancing,
fault tolerance, and scalability benchmarks.

Author: Mauro Risonho de Paula Assumpção
Date: December 5, 2025
License: MIT License
"""

import numpy as np
import time
import json
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

# Import Phase 5 components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from hardware.loihi2_simulator import Loihi2Simulator, ChipConfig
from hardware.brainscales2_simulator import BrainScaleS2Simulator
from scaling.distributed_cluster import (
    DistributedNeuromorphicCluster, 
    ChipType, 
    Transaction
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingTestSuite:
    """Comprehensive test suite for Phase 5 scaling capabilities."""
    
    def __init__(self, output_dir: str = "scaling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Scaling test suite initialized. Output: {self.output_dir}")
    
    def test_single_chip_throughput(self) -> Dict[str, Any]:
        """Test throughput of individual chip types."""
        logger.info("=" * 70)
        logger.info("TEST 1: Single Chip Throughput")
        logger.info("=" * 70)
        
        results = {}
        
        # Test Loihi 2
        logger.info("\nTesting Loihi 2...")
        loihi_sim = Loihi2Simulator()
        loihi_result = loihi_sim.benchmark(num_inferences=1000, input_size=30)
        results['loihi2'] = loihi_result
        
        # Test BrainScaleS-2
        logger.info("\nTesting BrainScaleS-2...")
        brainscales_sim = BrainScaleS2Simulator()
        brainscales_sim.load_fraud_detection_model()
        brainscales_result = brainscales_sim.wafers[0].benchmark(
            num_inferences=1000, input_size=30
        )
        results['brainscales2'] = brainscales_result
        
        # Save results
        with open(self.output_dir / "single_chip_throughput.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n✅ Single chip throughput test complete")
        return results
    
    def test_distributed_scaling(self, max_chips: int = 8) -> Dict[str, Any]:
        """Test how throughput scales with number of chips."""
        logger.info("=" * 70)
        logger.info("TEST 2: Distributed Scaling")
        logger.info("=" * 70)
        
        results = []
        
        for num_chips in [1, 2, 4, 8]:
            if num_chips > max_chips:
                break
                
            logger.info(f"\nTesting with {num_chips} chip(s)...")
            
            # Create cluster
            cluster = DistributedNeuromorphicCluster(
                load_balancing_strategy="least_loaded"
            )
            
            # Add chips (mix of types)
            for i in range(num_chips):
                if i % 3 == 0:
                    chip_type = ChipType.LOIHI2
                elif i % 3 == 1:
                    chip_type = ChipType.BRAINSCALES2
                else:
                    chip_type = ChipType.TRUENORTH
                
                cluster.add_chip(chip_type, max_capacity=500)
            
            # Start workers
            cluster.start_workers(num_workers=num_chips * 2)
            
            # Benchmark
            benchmark = cluster.benchmark(num_transactions=1000, batch_size=100)
            
            # Stop workers
            cluster.stop_workers()
            
            result = {
                'num_chips': num_chips,
                'throughput_tps': benchmark['throughput_tps'],
                'avg_latency_ms': benchmark['avg_latency_ms'],
                'total_energy_j': benchmark['total_energy_j'],
                'scaling_efficiency': benchmark['throughput_tps'] / num_chips
            }
            
            results.append(result)
            
            logger.info(f"  Throughput: {result['throughput_tps']:.0f} TPS")
            logger.info(f"  Scaling efficiency: {result['scaling_efficiency']:.0f} TPS/chip")
        
        # Save results
        with open(self.output_dir / "distributed_scaling.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot scaling curve
        self._plot_scaling_curve(results)
        
        logger.info("\n✅ Distributed scaling test complete")
        return results
    
    def test_load_balancing_strategies(self) -> Dict[str, Any]:
        """Compare different load balancing strategies."""
        logger.info("=" * 70)
        logger.info("TEST 3: Load Balancing Strategies")
        logger.info("=" * 70)
        
        strategies = ["round_robin", "least_loaded", "energy_efficient", "latency_optimized"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"\nTesting strategy: {strategy}...")
            
            cluster = DistributedNeuromorphicCluster(
                load_balancing_strategy=strategy
            )
            
            # Add heterogeneous chips
            cluster.add_chip(ChipType.LOIHI2, max_capacity=500)
            cluster.add_chip(ChipType.BRAINSCALES2, max_capacity=1000)
            cluster.add_chip(ChipType.TRUENORTH, max_capacity=300)
            
            cluster.start_workers(num_workers=6)
            
            # Benchmark
            benchmark = cluster.benchmark(num_transactions=1000, batch_size=100)
            
            cluster.stop_workers()
            
            results[strategy] = {
                'throughput_tps': benchmark['throughput_tps'],
                'avg_latency_ms': benchmark['avg_latency_ms'],
                'p95_latency_ms': benchmark['p95_latency_ms'],
                'total_energy_j': benchmark['total_energy_j'],
                'chip_utilization': benchmark['chip_utilization']
            }
            
            logger.info(f"  Throughput: {results[strategy]['throughput_tps']:.0f} TPS")
            logger.info(f"  Avg latency: {results[strategy]['avg_latency_ms']:.2f} ms")
        
        # Save results
        with open(self.output_dir / "load_balancing.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot comparison
        self._plot_load_balancing_comparison(results)
        
        logger.info("\n✅ Load balancing strategies test complete")
        return results
    
    def test_fault_tolerance(self) -> Dict[str, Any]:
        """Test cluster behavior with chip failures."""
        logger.info("=" * 70)
        logger.info("TEST 4: Fault Tolerance")
        logger.info("=" * 70)
        
        cluster = DistributedNeuromorphicCluster()
        
        # Add 4 chips
        for i in range(4):
            cluster.add_chip(ChipType.LOIHI2, f"loihi_{i}", max_capacity=500)
        
        cluster.start_workers(num_workers=8)
        
        # Baseline with all chips
        logger.info("\nBaseline (all chips healthy)...")
        baseline = cluster.benchmark(num_transactions=500, batch_size=50)
        
        # Simulate 1 chip failure
        logger.info("\nSimulating 1 chip failure...")
        cluster.chips[0].is_healthy = False
        result_1_fail = cluster.benchmark(num_transactions=500, batch_size=50)
        
        # Simulate 2 chip failures
        logger.info("\nSimulating 2 chip failures...")
        cluster.chips[1].is_healthy = False
        result_2_fail = cluster.benchmark(num_transactions=500, batch_size=50)
        
        cluster.stop_workers()
        
        results = {
            'baseline': {
                'throughput_tps': baseline['throughput_tps'],
                'avg_latency_ms': baseline['avg_latency_ms'],
                'healthy_chips': 4
            },
            'one_failure': {
                'throughput_tps': result_1_fail['throughput_tps'],
                'avg_latency_ms': result_1_fail['avg_latency_ms'],
                'throughput_degradation': (baseline['throughput_tps'] - result_1_fail['throughput_tps']) / baseline['throughput_tps'],
                'healthy_chips': 3
            },
            'two_failures': {
                'throughput_tps': result_2_fail['throughput_tps'],
                'avg_latency_ms': result_2_fail['avg_latency_ms'],
                'throughput_degradation': (baseline['throughput_tps'] - result_2_fail['throughput_tps']) / baseline['throughput_tps'],
                'healthy_chips': 2
            }
        }
        
        # Save results
        with open(self.output_dir / "fault_tolerance.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n✅ Fault tolerance test complete")
        logger.info(f"  1 failure: {results['one_failure']['throughput_degradation']*100:.1f}% degradation")
        logger.info(f"  2 failures: {results['two_failures']['throughput_degradation']*100:.1f}% degradation")
        
        return results
    
    def test_stress_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Sustained load stress test."""
        logger.info("=" * 70)
        logger.info(f"TEST 5: Stress Test ({duration_seconds}s)")
        logger.info("=" * 70)
        
        cluster = DistributedNeuromorphicCluster()
        
        # Create production-like cluster
        cluster.add_chip(ChipType.LOIHI2, "loihi_0", max_capacity=500)
        cluster.add_chip(ChipType.LOIHI2, "loihi_1", max_capacity=500)
        cluster.add_chip(ChipType.BRAINSCALES2, "brainscales_0", max_capacity=1000)
        
        cluster.start_workers(num_workers=8)
        
        # Sustained load
        start_time = time.time()
        total_transactions = 0
        throughput_samples = []
        
        logger.info("\nApplying sustained load...")
        
        while time.time() - start_time < duration_seconds:
            batch = []
            for i in range(100):
                txn = Transaction(
                    transaction_id=f"stress_{total_transactions}_{i}",
                    features=np.random.randn(30),
                    timestamp=time.time(),
                    priority=0
                )
                batch.append(txn)
            
            cluster.submit_batch(batch)
            total_transactions += len(batch)
            
            # Sample throughput
            time.sleep(0.5)
            results = cluster.get_results(timeout=0.1)
            
            if results:
                sample_tps = len(results) / 0.5
                throughput_samples.append(sample_tps)
            
            if int(time.time() - start_time) % 10 == 0:
                logger.info(f"  {int(time.time() - start_time)}s elapsed, {total_transactions} transactions submitted")
        
        # Wait for completion
        time.sleep(2)
        final_results = cluster.get_results(timeout=5.0)
        
        cluster.stop_workers()
        
        elapsed = time.time() - start_time
        
        results = {
            'duration_s': elapsed,
            'total_transactions': total_transactions,
            'avg_throughput_tps': total_transactions / elapsed,
            'peak_throughput_tps': max(throughput_samples) if throughput_samples else 0,
            'min_throughput_tps': min(throughput_samples) if throughput_samples else 0,
            'throughput_stability': np.std(throughput_samples) if throughput_samples else 0,
            'total_energy_j': cluster.total_energy_j
        }
        
        # Save results
        with open(self.output_dir / "stress_test.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n✅ Stress test complete")
        logger.info(f"  Average throughput: {results['avg_throughput_tps']:.0f} TPS")
        logger.info(f"  Peak throughput: {results['peak_throughput_tps']:.0f} TPS")
        logger.info(f"  Total energy: {results['total_energy_j']:.3f} J")
        
        return results
    
    def _plot_scaling_curve(self, results: List[Dict]):
        """Plot scaling efficiency curve."""
        num_chips = [r['num_chips'] for r in results]
        throughput = [r['throughput_tps'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(num_chips, throughput, 'o-', linewidth=2, markersize=8, label='Actual')
        
        # Ideal linear scaling
        ideal = [throughput[0] * n for n in num_chips]
        plt.plot(num_chips, ideal, '--', linewidth=2, alpha=0.5, label='Ideal Linear')
        
        plt.xlabel('Number of Chips', fontsize=12)
        plt.ylabel('Throughput (TPS)', fontsize=12)
        plt.title('Distributed Scaling Efficiency', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "scaling_curve.png", dpi=300)
        plt.close()
        
        logger.info(f"  Scaling curve saved to {self.output_dir}/scaling_curve.png")
    
    def _plot_load_balancing_comparison(self, results: Dict):
        """Plot load balancing strategy comparison."""
        strategies = list(results.keys())
        throughputs = [results[s]['throughput_tps'] for s in strategies]
        latencies = [results[s]['avg_latency_ms'] for s in strategies]
        energies = [results[s]['total_energy_j'] * 1000 for s in strategies]  # mJ
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Throughput
        axes[0].bar(strategies, throughputs, color='steelblue')
        axes[0].set_ylabel('Throughput (TPS)', fontsize=11)
        axes[0].set_title('Throughput Comparison', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Latency
        axes[1].bar(strategies, latencies, color='coral')
        axes[1].set_ylabel('Latency (ms)', fontsize=11)
        axes[1].set_title('Average Latency', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Energy
        axes[2].bar(strategies, energies, color='seagreen')
        axes[2].set_ylabel('Energy (mJ)', fontsize=11)
        axes[2].set_title('Total Energy', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "load_balancing_comparison.png", dpi=300)
        plt.close()
        
        logger.info(f"  Load balancing comparison saved to {self.output_dir}/load_balancing_comparison.png")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING COMPLETE PHASE 5 SCALING TEST SUITE")
        logger.info("=" * 70 + "\n")
        
        all_results = {}
        
        # Test 1: Single chip throughput
        all_results['single_chip'] = self.test_single_chip_throughput()
        
        # Test 2: Distributed scaling
        all_results['distributed_scaling'] = self.test_distributed_scaling(max_chips=8)
        
        # Test 3: Load balancing
        all_results['load_balancing'] = self.test_load_balancing_strategies()
        
        # Test 4: Fault tolerance
        all_results['fault_tolerance'] = self.test_fault_tolerance()
        
        # Test 5: Stress test
        all_results['stress_test'] = self.test_stress_test(duration_seconds=30)
        
        # Save complete results
        with open(self.output_dir / "complete_test_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {self.output_dir}/")
        
        return all_results


# Run tests
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 5 Scaling Test Suite")
    print("=" * 70)
    
    suite = ScalingTestSuite(output_dir="scaling_results")
    results = suite.run_all_tests()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    # Single chip best performer
    single_chip = results['single_chip']
    loihi_tps = 1.0 / single_chip['loihi2']['avg_latency_ms'] * 1000
    brainscales_tps = 1.0 / (single_chip['brainscales2']['avg_latency_us'] / 1000) * 1000
    print(f"\nSingle Chip Performance:")
    print(f"  Loihi 2: {loihi_tps:.0f} TPS")
    print(f"  BrainScaleS-2: {brainscales_tps:.0f} TPS")
    
    # Distributed scaling
    scaling = results['distributed_scaling']
    max_scaling = max(r['throughput_tps'] for r in scaling)
    print(f"\nDistributed Scaling:")
    print(f"  Max throughput: {max_scaling:.0f} TPS")
    
    # Best load balancing
    lb = results['load_balancing']
    best_strategy = max(lb.keys(), key=lambda s: lb[s]['throughput_tps'])
    print(f"\nBest Load Balancing Strategy: {best_strategy}")
    print(f"  Throughput: {lb[best_strategy]['throughput_tps']:.0f} TPS")
    
    # Stress test
    stress = results['stress_test']
    print(f"\nStress Test (sustained load):")
    print(f"  Average: {stress['avg_throughput_tps']:.0f} TPS")
    print(f"  Peak: {stress['peak_throughput_tps']:.0f} TPS")
    
    print("\n" + "=" * 70)
    print("Phase 5 validation complete! ✅")
    print("=" * 70)
