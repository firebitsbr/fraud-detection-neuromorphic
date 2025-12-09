"""
**Descrição:** Script de teste de carga para endpoints da API.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import asyncio
import aiohttp
import time
import random
import statistics
from typing import List, Dict
from dataclasses import dataclass, field
import json
from tqdm.auto import tqdm


@dataclass
class LoadTestResult:
    """Results from a load test."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_s: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        return self.total_requests / self.total_duration_s if self.total_duration_s > 0 else 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def p95_latency_ms(self) -> float:
        """Calculate P95 latency."""
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]
    
    @property
    def p99_latency_ms(self) -> float:
        """Calculate P99 latency."""
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]


class LoadTester:
    """Load testing client for the Fraud Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the load tester."""
        self.base_url = base_url.rstrip('/')
    
    def generate_transaction(self) -> Dict:
        """Generate a random transaction."""
        transaction = {
            "time": int(time.time()),
            "amount": round(random.uniform(1.0, 1000.0), 2)
        }
        
        # Generate random PCA components
        for i in range(1, 29):
            transaction[f"v{i}"] = round(random.gauss(0, 1), 6)
        
        return transaction
    
    async def send_request(self, session: aiohttp.ClientSession) -> tuple[bool, float, str]:
        """
        Send a single prediction request.
        
        Returns:
            (success, latency_ms, error_message)
        """
        transaction = self.generate_transaction()
        
        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/predict",
                json=transaction,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                await response.json()
                latency_ms = (time.time() - start_time) * 1000
                return (response.status == 200, latency_ms, "")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return (False, latency_ms, str(e))
    
    async def run_concurrent_requests(self, num_requests: int) -> LoadTestResult:
        """
        Run multiple concurrent requests.
        
        Args:
            num_requests: Number of concurrent requests to send
            
        Returns:
            LoadTestResult with performance metrics
        """
        result = LoadTestResult(total_requests=num_requests)
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Send all requests concurrently
            tasks = [self.send_request(session) for _ in range(num_requests)]
            responses = await asyncio.gather(*tasks)
            
            result.total_duration_s = time.time() - start_time
            
            # Process responses
            for success, latency_ms, error in responses:
                if success:
                    result.successful_requests += 1
                    result.latencies_ms.append(latency_ms)
                else:
                    result.failed_requests += 1
                    result.errors.append(error)
        
        return result
    
    async def run_sustained_load(
        self,
        duration_s: int,
        requests_per_second: int
    ) -> LoadTestResult:
        """
        Run sustained load test.
        
        Args:
            duration_s: Duration of test in seconds
            requests_per_second: Target request rate
            
        Returns:
            LoadTestResult with performance metrics
        """
        result = LoadTestResult()
        interval = 1.0 / requests_per_second
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            while time.time() - start_time < duration_s:
                batch_start = time.time()
                
                # Send batch of requests
                success, latency_ms, error = await self.send_request(session)
                result.total_requests += 1
                
                if success:
                    result.successful_requests += 1
                    result.latencies_ms.append(latency_ms)
                else:
                    result.failed_requests += 1
                    result.errors.append(error)
                
                # Sleep to maintain target rate
                elapsed = time.time() - batch_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
            
            result.total_duration_s = time.time() - start_time
        
        return result
    
    async def run_batch_test(
        self,
        batch_size: int,
        num_batches: int
    ) -> LoadTestResult:
        """
        Test batch prediction endpoint.
        
        Args:
            batch_size: Number of transactions per batch
            num_batches: Number of batches to send
            
        Returns:
            LoadTestResult with performance metrics
        """
        result = LoadTestResult(total_requests=num_batches)
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Progress bar for batch requests
            pbar = tqdm(range(num_batches), desc="Batch Requests", unit="batch")
            
            for _ in pbar:
                transactions = [self.generate_transaction() for _ in range(batch_size)]
                batch_start = time.time()
                
                try:
                    async with session.post(
                        f"{self.base_url}/predict/batch",
                        json={"transactions": transactions},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        await response.json()
                        latency_ms = (time.time() - batch_start) * 1000
                        
                        if response.status == 200:
                            result.successful_requests += 1
                            result.latencies_ms.append(latency_ms)
                            pbar.set_postfix({
                                'success': result.successful_requests,
                                'avg_latency': f'{statistics.mean(result.latencies_ms):.1f}ms'
                            })
                        else:
                            result.failed_requests += 1
                            result.errors.append(f"HTTP {response.status}")
                except Exception as e:
                    result.failed_requests += 1
                    result.errors.append(str(e))
            
            pbar.close()
            result.total_duration_s = time.time() - start_time
        
        return result


def print_results(test_name: str, result: LoadTestResult):
    """Print load test results."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Total Requests:       {result.total_requests}")
    print(f"Successful:           {result.successful_requests}")
    print(f"Failed:               {result.failed_requests}")
    print(f"Success Rate:         {result.success_rate*100:.2f}%")
    print(f"Total Duration:       {result.total_duration_s:.2f}s")
    print(f"Requests/Second:      {result.requests_per_second:.2f}")
    print(f"\nLatency Metrics:")
    print(f"  Average:            {result.avg_latency_ms:.2f}ms")
    print(f"  P95:                {result.p95_latency_ms:.2f}ms")
    print(f"  P99:                {result.p99_latency_ms:.2f}ms")
    
    if result.errors:
        print(f"\nErrors (showing first 5):")
        for error in result.errors[:5]:
            print(f"  - {error}")


async def main():
    """Run comprehensive load tests."""
    tester = LoadTester("http://localhost:8000")
    
    print("\n" + "="*60)
    print("FRAUD DETECTION API - LOAD TESTING SUITE")
    print("="*60)
    
    # Test 1: Warm-up
    print("\n[1/5] Warm-up Test (10 requests)...")
    result = await tester.run_concurrent_requests(10)
    print(f"      Completed in {result.total_duration_s:.2f}s")
    
    # Test 2: Burst load
    print("\n[2/5] Burst Load Test (100 concurrent requests)...")
    result = await tester.run_concurrent_requests(100)
    print_results("Burst Load (100 concurrent)", result)
    
    # Test 3: Sustained load
    print("\n[3/5] Sustained Load Test (10 req/s for 30s)...")
    result = await tester.run_sustained_load(
        duration_s=30,
        requests_per_second=10
    )
    print_results("Sustained Load (10 req/s)", result)
    
    # Test 4: High throughput
    print("\n[4/5] High Throughput Test (500 concurrent requests)...")
    result = await tester.run_concurrent_requests(500)
    print_results("High Throughput (500 concurrent)", result)
    
    # Test 5: Batch predictions
    print("\n[5/5] Batch Prediction Test (50 batches of 100)...")
    result = await tester.run_batch_test(
        batch_size=100,
        num_batches=50
    )
    print_results("Batch Predictions (50x100)", result)
    
    print("\n" + "="*60)
    print("LOAD TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
