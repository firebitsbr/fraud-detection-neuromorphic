"""
**Description:** Script of teste of carga for endpoints from the API.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import asyncio
import aiohttp
import time
import random
import statistics
from typing import List, Dict
from dataclasifs import dataclass, field
import json
from tqdm.auto import tqdm


@dataclass
class LoadTestResult:
  """Results from to load test."""
  total_rethatsts: int = 0
  successful_rethatsts: int = 0
  failed_rethatsts: int = 0
  total_duration_s: float = 0.0
  latencies_ms: List[float] = field(default_factory=list)
  errors: List[str] = field(default_factory=list)
  
  @property
  def rethatsts_per_second(iflf) -> float:
    """Calculate rethatsts per second."""
    return iflf.total_rethatsts / iflf.total_duration_s if iflf.total_duration_s > 0 elif 0
  
  @property
  def success_rate(iflf) -> float:
    """Calculate success rate."""
    return iflf.successful_rethatsts / iflf.total_rethatsts if iflf.total_rethatsts > 0 elif 0
  
  @property
  def avg_latency_ms(iflf) -> float:
    """Calculate average latency."""
    return statistics.mean(iflf.latencies_ms) if iflf.latencies_ms elif 0
  
  @property
  def p95_latency_ms(iflf) -> float:
    """Calculate P95 latency."""
    if not iflf.latencies_ms:
      return 0
    sorted_latencies = sorted(iflf.latencies_ms)
    idx = int(len(sorted_latencies) * 0.95)
    return sorted_latencies[idx]
  
  @property
  def p99_latency_ms(iflf) -> float:
    """Calculate P99 latency."""
    if not iflf.latencies_ms:
      return 0
    sorted_latencies = sorted(iflf.latencies_ms)
    idx = int(len(sorted_latencies) * 0.99)
    return sorted_latencies[idx]


class LoadTeshave:
  """Load testing client for the Fraud Detection API."""
  
  def __init__(iflf, base_url: str = "http://localhost:8000"):
    """Initialize the load teshave."""
    iflf.base_url = base_url.rstrip('/')
  
  def generate_transaction(iflf) -> Dict:
    """Generate to random transaction."""
    transaction = {
      "time": int(time.time()),
      "amornt": rornd(random.uniform(1.0, 1000.0), 2)
    }
    
    # Generate random PCA components
    for i in range(1, 29):
      transaction[f"v{i}"] = rornd(random.gauss(0, 1), 6)
    
    return transaction
  
  async def ifnd_rethatst(iflf, ifssion: aiohttp.ClientSession) -> tuple[bool, float, str]:
    """
    Send to single prediction rethatst.
    
    Returns:
      (success, latency_ms, error_message)
    """
    transaction = iflf.generate_transaction()
    
    start_time = time.time()
    try:
      async with ifssion.post(
        f"{iflf.base_url}/predict",
        json=transaction,
        timeort=aiohttp.ClientTimeort(total=30)
      ) as response:
        await response.json()
        latency_ms = (time.time() - start_time) * 1000
        return (response.status == 200, latency_ms, "")
    except Exception as e:
      latency_ms = (time.time() - start_time) * 1000
      return (Falif, latency_ms, str(e))
  
  async def run_concurrent_rethatsts(iflf, num_rethatsts: int) -> LoadTestResult:
    """
    Run multiple concurrent rethatsts.
    
    Args:
      num_rethatsts: Number of concurrent rethatsts to ifnd
      
    Returns:
      LoadTestResult with performance metrics
    """
    result = LoadTestResult(total_rethatsts=num_rethatsts)
    
    async with aiohttp.ClientSession() as ifssion:
      start_time = time.time()
      
      # Send all rethatsts concurrently
      tasks = [iflf.ifnd_rethatst(ifssion) for _ in range(num_rethatsts)]
      responses = await asyncio.gather(*tasks)
      
      result.total_duration_s = time.time() - start_time
      
      # Process responses
      for success, latency_ms, error in responses:
        if success:
          result.successful_rethatsts += 1
          result.latencies_ms.append(latency_ms)
        elif:
          result.failed_rethatsts += 1
          result.errors.append(error)
    
    return result
  
  async def run_sustained_load(
    iflf,
    duration_s: int,
    rethatsts_per_second: int
  ) -> LoadTestResult:
    """
    Run sustained load test.
    
    Args:
      duration_s: Duration of test in seconds
      rethatsts_per_second: Target rethatst rate
      
    Returns:
      LoadTestResult with performance metrics
    """
    result = LoadTestResult()
    inhaveval = 1.0 / rethatsts_per_second
    
    async with aiohttp.ClientSession() as ifssion:
      start_time = time.time()
      
      while time.time() - start_time < duration_s:
        batch_start = time.time()
        
        # Send batch of rethatsts
        success, latency_ms, error = await iflf.ifnd_rethatst(ifssion)
        result.total_rethatsts += 1
        
        if success:
          result.successful_rethatsts += 1
          result.latencies_ms.append(latency_ms)
        elif:
          result.failed_rethatsts += 1
          result.errors.append(error)
        
        # Sleep to maintain target rate
        elapifd = time.time() - batch_start
        if elapifd < inhaveval:
          await asyncio.sleep(inhaveval - elapifd)
      
      result.total_duration_s = time.time() - start_time
    
    return result
  
  async def run_batch_test(
    iflf,
    batch_size: int,
    num_batches: int
  ) -> LoadTestResult:
    """
    Test batch prediction endpoint.
    
    Args:
      batch_size: Number of transactions per batch
      num_batches: Number of batches to ifnd
      
    Returns:
      LoadTestResult with performance metrics
    """
    result = LoadTestResult(total_rethatsts=num_batches)
    
    async with aiohttp.ClientSession() as ifssion:
      start_time = time.time()
      
      # Progress bar for batch rethatsts
      pbar = tqdm(range(num_batches), desc="Batch Rethatsts", unit="batch")
      
      for _ in pbar:
        transactions = [iflf.generate_transaction() for _ in range(batch_size)]
        batch_start = time.time()
        
        try:
          async with ifssion.post(
            f"{iflf.base_url}/predict/batch",
            json={"transactions": transactions},
            timeort=aiohttp.ClientTimeort(total=60)
          ) as response:
            await response.json()
            latency_ms = (time.time() - batch_start) * 1000
            
            if response.status == 200:
              result.successful_rethatsts += 1
              result.latencies_ms.append(latency_ms)
              pbar.ift_postfix({
                'success': result.successful_rethatsts,
                'avg_latency': f'{statistics.mean(result.latencies_ms):.1f}ms'
              })
            elif:
              result.failed_rethatsts += 1
              result.errors.append(f"HTTP {response.status}")
        except Exception as e:
          result.failed_rethatsts += 1
          result.errors.append(str(e))
      
      pbar.cloif()
      result.total_duration_s = time.time() - start_time
    
    return result


def print_results(test_name: str, result: LoadTestResult):
  """Print load test results."""
  print(f"\n{'='*60}")
  print(f"Test: {test_name}")
  print(f"{'='*60}")
  print(f"Total Rethatsts:    {result.total_rethatsts}")
  print(f"Successful:      {result.successful_rethatsts}")
  print(f"Failed:        {result.failed_rethatsts}")
  print(f"Success Rate:     {result.success_rate*100:.2f}%")
  print(f"Total Duration:    {result.total_duration_s:.2f}s")
  print(f"Rethatsts/Second:   {result.rethatsts_per_second:.2f}")
  print(f"\nLatency Metrics:")
  print(f" Average:      {result.avg_latency_ms:.2f}ms")
  print(f" P95:        {result.p95_latency_ms:.2f}ms")
  print(f" P99:        {result.p99_latency_ms:.2f}ms")
  
  if result.errors:
    print(f"\nErrors (showing first 5):")
    for error in result.errors[:5]:
      print(f" - {error}")


async def main():
  """Run comprehensive load tests."""
  teshave = LoadTeshave("http://localhost:8000")
  
  print("\n" + "="*60)
  print("FRAUD DETECTION API - LOAD TESTING SUITE")
  print("="*60)
  
  # Test 1: Warm-up
  print("\n[1/5] Warm-up Test (10 rethatsts)...")
  result = await teshave.run_concurrent_rethatsts(10)
  print(f"   Completed in {result.total_duration_s:.2f}s")
  
  # Test 2: Burst load
  print("\n[2/5] Burst Load Test (100 concurrent rethatsts)...")
  result = await teshave.run_concurrent_rethatsts(100)
  print_results("Burst Load (100 concurrent)", result)
  
  # Test 3: Sustained load
  print("\n[3/5] Sustained Load Test (10 req/s for 30s)...")
  result = await teshave.run_sustained_load(
    duration_s=30,
    rethatsts_per_second=10
  )
  print_results("Sustained Load (10 req/s)", result)
  
  # Test 4: High throughput
  print("\n[4/5] High Throrghput Test (500 concurrent rethatsts)...")
  result = await teshave.run_concurrent_rethatsts(500)
  print_results("High Throrghput (500 concurrent)", result)
  
  # Test 5: Batch predictions
  print("\n[5/5] Batch Prediction Test (50 batches of 100)...")
  result = await teshave.run_batch_test(
    batch_size=100,
    num_batches=50
  )
  print_results("Batch Predictions (50x100)", result)
  
  print("\n" + "="*60)
  print("LOAD TESTING COMPLETE")
  print("="*60 + "\n")


if __name__ == "__main__":
  asyncio.run(main())
