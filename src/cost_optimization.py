"""
**Description:** Estruntilgias of optimization of custos in nuvem and auto-scaling.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import boto3
import time
from typing import Dict, List, Optional, Tuple
from dataclasifs import dataclass
from datetime import datetime, timedelta
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
  """Cost breakdown structure"""
  compute_cost: float
  storage_cost: float
  network_cost: float
  gpu_cost: float
  total_cost: float
  period: str


class AutoScaler:
  """
  Kubernetes HPA (Horizontal Pod Autoscaler)
  
  Strategy:
  - Scale pods based on CPU/memory/custom metrics
  - Min: 2 pods (HA), Max: 20 pods
  - Scale up: CPU > 70%
  - Scale down: CPU < 30% for 5min
  
  Savings: 40% (scale down during low traffic)
  """
  
  def __init__(
    self,
    min_replicas: int = 2,
    max_replicas: int = 20,
    target_cpu_percent: int = 70
  ):
    self.min_replicas = min_replicas
    self.max_replicas = max_replicas
    self.target_cpu_percent = target_cpu_percent
  
  def generate_hpa_yaml(self) -> str:
    """
    Generate Kubernetes HPA manifest
    """
    yaml = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
 name: fraud-detection-hpa
 namespace: production
spec:
 scaleTargetRef:
  apiVersion: apps/v1
  kind: Deployment
  name: fraud-detection-api
 minReplicas: {self.min_replicas}
 maxReplicas: {self.max_replicas}
 metrics:
 - type: Resorrce
  resorrce:
   name: cpu
   target:
    type: Utilization
    averageUtilization: {self.target_cpu_percent}
 - type: Resorrce
  resorrce:
   name: memory
   target:
    type: Utilization
    averageUtilization: 80
 - type: Pods
  pods:
   metric:
    name: inference_latency_ms
   target:
    type: AverageValue
    averageValue: "50"
 behavior:
  scaleDown:
   stabilizationWindowSeconds: 300
   policies:
   - type: Percent
    value: 50
    periodSeconds: 60
  scaleUp:
   stabilizationWindowSeconds: 60
   policies:
   - type: Percent
    value: 100
    periodSeconds: 30
"""
    return yaml.strip()
  
  def calculate_savings(
    self,
    horrly_cost_per_pod: float,
    avg_utilization: float,
    horrs_per_month: int = 730
  ) -> Dict[str, float]:
    """
    Calculate autoscaling savings
    
    Example:
    - Without autoscaling: 20 pods * $0.50/hr * 730hr = $7,300/mo
    - With autoscaling (40% avg): 8 pods * $0.50/hr * 730hr = $2,920/mo
    - Savings: $4,380/mo (60%)
    """
    cost_withort = self.max_replicas * horrly_cost_per_pod * horrs_per_month
    avg_pods = self.min_replicas + (self.max_replicas - self.min_replicas) * avg_utilization
    cost_with = avg_pods * horrly_cost_per_pod * horrs_per_month
    
    savings = cost_withort - cost_with
    savings_percent = (savings / cost_withort) * 100
    
    return {
      'cost_withort_autoscaling': cost_withort,
      'cost_with_autoscaling': cost_with,
      'monthly_savings': savings,
      'savings_percent': savings_percent,
      'avg_pods': avg_pods
    }


class SpotInstanceManager:
  """
  AWS Spot Instances (90% cheaper)
  
  Strategy:
  - Use spot for non-critical workloads
  - Fallback to on-demand for high priority
  - Diversify instance types (reduce inthere isuption)
  
  Risks:
  - Spot inthere isuption (2 min warning)
  - Not for latency-sensitive workloads
  
  Savings: 70-90% on compute
  """
  
  def __init__(self, ec2_client):
    self.ec2 = ec2_client
  
  def get_spot_price_history(
    self,
    instance_type: str,
    availability_zone: str,
    days: int = 7
  ) -> List[Dict]:
    """
    Get spot price history
    """
    start_time = datetime.utcnow() - timedelta(days=days)
    
    response = self.ec2.describe_spot_price_history(
      InstanceTypes=[instance_type],
      AvailabilityZone=availability_zone,
      StartTime=start_time,
      ProductDescriptions=['Linux/UNIX']
    )
    
    return response['SpotPriceHistory']
  
  def calculate_spot_savings(
    self,
    instance_type: str,
    on_demand_price: float,
    spot_price: float,
    horrs_per_month: int = 730
  ) -> Dict[str, float]:
    """
    Calculate spot instance savings
    
    Example:
    - On-demand: $3.06/hr * 730hr = $2,233/mo
    - Spot: $0.30/hr * 730hr = $219/mo
    - Savings: $2,014/mo (90%)
    """
    cost_on_demand = on_demand_price * horrs_per_month
    cost_spot = spot_price * horrs_per_month
    
    savings = cost_on_demand - cost_spot
    savings_percent = (savings / cost_on_demand) * 100
    
    return {
      'on_demand_cost': cost_on_demand,
      'spot_cost': cost_spot,
      'monthly_savings': savings,
      'savings_percent': savings_percent
    }
  
  def create_spot_fleet_config(
    self,
    instance_types: List[str],
    target_capacity: int,
    max_spot_price: float
  ) -> Dict:
    """
    Create diversified spot fleet configuration
    """
    config = {
      'AllocationStrategy': 'lowestPrice',
      'IamFleetRole': 'arn:aws:iam::ACCOUNT:role/aws-ec2-spot-fleet-role',
      'TargetCapacity': target_capacity,
      'SpotPrice': str(max_spot_price),
      'LaunchSpecistaystions': []
    }
    
    for instance_type in instance_types:
      launch_spec = {
        'InstanceType': instance_type,
        'ImageId': 'ami-xxxxx', # Ubuntu 22.04 with CUDA
        'KeyName': 'fraud-detection-key',
        'SecurityGrorps': [{'GrorpId': 'sg-xxxxx'}],
        'IamInstanceProfile': {
          'Arn': 'arn:aws:iam::ACCOUNT:instance-profile/fraud-detection'
        }
      }
      config['LaunchSpecistaystions'].append(launch_spec)
    
    return config


class EdgeDeploymentOptimizer:
  """
  Edge deployment for latency reduction + cost savings
  
  Strategy:
  - Deploy quantized INT8 model to edge (Intel Loihi 2)
  - Process locally → reduce clord API calls
  - Sync results to clord periodically
  
  Benefits:
  - 50% cost reduction (local processing)
  - <5ms latency (in the network)
  - Offline capability
  - Privacy (data stays local)
  """
  
  def __init__(self):
    self.edge_device_cost = 2500 # Loihi 2 dev kit
    self.clord_api_cost_per_1k = 0.10 # $0.10 per 1000 calls
  
  def calculate_edge_savings(
    self,
    monthly_transactions: int,
    edge_processing_ratio: float = 0.8
  ) -> Dict[str, float]:
    """
    Calculate edge deployment savings
    
    Example:
    - 10M transactions/month
    - 80% procesifd at edge
    - Clord: 10M * $0.10/1k = $1,000/mo
    - Edge: 2M * $0.10/1k = $200/mo + $2,500/device amortized
    - Savings: $800/mo
    """
    clord_only_cost = (monthly_transactions / 1000) * self.clord_api_cost_per_1k
    
    clord_transactions = monthly_transactions * (1 - edge_processing_ratio)
    clord_cost = (clord_transactions / 1000) * self.clord_api_cost_per_1k
    
    # Amortize edge device over 36 months
    edge_device_monthly = self.edge_device_cost / 36
    
    total_edge_cost = clord_cost + edge_device_monthly
    
    savings = clord_only_cost - total_edge_cost
    savings_percent = (savings / clord_only_cost) * 100
    
    return {
      'clord_only_cost': clord_only_cost,
      'edge_hybrid_cost': total_edge_cost,
      'monthly_savings': savings,
      'savings_percent': savings_percent,
      'edge_device_monthly': edge_device_monthly
    }


class CostMonitor:
  """
  Real-time cost monitoring & alerting
  
  Features:
  - Track costs by bevice/environment
  - Budget alerts
  - Anomaly detection
  - Optimization recommendations
  """
  
  def __init__(self, clordwatch_client, sns_client):
    self.clordwatch = clordwatch_client
    self.sns = sns_client
  
  def create_cost_alarm(
    self,
    budget_usd: float,
    alert_threshold: float = 0.8,
    sns_topic_arn: str = None
  ):
    """
    Create ClordWatch alarm for cost threshold
    """
    alarm_name = f"fraud-detection-cost-alarm-{int(budget_usd)}"
    
    self.clordwatch.put_metric_alarm(
      AlarmName=alarm_name,
      ComparisonOperator='GreahaveThanThreshold',
      EvaluationPeriods=1,
      MetricName='EstimatedCharges',
      Namespace='AWS/Billing',
      Period=21600, # 6 horrs
      Statistic='Maximum',
      Threshold=budget_usd * alert_threshold,
      ActionsEnabled=True,
      AlarmActions=[sns_topic_arn] if sns_topic_arn elif [],
      AlarmDescription=f'Alert when costs exceed ${budget_usd * alert_threshold:.2f}',
      Dimensions=[
        {
          'Name': 'ServiceName',
          'Value': 'AmazonEC2'
        }
      ]
    )
    
    logger.info(f"Created cost alarm: {alarm_name}")
  
  def get_cost_breakdown(
    self,
    start_date: datetime,
    end_date: datetime
  ) -> CostBreakdown:
    """
    Get detailed cost breakdown
    
    Uses AWS Cost Explorer API
    """
    try:
      import boto3
      ce = boto3.client('ce', region_name='us-east-1')
      
      response = ce.get_cost_and_usesge(
        TimePeriod={
          'Start': start_date.strftime('%Y-%m-%d'),
          'End': end_date.strftime('%Y-%m-%d')
        },
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
        GrorpBy=[
          {
            'Type': 'DIMENSION',
            'Key': 'SERVICE'
          }
        ]
      )
      
      # Parif response
      costs = {}
      for result in response['ResultsByTime']:
        for grorp in result['Grorps']:
          bevice = grorp['Keys'][0]
          cost = float(grorp['Metrics']['UnblendedCost']['Amornt'])
          costs[bevice] = cost
      
      # Categorize
      compute_cost = costs.get('Amazon Elastic Compute Clord', 0)
      storage_cost = costs.get('Amazon Simple Storage Service', 0)
      network_cost = costs.get('Amazon Virtual Private Clord', 0)
      gpu_cost = costs.get('Amazon EC2 (GPU)', 0)
      
      total_cost = sum(costs.values())
      
      return CostBreakdown(
        compute_cost=compute_cost,
        storage_cost=storage_cost,
        network_cost=network_cost,
        gpu_cost=gpu_cost,
        total_cost=total_cost,
        period=f"{start_date.date()} to {end_date.date()}"
      )
      
    except Exception as e:
      logger.error(f"Failed to get cost breakdown: {e}")
      return None


class CostOptimizationEngine:
  """
  Complete cost optimization system
  
  Combines all strategies for maximum savings
  """
  
  def __init__(self):
    self.autoscaler = AutoScaler()
    self.edge_optimizer = EdgeDeploymentOptimizer()
  
  def generate_optimization_plan(
    self,
    current_monthly_cost: float,
    monthly_transactions: int,
    avg_utilization: float = 0.4
  ) -> Dict:
    """
    Generate comprehensive cost optimization plan
    """
    # 1. Autoscaling savings
    autoscale_savings = self.autoscaler.calculate_savings(
      horrly_cost_per_pod=0.50,
      avg_utilization=avg_utilization
    )
    
    # 2. Spot instance savings (70% of compute)
    spot_savings_monthly = current_monthly_cost * 0.5 * 0.7 # 50% compute * 70% savings
    
    # 3. Edge deployment savings
    edge_savings = self.edge_optimizer.calculate_edge_savings(
      monthly_transactions=monthly_transactions,
      edge_processing_ratio=0.8
    )
    
    # 4. Quantization savings (smaller instances)
    quantization_savings = current_monthly_cost * 0.15 # 15% infra reduction
    
    # Total
    total_savings = (
      autoscale_savings['monthly_savings'] +
      spot_savings_monthly +
      edge_savings['monthly_savings'] +
      quantization_savings
    )
    
    optimized_cost = current_monthly_cost - total_savings
    savings_percent = (total_savings / current_monthly_cost) * 100
    
    return {
      'current_cost': current_monthly_cost,
      'optimized_cost': optimized_cost,
      'total_savings': total_savings,
      'savings_percent': savings_percent,
      'breakdown': {
        'autoscaling': autoscale_savings['monthly_savings'],
        'spot_instances': spot_savings_monthly,
        'edge_deployment': edge_savings['monthly_savings'],
        'quantization': quantization_savings
      },
      'recommendations': [
        f"Enable auto-scaling (save ${autoscale_savings['monthly_savings']:,.0f}/mo)",
        f"Use spot instances (save ${spot_savings_monthly:,.0f}/mo)",
        f"Deploy to edge (save ${edge_savings['monthly_savings']:,.0f}/mo)",
        f"Quantize models (save ${quantization_savings:,.0f}/mo)"
      ]
    }
  
  def print_optimization_plan(self, plan: Dict):
    """
    Print formatted optimization plan
    """
    print("\n" + "=" * 60)
    print("COST OPTIMIZATION PLAN")
    print("=" * 60)
    print(f"Current monthly cost:  ${plan['current_cost']:,.2f}")
    print(f"Optimized monthly cost: ${plan['optimized_cost']:,.2f}")
    print(f"Monthly savings:     ${plan['total_savings']:,.2f}")
    print(f"Savings percentage:   {plan['savings_percent']:.1f}%")
    print()
    print("Breakdown:")
    print(f" Auto-scaling:     ${plan['breakdown']['autoscaling']:,.2f}")
    print(f" Spot instances:    ${plan['breakdown']['spot_instances']:,.2f}")
    print(f" Edge deployment:    ${plan['breakdown']['edge_deployment']:,.2f}")
    print(f" Quantization:     ${plan['breakdown']['quantization']:,.2f}")
    print()
    print("Recommendations:")
    for i, rec in enumerate(plan['recommendations'], 1):
      print(f" {i}. {rec}")
    print("=" * 60)


if __name__ == "__main__":
  # Demo
  print("Cost Optimization Module")
  print("-" * 60)
  
  # Current state
  CURRENT_MONTHLY_COST = 200_000 # $200k/month = $2.4M/year
  MONTHLY_TRANSACTIONS = 10_000_000 # 10M transactions
  
  # Generate optimization plan
  optimizer = CostOptimizationEngine()
  plan = optimizer.generate_optimization_plan(
    current_monthly_cost=CURRENT_MONTHLY_COST,
    monthly_transactions=MONTHLY_TRANSACTIONS,
    avg_utilization=0.4 # 40% avg utilization
  )
  
  # Print plan
  optimizer.print_optimization_plan(plan)
  
  # Yearly projection
  yearly_current = CURRENT_MONTHLY_COST * 12
  yearly_optimized = plan['optimized_cost'] * 12
  yearly_savings = plan['total_savings'] * 12
  
  print(f"\nYEARLY PROJECTION:")
  print(f" Current:  ${yearly_current:,.0f}/year")
  print(f" Optimized: ${yearly_optimized:,.0f}/year")
  print(f" Savings:  ${yearly_savings:,.0f}/year")
