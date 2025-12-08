"""
Pydantic models for API request/response validation.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
Date: December 2025
License: MIT
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TransactionCategory(str, Enum):
    """Transaction category enumeration."""
    groceries = "groceries"
    retail = "retail"
    restaurant = "restaurant"
    gas = "gas"
    online = "online"
    travel = "travel"
    healthcare = "healthcare"
    entertainment = "entertainment"
    utilities = "utilities"
    other = "other"


class Transaction(BaseModel):
    """
    Transaction data model for fraud detection.
    
    Represents a single financial transaction with all features
    needed for fraud prediction.
    """
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in local currency")
    timestamp: int = Field(..., description="Unix timestamp of transaction")
    latitude: float = Field(..., ge=-90, le=90, description="Transaction latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Transaction longitude")
    category: Optional[TransactionCategory] = Field(
        TransactionCategory.other,
        description="Transaction category"
    )
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    card_last_4: Optional[str] = Field(None, description="Last 4 digits of card")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "amount": 129.99,
                "timestamp": 1733443200,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "category": "retail",
                "merchant_id": "merchant_abc123",
                "card_last_4": "4242"
            }
        }
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is reasonable."""
        now = int(datetime.utcnow().timestamp())
        # Allow timestamps from past 1 year to 1 day in future
        if v < now - 365*24*3600 or v > now + 24*3600:
            raise ValueError('Timestamp out of reasonable range')
        return v


class TransactionBatch(BaseModel):
    """
    Batch of transactions for bulk prediction.
    """
    batch_id: str = Field(..., description="Unique batch identifier")
    transactions: List[Transaction] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of transactions (max 1000)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_20251205_001",
                "transactions": [
                    {
                        "transaction_id": "txn_001",
                        "amount": 50.00,
                        "timestamp": 1733443200,
                        "latitude": 40.7128,
                        "longitude": -74.0060,
                        "category": "groceries"
                    },
                    {
                        "transaction_id": "txn_002",
                        "amount": 1500.00,
                        "timestamp": 1733443260,
                        "latitude": 51.5074,
                        "longitude": -0.1278,
                        "category": "online"
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Fraud prediction response for a single transaction.
    """
    transaction_id: str = Field(..., description="Transaction identifier")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    fraud_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of fraud (0-1)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model confidence in prediction"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    risk_factors: Optional[List[str]] = Field(
        None,
        description="List of identified risk factors"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "is_fraud": True,
                "fraud_probability": 0.89,
                "confidence": 0.95,
                "processing_time_ms": 8.5,
                "timestamp": "2025-12-05T12:00:00",
                "model_version": "2.0.0",
                "risk_factors": ["unusual_amount", "unusual_location", "rapid_sequence"]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response for batch prediction request.
    """
    batch_id: str = Field(..., description="Batch identifier")
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_transactions: int = Field(..., description="Total transactions processed")
    total_processing_time_ms: float = Field(
        ...,
        description="Total processing time"
    )
    timestamp: datetime = Field(..., description="Batch completion timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_20251205_001",
                "predictions": [
                    {
                        "transaction_id": "txn_001",
                        "is_fraud": False,
                        "fraud_probability": 0.12,
                        "confidence": 0.98,
                        "processing_time_ms": 8.5,
                        "timestamp": "2025-12-05T12:00:00",
                        "model_version": "2.0.0"
                    }
                ],
                "total_transactions": 2,
                "total_processing_time_ms": 17.0,
                "timestamp": "2025-12-05T12:00:01"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    pipeline_loaded: bool = Field(..., description="Whether ML pipeline is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-12-05T12:00:00",
                "version": "2.0.0",
                "pipeline_loaded": True,
                "uptime_seconds": 3600.5
            }
        }


class MetricsResponse(BaseModel):
    """
    System metrics response.
    """
    total_predictions: int = Field(..., description="Total predictions made")
    total_errors: int = Field(..., description="Total errors encountered")
    avg_latency_ms: float = Field(..., description="Average prediction latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    throughput_per_second: float = Field(..., description="Current throughput")
    fraud_rate: float = Field(..., description="Detected fraud rate")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_mb: float = Field(..., description="Memory usage in MB")
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 10000,
                "total_errors": 5,
                "avg_latency_ms": 8.5,
                "p95_latency_ms": 15.2,
                "p99_latency_ms": 22.1,
                "throughput_per_second": 120.5,
                "fraud_rate": 0.023,
                "cpu_percent": 45.2,
                "memory_mb": 512.8
            }
        }


class TrainingRequest(BaseModel):
    """
    Model training request.
    """
    dataset_path: Optional[str] = Field(None, description="Path to training dataset")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Hyperparameters for training"
    )
    validation_split: float = Field(0.2, ge=0, le=0.5, description="Validation split")
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_path": "/data/transactions.csv",
                "hyperparameters": {
                    "n_hidden1": 128,
                    "tau_m": 10.0,
                    "learning_rate": 0.01
                },
                "validation_split": 0.2
            }
        }


class TrainingResponse(BaseModel):
    """
    Model training response.
    """
    status: str = Field(..., description="Training status")
    model_version: str = Field(..., description="New model version")
    metrics: Dict[str, float] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Total training time")
    timestamp: datetime = Field(..., description="Training completion timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "completed",
                "model_version": "2.1.0",
                "metrics": {
                    "accuracy": 0.96,
                    "precision": 0.94,
                    "recall": 0.92,
                    "f1_score": 0.93
                },
                "training_time_seconds": 120.5,
                "timestamp": "2025-12-05T12:30:00"
            }
        }
