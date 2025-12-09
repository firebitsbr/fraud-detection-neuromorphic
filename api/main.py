"""
**Descrição:** FastAPI REST API for neuromorphic fraud detection.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uvicorn
import time
import asyncio
from datetime import datetime
import logging
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from main import FraudDetectionPipeline
from performance_profiler import PerformanceProfiler
from api.models import (Transaction, TransactionBatch, PredictionResponse,
                       BatchPredictionResponse, HealthResponse, MetricsResponse)
from api.monitoring import monitoring_service, metrics_collector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Neuromorphic Fraud Detection API",
    description="Real-time fraud detection using Spiking Neural Networks",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[FraudDetectionPipeline] = None
profiler = PerformanceProfiler()


@app.on_event("startup")
async def startup_event():
    """Initialize the fraud detection pipeline on startup."""
    global pipeline
    
    logger.info("Starting Neuromorphic Fraud Detection API...")
    
    try:
        # Initialize pipeline
        pipeline = FraudDetectionPipeline()
        
        # Load pre-trained model if available
        model_path = os.getenv("MODEL_PATH", "models/fraud_snn.pkl")
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            # pipeline.load_model(model_path)  # Implement if needed
        else:
            logger.warning("No pre-trained model found. Using untrained SNN (train via /train endpoint)")
        
        logger.info("API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Neuromorphic Fraud Detection API...")
    monitoring_service.stop()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Neuromorphic Fraud Detection API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/api/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        HealthResponse with system status
    """
    try:
        is_healthy = pipeline is not None
        
        return HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            timestamp=datetime.utcnow(),
            version="2.0.0",
            pipeline_loaded=is_healthy,
            uptime_seconds=time.time() - monitoring_service.start_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get system metrics for monitoring.
    
    Returns:
        MetricsResponse with performance metrics
    """
    try:
        metrics = metrics_collector.get_current_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_transaction(transaction: Transaction):
    """
    Predict fraud probability for a single transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        PredictionResponse with fraud prediction and confidence
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame format
        import pandas as pd
        df = pd.DataFrame([transaction.dict()])
        
        # Make prediction
        prediction = pipeline.predict(df)[0]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Record metrics
        metrics_collector.record_prediction(processing_time, prediction)
        
        logger.info(f"Transaction processed: {transaction.transaction_id}, "
                   f"fraud={prediction}, latency={processing_time:.2f}ms")
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=bool(prediction),
            fraud_probability=float(prediction),  # Simplified
            confidence=0.95,  # Placeholder
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            model_version="2.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        metrics_collector.record_error()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: TransactionBatch):
    """
    Predict fraud for multiple transactions in batch.
    
    Args:
        batch: Batch of transactions
        
    Returns:
        BatchPredictionResponse with predictions for all transactions
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame([t.dict() for t in batch.transactions])
        
        # Make predictions
        predictions = pipeline.predict(df)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        avg_time_per_transaction = processing_time / len(batch.transactions)
        
        # Create individual responses
        results = []
        for i, (transaction, prediction) in enumerate(zip(batch.transactions, predictions)):
            results.append(PredictionResponse(
                transaction_id=transaction.transaction_id,
                is_fraud=bool(prediction),
                fraud_probability=float(prediction),
                confidence=0.95,
                processing_time_ms=avg_time_per_transaction,
                timestamp=datetime.utcnow(),
                model_version="2.0.0"
            ))
            
            # Record metrics
            metrics_collector.record_prediction(avg_time_per_transaction, prediction)
        
        logger.info(f"Batch processed: {len(batch.transactions)} transactions, "
                   f"total_time={processing_time:.2f}ms")
        
        return BatchPredictionResponse(
            batch_id=batch.batch_id,
            predictions=results,
            total_transactions=len(batch.transactions),
            total_processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        metrics_collector.record_error()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/train", tags=["Management"])
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining (background task).
    
    Returns:
        Status message
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    def train_task():
        try:
            logger.info("Starting model retraining...")
            from main import generate_synthetic_transactions
            train_data = generate_synthetic_transactions(n_samples=5000)
            pipeline.train(train_data)
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    background_tasks.add_task(train_task)
    
    return {
        "status": "training_started",
        "message": "Model retraining initiated in background"
    }


@app.get("/model/info", tags=["Management"])
async def get_model_info():
    """
    Get information about the current model.
    
    Returns:
        Model information
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "model_type": "Spiking Neural Network",
        "version": "2.0.0",
        "framework": "Brian2",
        "architecture": {
            "n_input": 256,
            "n_hidden1": 128,
            "n_hidden2": 64,
            "n_output": 2
        },
        "learning_rule": "STDP",
        "encoding": "Multi-strategy (Rate, Temporal, Population, Latency)"
    }


@app.get("/stats", tags=["Monitoring"])
async def get_statistics():
    """
    Get API usage statistics.
    
    Returns:
        Usage statistics
    """
    stats = metrics_collector.get_statistics()
    return stats


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    run_api(reload=True)
