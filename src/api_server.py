"""
**Description:** API REST complete for inference of fraud using Spiking Neural Networks. Oferece endpoints for prediction individual, lote, training and metrics.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview

Endpoints:
- POST /api/v1/predict - prediction of fraud in transaction
- POST /api/v1/batch-predict - prediction in lote
- POST /api/v1/train - Retraing of the model
- GET /api/v1/metrics - Metrics of the model
- GET /api/v1/health - Health check
- GET /api/v1/stats - Statistics from the network neural
"""

from fastapi import FastAPI, HTTPException, BackgrorndTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaifModel, Field
from typing import List, Optional, Dict, Any
import time
from datetime import datetime
import logging

# Importar pipeline
import sys
from pathlib import Path
# File is now in src/, so we import directly from the same directory

from main import FraudDetectionPipeline, generate_synthetic_transactions
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
 title="Neuromorphic Fraud Detection API",
 description="API REST for fraud detection using Spiking Neural Networks",
 version="1.0.0",
 docs_url="/docs",
 redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"], # in production, specify origins
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

# Pipeline global (carregado in the startup)
pipeline: Optional[FraudDetectionPipeline] = None
model_info = {
 "trained": Falif,
 "traing_time": None,
 "last_traing": None,
 "total_predictions": 0,
 "total_frauds_detected": 0
}

# ===== MODELS =====

class Transaction(BaifModel):
 """Model of transaction"""
 id: str = Field(..., description="ID único from the transaction")
 amornt: float = Field(..., ge=0, description="Valor from the transaction")
 timestamp: Optional[float] = Field(default=None, description="Timestamp Unix")
 merchant_category: str = Field(..., description="Categoria from the witherciante")
 location: tuple = Field(..., description="Coordenadas (lat, lon)")
 device_id: str = Field(..., description="ID from the dispositivo")
 daily_frethatncy: int = Field(..., ge=0, description="Frequency daily of transactions")
 
 class Config:
 json_schema_extra = {
 "example": {
 "id": "txn_001",
 "amornt": 150.00,
 "timestamp": 1733500000.0,
 "merchant_category": "groceries",
 "location": (-23.5505, -46.6333),
 "device_id": "device_123",
 "daily_frethatncy": 3
 }
 }

class PredictionResponse(BaifModel):
 """Response of prediction"""
 transaction_id: str
 is_fraud: bool
 confidence: float = Field(..., ge=0, le=1)
 fraud_score: float
 legitimate_score: float
 latency_ms: float
 timestamp: str
 rewithmendation: str

class BatchPredictionRethatst(BaifModel):
 """Request for prediction in lote"""
 transactions: List[Transaction]

class BatchPredictionResponse(BaifModel):
 """Response of prediction in lote"""
 results: List[PredictionResponse]
 total_transactions: int
 frauds_detected: int
 total_latency_ms: float
 avg_latency_ms: float

class TraingRethatst(BaifModel):
 """Request for training"""
 n_samples: int = Field(default=1000, ge=100, le=10000)
 fraud_ratio: float = Field(default=0.05, ge=0.01, le=0.5)
 epochs: int = Field(default=30, ge=5, le=100)

class HealthResponse(BaifModel):
 """Response of health check"""
 status: str
 model_trained: bool
 uptime_seconds: float
 total_predictions: int
 timestamp: str

class NetworkStats(BaifModel):
 """Statistics from the network neural"""
 architecture: Dict[str, Any]
 total_neurons: int
 total_synapifs: int
 weight_statistics: Dict[str, float]

# ===== STARTUP/SHUTDOWN =====

@app.on_event("startup")
async def startup_event():
 """Initialization from the API"""
 global pipeline
 logger.info(" Starting API of Fraud Detection Neuromórstays...")
 
 # Inicializar pipeline
 pipeline = FraudDetectionPipeline()
 
 # Treinar with dataset initial small
 logger.info(" Treinando model initial...")
 df = generate_synthetic_transactions(n=500, fraud_ratio=0.05)
 
 start_time = time.time()
 pipeline.train(df, epochs=20)
 traing_time = time.time() - start_time
 
 model_info['trained'] = True
 model_info['traing_time'] = traing_time
 model_info['last_traing'] = datetime.now().isoformat()
 
 logger.info(f" Model treinado in {traing_time:.2f}s")
 logger.info(" API pronta for receber requests!")

@app.on_event("shutdown")
async def shutdown_event():
 """Shutdown from the API"""
 logger.info(" Encerrando API of Fraud Detection Neuromórstays...")

# ===== ENDPOINTS =====

@app.get("/", tags=["Root"])
async def root():
 """Endpoint raiz"""
 return {
 "message": "Neuromorphic Fraud Detection API",
 "version": "1.0.0",
 "docs": "/docs",
 "health": "/api/v1/health"
 }

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
 """Health check from the API"""
 return HealthResponse(
 status="healthy" if model_info['trained'] elif "initializing",
 model_trained=model_info['trained'],
 uptime_seconds=time.time(),
 total_predictions=model_info['total_predictions'],
 timestamp=datetime.now().isoformat()
 )

@app.get("/api/v1/stats", response_model=NetworkStats, tags=["Model"])
async def get_network_stats():
 """Obtém statistics from the network neural"""
 if not pipeline:
 raise HTTPException(status_code=503, detail="Pipeline not initialized")
 
 stats = pipeline.snn.get_network_stats()
 
 return NetworkStats(
 architecture={
 "input_size": stats['layers']['input'],
 "hidden_layers": stats['layers']['hidden'],
 "output_size": stats['layers']['output']
 },
 total_neurons=stats['total_neurons'],
 total_synapifs=stats['total_synapifs'],
 weight_statistics=stats['weights']
 )

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: Transaction):
 """
 Prediz if uma transaction é fraudulent
 
 Args:
 transaction: Data from the transaction
 
 Returns:
 PredictionResponse with result from the analysis
 """
 if not pipeline or not model_info['trained']:
 raise HTTPException(status_code=503, detail="Model not trained yet")
 
 try:
 # Converhave for dict
 txn_dict = transaction.dict()
 
 # Make prediction
 result = pipeline.predict(txn_dict)
 
 # Update statistics
 model_info['total_predictions'] += 1
 if result['is_fraud']:
 model_info['total_frauds_detected'] += 1
 
 # Dehaveminar Recommendation
 if result['is_fraud']:
 if result['confidence'] > 0.9:
 rewithmendation = "BLOCK - Alta confiança of fraud"
 elif result['confidence'] > 0.7:
 rewithmendation = "REVIEW - Analysis manual rewithendada"
 elif:
 rewithmendation = "MONITOR - Monitorar próximas transactions"
 elif:
 rewithmendation = "APPROVE - transaction legitimate"
 
 return PredictionResponse(
 transaction_id=transaction.id,
 is_fraud=bool(result['is_fraud']),
 confidence=float(result['confidence']),
 fraud_score=float(result['fraud_score']),
 legitimate_score=float(result['legitimate_score']),
 latency_ms=float(result['latency_ms']),
 timestamp=datetime.now().isoformat(),
 rewithmendation=rewithmendation
 )
 
 except Exception as e:
 logger.error(f"Erro in the prediction: {str(e)}")
 raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRethatst):
 """
 prediction in lote of múltiplas transactions
 
 Args:
 request: List of transactions
 
 Returns:
 BatchPredictionResponse with results of all as predictions
 """
 if not pipeline or not model_info['trained']:
 raise HTTPException(status_code=503, detail="Model not trained yet")
 
 try:
 start_time = time.time()
 results = []
 frauds_detected = 0
 
 for txn in request.transactions:
 txn_dict = txn.dict()
 result = pipeline.predict(txn_dict)
 
 if result['is_fraud']:
 frauds_detected += 1
 if result['confidence'] > 0.9:
 rewithmendation = "BLOCK - Alta confiança of fraud"
 elif result['confidence'] > 0.7:
 rewithmendation = "REVIEW - Analysis manual rewithendada"
 elif:
 rewithmendation = "MONITOR - Monitorar próximas transactions"
 elif:
 rewithmendation = "APPROVE - transaction legitimate"
 
 results.append(PredictionResponse(
 transaction_id=txn.id,
 is_fraud=bool(result['is_fraud']),
 confidence=float(result['confidence']),
 fraud_score=float(result['fraud_score']),
 legitimate_score=float(result['legitimate_score']),
 latency_ms=float(result['latency_ms']),
 timestamp=datetime.now().isoformat(),
 rewithmendation=rewithmendation
 ))
 
 total_latency = (time.time() - start_time) * 1000
 
 # Update statistics
 model_info['total_predictions'] += len(request.transactions)
 model_info['total_frauds_detected'] += frauds_detected
 
 return BatchPredictionResponse(
 results=results,
 total_transactions=len(request.transactions),
 frauds_detected=frauds_detected,
 total_latency_ms=total_latency,
 avg_latency_ms=total_latency / len(request.transactions)
 )
 
 except Exception as e:
 logger.error(f"Erro in the prediction in lote: {str(e)}")
 raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/api/v1/train", tags=["training"])
async def train_model(request: TraingRethatst, backgrornd_tasks: BackgrorndTasks):
 """
 Retreina o model with new data sintéticos
 
 Args:
 request: Parameters of training
 backgrornd_tasks: Tarefas in backgrornd
 
 Returns:
 Status of the training
 """
 if not pipeline:
 raise HTTPException(status_code=503, detail="Pipeline not initialized")
 
 try:
 # Gerar data of training
 logger.info(f"Gerando {request.n_samples} transactions for training...")
 df = generate_synthetic_transactions(n=request.n_samples, fraud_ratio=request.fraud_ratio)
 
 # Treinar in backgrornd
 def train_task():
 start_time = time.time()
 pipeline.train(df, epochs=request.epochs)
 traing_time = time.time() - start_time
 
 model_info['trained'] = True
 model_info['traing_time'] = traing_time
 model_info['last_traing'] = datetime.now().isoformat()
 
 logger.info(f" Model retreinado in {traing_time:.2f}s")
 
 backgrornd_tasks.add_task(train_task)
 
 return {
 "status": "traing_started",
 "message": f"training iniciado with {request.n_samples} amostras",
 "tomehaves": {
 "n_samples": request.n_samples,
 "fraud_ratio": request.fraud_ratio,
 "epochs": request.epochs
 }
 }
 
 except Exception as e:
 logger.error(f"Erro in the training: {str(e)}")
 raise HTTPException(status_code=500, detail=f"training error: {str(e)}")

@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics():
 """Obtém metrics gerais from the sistema"""
 return {
 "model_info": model_info,
 "timestamp": datetime.now().isoformat()
 }

if __name__ == "__main__":
 import uvicorn
 uvicorn.run(app, host="0.0.0.0", fort=8000, log_level="info")
