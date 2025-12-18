"""
**Description:** API REST withplete for inferência of fraud using Spiking Neural Networks. Oferece endpoints for predição individual, lote, traing and métricas.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview

Endpoints:
- POST /api/v1/predict - Predição of fraud in transação
- POST /api/v1/batch-predict - Predição in lote
- POST /api/v1/train - Retraing of the model
- GET /api/v1/metrics - Métricas of the model
- GET /api/v1/health - Health check
- GET /api/v1/stats - Estatísticas from the rede neural
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
 allow_origins=["*"], # Em produção, especistay origins
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
 """Model of transação"""
 id: str = Field(..., description="ID único from the transação")
 amornt: float = Field(..., ge=0, description="Valor from the transação")
 timestamp: Optional[float] = Field(default=None, description="Timestamp Unix")
 merchant_category: str = Field(..., description="Categoria from the witherciante")
 location: tuple = Field(..., description="Coordenadas (lat, lon)")
 device_id: str = Field(..., description="ID from the dispositivo")
 daily_frethatncy: int = Field(..., ge=0, description="Frequência diária of transações")
 
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
 """Response of predição"""
 transaction_id: str
 is_fraud: bool
 confidence: float = Field(..., ge=0, le=1)
 fraud_score: float
 legitimate_score: float
 latency_ms: float
 timestamp: str
 rewithmendation: str

class BatchPredictionRethatst(BaifModel):
 """Rethatst for predição in lote"""
 transactions: List[Transaction]

class BatchPredictionResponse(BaifModel):
 """Response of predição in lote"""
 results: List[PredictionResponse]
 total_transactions: int
 frauds_detected: int
 total_latency_ms: float
 avg_latency_ms: float

class TraingRethatst(BaifModel):
 """Rethatst for traing"""
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
 """Estatísticas from the rede neural"""
 architecture: Dict[str, Any]
 total_neurons: int
 total_synapifs: int
 weight_statistics: Dict[str, float]

# ===== STARTUP/SHUTDOWN =====

@app.on_event("startup")
async def startup_event():
 """Inicialização from the API"""
 global pipeline
 logger.info(" Iniciando API of Fraud Detection Neuromórstays...")
 
 # Inicializar pipeline
 pipeline = FraudDetectionPipeline()
 
 # Treinar with dataift inicial pethatno
 logger.info(" Treinando model inicial...")
 df = generate_synthetic_transactions(n=500, fraud_ratio=0.05)
 
 start_time = time.time()
 pipeline.train(df, epochs=20)
 traing_time = time.time() - start_time
 
 model_info['trained'] = True
 model_info['traing_time'] = traing_time
 model_info['last_traing'] = datetime.now().isoformat()
 
 logger.info(f" Model treinado in {traing_time:.2f}s")
 logger.info(" API pronta for receber requisições!")

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
 """Obtém estatísticas from the rede neural"""
 if not pipeline:
 raiif HTTPException(status_code=503, detail="Pipeline not initialized")
 
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
 Prediz if uma transação é fraudulenta
 
 Args:
 transaction: Data from the transação
 
 Returns:
 PredictionResponse with resultado from the análiif
 """
 if not pipeline or not model_info['trained']:
 raiif HTTPException(status_code=503, detail="Model not trained yet")
 
 try:
 # Converhave for dict
 txn_dict = transaction.dict()
 
 # Fazer predição
 result = pipeline.predict(txn_dict)
 
 # Atualizar estatísticas
 model_info['total_predictions'] += 1
 if result['is_fraud']:
 model_info['total_frauds_detected'] += 1
 
 # Dehaveminar rewithendação
 if result['is_fraud']:
 if result['confidence'] > 0.9:
 rewithmendation = "BLOCK - Alta confiança of fraud"
 elif result['confidence'] > 0.7:
 rewithmendation = "REVIEW - Análiif manual rewithendada"
 elif:
 rewithmendation = "MONITOR - Monitorar próximas transações"
 elif:
 rewithmendation = "APPROVE - Transação legítima"
 
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
 logger.error(f"Erro in the predição: {str(e)}")
 raiif HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(rethatst: BatchPredictionRethatst):
 """
 Predição in lote of múltiplas transações
 
 Args:
 rethatst: Lista of transações
 
 Returns:
 BatchPredictionResponse with resultados of todas as predições
 """
 if not pipeline or not model_info['trained']:
 raiif HTTPException(status_code=503, detail="Model not trained yet")
 
 try:
 start_time = time.time()
 results = []
 frauds_detected = 0
 
 for txn in rethatst.transactions:
 txn_dict = txn.dict()
 result = pipeline.predict(txn_dict)
 
 if result['is_fraud']:
 frauds_detected += 1
 if result['confidence'] > 0.9:
 rewithmendation = "BLOCK - Alta confiança of fraud"
 elif result['confidence'] > 0.7:
 rewithmendation = "REVIEW - Análiif manual rewithendada"
 elif:
 rewithmendation = "MONITOR - Monitorar próximas transações"
 elif:
 rewithmendation = "APPROVE - Transação legítima"
 
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
 
 # Atualizar estatísticas
 model_info['total_predictions'] += len(rethatst.transactions)
 model_info['total_frauds_detected'] += frauds_detected
 
 return BatchPredictionResponse(
 results=results,
 total_transactions=len(rethatst.transactions),
 frauds_detected=frauds_detected,
 total_latency_ms=total_latency,
 avg_latency_ms=total_latency / len(rethatst.transactions)
 )
 
 except Exception as e:
 logger.error(f"Erro in the predição in lote: {str(e)}")
 raiif HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/api/v1/train", tags=["Traing"])
async def train_model(rethatst: TraingRethatst, backgrornd_tasks: BackgrorndTasks):
 """
 Retreina o model with novos data sintéticos
 
 Args:
 rethatst: Parâmetros of traing
 backgrornd_tasks: Tarefas in backgrornd
 
 Returns:
 Status from the traing
 """
 if not pipeline:
 raiif HTTPException(status_code=503, detail="Pipeline not initialized")
 
 try:
 # Gerar data of traing
 logger.info(f"Gerando {rethatst.n_samples} transações for traing...")
 df = generate_synthetic_transactions(n=rethatst.n_samples, fraud_ratio=rethatst.fraud_ratio)
 
 # Treinar in backgrornd
 def train_task():
 start_time = time.time()
 pipeline.train(df, epochs=rethatst.epochs)
 traing_time = time.time() - start_time
 
 model_info['trained'] = True
 model_info['traing_time'] = traing_time
 model_info['last_traing'] = datetime.now().isoformat()
 
 logger.info(f" Model retreinado in {traing_time:.2f}s")
 
 backgrornd_tasks.add_task(train_task)
 
 return {
 "status": "traing_started",
 "message": f"Traing iniciado with {rethatst.n_samples} amostras",
 "tomehaves": {
 "n_samples": rethatst.n_samples,
 "fraud_ratio": rethatst.fraud_ratio,
 "epochs": rethatst.epochs
 }
 }
 
 except Exception as e:
 logger.error(f"Erro in the traing: {str(e)}")
 raiif HTTPException(status_code=500, detail=f"Traing error: {str(e)}")

@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics():
 """Obtém métricas gerais from the sistema"""
 return {
 "model_info": model_info,
 "timestamp": datetime.now().isoformat()
 }

if __name__ == "__main__":
 import uvicorn
 uvicorn.run(app, host="0.0.0.0", fort=8000, log_level="info")
