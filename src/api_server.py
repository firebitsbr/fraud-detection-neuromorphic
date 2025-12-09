"""
**Descrição:** API REST completa para inferência de fraude usando Spiking Neural Networks. Oferece endpoints para predição individual, lote, treinamento e métricas.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview

Endpoints:
- POST /api/v1/predict - Predição de fraude em transação
- POST /api/v1/batch-predict - Predição em lote
- POST /api/v1/train - Retreinamento do modelo
- GET /api/v1/metrics - Métricas do modelo
- GET /api/v1/health - Health check
- GET /api/v1/stats - Estatísticas da rede neural
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Neuromorphic Fraud Detection API",
    description="API REST para detecção de fraude usando Spiking Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline global (carregado no startup)
pipeline: Optional[FraudDetectionPipeline] = None
model_info = {
    "trained": False,
    "training_time": None,
    "last_training": None,
    "total_predictions": 0,
    "total_frauds_detected": 0
}


# ===== MODELS =====

class Transaction(BaseModel):
    """Modelo de transação"""
    id: str = Field(..., description="ID único da transação")
    amount: float = Field(..., ge=0, description="Valor da transação")
    timestamp: Optional[float] = Field(default=None, description="Timestamp Unix")
    merchant_category: str = Field(..., description="Categoria do comerciante")
    location: tuple = Field(..., description="Coordenadas (lat, lon)")
    device_id: str = Field(..., description="ID do dispositivo")
    daily_frequency: int = Field(..., ge=0, description="Frequência diária de transações")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "txn_001",
                "amount": 150.00,
                "timestamp": 1733500000.0,
                "merchant_category": "groceries",
                "location": (-23.5505, -46.6333),
                "device_id": "device_123",
                "daily_frequency": 3
            }
        }


class PredictionResponse(BaseModel):
    """Resposta de predição"""
    transaction_id: str
    is_fraud: bool
    confidence: float = Field(..., ge=0, le=1)
    fraud_score: float
    legitimate_score: float
    latency_ms: float
    timestamp: str
    recommendation: str


class BatchPredictionRequest(BaseModel):
    """Request para predição em lote"""
    transactions: List[Transaction]


class BatchPredictionResponse(BaseModel):
    """Resposta de predição em lote"""
    results: List[PredictionResponse]
    total_transactions: int
    frauds_detected: int
    total_latency_ms: float
    avg_latency_ms: float


class TrainingRequest(BaseModel):
    """Request para treinamento"""
    n_samples: int = Field(default=1000, ge=100, le=10000)
    fraud_ratio: float = Field(default=0.05, ge=0.01, le=0.5)
    epochs: int = Field(default=30, ge=5, le=100)


class HealthResponse(BaseModel):
    """Resposta de health check"""
    status: str
    model_trained: bool
    uptime_seconds: float
    total_predictions: int
    timestamp: str


class NetworkStats(BaseModel):
    """Estatísticas da rede neural"""
    architecture: Dict[str, Any]
    total_neurons: int
    total_synapses: int
    weight_statistics: Dict[str, float]


# ===== STARTUP/SHUTDOWN =====

@app.on_event("startup")
async def startup_event():
    """Inicialização da API"""
    global pipeline
    logger.info("🚀 Iniciando API de Detecção de Fraude Neuromórfica...")
    
    # Inicializar pipeline
    pipeline = FraudDetectionPipeline()
    
    # Treinar com dataset inicial pequeno
    logger.info("📊 Treinando modelo inicial...")
    df = generate_synthetic_transactions(n=500, fraud_ratio=0.05)
    
    start_time = time.time()
    pipeline.train(df, epochs=20)
    training_time = time.time() - start_time
    
    model_info['trained'] = True
    model_info['training_time'] = training_time
    model_info['last_training'] = datetime.now().isoformat()
    
    logger.info(f"✅ Modelo treinado em {training_time:.2f}s")
    logger.info("🎯 API pronta para receber requisições!")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown da API"""
    logger.info("👋 Encerrando API de Detecção de Fraude Neuromórfica...")


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
    """Health check da API"""
    return HealthResponse(
        status="healthy" if model_info['trained'] else "initializing",
        model_trained=model_info['trained'],
        uptime_seconds=time.time(),
        total_predictions=model_info['total_predictions'],
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/v1/stats", response_model=NetworkStats, tags=["Model"])
async def get_network_stats():
    """Obtém estatísticas da rede neural"""
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
        total_synapses=stats['total_synapses'],
        weight_statistics=stats['weights']
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: Transaction):
    """
    Prediz se uma transação é fraudulenta
    
    Args:
        transaction: Dados da transação
        
    Returns:
        PredictionResponse com resultado da análise
    """
    if not pipeline or not model_info['trained']:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        # Converter para dict
        txn_dict = transaction.dict()
        
        # Fazer predição
        result = pipeline.predict(txn_dict)
        
        # Atualizar estatísticas
        model_info['total_predictions'] += 1
        if result['is_fraud']:
            model_info['total_frauds_detected'] += 1
        
        # Determinar recomendação
        if result['is_fraud']:
            if result['confidence'] > 0.9:
                recommendation = "BLOCK - Alta confiança de fraude"
            elif result['confidence'] > 0.7:
                recommendation = "REVIEW - Análise manual recomendada"
            else:
                recommendation = "MONITOR - Monitorar próximas transações"
        else:
            recommendation = "APPROVE - Transação legítima"
        
        return PredictionResponse(
            transaction_id=transaction.id,
            is_fraud=bool(result['is_fraud']),
            confidence=float(result['confidence']),
            fraud_score=float(result['fraud_score']),
            legitimate_score=float(result['legitimate_score']),
            latency_ms=float(result['latency_ms']),
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/v1/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predição em lote de múltiplas transações
    
    Args:
        request: Lista de transações
        
    Returns:
        BatchPredictionResponse com resultados de todas as predições
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
                    recommendation = "BLOCK - Alta confiança de fraude"
                elif result['confidence'] > 0.7:
                    recommendation = "REVIEW - Análise manual recomendada"
                else:
                    recommendation = "MONITOR - Monitorar próximas transações"
            else:
                recommendation = "APPROVE - Transação legítima"
            
            results.append(PredictionResponse(
                transaction_id=txn.id,
                is_fraud=bool(result['is_fraud']),
                confidence=float(result['confidence']),
                fraud_score=float(result['fraud_score']),
                legitimate_score=float(result['legitimate_score']),
                latency_ms=float(result['latency_ms']),
                timestamp=datetime.now().isoformat(),
                recommendation=recommendation
            ))
        
        total_latency = (time.time() - start_time) * 1000
        
        # Atualizar estatísticas
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
        logger.error(f"Erro na predição em lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/api/v1/train", tags=["Training"])
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Retreina o modelo com novos dados sintéticos
    
    Args:
        request: Parâmetros de treinamento
        background_tasks: Tarefas em background
        
    Returns:
        Status do treinamento
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Gerar dados de treinamento
        logger.info(f"Gerando {request.n_samples} transações para treinamento...")
        df = generate_synthetic_transactions(n=request.n_samples, fraud_ratio=request.fraud_ratio)
        
        # Treinar em background
        def train_task():
            start_time = time.time()
            pipeline.train(df, epochs=request.epochs)
            training_time = time.time() - start_time
            
            model_info['trained'] = True
            model_info['training_time'] = training_time
            model_info['last_training'] = datetime.now().isoformat()
            
            logger.info(f"✅ Modelo retreinado em {training_time:.2f}s")
        
        background_tasks.add_task(train_task)
        
        return {
            "status": "training_started",
            "message": f"Treinamento iniciado com {request.n_samples} amostras",
            "parameters": {
                "n_samples": request.n_samples,
                "fraud_ratio": request.fraud_ratio,
                "epochs": request.epochs
            }
        }
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics():
    """Obtém métricas gerais do sistema"""
    return {
        "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
