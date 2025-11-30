from fastapi import FastAPI, Response
import numpy as np
import redis
import json
import time
import logging
import psutil
import os
import joblib
from typing import Dict
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Prometheus метрики - создаем один раз при запуске
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions', ['status'])
PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total prediction errors')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds')
PREDICTION_RISK_SCORE = Gauge('model_prediction_risk_score', 'Latest prediction risk score')
MODEL_SERVER_MEMORY = Gauge('model_server_memory_usage_bytes', 'Memory usage of model server')
MODEL_SERVER_REQUESTS = Counter('model_server_requests_total', 'Total requests to model server', ['method', 'endpoint', 'status'])
ACTIVE_REQUESTS = Gauge('model_server_active_requests', 'Active requests being processed')

# Глобальные переменные для модели
model = None
feature_columns = None
model_metadata = None

def load_model():
    """Загрузка обученной модели"""
    global model, feature_columns, model_metadata
    
    try:
        model = joblib.load('readmission_model.joblib')
        model_metadata = joblib.load('model_metadata.joblib')
        feature_columns = model_metadata['feature_columns']
        logger.info(f"Model loaded successfully. Features: {feature_columns}")
        logger.info(f"Model type: {model_metadata['model_type']}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_features(features: Dict) -> np.ndarray:
    """Предобработка фичей для инференса"""
    try:
        # Создаем вектор фичей в правильном порядке
        processed_features = []
        
        for column in feature_columns:
            if column in features:
                processed_features.append(features[column])
            else:
                # Значение по умолчанию если фича отсутствует
                if column in ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'medication_count', 'length_of_stay']:
                    processed_features.append(0)  # числовые фичи
                elif column in ['gender', 'discharge_destination', 'diabetes', 'hypertension']:
                    processed_features.append(0)  # категориальные фичи
                else:
                    processed_features.append(0.0)  # bmi
        
        return np.array(processed_features).reshape(1, -1)
    
    except Exception as e:
        logger.error(f"Error preprocessing features: {e}")
        raise

# Подключение к Redis
try:
    redis_client = redis.Redis(host='redis', port=6379, decode_responses=True, socket_connect_timeout=5)
    redis_client.ping()
    logger.info("Successfully connected to Redis")
except redis.ConnectionError as e:
    logger.error(f"Redis connection error: {e}")
    redis_client = None

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте сервера"""
    logger.info("Starting model server...")
    load_model()
    logger.info("Model server ready!")

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    """Middleware для сбора метрик запросов"""
    start_time = time.time()
    
    # Исключаем эндпоинт /metrics из метрик
    if request.url.path == "/metrics":
        return await call_next(request)
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # Логируем только успешные запросы к основным эндпоинтам
        MODEL_SERVER_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        # Логируем ошибки
        MODEL_SERVER_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise

@app.post("/predict")
async def predict(features: Dict):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        # Валидация входных данных
        required_fields = ['patient_id', 'age', 'blood_pressure', 'bmi']
        for field in required_fields:
            if field not in features:
                raise ValueError(f"Missing required field: {field}")
        
        # Преобразуем blood_pressure в systolic и diastolic
        bp_parts = features['blood_pressure'].split('/')
        if len(bp_parts) != 2:
            raise ValueError("blood_pressure must be in format 'systolic/diastolic'")
        
        systolic_bp = float(bp_parts[0])
        diastolic_bp = float(bp_parts[1])
        
        # Подготовка фичей для модели
        model_features = {
            'age': float(features['age']),
            'gender': features.get('gender', 'Other'),
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'cholesterol': float(features.get('cholesterol', 200)),  # дефолтное значение
            'bmi': float(features['bmi']),
            'diabetes': features.get('diabetes', 'No'),
            'hypertension': features.get('hypertension', 'No'),
            'medication_count': int(features.get('medication_count', 0)),
            'length_of_stay': int(features.get('length_of_stay', 3)),
            'discharge_destination': features.get('discharge_destination', 'Home')
        }
        
        # Кодирование категориальных переменных
        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        destination_map = {'Home': 0, 'Nursing_Facility': 1, 'Rehab': 2}
        
        model_features['gender'] = gender_map.get(model_features['gender'], 2)
        model_features['discharge_destination'] = destination_map.get(model_features['discharge_destination'], 0)
        model_features['diabetes'] = 1 if model_features['diabetes'] == 'Yes' else 0
        model_features['hypertension'] = 1 if model_features['hypertension'] == 'Yes' else 0
        
        # Препроцессинг и предсказание
        input_features = preprocess_features(model_features)
        prediction_proba = model.predict_proba(input_features)[0]
        risk_score = float(prediction_proba[1])  # Вероятность readmission
        
        # Детекция аномалий на основе комбинации факторов
        anomaly = False
        if (risk_score > 0.7 or 
            systolic_bp > 160 or 
            diastolic_bp > 100 or 
            features['bmi'] > 35):
            anomaly = True
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Сохранение в Redis для мониторинга
        if redis_client:
            feature_record = {
                **features,
                'risk_score': risk_score,
                'anomaly': anomaly,
                'timestamp': time.time()
            }
            redis_client.lpush(f"patient:{features['patient_id']}:predictions", json.dumps(feature_record))
            redis_client.ltrim(f"patient:{features['patient_id']}:predictions", 0, 99)
        
        # Обновление метрик
        PREDICTION_COUNTER.labels(status='success').inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_RISK_SCORE.set(risk_score)
        
        # Обновление использования памяти
        process = psutil.Process(os.getpid())
        MODEL_SERVER_MEMORY.set(process.memory_info().rss)
        
        logger.info(f"Prediction for patient {features['patient_id']}: risk={risk_score:.3f}, time={processing_time:.2f}ms")
        
        return {
            'risk_score': round(risk_score, 4),
            'anomaly': anomaly,
            'processing_time_ms': round(processing_time, 2),
            'model_version': model_metadata['version']
        }
        
    except Exception as e:
        PREDICTION_ERRORS.inc()
        PREDICTION_COUNTER.labels(status='error').inc()
        logger.error(f"Prediction error: {e}")
        return {
            "error": str(e),
            "risk_score": 0.0,
            "anomaly": False,
            "processing_time_ms": (time.time() - start_time) * 1000
        }
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/health")
async def health():
    redis_status = "connected" if redis_client and redis_client.ping() else "disconnected"
    model_status = "loaded" if model is not None else "not loaded"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "model": model_status,
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    """Эндпоинт для Prometheus метрик - возвращает правильный формат"""
    try:
        return Response(
            generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            f"Error generating metrics: {e}",
            status_code=500
        )

@app.get("/model-info")
async def model_info():
    """Информация о загруженной модели"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": model_metadata['model_type'],
        "version": model_metadata['version'],
        "feature_columns": feature_columns,
        "n_features": len(feature_columns)
    }

@app.get("/")
async def root():
    return {"message": "Patient Readmission Model Server"}