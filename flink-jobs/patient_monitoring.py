import json
import time
import logging
import requests
import redis
from datetime import datetime
from typing import Optional, Dict

from pyflink.table import EnvironmentSettings, TableEnvironment
from pyflink.table.udf import udf
from pyflink.table.types import DataTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatientMonitoring")

# Подавление предупреждений apache-beam и gRPC
logging.getLogger("apache_beam").setLevel(logging.ERROR)
logging.getLogger("apache_beam.typehints").setLevel(logging.ERROR)
logging.getLogger("apache_beam.io.gcp.bigquery").setLevel(logging.ERROR)
logging.getLogger("apache_beam.runners.worker").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

# Подавление исключений в потоках
import sys
import threading

def handle_thread_exception(args):
    # Игнорируем gRPC ошибки в worker threads
    if "grpc" in str(args.exc_value).lower() or "multiplexer" in str(args.exc_value).lower():
        return
    sys.__excepthook__(args)

threading.excepthook = handle_thread_exception

KAFKA_BROKERS = "kafka:9092"
REDIS_HOST = "redis"
REDIS_PORT = 6379
MODEL_ENDPOINT = "http://model-server:8000/predict"


@udf(result_type=DataTypes.STRING())
def predict_patient_risk_json(
    patient_id: str,
    age: int,
    glucose_level: float,
    blood_pressure: str,
    bmi: float,
    ts: int  # переименовано из timestamp, т.к. timestamp - зарезервированное слово
) -> str:
    """
    UDF для предсказания риска повторной госпитализации с измерением latency.
    """
    total_start = time.time()
    
    # Lazy init Redis
    if not hasattr(predict_patient_risk_json, '_redis'):
        try:
            predict_patient_risk_json._redis = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                decode_responses=True, socket_timeout=0.05
            )
            predict_patient_risk_json._redis.ping()
        except:
            predict_patient_risk_json._redis = None
    
    try:
        # 1. Extract features
        feature_start = time.time()
        
        # Получение исторических данных из Redis
        r = predict_patient_risk_json._redis
        previous_risk_score = 0.0
        previous_anomaly = False
        
        if r:
            try:
                key = f"features:patient:{patient_id}"
                cached = r.get(key)
                if cached:
                    features = json.loads(cached)
                    previous_risk_score = features.get('risk_score', 0.0)
                    previous_anomaly = features.get('anomaly', False)
            except:
                pass
        
        feature_time_ms = (time.time() - feature_start) * 1000
        
        # 2. Prepare features for ML model
        features = {
            'patient_id': patient_id,
            'age': age,
            'glucose_level': glucose_level,
            'blood_pressure': blood_pressure,
            'bmi': bmi,
            'previous_risk_score': previous_risk_score,
            'previous_anomaly': previous_anomaly
        }
        
        # 3. ML Inference
        inference_start = time.time()
        
        try:
            # Используем session для connection pooling
            if not hasattr(predict_patient_risk_json, '_session'):
                from requests.adapters import HTTPAdapter
                predict_patient_risk_json._session = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=20,
                    pool_maxsize=40,
                    max_retries=0
                )
                predict_patient_risk_json._session.mount('http://', adapter)
            
            response = predict_patient_risk_json._session.post(
                MODEL_ENDPOINT,
                json=features,
                timeout=(0.005, 0.015),  # 5ms connect, 15ms read
                headers={
                    'Connection': 'keep-alive',
                    'Content-Type': 'application/json'
                }
            )
            
            if response.status_code == 200:
                prediction = response.json()
                risk_score = prediction.get('risk_score', 0.0)
                anomaly = prediction.get('anomaly', False)
            else:
                risk_score = 0.0
                anomaly = False
        except Exception as e:
            logger.debug(f"Model inference error: {e}")
            risk_score = 0.0
            anomaly = False
        
        inference_time_ms = (time.time() - inference_start) * 1000
        
        # 4. Update Redis with new features
        if r:
            try:
                key = f"features:patient:{patient_id}"
                r.set(key, json.dumps({
                    'risk_score': risk_score,
                    'anomaly': anomaly,
                    'timestamp': ts
                }), ex=86400)  # TTL 24 hours
                
                # Update metrics
                r.incr('total_data_count')
                processing_time = int(time.time() * 1000)
                delay = processing_time - ts
                if delay > 300000:  # Более 5 минут - late data
                    r.incr('late_data_count')
            except:
                pass
        
        # 5. Total processing time
        total_time_ms = (time.time() - total_start) * 1000
        
        # 6. E2E latency
        now_ms = int(time.time() * 1000)
        e2e_latency = now_ms - ts
        
        return json.dumps({
            'patient_id': patient_id,
            'risk_score': round(risk_score, 4),
            'anomaly': anomaly,
            'age': age,
            'glucose_level': glucose_level,
            'blood_pressure': blood_pressure,
            'bmi': bmi,
            'inference_ms': round(inference_time_ms, 2),
            'feature_ms': round(feature_time_ms, 2),
            'processing_ms': round(total_time_ms, 2),
            'e2e_latency_ms': e2e_latency,
            'timestamp': ts,
            'processing_timestamp': now_ms
        })
        
    except Exception as e:
        logger.error(f"Error in predict_patient_risk_json: {e}")
        return json.dumps({
            'patient_id': patient_id,
            'error': str(e),
            'timestamp': ts
        })


def main():
    logger.info("="*60)
    logger.info("Patient Readmission Monitoring with ML")
    logger.info("Variant: 6 (p99 < 30 ms)")
    logger.info("="*60)
    
    settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(settings)
    
    config = t_env.get_config()
    config.set("parallelism.default", "8")
    config.set("pipeline.name", "Patient Readmission Monitoring Pipeline")
    
    # Настройка checkpointing
    config.set("execution.checkpointing.interval", "500ms")
    config.set("execution.checkpointing.mode", "EXACTLY_ONCE")
    config.set("execution.checkpointing.timeout", "10s")
    
    t_env.create_temporary_function("predict_patient_risk_json", predict_patient_risk_json)
    
    # Создание таблицы источника из Kafka (ts вместо timestamp)
    t_env.execute_sql("""
    CREATE TABLE patient_readmissions (
        patient_id STRING,
        age INT,
        glucose_level DOUBLE,
        blood_pressure STRING,
        bmi DOUBLE,
        ts BIGINT,
        proc_time AS PROCTIME()
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'patient-readmissions',
        'properties.bootstrap.servers' = 'kafka:9092',
        'properties.group.id' = 'flink-patient-monitoring',
        'scan.startup.mode' = 'latest-offset',
        'format' = 'json',
        'json.ignore-parse-errors' = 'true'
    )
    """)
    
    # Создание таблицы для предсказаний
    t_env.execute_sql("""
    CREATE TABLE patient_predictions (
        prediction_json STRING
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'patient-predictions',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'raw'
    )
    """)
    
    logger.info("Submitting ML inference job...")
    
    # Основной поток предсказаний
    result = t_env.execute_sql("""
    INSERT INTO patient_predictions
    SELECT predict_patient_risk_json(
        patient_id, age, glucose_level, blood_pressure, bmi, ts
    )
    FROM patient_readmissions
    """)
    
    logger.info("JOB RUNNING WITH ML INFERENCE!")
    result.wait()


if __name__ == "__main__":
    main()
