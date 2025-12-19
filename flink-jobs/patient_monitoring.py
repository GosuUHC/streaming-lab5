import json
import time
import logging
import os
import redis
import numpy as np
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
MODEL_PATH = "/opt/flink/models/readmission_model.onnx"
FEATURE_CONFIG_PATH = "/opt/flink/models/model_metadata.json"


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
    Использует ONNX модель напрямую для низкой задержки.
    """
    total_start = time.time()
    
    # Lazy init ONNX session
    if not hasattr(predict_patient_risk_json, '_onnx_session'):
        try:
            import onnxruntime as ort
            if os.path.exists(MODEL_PATH):
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.intra_op_num_threads = 1
                predict_patient_risk_json._onnx_session = ort.InferenceSession(
                    MODEL_PATH,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                # Загружаем feature columns из конфига
                if os.path.exists(FEATURE_CONFIG_PATH):
                    with open(FEATURE_CONFIG_PATH, 'r') as f:
                        config = json.load(f)
                        predict_patient_risk_json._feature_columns = config['feature_columns']
                else:
                    predict_patient_risk_json._feature_columns = [
                        'age', 'gender', 'systolic_bp', 'diastolic_bp', 'cholesterol',
                        'bmi', 'diabetes', 'hypertension', 'medication_count',
                        'length_of_stay', 'discharge_destination'
                    ]
            else:
                predict_patient_risk_json._onnx_session = None
                predict_patient_risk_json._feature_columns = None
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}")
            predict_patient_risk_json._onnx_session = None
            predict_patient_risk_json._feature_columns = None
    
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
        
        # Парсим blood_pressure
        bp_parts = blood_pressure.split('/')
        if len(bp_parts) != 2:
            systolic_bp = 120.0
            diastolic_bp = 80.0
        else:
            systolic_bp = float(bp_parts[0])
            diastolic_bp = float(bp_parts[1])
        
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
        
        # 2. Prepare features for ML model (в порядке feature_columns)
        feature_columns = predict_patient_risk_json._feature_columns
        if feature_columns:
            # Создаем фичи в правильном порядке
            feature_dict = {
                'age': float(age),
                'gender': 0,  # По умолчанию, можно добавить в данные
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'cholesterol': 200.0,  # По умолчанию
                'bmi': float(bmi),
                'diabetes': 0,  # По умолчанию
                'hypertension': 1 if systolic_bp > 140 or diastolic_bp > 90 else 0,
                'medication_count': 0,  # По умолчанию
                'length_of_stay': 3,  # По умолчанию
                'discharge_destination': 0  # По умолчанию (Home)
            }
            
            feature_vector = np.array([[
                feature_dict.get(col, 0.0) for col in feature_columns
            ]], dtype=np.float32)
        else:
            # Fallback если конфиг не загружен
            feature_vector = np.array([[
                float(age), 0.0, systolic_bp, diastolic_bp, 200.0,
                float(bmi), 0.0, 1 if systolic_bp > 140 else 0, 0.0, 3.0, 0.0
            ]], dtype=np.float32)
        
        # 3. ML Inference через ONNX
        inference_start = time.time()
        
        session = predict_patient_risk_json._onnx_session
        if session:
            try:
                input_name = session.get_inputs()[0].name
                result = session.run(None, {input_name: feature_vector})
                
                # ONNX модель возвращает вероятности для двух классов
                if len(result[0].shape) == 2 and result[0].shape[1] == 2:
                    risk_score = float(result[0][0][1])  # Вероятность readmission
                elif len(result[0].shape) == 2 and result[0].shape[1] == 1:
                    logit = float(result[0][0][0])
                    risk_score = float(1.0 / (1.0 + np.exp(-logit)))
                else:
                    risk_score = float(result[0][0][0] if result[0][0][0] <= 1.0 else 1.0 / (1.0 + np.exp(-result[0][0][0])))
            except Exception as e:
                logger.debug(f"ONNX inference error: {e}")
                risk_score = 0.0
        else:
            # Fallback если модель не загружена
            risk_score = 0.0
        
        # Детекция аномалий
        anomaly = False
        if (risk_score > 0.7 or 
            systolic_bp > 160 or 
            diastolic_bp > 100 or 
            bmi > 35):
            anomaly = True
        
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
    
    # Проверка модели
    import os
    model_path = "/opt/flink/models/readmission_model.onnx"
    if os.path.exists(model_path):
        logger.info(f"ONNX model found: {model_path}")
    else:
        logger.warning(f"ONNX model NOT found: {model_path}")
    
    logger.info("Creating EnvironmentSettings...")
    try:
        settings = EnvironmentSettings.in_streaming_mode()
        logger.info("EnvironmentSettings created")
        
        logger.info("Creating TableEnvironment...")
        t_env = TableEnvironment.create(settings)
        logger.info("TableEnvironment created successfully")
    except Exception as e:
        logger.error(f"Failed to create TableEnvironment: {e}", exc_info=True)
        raise
    
    logger.info("Configuring Flink settings...")
    try:
        config = t_env.get_config()
        config.set("parallelism.default", "8")
        config.set("pipeline.name", "Patient Readmission Monitoring Pipeline")
        
        # Настройка checkpointing
        config.set("execution.checkpointing.interval", "500ms")
        config.set("execution.checkpointing.mode", "EXACTLY_ONCE")
        config.set("execution.checkpointing.timeout", "10s")
        logger.info("Flink settings configured")
    except Exception as e:
        logger.error(f"Failed to configure Flink settings: {e}", exc_info=True)
        raise
    
    logger.info("Creating UDF...")
    try:
        t_env.create_temporary_function("predict_patient_risk_json", predict_patient_risk_json)
        logger.info("UDF created successfully")
    except Exception as e:
        logger.error(f"Failed to create UDF: {e}", exc_info=True)
        raise
    
    # Создание таблицы источника из Kafka (ts вместо timestamp)
    logger.info("Creating source table...")
    try:
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
        logger.info("Source table created successfully")
    except Exception as e:
        logger.error(f"Failed to create source table: {e}", exc_info=True)
        raise
    
    # Создание таблицы для предсказаний
    logger.info("Creating sink table...")
    try:
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
        logger.info("Sink table created successfully")
    except Exception as e:
        logger.error(f"Failed to create sink table: {e}", exc_info=True)
        raise
    
    logger.info("Submitting ML inference job...")
    
    # Основной поток предсказаний
    try:
        result = t_env.execute_sql("""
    INSERT INTO patient_predictions
    SELECT predict_patient_risk_json(
        patient_id, age, glucose_level, blood_pressure, bmi, ts
    )
    FROM patient_readmissions
    """)
        logger.info("SQL query submitted successfully")
    except Exception as e:
        logger.error(f"Failed to submit SQL query: {e}", exc_info=True)
        raise
    
    logger.info("JOB RUNNING WITH ML INFERENCE!")
    try:
        result.wait()
    except Exception as e:
        logger.error(f"Job execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
