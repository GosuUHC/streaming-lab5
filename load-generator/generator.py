import json
import time
import random
import logging
import os
import numpy as np
from kafka import KafkaProducer
import pandas as pd
import requests
from prometheus_client import push_to_gateway, CollectorRegistry, Gauge, Histogram, Counter
from collections import deque

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus метрики
registry = CollectorRegistry()
LATENCY_P50 = Gauge('load_generator_latency_p50_ms', 'P50 latency in milliseconds', registry=registry)
LATENCY_P95 = Gauge('load_generator_latency_p95_ms', 'P95 latency in milliseconds', registry=registry)
LATENCY_P99 = Gauge('load_generator_latency_p99_ms', 'P99 latency in milliseconds', registry=registry)
LATENCY_AVG = Gauge('load_generator_latency_avg_ms', 'Average latency in milliseconds', registry=registry)
REQUEST_COUNTER = Counter('load_generator_requests_total', 'Total number of requests', ['status'], registry=registry)
REQUEST_LATENCY = Histogram('load_generator_request_latency_seconds', 'Request latency in seconds', registry=registry)

class PatientDataGenerator:
    def __init__(self):
        # Загружаем шаблон данных для реалистичных значений
        self._load_data_template()
        self._wait_for_kafka()
        self.producer = KafkaProducer(
            bootstrap_servers=['kafka:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            retries=5,
            request_timeout_ms=30000
        )
        logger.info("Kafka producer initialized")
        
        # Настройка для HTTP запросов к model-server
        self.model_server_url = os.getenv('MODEL_SERVER_URL', 'http://model-server:8000/predict')
        self.prometheus_gateway = os.getenv('PROMETHEUS_GATEWAY', 'pushgateway:9091')
        
        # Буфер для метрик latency (последние 1000 запросов)
        self.latency_buffer = deque(maxlen=1000)
        
        # Счетчики для метрик
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Проверка доступности model-server
        self._wait_for_model_server()
        
    def _load_data_template(self):
        """Загрузка шаблона данных для реалистичных значений"""
        try:
            # Создаем базовые распределения на основе предоставленных данных
            self.age_dist = lambda: random.randint(18, 90)
            self.gender_dist = lambda: random.choice(['Male', 'Female', 'Other'])
            self.bp_dist = lambda: f"{random.randint(100, 160)}/{random.randint(60, 100)}"
            self.cholesterol_dist = lambda: random.randint(150, 300)
            self.bmi_dist = lambda: round(random.uniform(18.0, 40.0), 1)
            self.diabetes_dist = lambda: random.choice(['Yes', 'No'])
            self.hypertension_dist = lambda: random.choice(['Yes', 'No'])
            self.medication_dist = lambda: random.randint(0, 10)
            self.stay_dist = lambda: random.randint(1, 10)
            self.destination_dist = lambda: random.choice(['Home', 'Nursing_Facility', 'Rehab'])
            
        except Exception as e:
            logger.error(f"Error loading data template: {e}")
            raise
        
    def _wait_for_kafka(self):
        """Ожидание готовности Kafka"""
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=['kafka:9092'],
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    request_timeout_ms=5000
                )
                producer.close()
                logger.info("Kafka is ready!")
                return
            except Exception as e:
                retry_count += 1
                logger.info(f"Waiting for Kafka... ({retry_count}/{max_retries})")
                time.sleep(2)
        
        raise Exception("Kafka not ready after waiting")
    
    def _wait_for_model_server(self):
        """Ожидание готовности model-server"""
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(
                    self.model_server_url.replace('/predict', '/health'),
                    timeout=2
                )
                if response.status_code == 200:
                    logger.info("Model server is ready!")
                    return
            except Exception as e:
                retry_count += 1
                logger.info(f"Waiting for model server... ({retry_count}/{max_retries})")
                time.sleep(2)
        
        logger.warning("Model server not ready, but continuing anyway")
        
    def generate_patient_data(self, patient_id):
        """Генерация реалистичных данных пациента"""
        return {
            'patient_id': patient_id,
            'age': self.age_dist(),
            'gender': self.gender_dist(),
            'blood_pressure': self.bp_dist(),
            'cholesterol': self.cholesterol_dist(),
            'bmi': self.bmi_dist(),
            'diabetes': self.diabetes_dist(),
            'hypertension': self.hypertension_dist(),
            'medication_count': self.medication_dist(),
            'length_of_stay': self.stay_dist(),
            'discharge_destination': self.destination_dist(),
            'timestamp': int(time.time() * 1000)
        }
    
    def generate_late_data(self, count=100):
        """Генерация late-arriving данных"""
        base_time = int(time.time() * 1000) - 360000  # 6 минут назад
        logger.info(f"Generating {count} late events")
        
        for i in range(count):
            data = self.generate_patient_data(random.randint(1, 1000))
            data['timestamp'] = base_time + random.randint(0, 300000)  # Разброс 5 минут
            self.producer.send('patient-readmissions', data)
            
        self.producer.flush()
        logger.info("Late data generation completed")
    
    def _make_http_request(self, data):
        """Выполнение HTTP запроса к model-server с измерением latency"""
        request_start = time.time()
        latency_ms = 0
        status = 'error'
        
        try:
            response = requests.post(
                self.model_server_url,
                json=data,
                timeout=1.0
            )
            latency_ms = (time.time() - request_start) * 1000
            
            if response.status_code == 200:
                status = 'success'
                self.successful_requests += 1
                logger.info(f"Request to model-server: patient_id={data['patient_id']}, "
                          f"latency={latency_ms:.2f}ms, risk_score={response.json().get('risk_score', 'N/A')}")
            else:
                status = 'error'
                self.failed_requests += 1
                logger.warning(f"Request failed: status={response.status_code}, latency={latency_ms:.2f}ms")
                
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - request_start) * 1000
            status = 'timeout'
            self.failed_requests += 1
            logger.warning(f"Request timeout: latency={latency_ms:.2f}ms")
        except Exception as e:
            latency_ms = (time.time() - request_start) * 1000
            status = 'error'
            self.failed_requests += 1
            logger.error(f"Request error: {e}, latency={latency_ms:.2f}ms")
        
        self.total_requests += 1
        
        # Добавляем latency в буфер для расчета перцентилей
        if latency_ms > 0:
            self.latency_buffer.append(latency_ms)
            REQUEST_LATENCY.observe(latency_ms / 1000.0)  # в секундах для Histogram
        
        REQUEST_COUNTER.labels(status=status).inc()
        
        return latency_ms
    
    def _push_metrics_to_prometheus(self):
        """Расчет и отправка метрик в Prometheus через pushgateway"""
        if len(self.latency_buffer) == 0:
            return
        
        try:
            # Расчет перцентилей
            latencies = np.array(list(self.latency_buffer))
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            avg_latency = np.mean(latencies)
            
            # Установка значений метрик
            LATENCY_P50.set(p50)
            LATENCY_P95.set(p95)
            LATENCY_P99.set(p99)
            LATENCY_AVG.set(avg_latency)
            
            # Отправка в pushgateway
            push_to_gateway(
                self.prometheus_gateway,
                job='load_generator',
                registry=registry
            )
            
            logger.info(f"Metrics pushed: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms, avg={avg_latency:.2f}ms")
            
        except Exception as e:
            logger.warning(f"Failed to push metrics to Prometheus: {e}")
    
    def start_normal_load(self, events_per_second=500):
        """Генерация нормальной нагрузки"""
        logger.info(f"Starting load generation: {events_per_second} events/sec")
        logger.info(f"Model server URL: {self.model_server_url}")
        
        late_data_counter = 0
        patient_counter = 1
        metrics_push_counter = 0
        
        while True:
            batch_start = time.time()
            
            for i in range(events_per_second):
                data = self.generate_patient_data(patient_counter)
                
                # Отправка в Kafka
                self.producer.send('patient-readmissions', data)
                
                # HTTP запрос к model-server
                self._make_http_request(data)
                
                patient_counter += 1
            
            # Периодическая генерация late данных
            late_data_counter += 1
            if late_data_counter >= 10:  # Каждые 10 секунд
                self.generate_late_data(20)
                late_data_counter = 0
            
            self.producer.flush()
            
            # Периодическая отправка метрик в Prometheus (каждые 5 секунд)
            metrics_push_counter += 1
            if metrics_push_counter >= 5:
                self._push_metrics_to_prometheus()
                metrics_push_counter = 0
            
            # Поддержание стабильной скорости
            batch_time = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_time)
            time.sleep(sleep_time)

if __name__ == '__main__':
    try:
        generator = PatientDataGenerator()
        # Даем время другим сервисам запуститься
        time.sleep(10)
        generator.start_normal_load(events_per_second=1500)
    except Exception as e:
        logger.error(f"Generator failed: {e}")
        raise