"""
Сервис для непрерывного мониторинга data drift
Интегрируется с Kafka для чтения данных и отправки метрик в Prometheus
"""
import os
import time
import logging
import pandas as pd
import json
from kafka import KafkaConsumer
from prometheus_client import push_to_gateway, CollectorRegistry, Gauge
import redis
from data_drift_monitor import DataDriftMonitor, load_reference_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus метрики
registry = CollectorRegistry()
DRIFT_SCORE = Gauge('data_drift_score', 'Data drift score (0-1)', registry=registry)
DRIFT_DETECTED = Gauge('data_drift_detected', 'Data drift detected (1=yes, 0=no)', registry=registry)
LATE_DATA_RATIO = Gauge('late_data_ratio', 'Ratio of late-arriving data', registry=registry)
KAFKA_LAG = Gauge('kafka_consumer_lag', 'Kafka consumer lag', ['topic', 'partition'], registry=registry)
PROCESSING_LATENCY = Gauge('processing_latency_ms', 'Processing latency in milliseconds', registry=registry)


class MonitoringService:
    """Сервис для непрерывного мониторинга"""
    
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.prometheus_gateway = os.getenv('PROMETHEUS_GATEWAY', 'pushgateway:9091')
        self.reference_data_path = os.getenv('REFERENCE_DATA_PATH', '/data/hospital_readmissions_30k.csv')
        self.kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'patient-readmissions')
        
        # Подключение к Redis
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Инициализация монитора data drift
        self.drift_monitor = DataDriftMonitor(
            redis_client=self.redis_client,
            prometheus_gateway=self.prometheus_gateway
        )
        
        # Загрузка референсных данных
        try:
            if os.path.exists(self.reference_data_path):
                reference_data = load_reference_data(self.reference_data_path)
                self.drift_monitor.set_reference_data(reference_data)
                logger.info(f"Reference data loaded: {reference_data.shape}")
            else:
                logger.warning(f"Reference data file not found: {self.reference_data_path}. "
                             f"Data drift monitoring will use simple detection.")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}. "
                        f"Data drift monitoring will use simple detection.")
        
        # Буфер для накопления данных перед анализом
        self.data_buffer = []
        self.buffer_size = 1000  # Размер буфера для анализа drift
        self.last_drift_check = time.time()
        self.drift_check_interval = 60  # Проверка drift каждую минуту
        
        # Статистика для late data
        self.late_data_count = 0
        self.total_data_count = 0
    
    def process_message(self, message):
        """Обработка сообщения из Kafka"""
        try:
            # Парсинг сообщения
            if isinstance(message.value, bytes):
                data = json.loads(message.value.decode('utf-8'))
            else:
                data = message.value
            
            # Обновление счетчиков
            self.total_data_count += 1
            
            # Проверка на late data (если есть поле processing_timestamp и event_timestamp)
            if 'processing_timestamp' in data and 'event_timestamp' in data:
                delay = data['processing_timestamp'] - data['event_timestamp']
                if delay > 300000:  # Более 5 минут задержки
                    self.late_data_count += 1
            
            # Добавление в буфер
            self.data_buffer.append(data)
            
            # Проверка drift при достижении размера буфера или интервала
            current_time = time.time()
            if (len(self.data_buffer) >= self.buffer_size or 
                current_time - self.last_drift_check >= self.drift_check_interval):
                self.check_data_drift()
                self.last_drift_check = current_time
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def check_data_drift(self):
        """Проверка data drift на накопленных данных"""
        if len(self.data_buffer) < 100:  # Минимум 100 записей для анализа
            return
        
        try:
            # Преобразование буфера в DataFrame
            current_data = pd.DataFrame(self.data_buffer)
            
            # Мониторинг drift
            drift_result = self.drift_monitor.monitor_data_drift(current_data)
            
            logger.info(f"Drift check: score={drift_result.get('drift_score', 0):.3f}, "
                       f"detected={drift_result.get('drift_detected', False)}")
            
            # Обновление метрик late data
            late_ratio = self.late_data_count / self.total_data_count if self.total_data_count > 0 else 0.0
            LATE_DATA_RATIO.set(late_ratio)
            
            # Push метрик в Prometheus
            try:
                push_to_gateway(
                    self.prometheus_gateway,
                    job='data_drift_monitor',
                    registry=registry
                )
            except Exception as e:
                logger.warning(f"Failed to push metrics to Prometheus: {e}")
            
            # Очистка буфера (оставляем последние 100 записей для контекста)
            self.data_buffer = self.data_buffer[-100:]
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
    
    def run(self):
        """Запуск сервиса мониторинга"""
        logger.info("Starting monitoring service...")
        logger.info(f"Kafka: {self.kafka_bootstrap}, Topic: {self.kafka_topic}")
        
        consumer = None
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                consumer = KafkaConsumer(
                    self.kafka_topic,
                    bootstrap_servers=self.kafka_bootstrap,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    consumer_timeout_ms=10000,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id='data-drift-monitor'
                )
                logger.info("Connected to Kafka")
                break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Failed to connect to Kafka (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(5)
                else:
                    logger.error("Failed to connect to Kafka after all retries")
                    return
        
        if consumer is None:
            logger.error("Could not create Kafka consumer")
            return
        
        # Основной цикл обработки
        try:
            for message in consumer:
                self.process_message(message)
                
                # Периодическая проверка drift даже если буфер не заполнен
                current_time = time.time()
                if current_time - self.last_drift_check >= self.drift_check_interval:
                    self.check_data_drift()
                    self.last_drift_check = current_time
                    
        except KeyboardInterrupt:
            logger.info("Monitoring service stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            if consumer:
                consumer.close()
            logger.info("Monitoring service stopped")


def main():
    """Точка входа"""
    try:
        service = MonitoringService()
        service.run()
    except Exception as e:
        logger.error(f"Monitoring service failed: {e}")
        raise


if __name__ == '__main__':
    main()

